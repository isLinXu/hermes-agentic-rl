"""Tests for the atropos env adapter.

We keep these opt-in: if atroposlib / transformers aren't importable the
whole file is skipped. On Box's shipped env (with torch + transformers + the
atropos/ sibling directory) they run as part of the normal pytest pass.

Three tiers covered:

1. Public imports survive without atroposlib (sanity).
2. ``HermesAPIServer`` serves all three atropos entry points end-to-end.
3. ``AtroposEnvAdapter`` bridges a custom atropos env → hermes trainer
   interface, with ``AtroposRewardComponent`` producing meaningful scores.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# 1) Make the sibling ./atropos source directory importable without pip install -e.
ATROPOS_SRC = Path(__file__).resolve().parents[1] / "atropos"
if ATROPOS_SRC.exists() and str(ATROPOS_SRC) not in sys.path:
    sys.path.insert(0, str(ATROPOS_SRC))

# 2) Always-on import: the integration module itself must be importable
#    even if atroposlib is absent (it defers the heavy imports to first use).
from hermes_agentic_rl.integrations import (
    AtroposEnvAdapter,
    AtroposEnvAdapterConfig,
    AtroposRewardComponent,
    AtroposUnavailableError,
    HermesAPIServer,
)


def test_public_symbols_are_importable() -> None:
    # sanity: these are classes / callables
    assert callable(HermesAPIServer)
    assert isinstance(AtroposUnavailableError, type) and issubclass(
        AtroposUnavailableError, RuntimeError
    )
    assert AtroposEnvAdapter is not None
    assert AtroposRewardComponent is not None


# ---------------------------------------------------------------------------
# Heavy tiers — require atroposlib (+ openai types) + transformers + torch
# ---------------------------------------------------------------------------

atroposlib = pytest.importorskip("atroposlib.envs.server_handling.server_baseline")
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

try:  # the env-level tier also needs this — guarded separately below
    from atroposlib.envs.base import BaseEnv as _AtroposBase
    from atroposlib.envs.base import BaseEnvConfig as _AtroposBaseEnvCfg
    _HAS_BASEENV = True
except Exception:
    _HAS_BASEENV = False


from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend


@pytest.fixture(scope="module")
def hf_tokenizer():
    """Module-scoped GPT-2 tokenizer (cached by HF)."""
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def tiny_backend():
    return TinyCausalLMBackend(
        TinyBackendConfig(seed=0, dim=16, n_heads=2, n_layers=1, max_len=64)
    )


# --- HermesAPIServer ---------------------------------------------------------


def test_hermes_api_server_chat_completion_shape(tiny_backend, hf_tokenizer):
    srv = HermesAPIServer(
        backend=tiny_backend,
        hf_tokenizer=hf_tokenizer,
        model_name="tiny",
        max_new_tokens=4,
    )
    resp = asyncio.run(
        srv._chat_completion_wrapper(
            messages=[{"role": "user", "content": "Hello"}],
            n=3,
            max_tokens=3,
            temperature=0.0,
            seed=11,
        )
    )
    assert len(resp.choices) == 3
    for ch in resp.choices:
        assert ch.message.role == "assistant"
        assert isinstance(ch.message.content, str)
        assert ch.finish_reason in {"stop", "length"}
    assert resp.usage.prompt_tokens > 0


def test_hermes_api_server_tokens_and_logprobs_alignment(tiny_backend, hf_tokenizer):
    srv = HermesAPIServer(
        backend=tiny_backend,
        hf_tokenizer=hf_tokenizer,
        model_name="tiny",
        max_new_tokens=5,
    )
    pt, ot, olp, fr = asyncio.run(
        srv._tokens_and_logprobs_completion_wrapper(
            prompt="Q: 2+2=?", n=2, max_tokens=4, temperature=1.0, seed=7,
        )
    )
    assert isinstance(pt, list) and len(pt) >= 1
    assert len(ot) == 2 == len(olp) == len(fr)
    # Bridge path: tokens must match logprobs length exactly.
    for toks, lps in zip(ot, olp, strict=False):
        assert len(toks) == len(lps) >= 1
    # All token ids must live in the HF vocab (since backend has its own vocab
    # but the bridge re-encodes to HF vocab for atropos compatibility).
    vocab = hf_tokenizer.vocab_size
    for toks in ot:
        assert all(0 <= t < vocab for t in toks)


def test_hermes_api_server_classic_completion(tiny_backend, hf_tokenizer):
    srv = HermesAPIServer(
        backend=tiny_backend, hf_tokenizer=hf_tokenizer, max_new_tokens=3
    )
    resp = asyncio.run(
        srv._completion_wrapper(prompt="once upon a", n=1, max_tokens=3, temperature=0.0)
    )
    assert len(resp.choices) == 1
    assert isinstance(resp.choices[0].text, str)


# --- AtroposEnvAdapter + a self-contained mini atropos env ------------------


@pytest.mark.skipif(not _HAS_BASEENV, reason="atroposlib.envs.base deps not installed")
def test_adapter_wraps_mini_atropos_env(tiny_backend, hf_tokenizer):
    """Build a tiny atropos BaseEnv inline and verify the adapter bridges it."""
    from atroposlib.envs.base import BaseEnv, BaseEnvConfig

    class EchoAtroposEnv(BaseEnv):
        name = "mini-echo"

        async def setup(self):
            self._items = [
                ("say yes", "yes"),
                ("say no", "no"),
                ("say maybe", "maybe"),
            ]
            self._idx = 0

        async def get_next_item(self):
            item = self._items[self._idx % len(self._items)]
            self._idx += 1
            return item

        async def collect_trajectory(self, item):
            return None, []

        async def evaluate(self, *args, **kwargs):
            return None

        async def score(self, rollout_group_data):
            # rollout_group_data: [(messages, gold)]
            (messages, gold) = rollout_group_data[0]
            reply = messages[-1].get("content", "")
            s = 1.0 if str(gold).lower() in reply.lower() else 0.0
            return {"tokens": [[]], "masks": [[]], "scores": [s]}

    env_cfg = BaseEnvConfig(
        group_size=2,
        max_token_length=32,
        tokenizer_name="gpt2",
    )
    adapter = AtroposEnvAdapter(
        env_cls=EchoAtroposEnv,
        env_config=env_cfg,
        backend=tiny_backend,
        adapter_config=AtroposEnvAdapterConfig(
            max_new_tokens=6,
            override_tokenizer=hf_tokenizer,  # skip HF download
        ),
    )
    # The adapter must:
    #   1. proxy setup/get_next_item
    #   2. expose an HF-compatible tokenizer to the hermes side
    #   3. have hot-swapped atropos_env.server.servers to our HermesAPIServer
    asyncio.run(adapter.setup())
    item = asyncio.run(adapter.get_next_item())
    assert "task_id" in item and "instruction" in item and "raw" in item
    assert item["instruction"] == "say yes"

    # Adapter exposes an HF-compatible tokenizer protocol to hermes.
    # This tokenizer is for *atropos-vocab* token ids (GPT-2 here); the
    # tiny backend lives in its own vocab, so we don't cross them here.
    htok = adapter.tokenizer
    assert htok.vocab_size == hf_tokenizer.vocab_size
    ids = htok.encode("hi")
    assert all(0 <= i < htok.vocab_size for i in ids)

    # Hot-swap verification: the atropos env's server manager must now
    # route through our in-process HermesAPIServer.
    assert len(adapter.atropos_env.server.servers) == 1
    assert type(adapter.atropos_env.server.servers[0]).__name__ == "HermesAPIServer"

    # score_rollout must succeed and return a float in [0, 1]
    hit = asyncio.run(adapter.score_rollout(item, "yes"))
    assert hit == 1.0
    miss = asyncio.run(adapter.score_rollout(item, "nothing"))
    assert miss == 0.0


@pytest.mark.skipif(not _HAS_BASEENV, reason="atroposlib.envs.base deps not installed")
def test_atropos_reward_component_plugs_into_reward_manager(tiny_backend, hf_tokenizer):
    from atroposlib.envs.base import BaseEnv, BaseEnvConfig

    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.core.types import Trajectory

    class KeywordEnv(BaseEnv):
        name = "kw"

        async def setup(self):
            self._i = 0

        async def get_next_item(self):
            self._i += 1
            return ("please include the word apple", "apple")

        async def collect_trajectory(self, item):
            return None, []

        async def evaluate(self, *a, **k):
            return None

        async def score(self, group):
            (messages, gold) = group[0]
            txt = messages[-1]["content"]
            s = 1.0 if str(gold) in txt else 0.0
            return {"scores": [s]}

    cfg = BaseEnvConfig(group_size=1, max_token_length=32, tokenizer_name="gpt2")
    adapter = AtroposEnvAdapter(
        env_cls=KeywordEnv,
        env_config=cfg,
        backend=tiny_backend,
        adapter_config=AtroposEnvAdapterConfig(
            max_new_tokens=4, override_tokenizer=hf_tokenizer
        ),
    )
    asyncio.run(adapter.setup())
    item = asyncio.run(adapter.get_next_item())

    rm = RewardManager([AtroposRewardComponent(adapter, weight=1.0)])

    traj_hit = Trajectory(
        task_id=item["task_id"], prompt=item["instruction"], steps=[],
        final_output="an apple a day", finished_naturally=True, turns_used=1,
        metadata={},
    )
    traj_miss = Trajectory(
        task_id=item["task_id"], prompt=item["instruction"], steps=[],
        final_output="banana", finished_naturally=True, turns_used=1,
        metadata={},
    )
    s_hit = asyncio.run(rm.evaluate(item, traj_hit, tool_context=None)).final_score
    s_miss = asyncio.run(rm.evaluate(item, traj_miss, tool_context=None)).final_score
    assert s_hit > s_miss
    assert s_hit == 1.0 and s_miss == 0.0
