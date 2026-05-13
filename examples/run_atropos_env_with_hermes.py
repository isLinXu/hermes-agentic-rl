"""End-to-end example: run a real atropos environment with the hermes GRPO trainer.

Zero HTTP server, zero vLLM, zero GPU — all in-process on CPU. The
``HermesAPIServer`` satisfies atropos's ``APIServer`` contract by
delegating generation to any ``LLMBackend`` (here: ``TinyCausalLM``).

Usage:

    # Make the Atropos subproject importable (no pip install needed)
    PYTHONPATH=./subprojects/atropos python examples/run_atropos_env_with_hermes.py

Swap ``LettersEnv`` for ``atroposlib/environments/gsm8k_server.py::GSM8kEnv``
to train on real GSM8K (requires the extra deps: datasets/math-verify).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Auto-inject the Atropos subproject path so this script is zero-install.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_ATROPOS_SRC = _ROOT / "subprojects" / "atropos"
if _ATROPOS_SRC.exists() and str(_ATROPOS_SRC) not in sys.path:
    sys.path.insert(0, str(_ATROPOS_SRC))

from atroposlib.envs.base import BaseEnv, BaseEnvConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from hermes_agentic_rl.backends.tiny import (  # noqa: E402
    TinyBackendConfig,
    TinyCausalLMBackend,
)
from hermes_agentic_rl.core.reward_manager import RewardManager  # noqa: E402
from hermes_agentic_rl.integrations import (  # noqa: E402
    AtroposEnvAdapter,
    AtroposEnvAdapterConfig,
    AtroposRewardComponent,
)
from hermes_agentic_rl.trainers import GRPOTrainer, GRPOTrainerConfig  # noqa: E402


class LettersEnv(BaseEnv):
    """Dense-reward toy env: reward = fraction of vowels in the response."""

    name = "letters"

    async def setup(self):
        self._i = 0

    async def get_next_item(self):
        self._i += 1
        return ("write anything", None)

    async def collect_trajectory(self, item):  # unused by hermes path
        return None, []

    async def evaluate(self, *args, **kwargs):
        return None

    async def score(self, rollout_group_data):
        (messages, _) = rollout_group_data[0]
        text = messages[-1]["content"] if messages else ""
        if not text:
            return {"scores": [0.0]}
        vowels = sum(1 for ch in text.lower() if ch in "aeiou")
        return {"scores": [vowels / max(1, len(text))]}


def main() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    backend = TinyCausalLMBackend(
        TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2)
    )
    env_cfg = BaseEnvConfig(
        group_size=4, max_token_length=32, tokenizer_name="gpt2"
    )
    adapter = AtroposEnvAdapter(
        env_cls=LettersEnv,
        env_config=env_cfg,
        backend=backend,
        adapter_config=AtroposEnvAdapterConfig(
            max_new_tokens=8, override_tokenizer=tok
        ),
    )
    rm = RewardManager([AtroposRewardComponent(adapter, weight=1.0)])
    trainer = GRPOTrainer(
        backend,
        adapter,
        rm,
        cfg=GRPOTrainerConfig(
            n_iters=6,
            group_size=4,
            prompts_per_iter=1,
            lr=0.01,
            max_new_tokens=8,
            log_every=1,
            seed=0,
        ),
    )
    stats = trainer.train()

    print("\n== atropos env x hermes GRPO - training summary ==")
    for r in stats.iters:
        print(
            f"  iter={r['iter']}  "
            f"mean_reward={r['mean_reward']:.4f}  "
            f"policy_loss={r['policy_loss']:.5f}  "
            f"n_updated={r['n_updated']}"
        )


if __name__ == "__main__":
    main()
