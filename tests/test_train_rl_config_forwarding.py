from __future__ import annotations

import asyncio

from hermes_agentic_rl.backends.base import GenerationOutput
from hermes_agentic_rl.cli.train_rl import (
    _build_backend,
    _build_grpo,
    _build_ppo,
    _make_agent_loop_factory,
)


class _FakeTokenizer:
    vocab_size = 8
    pad_id = 0
    eos_id = 1
    bos_id = 2

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [3 if ch else 4 for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        _ = ids
        return "</tool_call>"


class _FakeBackend:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.calls: list[dict] = []

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        return GenerationOutput(response_ids=[3, 4], logprobs=[-0.1, -0.2], finished=True)


def test_build_grpo_forwards_sft_and_advantage_knobs() -> None:
    cfg = {
        "train_rl": {
            "advantage_eps": 1e-5,
            "interleave_sft_every": 3,
            "interleave_sft_samples": 7,
            "interleave_sft_lr": 2e-4,
            "interleave_sft_epochs": 4,
            "interleave_sft_batch_size": 5,
            "bootstrap_sft_rounds": 2,
            "bootstrap_sft_samples": 9,
            "bootstrap_sft_lr": 3e-4,
            "bootstrap_sft_epochs": 6,
            "batch_generate": True,
            "target_kl": 0.03,
            "adaptive_kl": True,
            "normalize_reward": True,
            "grad_accum_steps": 2,
            "amp_dtype": "fp32",
            "save_best_checkpoint": True,
        }
    }

    tcfg = _build_grpo(cfg, None)

    assert tcfg.advantage_eps == 1e-5
    assert tcfg.interleave_sft_every == 3
    assert tcfg.interleave_sft_samples == 7
    assert tcfg.interleave_sft_lr == 2e-4
    assert tcfg.interleave_sft_epochs == 4
    assert tcfg.interleave_sft_batch_size == 5
    assert tcfg.bootstrap_sft_rounds == 2
    assert tcfg.bootstrap_sft_samples == 9
    assert tcfg.bootstrap_sft_lr == 3e-4
    assert tcfg.bootstrap_sft_epochs == 6
    assert tcfg.batch_generate is True
    assert tcfg.target_kl == 0.03
    assert tcfg.adaptive_kl is True
    assert tcfg.normalize_reward is True
    assert tcfg.grad_accum_steps == 2
    assert tcfg.amp_dtype == "fp32"
    assert tcfg.save_best_checkpoint is True


def test_build_ppo_forwards_sft_and_batch_generate_knobs() -> None:
    cfg = {
        "train_rl": {
            "batch_generate": True,
            "interleave_sft_every": 3,
            "interleave_sft_samples": 7,
            "interleave_sft_lr": 2e-4,
            "interleave_sft_epochs": 4,
            "interleave_sft_batch_size": 5,
            "bootstrap_sft_rounds": 2,
            "bootstrap_sft_samples": 9,
            "bootstrap_sft_lr": 3e-4,
            "bootstrap_sft_epochs": 6,
            "target_kl": 0.02,
            "adaptive_kl": True,
            "normalize_reward": True,
            "grad_accum_steps": 3,
            "amp_dtype": "fp32",
        }
    }

    tcfg = _build_ppo(cfg, None)

    assert tcfg.batch_generate is True
    assert tcfg.interleave_sft_every == 3
    assert tcfg.interleave_sft_samples == 7
    assert tcfg.interleave_sft_lr == 2e-4
    assert tcfg.interleave_sft_epochs == 4
    assert tcfg.interleave_sft_batch_size == 5
    assert tcfg.bootstrap_sft_rounds == 2
    assert tcfg.bootstrap_sft_samples == 9
    assert tcfg.bootstrap_sft_lr == 3e-4
    assert tcfg.bootstrap_sft_epochs == 6
    assert tcfg.target_kl == 0.02
    assert tcfg.adaptive_kl is True
    assert tcfg.normalize_reward is True
    assert tcfg.grad_accum_steps == 3
    assert tcfg.amp_dtype == "fp32"


def test_build_tiny_backend_forwards_device_and_dtype() -> None:
    cfg = {
        "backend": {
            "name": "tiny",
            "dim": 16,
            "n_heads": 2,
            "n_layers": 1,
            "max_len": 32,
            "device": "cpu",
            "dtype": "float32",
            "seed": 0,
        }
    }

    backend = _build_backend(cfg, need_value_head=False)

    assert str(backend.cfg.device) == "cpu"
    assert backend.cfg.dtype == "float32"


def test_policy_agent_loop_forwards_stop_strings() -> None:
    cfg = {
        "agent_loop": {
            "type": "policy",
            "stop_strings": ["</tool_call>"],
        },
        "train_rl": {
            "max_new_tokens": 12,
            "temperature": 0.7,
        },
    }

    factory = _make_agent_loop_factory(cfg)
    loop = factory(backend=_FakeBackend(), seed=7)

    asyncio.run(loop.run("Solve it"))

    assert loop.stop_strings == ["</tool_call>"]
    assert loop.backend.calls[0]["stop_strings"] == ["</tool_call>"]
