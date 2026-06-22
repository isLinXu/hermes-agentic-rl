"""Fast, network-free smoke tests using TinyCausalLM only.

These tests exist so CI and local dev loops do not depend on HuggingFace model
downloads (GPT-2, etc.). They cover the algorithm registry, config validation,
and a minimal train-rl path end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.algos.ppo import PPO, PPOConfig
from hermes_agentic_rl.algos.rloo import RLOOAlgo, RLOOConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.config import ConfigValidationError, load_config, validate_config
from hermes_agentic_rl.utils.coerce import coerce_float


def _tiny_backend(*, with_value_head: bool = False) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(
            dim=16,
            n_heads=2,
            n_layers=1,
            max_len=32,
            seed=0,
            with_value_head=with_value_head,
        )
    )


def _record(reward: float = 1.0, *, group_id: str = "g0") -> RolloutRecord:
    return RolloutRecord(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        old_logprobs=[-0.1, -0.2, -0.3],
        reward=reward,
        group_id=group_id,
    )


@pytest.mark.parametrize(
    "algo_cls,cfg,needs_value_head",
    [
        (GRPO, GRPOConfig(kl_coef=0.0), False),
        (RLOOAlgo, RLOOConfig(), False),
        (PPO, PPOConfig(), True),
    ],
)
def test_algo_compute_loss_smoke(algo_cls, cfg, needs_value_head) -> None:
    backend = _tiny_backend(with_value_head=needs_value_head)
    algo = algo_cls(cfg)
    batch = RolloutBatch([_record(reward=1.0), _record(reward=0.0)])
    loss, stats = algo.compute_loss(backend, None, batch)
    assert loss.requires_grad or float(loss.item()) == 0.0
    assert stats.n_records == 2


def test_coerce_float_handles_bad_values() -> None:
    assert coerce_float("1.5") == pytest.approx(1.5)
    assert coerce_float(None, default=0.0) == 0.0
    assert coerce_float("not-a-number", default=-1.0) == -1.0


def test_load_config_validates_echo_mvp() -> None:
    cfg_path = Path("configs/echo_grpo_mvp.yaml")
    if not cfg_path.exists():
        pytest.skip("echo_grpo_mvp.yaml not present")
    cfg = load_config(cfg_path)
    assert "runtime" in cfg
    assert cfg["runtime"]["integration"] == "fake"


def test_validate_config_rejects_bad_integration() -> None:
    with pytest.raises(ConfigValidationError):
        validate_config({"runtime": {"integration": "unknown", "max_agent_turns": 1}})


def test_train_rl_tiny_grpo_one_iter(tmp_path: Path) -> None:
    """End-to-end train-rl without HF weights or network."""
    from hermes_agentic_rl.cli.train_rl import run_train_rl

    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        """
backend:
  name: tiny
  dim: 16
  n_heads: 2
  n_layers: 1
  max_len: 32
  seed: 0
environment:
  type: echo
train_rl:
  n_iters: 1
  group_size: 2
  prompts_per_iter: 1
  lr: 0.01
  max_new_tokens: 4
  log_every: 1
  save_every: 0
  seed: 0
""",
        encoding="utf-8",
    )
    rc = run_train_rl(str(config_path), output_dir=str(tmp_path / "out"))
    assert rc == 0
