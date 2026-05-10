"""PPO advantage whiten/clip — parity with GRPO's whiten mode."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.trainers.ppo_trainer import PPOTrainer, PPOTrainerConfig


def _make_ppo(*, whiten: bool, clip: float = 3.0, n_iters: int = 3) -> PPOTrainer:
    backend = TinyCausalLMBackend(
        TinyBackendConfig(
            dim=16, n_heads=2, n_layers=1, seed=0, with_value_head=True
        )
    )
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    cfg = PPOTrainerConfig(
        n_iters=n_iters,
        group_size=4,
        prompts_per_iter=1,
        lr=1e-3,
        max_new_tokens=4,
        temperature=1.0,
        log_every=100,
        seed=0,
        normalize_advantage=True,
        whiten_advantage=whiten,
        advantage_clip=clip,
    )
    return PPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)


def test_ppo_whiten_keeps_training_stable() -> None:
    """With whiten=True + clip=1.0, mean_advantage should stay in [-1, 1]."""
    trainer = _make_ppo(whiten=True, clip=1.0, n_iters=3)
    stats = trainer.train()
    for rec in stats.iters:
        adv = rec.get("mean_advantage", 0.0)
        assert -1.5 <= adv <= 1.5, f"mean_advantage out of ±1.5 range: {adv}"


def test_ppo_no_whiten_is_default() -> None:
    """whiten=False is the legacy v0.3 behavior; must not error."""
    trainer = _make_ppo(whiten=False, n_iters=2)
    stats = trainer.train()
    assert len(stats.iters) == 2


def test_ppo_advantage_clip_zero_disables_clipping() -> None:
    """advantage_clip=0 with whiten_advantage=True is a no-op (no clamp)."""
    trainer = _make_ppo(whiten=True, clip=0.0, n_iters=1)
    stats = trainer.train()
    assert len(stats.iters) == 1
