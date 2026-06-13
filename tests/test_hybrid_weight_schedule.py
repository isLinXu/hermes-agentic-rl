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
from hermes_agentic_rl.trainers.hybrid_trainer import HybridTrainer, HybridTrainerConfig
from hermes_agentic_rl.trainers.on_policy import _scheduled_scalar


def test_scheduled_scalar_linear_and_cosine() -> None:
    linear = {"mode": "linear", "start": 1.0, "end": 0.5, "total_steps": 4}
    assert _scheduled_scalar(linear, default=1.0, iter_idx=0, total_steps=4) == 1.0
    assert _scheduled_scalar(linear, default=1.0, iter_idx=4, total_steps=4) == 0.5

    cosine = {"mode": "cosine", "start": 0.0, "end": 1.0, "total_steps": 2}
    assert _scheduled_scalar(cosine, default=0.0, iter_idx=0, total_steps=2) == 0.0
    assert _scheduled_scalar(cosine, default=0.0, iter_idx=2, total_steps=2) == 1.0


def test_hybrid_trainer_applies_weight_schedules() -> None:
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    trainer = HybridTrainer(
        policy=backend,
        env=EchoTaskEnv(build_default_echo_dataset()),
        reward_manager=RewardManager([EchoRewardComponent(weight=1.0)]),
        cfg=HybridTrainerConfig(
            n_iters=2,
            group_size=2,
            prompts_per_iter=1,
            max_new_tokens=2,
            lr=1e-3,
            log_every=0,
            seed=0,
            w_rl=1.0,
            w_opd=0.0,
            opd_teacher_fill=False,
            w_rl_schedule={"mode": "linear", "start": 1.0, "end": 0.5},
            w_opd_schedule={"mode": "linear", "start": 0.0, "end": 1.0},
        ),
    )

    stats = trainer.train()

    assert stats.iters[0]["w_rl"] == 1.0
    assert stats.iters[0]["w_opd"] == 0.0
    assert stats.iters[1]["w_rl"] == 0.5
    assert stats.iters[1]["w_opd"] == 1.0
    assert stats.iters[1]["w_rl_schedule_active"] == 1.0
    assert stats.iters[1]["w_opd_schedule_active"] == 1.0
