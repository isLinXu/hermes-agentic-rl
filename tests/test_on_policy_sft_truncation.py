from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import EchoRewardComponent, EchoTaskEnv
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


def test_bootstrap_sft_truncates_overlong_sequences_for_small_backends() -> None:
    env = EchoTaskEnv(
        [
            {
                "task_id": "echo-long-1",
                "instruction": "Say: " + ("A" * 180),
                "target": "B" * 120,
            }
        ]
    )
    reward_manager = RewardManager([EchoRewardComponent(weight=1.0)])
    backend = TinyCausalLMBackend(
        TinyBackendConfig(
            dim=16,
            n_heads=2,
            n_layers=2,
            max_len=64,
            seed=0,
        )
    )
    trainer = GRPOTrainer(
        policy=backend,
        env=env,
        reward_manager=reward_manager,
        cfg=GRPOTrainerConfig(
            n_iters=0,
            bootstrap_sft_rounds=1,
            bootstrap_sft_samples=1,
            bootstrap_sft_lr=5e-4,
            bootstrap_sft_epochs=1,
        ),
    )

    stats = trainer.train()

    assert len(stats.iters) == 1
    assert stats.iters[0]["algo"] == "sft_bootstrap"
    assert stats.iters[0]["n_sft_samples"] == 1
