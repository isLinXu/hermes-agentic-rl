from __future__ import annotations

from types import MethodType
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.base import RolloutBatch
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.echo_task_env import EchoTaskEnv, build_default_echo_dataset
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


class _DenseTokenReward:
    name = "dense_token_reward"
    weight = 1.0

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del item, tool_context
        rl_meta = trajectory.metadata["runtime"]["rl"]
        n_tokens = len(rl_meta["response_ids"])
        rl_meta["token_rewards"] = [0.25 for _ in range(n_tokens)]
        return RewardResult(
            name=self.name,
            score=1.0,
            reason="dense",
            weight=self.weight,
        )


def _make_trainer(*, batch_generate: bool = False) -> GRPOTrainer:
    policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    return GRPOTrainer(
        policy=policy,
        env=EchoTaskEnv(build_default_echo_dataset()),
        reward_manager=RewardManager([_DenseTokenReward()]),
        cfg=GRPOTrainerConfig(
            n_iters=1,
            group_size=4,
            prompts_per_iter=1,
            max_new_tokens=4,
            log_every=100,
            batch_generate=batch_generate,
        ),
    )


def _capture_update_batch(trainer: GRPOTrainer) -> list[RolloutBatch]:
    batches: list[RolloutBatch] = []
    original = trainer.algo.compute_loss

    def _spy(self, policy, ref_policy, batch):
        batches.append(batch)
        return original(policy, ref_policy, batch)

    trainer.algo.compute_loss = MethodType(_spy, trainer.algo)
    return batches


def test_trainer_copies_reward_token_rewards_to_rollout_records() -> None:
    trainer = _make_trainer()
    batches = _capture_update_batch(trainer)

    trainer.train()

    assert batches
    for record in batches[0].records:
        assert record.metadata["token_rewards"] == [0.25] * len(record.response_ids)


def test_batch_rollout_copies_reward_token_rewards_to_rollout_records() -> None:
    trainer = _make_trainer(batch_generate=True)
    batches = _capture_update_batch(trainer)

    trainer.train()

    assert batches
    for record in batches[0].records:
        assert record.metadata["token_rewards"] == [0.25] * len(record.response_ids)
