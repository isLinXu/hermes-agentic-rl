from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardSummary, Trajectory
from hermes_agentic_rl.trainers.base import BaseTrainer


class TrainerBridge:
    def __init__(self, trainer: BaseTrainer) -> None:
        self.trainer = trainer

    async def submit(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        return await self.trainer.submit(item, trajectory, reward_summary)
