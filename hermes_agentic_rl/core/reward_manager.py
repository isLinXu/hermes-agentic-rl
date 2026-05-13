from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardSummary, Trajectory
from hermes_agentic_rl.rewards.aggregate import weighted_sum
from hermes_agentic_rl.rewards.base import BaseReward


class RewardManager:
    def __init__(self, rewards: list[BaseReward]) -> None:
        self.rewards = rewards

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardSummary:
        results = []
        for reward in self.rewards:
            results.append(await reward.evaluate(item, trajectory, tool_context))
        return weighted_sum(results)
