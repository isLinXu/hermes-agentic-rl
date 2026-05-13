from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward


class TerminalTaskEnv(BaseEnv):
    def __init__(self, dataset: list[dict[str, Any]]) -> None:
        self.dataset = dataset
        self._index = 0
        self._outcome_reward = OutcomeReward(weight=0.7)
        self._toolcall_reward = ToolcallReward(weight=0.3)

    async def setup(self) -> None:
        self._index = 0

    async def get_next_item(self) -> dict[str, Any]:
        item = self.dataset[self._index % len(self.dataset)]
        self._index += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        return item["instruction"]

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        return [
            await self._outcome_reward.evaluate(item, trajectory, tool_context),
            await self._toolcall_reward.evaluate(item, trajectory, tool_context),
        ]
