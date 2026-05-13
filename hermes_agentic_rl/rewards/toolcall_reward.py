from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward


def _extract_tool_name(call: dict[str, Any]) -> str | None:
    name = call.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()

    func = call.get("function")
    if isinstance(func, dict):
        func_name = func.get("name")
        if isinstance(func_name, str) and func_name.strip():
            return func_name.strip()

    return None


class ToolcallReward(BaseReward):
    name = "toolcall_reward"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        call_count = sum(len(step.tool_calls) for step in trajectory.steps)
        if call_count == 0:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no tool calls found in trajectory",
                weight=self.weight,
                metadata={"tool_call_count": 0},
            )

        invalid_calls = 0
        for step in trajectory.steps:
            for call in step.tool_calls:
                if _extract_tool_name(call) is None:
                    invalid_calls += 1

        score = max(0.0, 1.0 - (invalid_calls / call_count))
        reason = (
            "all tool calls contain name"
            if invalid_calls == 0
            else "some tool calls are missing name"
        )
        return RewardResult(
            name=self.name,
            score=score,
            reason=reason,
            weight=self.weight,
            metadata={
                "tool_call_count": call_count,
                "invalid_calls": invalid_calls,
            },
        )
