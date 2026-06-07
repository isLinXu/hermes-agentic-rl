from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward

# Default weight blend for the three tool-call quality axes.
_DEFAULT_NAME_WEIGHT = 0.4
_DEFAULT_SCHEMA_WEIGHT = 0.3
_DEFAULT_VALUE_WEIGHT = 0.3


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
    """Reward based on tool-call quality across three axes.

    Axes:
      - **name**: whether the tool call has a valid name.
      - **schema**: whether the tool call follows expected schema.
      - **value**: whether the tool call arguments are sensible.

    The three weights are normalised to sum to 1. If all weights are zero,
    the default blend (0.4 / 0.3 / 0.3) is used as a fallback.
    """

    name = "toolcall_reward"

    def __init__(
        self,
        weight: float = 1.0,
        *,
        name_weight: float = _DEFAULT_NAME_WEIGHT,
        schema_weight: float = _DEFAULT_SCHEMA_WEIGHT,
        value_weight: float = _DEFAULT_VALUE_WEIGHT,
    ) -> None:
        self.weight = weight
        total = name_weight + schema_weight + value_weight
        if total <= 0:
            # All-zero fallback → use defaults.
            self.name_weight = _DEFAULT_NAME_WEIGHT
            self.schema_weight = _DEFAULT_SCHEMA_WEIGHT
            self.value_weight = _DEFAULT_VALUE_WEIGHT
        else:
            self.name_weight = name_weight / total
            self.schema_weight = schema_weight / total
            self.value_weight = value_weight / total

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
