from __future__ import annotations

import re
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward

_SUCCESS_MARKERS = {"done", "ok", "okay", "success", "successful", "completed", "complete"}
_SUCCESS_KEYWORDS = ("success", "successfully", "created", "completed", "done", "written")


def _extract_filenames(text: str) -> list[str]:
    """
    保守提取形如 "hello.txt" 或 "notes/todo.md" 的 token。

    注意：这里是启发式，不追求严格文件路径语义；用于 OutcomeReward 的短成功标记兼容。
    """

    return re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8}", text)


def _looks_successful(final_output: str, instruction: str | None) -> bool:
    low = final_output.lower()
    if any(keyword in low for keyword in _SUCCESS_KEYWORDS):
        return True

    if instruction:
        instruction_tokens = _extract_filenames(instruction)
        low_instruction_tokens = {t.lower() for t in instruction_tokens if t}
        for token in low_instruction_tokens:
            if token and token in low:
                return True

    return False


class OutcomeReward(BaseReward):
    name = "outcome_reward"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        expected_output = item.get("expected_output")
        if expected_output is None:
            score = 1.0 if trajectory.final_output else 0.0
            reason = "no expected_output provided; used final_output presence"
        elif trajectory.final_output == expected_output:
            score = 1.0
            reason = "final_output matched expected_output"
        elif isinstance(expected_output, str) and expected_output.strip().lower() in _SUCCESS_MARKERS:
            if trajectory.final_output and _looks_successful(trajectory.final_output, item.get("instruction")):
                score = 1.0
                reason = "expected_output is a success marker; inferred success from final_output"
            else:
                score = 0.0
                reason = "expected_output is a success marker; final_output did not look successful"
        else:
            score = 0.0
            reason = "final_output mismatched expected_output"

        return RewardResult(
            name=self.name,
            score=score,
            reason=reason,
            weight=self.weight,
        )
