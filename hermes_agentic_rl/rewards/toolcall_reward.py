from __future__ import annotations

import json
import math
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


def _extract_arguments(call: dict[str, Any]) -> tuple[Any, bool]:
    func = call.get("function")
    if isinstance(func, dict) and "arguments" in func:
        return func.get("arguments"), True
    if "arguments" in call:
        return call.get("arguments"), True
    if "args" in call:
        return call.get("args"), True
    return None, False


def _decode_arguments(
    value: Any,
    *,
    present: bool,
) -> tuple[dict[str, Any], float, float, bool, bool]:
    if not present:
        return {}, 0.0, 0.0, False, False
    if isinstance(value, dict):
        return value, 1.0, _value_quality(value), True, True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}, 0.0, 0.0, False, False
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"arg": value}, 0.0, 0.0, False, False
        if isinstance(parsed, dict):
            return parsed, 1.0, _value_quality(parsed), True, True
        return {"arg": parsed}, 0.5, _value_quality(parsed), True, False
    if isinstance(value, list):
        return {"items": value}, 0.5, _value_quality(value), True, False
    return {"arg": value}, 0.25, _value_quality(value), True, False


def _value_quality(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        return 1.0 if value.strip() else 0.0
    if isinstance(value, bool):
        return 1.0
    if isinstance(value, int | float):
        return 1.0 if math.isfinite(float(value)) else 0.0
    if isinstance(value, list):
        if not value:
            return 1.0
        return sum(_value_quality(item) for item in value) / len(value)
    if isinstance(value, dict):
        if not value:
            return 1.0
        key_scores = [1.0 if str(key).strip() else 0.0 for key in value]
        value_scores = [_value_quality(item) for item in value.values()]
        return (sum(key_scores) + sum(value_scores)) / (len(key_scores) + len(value_scores))
    return 1.0


def score_tool_call(call: Any, *, weights: tuple[float, float, float]) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {
            "score": 0.0,
            "name": None,
            "name_score": 0.0,
            "schema_score": 0.0,
            "value_score": 0.0,
            "arguments_present": False,
            "arguments_json_valid": False,
            "arguments_object_like": False,
            "arguments": {},
        }
    name_weight, schema_weight, value_weight = weights
    name = _extract_tool_name(call)
    name_score = 1.0 if name is not None else 0.0
    raw_arguments, arguments_present = _extract_arguments(call)
    arguments, schema_score, value_score, arguments_json_valid, arguments_object_like = (
        _decode_arguments(raw_arguments, present=arguments_present)
    )
    if name_score <= 0.0:
        score = 0.0
    else:
        score = name_weight * name_score + schema_weight * schema_score + value_weight * value_score
    return {
        "score": max(0.0, min(1.0, score)),
        "name": name,
        "name_score": name_score,
        "schema_score": schema_score,
        "value_score": value_score,
        "arguments_present": arguments_present,
        "arguments_json_valid": arguments_json_valid,
        "arguments_object_like": arguments_object_like,
        "arguments": arguments,
    }


def score_tool_calls(
    calls: list[Any],
    *,
    weights: tuple[float, float, float] = (
        _DEFAULT_NAME_WEIGHT,
        _DEFAULT_SCHEMA_WEIGHT,
        _DEFAULT_VALUE_WEIGHT,
    ),
) -> dict[str, Any]:
    if not calls:
        return {
            "score": 0.0,
            "call_count": 0,
            "invalid_calls": 0,
            "missing_name_calls": 0,
            "invalid_argument_json_calls": 0,
            "invalid_argument_schema_calls": 0,
            "low_value_calls": 0,
            "mean_name_score": 0.0,
            "mean_schema_score": 0.0,
            "mean_value_score": 0.0,
            "call_scores": [],
        }

    call_scores = [score_tool_call(call, weights=weights) for call in calls]
    count = len(call_scores)
    scores = [float(row["score"]) for row in call_scores]
    return {
        "score": sum(scores) / count,
        "call_count": count,
        "invalid_calls": sum(1 for score in scores if score < 0.999),
        "missing_name_calls": sum(1 for row in call_scores if row["name_score"] <= 0.0),
        "invalid_argument_json_calls": sum(
            1 for row in call_scores if not row["arguments_json_valid"]
        ),
        "invalid_argument_schema_calls": sum(
            1 for row in call_scores if not row["arguments_object_like"]
        ),
        "low_value_calls": sum(1 for row in call_scores if row["value_score"] < 0.999),
        "mean_name_score": sum(float(row["name_score"]) for row in call_scores) / count,
        "mean_schema_score": sum(float(row["schema_score"]) for row in call_scores) / count,
        "mean_value_score": sum(float(row["value_score"]) for row in call_scores) / count,
        "call_scores": call_scores,
    }


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
        calls = [call for step in trajectory.steps for call in step.tool_calls]
        call_count = len(calls)
        if call_count == 0:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no tool calls found in trajectory",
                weight=self.weight,
                metadata={"tool_call_count": 0},
            )

        summary = score_tool_calls(
            calls,
            weights=(self.name_weight, self.schema_weight, self.value_weight),
        )
        score = float(summary["score"])
        invalid_calls = int(summary["invalid_calls"])
        reason = "all tool calls valid" if invalid_calls == 0 else "some tool calls invalid"
        return RewardResult(
            name=self.name,
            score=score,
            reason=reason,
            weight=self.weight,
            metadata={
                "tool_call_count": call_count,
                "invalid_calls": invalid_calls,
                "missing_name_calls": summary["missing_name_calls"],
                "invalid_argument_json_calls": summary["invalid_argument_json_calls"],
                "invalid_argument_schema_calls": summary["invalid_argument_schema_calls"],
                "low_value_calls": summary["low_value_calls"],
                "mean_name_score": summary["mean_name_score"],
                "mean_schema_score": summary["mean_schema_score"],
                "mean_value_score": summary["mean_value_score"],
            },
        )
