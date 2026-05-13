from __future__ import annotations

from typing import Any

from hermes_agentic_rl.collectors.conversation_collector import SessionTurnSample
from hermes_agentic_rl.core.types import RewardResult, RewardSummary
from hermes_agentic_rl.rewards.aggregate import weighted_sum
from hermes_agentic_rl.rewards.next_turn_feedback import (
    NextTurnFeedbackConfig,
    score_assistant_toolcall_message,
    score_feedback_messages,
    score_tool_feedback_messages,
    score_user_feedback_messages,
)


def _component_name(raw: Any) -> str:
    return str(raw or "").strip()


def build_session_reward_results(
    sample: SessionTurnSample,
    cfg: dict[str, Any] | None = None,
) -> list[RewardResult]:
    config = cfg or {}
    component_specs = config.get("components")
    if not isinstance(component_specs, list) or not component_specs:
        component_specs = [
            {"name": "session_user_feedback_reward", "weight": 0.5},
            {"name": "session_tool_feedback_reward", "weight": 0.3},
            {"name": "session_toolcall_reward", "weight": 0.2},
        ]

    feedback_cfg = NextTurnFeedbackConfig()
    results: list[RewardResult] = []
    for spec in component_specs:
        if not isinstance(spec, dict):
            continue
        name = _component_name(spec.get("name"))
        weight = float(spec.get("weight", 1.0))
        if name == "session_user_feedback_reward":
            score, reason = score_user_feedback_messages(sample.feedback_messages, feedback_cfg)
        elif name == "session_tool_feedback_reward":
            score, reason = score_tool_feedback_messages(sample.feedback_messages, feedback_cfg)
        elif name == "session_toolcall_reward":
            score, reason = score_assistant_toolcall_message(sample.assistant_message)
        elif name == "next_turn_feedback_reward":
            score, reason = score_feedback_messages(sample.feedback_messages, feedback_cfg)
        else:
            continue
        results.append(
            RewardResult(
                name=name,
                score=float(score),
                reason=reason,
                weight=weight,
                metadata={"turn_index": sample.turn_index, "session_id": sample.session_id},
            )
        )
    return results


def judge_session_turn_sample(
    sample: SessionTurnSample,
    cfg: dict[str, Any] | None = None,
) -> RewardSummary:
    return weighted_sum(build_session_reward_results(sample, cfg))
