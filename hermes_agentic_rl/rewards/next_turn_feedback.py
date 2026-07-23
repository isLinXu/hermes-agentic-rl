from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hermes_agentic_rl.collectors.trajectory_adapter import (
    trajectory_to_session_turn_samples,
)
from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward
from hermes_agentic_rl.rewards.toolcall_reward import score_tool_calls


@dataclass(slots=True)
class NextTurnFeedbackConfig:
    positive_markers: tuple[str, ...] = (
        "thanks",
        "thank you",
        "great",
        "correct",
        "works",
        "solved",
        "nice",
        "good",
    )
    negative_markers: tuple[str, ...] = (
        "wrong",
        "incorrect",
        "doesn't work",
        "does not work",
        "failed",
        "failure",
        "bad",
        "not what i asked",
        "try again",
    )
    tool_error_markers: tuple[str, ...] = (
        "error",
        "exception",
        "traceback",
        "not found",
        "failed",
    )
    tool_success_markers: tuple[str, ...] = (
        "ok",
        "success",
        "done",
        "completed",
    )
    missing_feedback_score: float = 0.0
    tool_success_bonus: float = 0.25
    tool_error_penalty: float = -0.5
    positive_user_score: float = 1.0
    negative_user_score: float = -1.0


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def score_user_feedback_messages(
    feedback_messages: list[dict[str, Any]],
    cfg: NextTurnFeedbackConfig | None = None,
) -> tuple[float, str]:
    cfg = cfg or NextTurnFeedbackConfig()
    score = 0.0
    reasons: list[str] = []
    for message in feedback_messages:
        if str(message.get("role", "")) != "user":
            continue
        text = _stringify_content(message.get("content", "")).lower()
        if any(marker in text for marker in cfg.negative_markers):
            score += cfg.negative_user_score
            reasons.append("user_negative")
        elif any(marker in text for marker in cfg.positive_markers):
            score += cfg.positive_user_score
            reasons.append("user_positive")
    if not reasons:
        return cfg.missing_feedback_score, "user_feedback_neutral"
    return max(-1.0, min(1.0, score)), ",".join(reasons)


def score_tool_feedback_messages(
    feedback_messages: list[dict[str, Any]],
    cfg: NextTurnFeedbackConfig | None = None,
) -> tuple[float, str]:
    cfg = cfg or NextTurnFeedbackConfig()
    score = 0.0
    reasons: list[str] = []
    for message in feedback_messages:
        if str(message.get("role", "")) != "tool":
            continue
        text = _stringify_content(message.get("content", "")).lower()
        if any(marker in text for marker in cfg.tool_error_markers):
            score += cfg.tool_error_penalty
            reasons.append("tool_error")
        elif any(marker in text for marker in cfg.tool_success_markers):
            score += cfg.tool_success_bonus
            reasons.append("tool_success")
    if not reasons:
        return cfg.missing_feedback_score, "tool_feedback_neutral"
    return max(-1.0, min(1.0, score)), ",".join(reasons)


def score_assistant_toolcall_message(assistant_message: dict[str, Any]) -> tuple[float, str]:
    tool_calls = assistant_message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return 0.0, "no_tool_calls"

    summary = score_tool_calls(tool_calls)
    score = float(summary["score"])
    if int(summary["invalid_calls"]) == 0:
        return score, "valid_tool_calls"
    if int(summary["invalid_argument_json_calls"]) > 0:
        return score, "invalid_tool_call_json"
    if int(summary["invalid_argument_schema_calls"]) > 0:
        return score, "invalid_tool_call_schema"
    return score, "invalid_tool_calls"


def score_feedback_messages(
    feedback_messages: list[dict[str, Any]],
    cfg: NextTurnFeedbackConfig | None = None,
) -> tuple[float, str]:
    cfg = cfg or NextTurnFeedbackConfig()
    if not feedback_messages:
        return cfg.missing_feedback_score, "no_followup_feedback"

    user_score, user_reason = score_user_feedback_messages(feedback_messages, cfg)
    tool_score, tool_reason = score_tool_feedback_messages(feedback_messages, cfg)
    score = user_score + tool_score
    reasons = [reason for reason in [user_reason, tool_reason] if "neutral" not in reason]
    if not reasons:
        return cfg.missing_feedback_score, "feedback_neutral"
    score = max(-1.0, min(1.0, score))
    return score, ",".join(reasons)


class NextTurnFeedbackReward(BaseReward):
    name = "next_turn_feedback_reward"

    def __init__(
        self,
        *,
        weight: float = 1.0,
        config: NextTurnFeedbackConfig | None = None,
    ) -> None:
        self.weight = weight
        self.config = config or NextTurnFeedbackConfig()

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        samples = trajectory_to_session_turn_samples(trajectory)
        if not samples:
            return RewardResult(
                name=self.name,
                score=self.config.missing_feedback_score,
                reason="no_session_turn_samples",
                weight=self.weight,
            )

        per_turn_scores: list[float] = []
        reasons: list[str] = []
        for sample in samples:
            score, reason = score_feedback_messages(sample.feedback_messages, self.config)
            per_turn_scores.append(score)
            reasons.append(f"turn{sample.turn_index}:{reason}")

        final_score = sum(per_turn_scores) / len(per_turn_scores)
        return RewardResult(
            name=self.name,
            score=final_score,
            reason=";".join(reasons),
            weight=self.weight,
            metadata={"turn_scores": per_turn_scores, "n_turns": len(samples)},
        )
