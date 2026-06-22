from __future__ import annotations

from collections import Counter
from typing import Any

from hermes_agentic_rl.collectors.conversation_collector import SessionTurnSample

POSITIVE_FEEDBACK_KEYWORDS = (
    "great",
    "thanks",
    "thank you",
    "success",
    "works",
    "done",
    "correct",
    "很好",
    "成功",
    "通过",
)
NEGATIVE_FEEDBACK_KEYWORDS = (
    "wrong",
    "error",
    "failed",
    "fail",
    "try again",
    "not correct",
    "fix",
    "bug",
    "不对",
    "错误",
    "失败",
    "修复",
)
PROCEDURE_KEYWORDS = (
    "create",
    "created",
    "update",
    "updated",
    "fix",
    "fixed",
    "write",
    "wrote",
    "run",
    "ran",
    "check",
    "checked",
    "implement",
    "implemented",
    "生成",
    "更新",
    "修复",
    "执行",
)


def normalize_replay_mining_config(raw: Any) -> dict[str, Any]:
    if raw is False:
        return {"enabled": False}
    cfg = dict(raw) if isinstance(raw, dict) else {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "positive_reward_threshold": float(cfg.get("positive_reward_threshold", 0.0)),
        "negative_reward_threshold": float(cfg.get("negative_reward_threshold", -0.01)),
        "min_skill_reward": float(cfg.get("min_skill_reward", 0.5)),
        "long_context_messages": max(0, int(cfg.get("long_context_messages", 8))),
        "long_context_chars": max(0, int(cfg.get("long_context_chars", 3000))),
    }


def mine_session_turn_sample(
    sample: SessionTurnSample,
    *,
    reward: float,
    metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    cfg = normalize_replay_mining_config(config)
    if not cfg.get("enabled", True):
        return None

    metadata = metadata or {}
    feedback_text = _messages_text(sample.feedback_messages)
    prompt_text = _messages_text(sample.prompt_messages)
    assistant_text = _message_text(sample.assistant_message)
    full_text = " ".join([prompt_text, assistant_text, feedback_text]).lower()
    feedback_roles = [str(item.get("role", "")) for item in sample.feedback_messages]

    assistant_tool_call = _has_assistant_tool_call(sample.assistant_message)
    tool_feedback = "tool" in feedback_roles
    tool_component = any(
        "tool" in str(component.get("name", "")).lower() and _positive_component_score(component)
        for component in metadata.get("reward_components", [])
        if isinstance(component, dict)
    )
    positive_feedback = reward > float(cfg["positive_reward_threshold"]) or _contains_any(
        feedback_text,
        POSITIVE_FEEDBACK_KEYWORDS,
    )
    negative_feedback = reward < float(cfg["negative_reward_threshold"]) or _contains_any(
        feedback_text,
        NEGATIVE_FEEDBACK_KEYWORDS,
    )
    long_context = _is_long_context(sample, prompt_text, cfg)
    procedural_pattern = _contains_any(full_text, PROCEDURE_KEYWORDS)
    multi_turn_recovery = sample.turn_index > 0 and positive_feedback
    skill_candidate = (
        reward >= float(cfg["min_skill_reward"])
        and positive_feedback
        and procedural_pattern
        and not negative_feedback
    )

    axes: list[str] = []
    reasons: list[str] = []
    recommended_uses: list[str] = []

    if positive_feedback:
        _append_unique(axes, "task_success")
        _append_unique(reasons, "positive_feedback")
        _append_unique(recommended_uses, "positive_replay")
    if assistant_tool_call:
        _append_unique(axes, "tool_use_reliability")
        _append_unique(reasons, "assistant_tool_call")
        _append_unique(recommended_uses, "tool_reliability_replay")
    if tool_feedback:
        _append_unique(axes, "tool_use_reliability")
        _append_unique(reasons, "tool_feedback")
        _append_unique(recommended_uses, "tool_reliability_replay")
    if tool_component:
        _append_unique(axes, "tool_use_reliability")
        _append_unique(reasons, "tool_reward_component")
    if negative_feedback:
        _append_unique(axes, "interaction_control")
        _append_unique(reasons, "negative_feedback")
        _append_unique(recommended_uses, "failure_recovery_replay")
    if multi_turn_recovery:
        _append_unique(axes, "interaction_control")
        _append_unique(reasons, "multi_turn_recovery")
        _append_unique(recommended_uses, "failure_recovery_replay")
    if long_context:
        _append_unique(axes, "prompt_context")
        _append_unique(reasons, "long_context")
        _append_unique(recommended_uses, "context_benchmark_seed")
    if skill_candidate:
        _append_unique(axes, "skill_learning")
        _append_unique(reasons, "skill_candidate")
        _append_unique(recommended_uses, "skill_candidate")
    if reasons:
        _append_unique(axes, "self_evolution_signal")
    if not recommended_uses:
        _append_unique(recommended_uses, "review")

    usefulness_score = _usefulness_score(
        reward=reward,
        axes=axes,
        skill_candidate=skill_candidate,
        negative_feedback=negative_feedback,
    )
    return {
        "axes": axes,
        "reasons": reasons,
        "recommended_uses": recommended_uses,
        "skill_candidate": skill_candidate,
        "usefulness_score": usefulness_score,
        "signals": {
            "assistant_tool_call": assistant_tool_call,
            "tool_feedback": tool_feedback,
            "tool_reward_component": tool_component,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "long_context": long_context,
            "multi_turn_recovery": multi_turn_recovery,
        },
        "context": {
            "prompt_messages": len(sample.prompt_messages),
            "feedback_messages": len(sample.feedback_messages),
            "prompt_chars": len(prompt_text),
        },
    }


def summarize_replay_mining(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = normalize_replay_mining_config(config)
    by_axis: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_recommended_use: Counter[str] = Counter()
    skill_candidates = 0
    usefulness_scores: list[float] = []

    for record in records:
        metadata = record.get("metadata", {})
        mining = metadata.get("replay_mining", {}) if isinstance(metadata, dict) else {}
        if not isinstance(mining, dict) or not mining:
            continue
        for axis in mining.get("axes", []):
            by_axis[str(axis)] += 1
        for reason in mining.get("reasons", []):
            by_reason[str(reason)] += 1
        for use in mining.get("recommended_uses", []):
            by_recommended_use[str(use)] += 1
        if mining.get("skill_candidate"):
            skill_candidates += 1
        try:
            usefulness_scores.append(float(mining.get("usefulness_score", 0.0)))
        except (TypeError, ValueError):
            continue

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "config": cfg,
        "samples_scored": len(usefulness_scores),
        "skill_candidates": skill_candidates,
        "mean_usefulness_score": (
            sum(usefulness_scores) / len(usefulness_scores) if usefulness_scores else 0.0
        ),
        "by_axis": dict(sorted(by_axis.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_recommended_use": dict(sorted(by_recommended_use.items())),
    }


def _has_assistant_tool_call(message: dict[str, Any]) -> bool:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return True
    content = _message_text(message).lower()
    return "<tool_call" in content or ('"arguments"' in content and '"name"' in content)


def _is_long_context(
    sample: SessionTurnSample,
    prompt_text: str,
    cfg: dict[str, Any],
) -> bool:
    message_threshold = int(cfg.get("long_context_messages", 0))
    char_threshold = int(cfg.get("long_context_chars", 0))
    return (message_threshold > 0 and len(sample.prompt_messages) >= message_threshold) or (
        char_threshold > 0 and len(prompt_text) >= char_threshold
    )


def _messages_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_message_text(message) for message in messages)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _positive_component_score(component: dict[str, Any]) -> bool:
    try:
        return float(component.get("score", 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _usefulness_score(
    *,
    reward: float,
    axes: list[str],
    skill_candidate: bool,
    negative_feedback: bool,
) -> float:
    score = max(0.0, float(reward)) + 0.05 * len(axes)
    if skill_candidate:
        score += 0.25
    if negative_feedback:
        score += 0.10
    return round(score, 6)
