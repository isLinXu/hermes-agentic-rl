from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from typing import Any, Literal

from hermes_agentic_rl.collectors.conversation_collector import (
    collect_session_turn_samples,
)
from hermes_agentic_rl.collectors.session_judge import judge_session_turn_sample
from hermes_agentic_rl.core.types import Trajectory

TurnCreditMode = Literal["shared", "terminal", "discounted", "judge", "hybrid"]


@dataclass(slots=True)
class MultiTurnCreditConfig:
    mode: TurnCreditMode = "shared"
    gamma: float = 0.9
    final_reward_weight: float = 1.0
    local_reward_weight: float = 0.0
    judge_weight: float = 1.0
    teacher_weight: float = 1.0
    judge: dict[str, Any] | None = None


def parse_multi_turn_credit_config(cfg: dict[str, Any] | None) -> MultiTurnCreditConfig:
    raw = cfg or {}
    mode = str(raw.get("mode", "shared")).strip().lower() or "shared"
    if mode not in {"shared", "terminal", "discounted", "judge", "hybrid"}:
        raise ValueError(f"unsupported multi_turn_credit.mode: {mode!r}")

    if "final_reward_weight" in raw:
        final_reward_weight = float(raw["final_reward_weight"])
    elif mode == "hybrid":
        final_reward_weight = 0.7
    elif mode == "judge":
        final_reward_weight = 0.0
    else:
        final_reward_weight = 1.0

    if "local_reward_weight" in raw:
        local_reward_weight = float(raw["local_reward_weight"])
    elif mode == "hybrid":
        local_reward_weight = 0.3
    elif mode == "judge":
        local_reward_weight = 1.0
    else:
        local_reward_weight = 0.0

    judge_cfg = raw.get("judge")
    if not isinstance(judge_cfg, dict):
        judge_cfg = None

    gamma = float(raw.get("gamma", 0.9))
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError(
            f"multi_turn_credit.gamma must be a finite non-negative float, got {gamma!r}"
        )
    if not math.isfinite(final_reward_weight):
        raise ValueError("multi_turn_credit.final_reward_weight must be finite")
    if not math.isfinite(local_reward_weight):
        raise ValueError("multi_turn_credit.local_reward_weight must be finite")
    judge_weight = float(raw.get("judge_weight", 1.0))
    teacher_weight = float(raw.get("teacher_weight", 1.0))
    if not math.isfinite(judge_weight) or judge_weight < 0.0:
        raise ValueError("multi_turn_credit.judge_weight must be a finite non-negative float")
    if not math.isfinite(teacher_weight) or teacher_weight < 0.0:
        raise ValueError("multi_turn_credit.teacher_weight must be a finite non-negative float")

    return MultiTurnCreditConfig(
        mode=mode,  # type: ignore[arg-type]
        gamma=gamma,
        final_reward_weight=final_reward_weight,
        local_reward_weight=local_reward_weight,
        judge_weight=judge_weight,
        teacher_weight=teacher_weight,
        judge=judge_cfg,
    )


def assign_multi_turn_rewards(
    trajectory: Trajectory,
    *,
    final_reward: float,
    n_turns: int,
    cfg: dict[str, Any] | None = None,
    teacher_responses: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    parsed = parse_multi_turn_credit_config(cfg)
    safe_turns = max(1, int(n_turns))
    local_scores, local_metadata = _local_turn_scores(
        trajectory,
        n_turns=safe_turns,
        judge_cfg=parsed.judge,
        judge_weight=parsed.judge_weight,
        teacher_weight=parsed.teacher_weight,
        teacher_responses=teacher_responses,
    )

    rows: list[dict[str, Any]] = []
    for turn_index in range(safe_turns):
        final_component = _final_component(
            final_reward=final_reward,
            turn_index=turn_index,
            n_turns=safe_turns,
            cfg=parsed,
        )
        local_component = local_scores[turn_index]
        weighted_final = parsed.final_reward_weight * final_component
        weighted_local = parsed.local_reward_weight * local_component
        reward = weighted_final + weighted_local
        judge_component = float(local_metadata[turn_index].get("judge_score", 0.0))
        teacher_component = float(local_metadata[turn_index].get("teacher_score", 0.0))
        rows.append(
            {
                "reward": float(reward),
                "final_component": float(final_component),
                "local_component": float(local_component),
                "judge_component": judge_component,
                "teacher_component": teacher_component,
                "weighted_final_component": float(weighted_final),
                "weighted_local_component": float(weighted_local),
                "mode": parsed.mode,
                "gamma": float(parsed.gamma),
                "final_reward_weight": float(parsed.final_reward_weight),
                "local_reward_weight": float(parsed.local_reward_weight),
                "judge_weight": float(parsed.judge_weight),
                "teacher_weight": float(parsed.teacher_weight),
                "judge_metadata": dict(local_metadata[turn_index]),
            }
        )
    return rows


def _local_turn_scores(
    trajectory: Trajectory,
    *,
    n_turns: int,
    judge_cfg: dict[str, Any] | None,
    judge_weight: float,
    teacher_weight: float,
    teacher_responses: list[str | None] | None,
) -> tuple[list[float], list[dict[str, Any]]]:
    scores = [0.0] * n_turns
    metadata: list[dict[str, Any]] = [{} for _ in range(n_turns)]

    messages = trajectory.metadata.get("messages")
    if not isinstance(messages, list) or not messages:
        return scores, metadata

    assistant_texts = _assistant_turn_texts(messages, n_turns=n_turns)
    teacher_scores, teacher_metadata = _teacher_turn_scores(
        assistant_texts,
        teacher_responses=teacher_responses,
        n_turns=n_turns,
    )
    judge_scores = [0.0] * n_turns
    judge_metadata_rows: list[dict[str, Any]] = [{} for _ in range(n_turns)]

    runtime_block = trajectory.metadata.get("runtime")
    runtime_task_id = None
    if isinstance(runtime_block, dict):
        runtime_task_id = runtime_block.get("task_id") or runtime_block.get("session_id")
    session_id = str(runtime_task_id or trajectory.task_id or "session")
    samples = collect_session_turn_samples(
        [dict(m) for m in messages if isinstance(m, dict)],
        session_id=session_id,
        task_id=trajectory.task_id,
    )
    for sample in samples:
        if sample.turn_index >= n_turns:
            continue
        summary = judge_session_turn_sample(sample, judge_cfg)
        judge_scores[sample.turn_index] = float(summary.final_score)
        judge_metadata_rows[sample.turn_index] = {
            "session_id": sample.session_id,
            "task_id": sample.task_id,
            "turn_index": sample.turn_index,
            "feedback_roles": list(sample.metadata.get("feedback_roles", [])),
            "reward_components": [
                {
                    "name": component.name,
                    "score": float(component.score),
                    "weight": float(component.weight),
                    "reason": component.reason,
                }
                for component in summary.components
            ],
            "reward_summary": dict(summary.metadata),
        }

    for turn_index in range(n_turns):
        weighted_parts: list[tuple[float, float]] = []
        if judge_weight > 0.0:
            weighted_parts.append((judge_weight, judge_scores[turn_index]))
        if teacher_weight > 0.0 and teacher_metadata[turn_index]:
            weighted_parts.append((teacher_weight, teacher_scores[turn_index]))
        if weighted_parts:
            denom = sum(weight for weight, _score in weighted_parts)
            scores[turn_index] = sum(weight * score for weight, score in weighted_parts) / max(
                1e-8, denom
            )
        metadata[turn_index] = {
            "judge_score": float(judge_scores[turn_index]),
            "teacher_score": float(teacher_scores[turn_index]),
            "teacher_metadata": dict(teacher_metadata[turn_index]),
            **dict(judge_metadata_rows[turn_index]),
        }
    return scores, metadata


def _assistant_turn_texts(messages: list[dict[str, Any]], *, n_turns: int) -> list[str]:
    out: list[str] = []
    for message in messages:
        if str(message.get("role", "")) != "assistant":
            continue
        out.append(str(message.get("content", "") or ""))
        if len(out) >= n_turns:
            break
    while len(out) < n_turns:
        out.append("")
    return out


def _teacher_turn_scores(
    assistant_texts: list[str],
    *,
    teacher_responses: list[str | None] | None,
    n_turns: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    scores = [0.0] * n_turns
    metadata: list[dict[str, Any]] = [{} for _ in range(n_turns)]
    if not teacher_responses:
        return scores, metadata

    for turn_index in range(min(n_turns, len(teacher_responses))):
        expected = teacher_responses[turn_index]
        if not isinstance(expected, str) or not expected.strip():
            continue
        actual = assistant_texts[turn_index]
        similarity = _text_similarity(actual, expected)
        scores[turn_index] = similarity
        metadata[turn_index] = {
            "expected_response": expected,
            "actual_response": actual,
            "similarity": float(similarity),
        }
    return scores, metadata


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _text_similarity(actual: str, expected: str) -> float:
    actual_norm = _normalize_text(actual)
    expected_norm = _normalize_text(expected)
    if not expected_norm:
        return 0.0
    if actual_norm == expected_norm:
        return 1.0
    if expected_norm in actual_norm:
        return 1.0
    return float(difflib.SequenceMatcher(a=actual_norm, b=expected_norm).ratio())


def _final_component(
    *,
    final_reward: float,
    turn_index: int,
    n_turns: int,
    cfg: MultiTurnCreditConfig,
) -> float:
    if cfg.mode == "shared":
        return float(final_reward)
    if cfg.mode == "terminal":
        return float(final_reward if turn_index == (n_turns - 1) else 0.0)
    if cfg.mode in {"discounted", "hybrid"}:
        exponent = max(0, (n_turns - 1) - turn_index)
        return float(final_reward) * (float(cfg.gamma) ** exponent)
    if cfg.mode == "judge":
        return 0.0
    raise ValueError(f"unsupported multi_turn_credit.mode: {cfg.mode!r}")
