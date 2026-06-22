"""Token-Level Reward Component for RewardComposer
=================================================

This module provides a ``BaseReward`` implementation that integrates
the token-level reward computation from ``token_level_reward.py`` into
the ``RewardComposer`` framework.

This bridges the gap between:
  - Fine-grained token-level credit assignment (per-token advantages)
  - The existing reward composition pipeline (RewardComposer + BaseReward)

The component extracts tool call spans from the trajectory, computes
per-token scores, and returns a trajectory-level summary score that
can be combined with other reward components.
"""

from __future__ import annotations

import json
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward
from hermes_agentic_rl.rewards.token_level_reward import (
    TokenRewardConfig,
    compute_combined_reward,
    compute_length_penalty,
    compute_outcome_reward,
)


def _extract_tool_call_spans_from_trajectory(
    trajectory: Trajectory,
) -> list[tuple[int, int, float]]:
    """Extract tool call spans with quality scores from a trajectory.

    Returns list of (start_token_idx, end_token_idx, quality_score)
    tuples indicating where tool calls appear in the token sequence.
    """
    spans: list[tuple[int, int, float]] = []
    token_offset = 0

    for step in trajectory.steps:
        for call in step.tool_calls:
            # Estimate token span from the tool call's string representation
            call_str = json.dumps(call, ensure_ascii=False) if isinstance(call, dict) else str(call)
            # Rough estimate: ~4 chars per token for English, ~2 for CJK
            estimated_tokens = max(1, len(call_str) // 3)

            # Score the tool call quality
            quality = _score_single_tool_call(call)

            spans.append((token_offset, token_offset + estimated_tokens, quality))
            token_offset += estimated_tokens

    return spans


def _score_single_tool_call(call: Any) -> float:
    """Score a single tool call's quality.

    Returns a score in [0.0, 1.0] based on:
    - Has a valid name: +0.4
    - Has valid arguments (JSON): +0.3
    - Arguments are well-structured: +0.3
    """
    if not isinstance(call, dict):
        return 0.0

    score = 0.0

    # Check name
    name = call.get("name")
    func = call.get("function")
    if (isinstance(name, str) and name.strip()) or (
        isinstance(func, dict) and isinstance(func.get("name"), str) and func["name"].strip()
    ):
        score += 0.4

    # Check arguments
    raw_args = None
    if isinstance(func, dict) and "arguments" in func:
        raw_args = func["arguments"]
    elif "arguments" in call:
        raw_args = call["arguments"]

    if raw_args is None:
        return score

    # Arguments present
    score += 0.1

    # Arguments are valid JSON
    if isinstance(raw_args, dict):
        score += 0.2
        # Arguments are well-structured (non-empty dict)
        if raw_args:
            score += 0.3
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args.strip())
            score += 0.2
            if isinstance(parsed, dict) and parsed:
                score += 0.3
        except json.JSONDecodeError:
            pass

    return min(1.0, score)


class TokenLevelReward(BaseReward):
    """Token-level reward component for RewardComposer integration.

    This component evaluates a trajectory by:
    1. Extracting tool call spans from the trajectory
    2. Computing per-token quality scores at tool call positions
    3. Computing outcome reward (final answer correctness)
    4. Computing length penalty
    5. Combining all components into a single score

    The resulting score reflects both the quality of tool calls
    (fine-grained) and the overall outcome (coarse-grained).

    Usage with RewardComposer::

        from hermes_agentic_rl.rewards.composer import RewardComposer
        from hermes_agentic_rl.rewards.token_level_reward_component import TokenLevelReward

        composer = RewardComposer(
            components=[
                TokenLevelReward(weight=0.5),
                ToolcallReward(weight=0.3),
                OutcomeReward(weight=0.2),
            ],
            config={"normalize": {"token_level_reward": True}},
        )
    """

    name = "token_level_reward"

    def __init__(
        self,
        weight: float = 1.0,
        *,
        cfg: TokenRewardConfig | None = None,
    ) -> None:
        self.weight = weight
        self.cfg = cfg or TokenRewardConfig()

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        # 1. Extract tool call spans with quality scores
        tool_call_spans = _extract_tool_call_spans_from_trajectory(trajectory)

        # 2. Compute tool call quality score (average of span scores)
        if tool_call_spans:
            tool_quality = sum(score for _, _, score in tool_call_spans) / len(tool_call_spans)
        else:
            tool_quality = 0.0

        # 3. Compute outcome reward
        gold_answer = str(item.get("expected_output", item.get("answer", "")))
        final_output = trajectory.final_output or ""
        outcome = compute_outcome_reward(final_output, gold_answer, partial_credit=True)

        # 4. Compute length penalty
        response_length = sum(len(step.assistant_message or "") for step in trajectory.steps)
        # Rough token estimate
        estimated_tokens = max(1, response_length // 3)
        length_pen = compute_length_penalty(
            estimated_tokens,
            target_length=self.cfg.target_length,
            coef=self.cfg.length_coef,
            cosine_schedule=self.cfg.length_cosine_schedule,
        )

        # 5. Compute combined reward
        # Base trajectory reward is a blend of tool quality and outcome
        trajectory_reward = 0.5 * tool_quality + 0.5 * outcome

        combined = compute_combined_reward(
            trajectory_reward,
            length_penalty=length_pen,
            outcome_bonus=self.cfg.outcome_bonus if outcome >= 0.9 else 0.0,
            reward_floor=self.cfg.reward_floor,
        )

        # Build metadata for logging
        n_tool_calls = len(tool_call_spans)
        n_valid_calls = sum(1 for _, _, s in tool_call_spans if s >= 0.9)

        reason = (
            f"tool_quality={tool_quality:.3f}, outcome={outcome:.3f}, "
            f"length_pen={length_pen:.3f}, combined={combined:.3f}"
        )

        return RewardResult(
            name=self.name,
            score=combined,
            reason=reason,
            weight=self.weight,
            metadata={
                "tool_call_count": n_tool_calls,
                "valid_tool_calls": n_valid_calls,
                "tool_quality": float(tool_quality),
                "outcome_score": float(outcome),
                "length_penalty": float(length_pen),
                "estimated_tokens": estimated_tokens,
                "tool_call_spans": [
                    {"start": s, "end": e, "score": float(sc)} for s, e, sc in tool_call_spans
                ],
            },
        )
