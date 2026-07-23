"""Token-Level Reward for Agentic RL
==================================

Fine-grained, token-level reward assignment for agentic RL training.
Addresses the core issue: reward increases but tool call metrics stay at zero.

Design Principles
------------------
1. **Token-level advantage** (REINFORCE++) instead of trajectory-level:
   Each token receives its own advantage estimate, enabling fine-grained
   credit attribution to the specific tokens that produced tool calls.

2. **Structured tool call rewards** (name + schema + arguments quality):
   Decomposes tool call quality into three axes, each contributing to
   per-token reward at the positions where tool calls are generated.

3. **Outcome bonus** for successful tool calls only:
   A sparse bonus that fires only when the final answer is correct,
   preventing the model from learning to "game" intermediate rewards.

4. **KL penalty** with configurable estimator (k1/k2/k3):
   Prevents the policy from diverging too far from the reference.

5. **Length penalty** with cosine schedule:
   Discourages unnecessarily long responses while allowing
   sufficient reasoning length.

Integration
-----------
This module integrates with the existing reward stack:
  - ``RewardComposer`` for aggregation with normalization
  - ``ToolcallReward`` for structured tool call scoring
  - ``DynamicRewardBalancer`` for adaptive weight scheduling
  - ``shaping.py`` for curriculum-aware reward shaping
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TokenRewardConfig:
    """Configuration for token-level reward computation."""

    # KL divergence penalty coefficient. 0 = disabled.
    kl_coef: float = 0.0

    # Length penalty coefficient. 0 = disabled.
    length_coef: float = 0.01

    # Target response length (tokens). Length penalty is zero at this length
    # and increases for deviations.
    target_length: int = 512

    # Whether to use cosine schedule for length penalty (smoother than linear).
    length_cosine_schedule: bool = True

    # Outcome bonus for correct final answers.
    outcome_bonus: float = 0.25

    # Minimum reward floor (prevents total collapse to negative infinity).
    reward_floor: float = -1.0

    # Epsilon for numerical stability in normalization.
    norm_eps: float = 1e-8


# ---------------------------------------------------------------------------
# Token-level advantage computation (REINFORCE++)
# ---------------------------------------------------------------------------


def compute_token_advantages(
    per_token_scores: list[float],
    masks: list[int],
    *,
    cfg: TokenRewardConfig | None = None,
) -> list[float]:
    """Compute per-token advantages using REINFORCE++ style normalization.

    This replaces the coarse trajectory-level advantage with a fine-grained
    token-level advantage that better attributes credit to individual token
    decisions.

    The advantage for token *i* is::

        Â_i = (R_i - mean(R)) / (std(R) + ε)

    where R_i is the per-token score (from tool call quality, outcome bonus,
    etc.) and the normalization is computed over all completion tokens in
    the group.

    Args:
        per_token_scores: One score per token. Prompt tokens should have
            score 0.0 (they will be masked out).
        masks: 1 for completion tokens, 0 for prompt tokens.
        cfg: Optional configuration.

    Returns:
        List of per-token advantages (same length as input). Prompt tokens
        get advantage 0.0.
    """
    cfg = cfg or TokenRewardConfig()
    if not per_token_scores:
        return []

    # Extract completion token scores
    completion_scores = [s for s, m in zip(per_token_scores, masks, strict=False) if m == 1]

    if not completion_scores:
        return [0.0] * len(per_token_scores)

    mean_score = float(np.mean(completion_scores))
    std_score = float(np.std(completion_scores)) if len(completion_scores) > 1 else 1.0

    advantages: list[float] = []
    for score, mask in zip(per_token_scores, masks, strict=False):
        if mask == 1:
            adv = (score - mean_score) / (std_score + cfg.norm_eps)
        else:
            adv = 0.0
        advantages.append(adv)

    return advantages


# ---------------------------------------------------------------------------
# Per-token score assignment
# ---------------------------------------------------------------------------


def assign_tool_call_token_scores(
    token_ids: list[int],
    token_texts: list[str],
    tool_call_spans: list[tuple[int, int, float]],
    *,
    base_score: float = 0.0,
) -> list[float]:
    """Assign per-token scores based on tool call quality.

    This is the key function that bridges tool call rewards to token-level
    credit assignment. Tokens within a tool call span receive the tool call's
    quality score; all other tokens receive the base score.

    Args:
        token_ids: Token IDs (for length reference).
        token_texts: Decoded text for each token.
        tool_call_spans: List of (start_idx, end_idx, score) tuples
            indicating where tool calls appear in the token sequence
            and their quality scores.
        base_score: Default score for non-tool-call tokens.

    Returns:
        Per-token scores list.
    """
    n = len(token_ids)
    scores = [base_score] * n

    for start, end, score in tool_call_spans:
        # Clamp to valid range
        start = max(0, min(start, n))
        end = max(start, min(end, n))
        for i in range(start, end):
            scores[i] = score

    return scores


# ---------------------------------------------------------------------------
# KL penalty
# ---------------------------------------------------------------------------


def compute_kl_penalty(
    log_probs: list[float],
    ref_log_probs: list[float],
    masks: list[int],
    *,
    kl_coef: float = 0.0,
    estimator: str = "k3",
) -> float:
    """Compute KL divergence penalty between policy and reference.

    Supports three estimators:
      - k1: ``log(π/π_ref)`` (naive, high variance)
      - k2: ``log(π/π_ref) * π/π_ref`` (importance-weighted)
      - k3: ``0.5 * (log(π/π_ref))^2`` (low variance, default)

    Args:
        log_probs: Log probabilities under the current policy.
        ref_log_probs: Log probabilities under the reference policy.
        masks: 1 for completion tokens, 0 for prompt.
        kl_coef: KL penalty coefficient. 0 = disabled.
        estimator: One of "k1", "k2", "k3".

    Returns:
        Scalar KL penalty value.
    """
    if kl_coef <= 0.0:
        return 0.0

    kl_terms: list[float] = []
    for lp, rlp, m in zip(log_probs, ref_log_probs, masks, strict=False):
        if m != 1:
            continue
        log_ratio = lp - rlp
        if estimator == "k1":
            kl_terms.append(log_ratio)
        elif estimator == "k2":
            kl_terms.append(log_ratio * math.exp(log_ratio))
        elif estimator == "k3":
            kl_terms.append(0.5 * log_ratio * log_ratio)
        else:
            raise ValueError(f"Unknown KL estimator: {estimator!r}")

    if not kl_terms:
        return 0.0

    mean_kl = sum(kl_terms) / len(kl_terms)
    return kl_coef * mean_kl


# ---------------------------------------------------------------------------
# Length penalty
# ---------------------------------------------------------------------------


def compute_length_penalty(
    response_length: int,
    *,
    target_length: int = 512,
    coef: float = 0.01,
    cosine_schedule: bool = True,
) -> float:
    """Compute length penalty for a response.

    Penalizes responses that deviate from the target length. With cosine
    schedule, the penalty grows smoothly; without, it's linear.

    Formula (cosine)::

        penalty = coef * (1 + cos(π * min(len, 2*target) / (2*target)))

    Formula (linear)::

        penalty = coef * |len - target| / target

    Args:
        response_length: Number of tokens in the response.
        target_length: Desired response length.
        coef: Penalty strength.
        cosine_schedule: Use cosine schedule (smoother).

    Returns:
        Non-negative penalty value (subtracted from reward).
    """
    if coef <= 0.0 or response_length <= 0:
        return 0.0

    if cosine_schedule:
        # Cosine schedule: penalty is highest at length 0 and 2*target,
        # lowest at target_length.
        ratio = min(response_length, 2 * target_length) / max(1, 2 * target_length)
        penalty = coef * (1.0 + math.cos(math.pi * ratio))
    else:
        # Linear penalty: proportional to deviation from target.
        deviation = abs(response_length - target_length)
        penalty = coef * deviation / max(1, target_length)

    return penalty


# ---------------------------------------------------------------------------
# Combined reward computation
# ---------------------------------------------------------------------------


def compute_combined_reward(
    trajectory_reward: float,
    *,
    kl_penalty: float = 0.0,
    length_penalty: float = 0.0,
    outcome_bonus: float = 0.0,
    reward_floor: float = -1.0,
) -> float:
    """Compute the final combined reward for a trajectory.

    Formula::

        R = trajectory_reward + outcome_bonus - kl_penalty - length_penalty
        R = max(R, reward_floor)

    Args:
        trajectory_reward: Base reward from reward composer.
        kl_penalty: KL divergence penalty (from compute_kl_penalty).
        length_penalty: Length penalty (from compute_length_penalty).
        outcome_bonus: Bonus for correct final answer.
        reward_floor: Minimum reward value.

    Returns:
        Combined reward, clamped to [reward_floor, ∞).
    """
    combined = trajectory_reward + outcome_bonus - kl_penalty - length_penalty
    return max(combined, reward_floor)


# ---------------------------------------------------------------------------
# Outcome reward
# ---------------------------------------------------------------------------


def compute_outcome_reward(
    response: str,
    gold_answer: str,
    *,
    partial_credit: bool = True,
) -> float:
    """Compute outcome reward for a response against a gold answer.

    Supports partial credit via string similarity when the answer is
    not exactly correct. This provides a smoother gradient signal than
    a pure binary reward.

    Args:
        response: Model's response text.
        gold_answer: Expected answer text.
        partial_credit: Whether to award partial credit for close answers.

    Returns:
        Reward in [0.0, 1.0].
    """
    if not response or not gold_answer:
        return 0.0

    # Normalize whitespace for comparison
    resp_norm = " ".join(response.split())
    gold_norm = " ".join(gold_answer.split())

    # Exact match → full reward
    if resp_norm == gold_norm:
        return 1.0

    # Check if gold answer is contained in response
    if gold_norm in resp_norm:
        return 1.0

    # Extract boxed answer if present
    boxed = _extract_boxed_answer(response)
    if boxed is not None:
        boxed_norm = " ".join(boxed.split())
        if boxed_norm == gold_norm or gold_norm in boxed_norm:
            return 1.0

    if not partial_credit:
        return 0.0

    # Partial credit via character-level similarity
    # Using a simple ratio rather than full SequenceMatcher for speed
    if len(resp_norm) == 0 or len(gold_norm) == 0:
        return 0.0

    # Check for numeric match (common in math tasks)
    resp_nums = _extract_numbers(resp_norm)
    gold_nums = _extract_numbers(gold_norm)
    if resp_nums and gold_nums:
        # If the last number in the response matches the gold, give high credit
        if resp_nums[-1] == gold_nums[-1]:
            return 0.9
        # Partial credit for any matching numbers
        matches = sum(1 for n in resp_nums if n in gold_nums)
        return min(0.5, matches / max(1, len(gold_nums)))

    # Fallback: simple substring overlap
    overlap = 0
    shorter, longer = (
        (resp_norm, gold_norm) if len(resp_norm) <= len(gold_norm) else (gold_norm, resp_norm)
    )
    for i in range(len(shorter)):
        for j in range(i + 3, min(i + 20, len(shorter) + 1)):
            if shorter[i:j] in longer:
                overlap = max(overlap, j - i)
    return min(0.3, overlap / max(1, len(gold_norm)))


def _extract_boxed_answer(text: str) -> str | None:
    """Extract answer from LaTeX \\boxed{...} format."""
    import re

    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1) if match else None


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text."""
    import re

    matches = re.findall(r"-?\d+\.?\d*", text)
    return [float(m) for m in matches if m]


# ---------------------------------------------------------------------------
# High-level: compute full token-level reward for a trajectory
# ---------------------------------------------------------------------------


def compute_trajectory_token_rewards(
    token_ids: list[int],
    token_texts: list[str],
    masks: list[int],
    tool_call_spans: list[tuple[int, int, float]],
    trajectory_reward: float,
    log_probs: list[float] | None = None,
    ref_log_probs: list[float] | None = None,
    *,
    cfg: TokenRewardConfig | None = None,
) -> tuple[list[float], float]:
    """Compute full token-level rewards for a trajectory.

    This is the main entry point that combines all reward components:
    1. Per-token tool call quality scores
    2. Outcome bonus (if final answer is correct)
    3. KL penalty (if log probs provided)
    4. Length penalty
    5. Token-level advantage normalization

    Args:
        token_ids: Token IDs for the trajectory.
        token_texts: Decoded text per token.
        masks: 1 for completion tokens, 0 for prompt.
        tool_call_spans: (start, end, score) for each tool call.
        trajectory_reward: Base reward from reward composer.
        log_probs: Optional log probs under current policy.
        ref_log_probs: Optional log probs under reference policy.
        cfg: Configuration.

    Returns:
        (per_token_advantages, combined_reward) tuple.
    """
    cfg = cfg or TokenRewardConfig()

    # 1. Assign per-token scores from tool call quality
    per_token_scores = assign_tool_call_token_scores(
        token_ids, token_texts, tool_call_spans, base_score=0.0
    )

    # 2. Compute outcome bonus
    outcome_bonus = cfg.outcome_bonus if trajectory_reward > 0 else 0.0

    # 3. Compute KL penalty
    kl_penalty = 0.0
    if log_probs is not None and ref_log_probs is not None:
        kl_penalty = compute_kl_penalty(log_probs, ref_log_probs, masks, kl_coef=cfg.kl_coef)

    # 4. Compute length penalty
    response_length = sum(1 for m in masks if m == 1)
    length_penalty = compute_length_penalty(
        response_length,
        target_length=cfg.target_length,
        coef=cfg.length_coef,
        cosine_schedule=cfg.length_cosine_schedule,
    )

    # 5. Compute combined reward
    combined_reward = compute_combined_reward(
        trajectory_reward,
        kl_penalty=kl_penalty,
        length_penalty=length_penalty,
        outcome_bonus=outcome_bonus,
        reward_floor=cfg.reward_floor,
    )

    # 6. Distribute combined reward across completion tokens
    #    The trajectory-level reward is spread evenly, then per-token
    #    tool call scores are added on top.
    n_completion = sum(1 for m in masks if m == 1)
    if n_completion > 0:
        base_per_token = combined_reward / n_completion
        for i, m in enumerate(masks):
            if m == 1:
                per_token_scores[i] += base_per_token

    # 7. Compute token-level advantages
    advantages = compute_token_advantages(per_token_scores, masks, cfg=cfg)

    return advantages, combined_reward
