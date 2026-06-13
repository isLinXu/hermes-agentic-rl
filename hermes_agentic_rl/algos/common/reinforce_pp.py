"""REINFORCE++ style per-token advantage for GRPO.

Standard GRPO uses a scalar advantage per rollout, broadcast to all tokens.
This loses fine-grained signal: the model can't tell *which* tokens contributed
to the reward.

REINFORCE++ (Hu 2025) assigns per-token advantages using a learned or
heuristic credit-assignment function. Here we implement a simple heuristic:
tokens after the `<answer>` tag get higher advantage weight than tokens before.

Additionally, we add:
- **Token-level reward decomposition**: for letter_counting, we can identify
  which tokens encode the final number/JSON and assign reward only to those.
- **Return normalization across the whole batch** (not just per-group) as an
  alternative to group normalization — useful when group variance is zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Per-token advantage construction
# ---------------------------------------------------------------------------


def reinforce_plusplus_advantage(
    response_ids: Sequence[int],
    old_logprobs: Sequence[float],
    reward: float,
    *,
    answer_start_id: int | None = None,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> list[float]:
    """REINFORCE++ style per-token advantage.

    When ``answer_start_id`` is provided, tokens from that position onward
    get the full reward as advantage; preceding tokens get exponentially
    decayed advantage (gamma-weighted).

    When ``answer_start_id`` is None, all tokens get the same reward as
    advantage (standard GRPO behavior, but per-token instead of scalar).

    Args:
        response_ids: token ids of the response.
        old_logprobs: per-token logprobs from rollout.
        reward: scalar reward for this rollout.
        answer_start_id: if set, advantage is concentrated on tokens after
            the first occurrence of this id.
        gamma: decay factor for tokens before answer_start_id.
        eps: epsilon for numerical stability.

    Returns:
        Per-token advantages [T].
    """
    T = len(response_ids)
    if T == 0 or reward == 0.0:
        return [0.0] * T

    advs = [0.0] * T

    if answer_start_id is not None:
        # Find the first occurrence of the answer-start token
        ans_pos = None
        for i, tid in enumerate(response_ids):
            if tid == answer_start_id:
                ans_pos = i
                break

        if ans_pos is not None:
            # Tokens from ans_pos onward: full reward
            for i in range(ans_pos, T):
                advs[i] = reward
            # Tokens before: exponentially decayed
            for i in range(ans_pos - 1, -1, -1):
                advs[i] = advs[i + 1] * gamma
        else:
            # No answer tag found: uniform distribution
            for i in range(T):
                advs[i] = reward / max(T, 1)
    else:
        # Uniform per-token advantage
        for i in range(T):
            advs[i] = reward

    return advs


def token_level_advantage_from_regex(
    response_text: str,
    response_ids: Sequence[int],
    reward: float,
    *,
    tokenizer_decode: Callable[[list[int]], str],
    pattern: str = r"<answer>.*?</answer>",
) -> list[float]:
    """Construct per-token advantage by matching a regex in the decoded text.

    Tokens that fall within the regex match get the full reward; others get 0.
    This is more precise than answer_start_id because it handles multi-token
    answer patterns.

    Args:
        response_text: decoded response string.
        response_ids: token ids (for length reference).
        reward: scalar reward.
        tokenizer_decode: function that decodes a slice of token ids → string.
        pattern: regex pattern to match.

    Returns:
        Per-token advantages [T].
    """
    import re

    T = len(response_ids)
    if T == 0 or reward == 0.0:
        return [0.0] * T

    m = re.search(pattern, response_text, re.DOTALL)
    if m is None:
        # No match: distribute reward uniformly as fallback
        return [reward / max(T, 1)] * T

    start_char = m.start()
    end_char = m.end()

    # Find which token indices cover the matched span
    advs = [0.0] * T
    char_pos = 0
    for i in range(T):
        token_text = tokenizer_decode([response_ids[i]])
        token_len = len(token_text)
        token_start = char_pos
        token_end = char_pos + token_len

        # Overlap with match region
        overlap_start = max(token_start, start_char)
        overlap_end = min(token_end, end_char)
        if overlap_end > overlap_start:
            overlap_frac = (overlap_end - overlap_start) / max(token_len, 1)
            advs[i] = reward * overlap_frac

        char_pos = token_end

    return advs


# ---------------------------------------------------------------------------
# Batch-level advantage normalization
# ---------------------------------------------------------------------------


def batch_normalize_advantage(
    rewards: Sequence[float],
    eps: float = 1e-6,
) -> list[float]:
    """Global z-score normalization across the entire batch.

    Unlike group_normalize_advantage (per-prompt), this normalizes across
    all rollouts in the batch. Useful when groups are small or have zero
    variance.

    Returns:
        Normalized advantages (same order as rewards).
    """
    n = len(rewards)
    if n <= 1:
        return [0.0] * n
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var**0.5
    if std < eps:
        return [0.0] * n
    return [(r - mean) / (std + eps) for r in rewards]


def whitened_advantage(
    rewards: Sequence[float],
    eps: float = 1e-6,
) -> list[float]:
    """Whiten advantages: center + scale + clip to [-3, 3].

    This is more aggressive than z-score and helps when rewards are very
    sparse (only 0 or 1).
    """
    n = len(rewards)
    if n <= 1:
        return [0.0] * n
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var**0.5
    if std < eps:
        return [0.0] * n
    advs = [(r - mean) / (std + eps) for r in rewards]
    return [max(-3.0, min(3.0, a)) for a in advs]


def expand_per_token_advantage(
    records_with_adv: list[tuple[Any, list[float]]],
    *,
    answer_start_token_id: int | None = None,
    gamma: float = 0.95,
    eps: float = 1e-6,
) -> list[tuple[Any, list[float]]]:
    """Expand scalar advantages into per-token advantages via REINFORCE++."""
    expanded: list[tuple[Any, list[float]]] = []
    for rec, adv_list in records_with_adv:
        if len(rec.response_ids) == 0 or len(adv_list) == 0:
            expanded.append((rec, adv_list))
            continue
        reward = adv_list[0]
        per_tok = reinforce_plusplus_advantage(
            response_ids=rec.response_ids,
            old_logprobs=rec.old_logprobs,
            reward=reward,
            answer_start_id=answer_start_token_id,
            gamma=gamma,
            eps=eps,
        )
        expanded.append((rec, per_tok))
    return expanded
