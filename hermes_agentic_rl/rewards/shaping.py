"""Built-in reward shaping helpers for OnPolicyTrainer.

Pass any of these (or a custom function) as ``reward_shaping_fn`` to
``OnPolicyTrainer``/``GRPOTrainer``/``PPOTrainer``::

    from hermes_agentic_rl.rewards.shaping import length_penalty_shaping

    trainer = GRPOTrainer(
        ...,
        reward_shaping_fn=length_penalty_shaping(
            target_len=64,
            penalty_coef=0.01,
        ),
    )

Shaping functions have signature::

    fn(records: list[RolloutRecord]) -> list[RolloutRecord]

They mutate ``record.reward`` in-place and optionally store the raw reward
under ``record.metadata["pre_shaping_reward"]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

RolloutRecord = Any   # avoid circular import; duck-typed


# ---------------------------------------------------------------------------
# Length penalty
# ---------------------------------------------------------------------------


def length_penalty_shaping(
    target_len: int = 64,
    penalty_coef: float = 0.01,
    use_response_tokens: bool = True,
) -> Callable[[list[Any]], list[Any]]:
    """Penalize responses that deviate from ``target_len`` tokens.

    Penalty formula::

        shaped_reward = reward - penalty_coef * |response_len - target_len|

    Args:
        target_len: desired response length in tokens (or chars).
        penalty_coef: strength of the penalty per-token deviation.
        use_response_tokens: if True, measure ``len(record.response_ids)``;
            otherwise measure ``record.metadata.get("response_tokens", 0)``.
    """
    def _shape(records: list[Any]) -> list[Any]:
        for rec in records:
            if use_response_tokens:
                resp_len = len(rec.response_ids)
            else:
                resp_len = int(rec.metadata.get("response_tokens", len(rec.response_ids)))
            penalty = penalty_coef * abs(resp_len - target_len)
            rec.metadata.setdefault("pre_shaping_reward", float(rec.reward))
            rec.metadata["length_penalty"] = float(penalty)
            rec.reward = float(rec.reward) - penalty
        return records

    return _shape


# ---------------------------------------------------------------------------
# Format bonus (regex / substring match)
# ---------------------------------------------------------------------------


def format_bonus_shaping(
    match_fn: Callable[[str], bool],
    bonus: float = 0.1,
    key: str = "format_bonus",
) -> Callable[[list[Any]], list[Any]]:
    """Add a flat bonus when the response matches a format predicate.

    Args:
        match_fn: callable that takes the decoded response text and returns
            True if the format is correct.
        bonus: reward bonus to add (can be negative to penalize).
        key: metadata key for logging.

    Example — reward JSON responses::

        import json

        def is_json(text):
            try:
                json.loads(text)
                return True
            except Exception:
                return False

        trainer = GRPOTrainer(
            ...,
            reward_shaping_fn=format_bonus_shaping(is_json, bonus=0.2),
        )
    """
    def _shape(records: list[Any]) -> list[Any]:
        for rec in records:
            # Try to get decoded text from metadata, fall back to empty string.
            text: str = rec.metadata.get("final_output") or ""
            matched = False
            try:
                matched = bool(match_fn(text))
            except Exception:
                pass
            rec.metadata.setdefault("pre_shaping_reward", float(rec.reward))
            rec.metadata[key] = float(bonus) if matched else 0.0
            if matched:
                rec.reward = float(rec.reward) + bonus
        return records

    return _shape


# ---------------------------------------------------------------------------
# Composite: chain multiple shapers
# ---------------------------------------------------------------------------


def compose_shaping(
    *shapers: Callable[[list[Any]], list[Any]],
) -> Callable[[list[Any]], list[Any]]:
    """Apply multiple shaping functions sequentially.

    Example::

        fn = compose_shaping(
            length_penalty_shaping(target_len=64, penalty_coef=0.005),
            format_bonus_shaping(is_json, bonus=0.1),
        )
        trainer = GRPOTrainer(..., reward_shaping_fn=fn)
    """
    def _shape(records: list[Any]) -> list[Any]:
        for shaper in shapers:
            records = list(shaper(records))
        return records

    return _shape
