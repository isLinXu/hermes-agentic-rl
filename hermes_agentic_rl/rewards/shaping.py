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


# ---------------------------------------------------------------------------
# Curriculum-aware shaping: coefficients anneal with training progress
# ---------------------------------------------------------------------------


def curriculum_shaping(
    inner: Callable[[list[Any]], list[Any]],
    *,
    warmup_iters: int = 0,
    cooldown_start: int | None = None,
    total_iters: int = 0,
    min_scale: float = 0.0,
) -> Callable[[list[Any]], list[Any]]:
    """Wrap any shaping function so its effect scales with training progress.

    Timeline::

        [0, warmup_iters)          : scale ramps 0 → 1 (shaping fades in)
        [warmup_iters, cooldown)   : scale = 1 (full shaping)
        [cooldown, total_iters)    : scale ramps 1 → min_scale (shaping fades out)
        [total_iters, ∞)           : scale = min_scale

    This enables curricula like "start with strong length penalties, then relax
    them as the model learns to generate concisely."

    The inner shaper receives the original records; curriculum_shaping rescales
    the *delta* between original and shaped rewards by the current scale factor.

    Args:
        inner: the base shaping function to wrap.
        warmup_iters: iterations over which to ramp shaping from 0 → 1.
        cooldown_start: iteration at which to start fading out. None = no fade.
        total_iters: total training iterations (for fade-out endpoint).
        min_scale: minimum scale factor (0 = shaping fully disabled at end).

    Example::

        fn = curriculum_shaping(
            length_penalty_shaping(target_len=64, penalty_coef=0.01),
            warmup_iters=5,
            cooldown_start=50,
            total_iters=100,
            min_scale=0.1,
        )
    """
    cooldown_start = cooldown_start if cooldown_start is not None else total_iters

    def _shape(records: list[Any]) -> list[Any]:
        # Determine current iteration from the first record's metadata, falling
        # back to 0 (which puts us in warm-up → scale=0 → no shaping).
        it = 0
        for rec in records:
            candidate = rec.metadata.get("_trainer_iter")
            if isinstance(candidate, (int, float)):
                it = int(candidate)
                break

        # Compute scale factor.
        if it < warmup_iters:
            scale = it / max(1, warmup_iters)
        elif it < cooldown_start:
            scale = 1.0
        elif it < total_iters:
            fade_len = max(1, total_iters - cooldown_start)
            scale = 1.0 - (1.0 - min_scale) * ((it - cooldown_start) / fade_len)
        else:
            scale = min_scale

        if scale <= 0.0:
            return records

        # Snapshot original rewards, apply inner shaping, then rescale delta.
        orig_rewards = [float(rec.reward) for rec in records]
        shaped = inner(records)
        for rec, orig_r in zip(shaped, orig_rewards, strict=False):
            delta = float(rec.reward) - orig_r
            rec.reward = orig_r + delta * scale
            rec.metadata["curriculum_shaping_scale"] = scale
        return shaped

    return _shape


# ---------------------------------------------------------------------------
# Progress-aware shaping: coefficients depend on running reward stats
# ---------------------------------------------------------------------------


def adaptive_shaping(
    inner: Callable[[list[Any]], list[Any]],
    *,
    target_reward: float = 1.0,
    gain: float = 1.0,
) -> Callable[[list[Any]], list[Any]]:
    """Wrap a shaping function so its effect scales inversely with reward.

    When the mean reward is far below ``target_reward``, shaping is strong
    (guides exploration). As reward approaches the target, shaping fades
    (lets the intrinsic reward dominate). Formula::

        scale = gain * max(0, 1 - mean_reward / target_reward)

    This is useful for "training wheels" shaping (e.g. format bonuses) that
    should disappear once the model has learned the basic behavior.

    Args:
        inner: the base shaping function to wrap.
        target_reward: reward level at which shaping becomes zero.
        gain: multiplier on the scale factor.
    """
    def _shape(records: list[Any]) -> list[Any]:
        if not records:
            return records
        mean_r = sum(float(rec.reward) for rec in records) / len(records)
        scale = gain * max(0.0, 1.0 - mean_r / target_reward)
        if scale <= 0.0:
            return records

        orig_rewards = [float(rec.reward) for rec in records]
        shaped = inner(records)
        for rec, orig_r in zip(shaped, orig_rewards, strict=False):
            delta = float(rec.reward) - orig_r
            rec.reward = orig_r + delta * scale
            rec.metadata["adaptive_shaping_scale"] = scale
        return shaped

    return _shape
