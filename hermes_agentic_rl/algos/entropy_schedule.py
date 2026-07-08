"""Entropy regularization scheduler for RL training.

Problem
-------
A fixed ``entropy_coef`` is a poor fit for long training runs:
  - Too large early on → policy never commits to good actions.
  - Too small later → policy collapses prematurely.

This module provides three scheduling strategies:

1. **LinearEntropySchedule**: linearly anneal ``entropy_coef`` from
   ``start`` to ``end`` over ``total_steps`` training steps.

2. **ExponentialEntropySchedule**: ``coef_t = start * decay^t`` with
   optional ``floor`` to prevent full collapse.

3. **TargetEntropyPID**: closed-loop controller. Measures the batch's
   mean per-token entropy (``H_t``) and adjusts ``entropy_coef`` so
   that ``H_t → H_target``.  Uses a simple proportional controller
   (P-only, with clamp). This mirrors SAC's temperature auto-tuning
   but applied to a per-step coefficient rather than a Lagrange
   multiplier.

All three expose the same interface::

    scheduler = LinearEntropySchedule(start=0.05, end=0.001, total_steps=1000)
    coef = scheduler.get_coef(step=t)           # query without update
    coef = scheduler.step(step=t)               # advance internal state (same for these)

    # PID variant also accepts current entropy:
    coef = pid.update(current_entropy=H_t)

Usage in Trainer
----------------
At the start of each training step::

    entropy_coef = scheduler.step(step=global_step)
    algo.cfg.entropy_coef = entropy_coef

Or for PID (after computing the algo stats)::

    entropy_coef = pid.update(current_entropy=stats.entropy)
    algo.cfg.entropy_coef = entropy_coef

The PID scheduler is stateful; the linear/exp schedulers are stateless
(same ``get_coef`` result for the same step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Linear schedule
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LinearEntropySchedule:
    """Linearly anneal ``entropy_coef`` from ``start`` to ``end``."""

    start: float = 0.05
    end: float = 0.0
    total_steps: int = 1000

    def get_coef(self, step: int) -> float:
        if self.total_steps <= 0:
            return self.end
        frac = min(1.0, max(0.0, step / self.total_steps))
        return self.start + frac * (self.end - self.start)

    # alias so all schedulers share the same call interface
    def step(self, step: int) -> float:
        return self.get_coef(step)


# ---------------------------------------------------------------------------
# Exponential schedule
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExponentialEntropySchedule:
    """Exponentially decay ``entropy_coef``.

    ``coef_t = max(floor, start * decay^step)``
    """

    start: float = 0.05
    decay: float = 0.995
    floor: float = 1e-4

    def get_coef(self, step: int) -> float:
        raw = self.start * (self.decay ** max(0, step))
        return max(self.floor, raw)

    def step(self, step: int) -> float:
        return self.get_coef(step)


# ---------------------------------------------------------------------------
# Cosine schedule (warm-up → peak → anneal)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CosineEntropySchedule:
    """Cosine-annealed entropy coef with optional warm-up.

    Phases:
      [0, warmup_steps)       : linear warm-up from 0 → peak
      [warmup_steps, total)   : cosine decay from peak → end
    """

    peak: float = 0.05
    end: float = 0.0
    total_steps: int = 1000
    warmup_steps: int = 0

    def get_coef(self, step: int) -> float:
        if step < self.warmup_steps:
            # linear warm-up
            frac = step / max(1, self.warmup_steps)
            return self.peak * frac
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        t = min(1.0, (step - self.warmup_steps) / decay_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.end + (self.peak - self.end) * cos

    def step(self, step: int) -> float:
        return self.get_coef(step)


# ---------------------------------------------------------------------------
# Target-entropy PID controller
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TargetEntropyPID:
    """PID-style controller that drives policy entropy toward a target.

    The update rule is::

        error   = H_target - H_current
        delta   = kp * error + ki * integral(error) + kd * derivative(error)
        coef_t+1 = clip(coef_t + delta, lo, hi)

    The coefficient increases when entropy is *below* target (encouraging
    more exploration) and decreases when entropy is *above* target.

    Args:
        target_entropy: desired mean per-token entropy (nats).  A common
            heuristic is ``0.5 * log(vocab_size)`` for the initial policy.
        init_coef: starting value of ``entropy_coef``.
        lr: legacy alias for ``kp``.
        kp: proportional gain. If omitted, ``lr`` is used.
        ki: integral gain.
        kd: derivative gain.
        lo: minimum allowed coefficient.
        hi: maximum allowed coefficient.
        max_integral: anti-windup clamp for the accumulated integral error.
    """

    target_entropy: float = 1.0
    init_coef: float = 0.01
    lr: float = 1e-3
    kp: float | None = None
    ki: float = 0.0
    kd: float = 0.0
    lo: float = 0.0
    hi: float = 1.0
    max_integral: float = 10.0
    # internal state — not a constructor arg for external callers
    _coef: float = field(default=0.0, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._coef = self.init_coef

    @property
    def coef(self) -> float:
        return self._coef

    def update(self, current_entropy: float) -> float:
        """Feed the latest batch entropy and return the new coefficient.

        Args:
            current_entropy: mean per-token entropy from the last training
                step (available in ``AlgoUpdateStats.entropy`` if
                ``entropy_coef > 0``, or computed separately).

        Returns:
            Updated ``entropy_coef`` to set on the algo config.
        """
        error = self.target_entropy - current_entropy
        # Anti-windup: cap the integral so that its *contribution*
        # (ki * integral) cannot exceed ``max_integral`` in coefficient units.
        # Without this linkage a large ``ki`` turns the configured
        # ``max_integral`` (which is intended as a coefficient-space bound)
        # into a runaway term that snaps ``_coef`` to its clamp.
        if self.ki > 0.0:
            integral_cap = min(float(self.max_integral), float(self.max_integral) / self.ki)
        else:
            integral_cap = float(self.max_integral)
        self._integral = _clamp(
            self._integral + error,
            -integral_cap,
            integral_cap,
        )
        derivative = 0.0 if self._prev_error is None else error - self._prev_error
        gain_p = self.lr if self.kp is None else self.kp
        delta = gain_p * error + self.ki * self._integral + self.kd * derivative
        self._coef = float(_clamp(self._coef + delta, self.lo, self.hi))
        self._prev_error = error
        self._step += 1
        return self._coef

    def reset(self) -> None:
        self._coef = self.init_coef
        self._step = 0
        self._integral = 0.0
        self._prev_error = None

    # Compatibility with step-based schedulers (ignores step arg).
    def step(self, step: int = 0) -> float:
        return self._coef


def _clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Unified scheduler type alias + factory
# ---------------------------------------------------------------------------

EntropyScheduler = (
    LinearEntropySchedule | ExponentialEntropySchedule | CosineEntropySchedule | TargetEntropyPID
)


def make_entropy_scheduler(
    kind: str = "linear",
    **kwargs: object,
) -> EntropyScheduler:
    """Factory for entropy schedulers.

    Args:
        kind: one of ``"linear"``, ``"exp"``, ``"cosine"``, ``"pid"``.
        **kwargs: forwarded to the dataclass constructor.

    Example::

        sched = make_entropy_scheduler("cosine", peak=0.05, total_steps=2000)
        coef = sched.step(t)
    """
    _MAP = {
        "linear": LinearEntropySchedule,
        "exp": ExponentialEntropySchedule,
        "exponential": ExponentialEntropySchedule,
        "cosine": CosineEntropySchedule,
        "pid": TargetEntropyPID,
    }
    cls = _MAP.get(kind)
    if cls is None:
        raise ValueError(f"Unknown entropy scheduler kind: {kind!r}. Choose from {list(_MAP)}")
    return cls(**kwargs)  # type: ignore[return-value]
