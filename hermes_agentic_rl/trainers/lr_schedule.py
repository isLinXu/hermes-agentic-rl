"""Learning-rate schedulers for on-policy RL training.

Three strategies, all sharing the same step-based interface:

1. **ConstantLR**     — no change (legacy behaviour, the default).
2. **LinearLR**       — linear warm-up from ``start_lr`` → ``lr``, then
                        optionally linear decay back to ``end_lr``.
3. **CosineLR**       — cosine annealing with optional linear warm-up.
4. **WarmupCosine**   — alias for CosineLR with warmup (most common in LLM RL).

All schedulers implement::

    scheduler.get_lr(step: int) -> float   # query without side-effect
    scheduler.step(step: int) -> float     # same (stateless callers)

``OnPolicyTrainer`` calls ``scheduler.get_lr(iter_idx)`` at the start of
each iteration and writes the result to ``optim.param_groups[*]["lr"]``.

Usage::

    from hermes_agentic_rl.trainers.lr_schedule import make_lr_scheduler

    # warm-up for 10 steps, then cosine decay over 100 steps
    sched = make_lr_scheduler(
        "warmup_cosine",
        lr=1e-4,
        warmup_steps=10,
        total_steps=100,
    )
    for it in range(100):
        lr = sched.get_lr(it)
        for pg in optim.param_groups:
            pg["lr"] = lr
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constant (baseline / pass-through)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConstantLR:
    """Always returns the base learning rate (legacy)."""

    lr: float = 1e-3

    def get_lr(self, step: int) -> float:
        return self.lr

    def step(self, step: int) -> float:
        return self.get_lr(step)


# ---------------------------------------------------------------------------
# Linear warm-up → constant → optional linear decay
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LinearLR:
    """Linear warm-up from ``warmup_start_lr`` to ``lr``, then optional
    linear decay to ``end_lr`` over ``total_steps``.

    Timeline::

        [0, warmup_steps)           : linear ramp  warmup_start_lr → lr
        [warmup_steps, total_steps) : linear decay  lr → end_lr
        [total_steps, ∞)            : constant end_lr
    """

    lr: float = 1e-3
    warmup_steps: int = 0
    total_steps: int = 0            # 0 = no decay after warmup
    warmup_start_lr: float = 0.0
    end_lr: float = 0.0

    def get_lr(self, step: int) -> float:
        s = max(0, int(step))
        # warm-up phase
        if self.warmup_steps > 0 and s < self.warmup_steps:
            frac = s / self.warmup_steps
            return self.warmup_start_lr + frac * (self.lr - self.warmup_start_lr)
        # decay phase
        decay_start = self.warmup_steps
        decay_end = self.total_steps
        if decay_end > decay_start and s < decay_end:
            frac = (s - decay_start) / (decay_end - decay_start)
            return self.lr + frac * (self.end_lr - self.lr)
        # flat tail
        if decay_end > decay_start and s >= decay_end:
            return self.end_lr
        return self.lr

    def step(self, step: int) -> float:
        return self.get_lr(step)


# ---------------------------------------------------------------------------
# Cosine annealing with optional warm-up
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CosineLR:
    """Cosine decay from ``lr`` to ``end_lr`` over ``total_steps``.

    If ``warmup_steps > 0`` a linear ramp from ``warmup_start_lr`` → ``lr``
    precedes the cosine phase.

    Timeline::

        [0, warmup_steps)              : linear  warmup_start_lr → lr
        [warmup_steps, total_steps)    : cosine  lr → end_lr
        [total_steps, ∞)               : constant end_lr
    """

    lr: float = 1e-3
    total_steps: int = 100
    warmup_steps: int = 0
    warmup_start_lr: float = 0.0
    end_lr: float = 0.0

    def get_lr(self, step: int) -> float:
        s = max(0, int(step))
        # warm-up
        if self.warmup_steps > 0 and s < self.warmup_steps:
            frac = s / self.warmup_steps
            return self.warmup_start_lr + frac * (self.lr - self.warmup_start_lr)
        # cosine decay
        decay_start = self.warmup_steps
        decay_steps = max(1, self.total_steps - decay_start)
        t = min(1.0, (s - decay_start) / decay_steps)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.end_lr + (self.lr - self.end_lr) * cos_val

    def step(self, step: int) -> float:
        return self.get_lr(step)


# WarmupCosine is just CosineLR with warmup_steps > 0; factory creates it.
WarmupCosineLR = CosineLR


# ---------------------------------------------------------------------------
# Type alias + factory
# ---------------------------------------------------------------------------

LRScheduler = ConstantLR | LinearLR | CosineLR


def make_lr_scheduler(
    kind: str = "constant",
    *,
    lr: float = 1e-3,
    total_steps: int = 0,
    warmup_steps: int = 0,
    warmup_start_lr: float = 0.0,
    end_lr: float = 0.0,
) -> LRScheduler:
    """Factory for LR schedulers.

    Args:
        kind: ``"constant"`` | ``"linear"`` | ``"cosine"`` | ``"warmup_cosine"``
        lr: peak / base learning rate.
        total_steps: total number of training iterations (used by linear/cosine).
        warmup_steps: number of warm-up iterations (0 = no warm-up).
        warmup_start_lr: initial LR during warm-up (default 0.0).
        end_lr: final LR after decay (default 0.0).

    Example::

        sched = make_lr_scheduler("warmup_cosine", lr=1e-4,
                                  warmup_steps=50, total_steps=500)
        lr_t = sched.get_lr(t)
    """
    k = kind.lower().replace("-", "_")
    if k == "constant":
        return ConstantLR(lr=lr)
    elif k == "linear":
        return LinearLR(
            lr=lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            warmup_start_lr=warmup_start_lr,
            end_lr=end_lr,
        )
    elif k in ("cosine", "warmup_cosine"):
        return CosineLR(
            lr=lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            end_lr=end_lr,
        )
    else:
        raise ValueError(
            f"Unknown LR scheduler kind: {kind!r}. "
            "Choose from: constant, linear, cosine, warmup_cosine"
        )
