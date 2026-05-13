"""Numerical utilities for on-policy RL training.

Contents:
  - :class:`RunningMeanStd` — Welford online mean/variance, used to
    normalize rewards across iterations without biasing the gradient.
  - :class:`AdaptiveKLController` — InstructGPT (Ouyang 2022, Appendix A.2)
    style β-adaptive KL coefficient. Scales ``kl_coef`` up when KL drifts
    above ``target_kl``, down when it dips below — keeps PPO inside a
    trust region without hand-tuning.

Both are pure-Python, stateless w.r.t. torch, and fully serializable via
``state_dict`` / ``load_state_dict`` so they survive checkpoint resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunningMeanStd:
    """Welford online mean + variance.

    Used to whiten scalar rewards across an iteration window. Updating by
    batch is O(n); reading mean/std is O(1). Numerically stable for
    sequential batches of widely different sizes.
    """

    mean: float = 0.0
    var: float = 1.0
    count: float = 1e-4
    eps: float = 1e-8

    def update(self, values: list[float]) -> None:
        if not values:
            return
        batch_mean = sum(values) / len(values)
        batch_var = sum((v - batch_mean) ** 2 for v in values) / max(1, len(values))
        batch_count = float(len(values))

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return (self.var + self.eps) ** 0.5

    def normalize(self, value: float) -> float:
        return (value - self.mean) / self.std

    def state_dict(self) -> dict[str, float]:
        return {
            "mean": float(self.mean),
            "var": float(self.var),
            "count": float(self.count),
            "eps": float(self.eps),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = float(state.get("mean", 0.0))
        self.var = float(state.get("var", 1.0))
        self.count = float(state.get("count", 1e-4))
        self.eps = float(state.get("eps", 1e-8))


@dataclass
class AdaptiveKLController:
    """InstructGPT β-adaptive KL coefficient.

    Reference: Ouyang et al., "Training language models to follow
    instructions with human feedback" (2022), Appendix A.2::

        proportional_error = clip(kl / target_kl - 1, -0.2, 0.2)
        beta_next = beta * (1 + proportional_error * horizon_scale)

    Where ``horizon_scale = n_steps / horizon``. With default ``horizon =
    10000`` and one call per iter, β doubles/halves roughly every 10k
    iters of sustained over/under-shoot. Tunable via ``horizon``.

    Set ``target_kl <= 0`` to disable adaptive behavior (β stays fixed).
    """

    init_kl_coef: float = 0.2
    target_kl: float = 0.1
    horizon: float = 10000.0
    value: float = field(init=False)
    min_coef: float = 1e-4
    max_coef: float = 10.0
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.value = float(self.init_kl_coef)

    def update(self, current_kl: float, n_steps: int = 1) -> float:
        """Update β given observed KL; returns new β."""
        self._step_count += int(max(1, n_steps))
        if self.target_kl <= 0:
            return self.value
        # Proportional error in [-0.2, 0.2]
        pe = current_kl / max(self.target_kl, 1e-8) - 1.0
        pe = max(-0.2, min(0.2, pe))
        mult = 1.0 + pe * (n_steps / max(1.0, self.horizon))
        self.value = float(self.value * mult)
        self.value = max(self.min_coef, min(self.max_coef, self.value))
        return self.value

    def state_dict(self) -> dict[str, Any]:
        return {
            "init_kl_coef": self.init_kl_coef,
            "target_kl": self.target_kl,
            "horizon": self.horizon,
            "value": self.value,
            "min_coef": self.min_coef,
            "max_coef": self.max_coef,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.init_kl_coef = float(state.get("init_kl_coef", self.init_kl_coef))
        self.target_kl = float(state.get("target_kl", self.target_kl))
        self.horizon = float(state.get("horizon", self.horizon))
        self.value = float(state.get("value", self.value))
        self.min_coef = float(state.get("min_coef", self.min_coef))
        self.max_coef = float(state.get("max_coef", self.max_coef))
        self._step_count = int(state.get("step_count", 0))
