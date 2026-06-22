"""Numerical utilities for on-policy RL training.

Contents:
  - :class:`RunningMeanStd` — Welford online mean/variance, used to
    normalize rewards across iterations without biasing the gradient.
  - :class:`AdaptiveKLController` — re-exported from ``kl_controller`` for
    backward compatibility. New code should import from
    ``hermes_agentic_rl.trainers.kl_controller`` directly and use
    ``build_kl_controller`` / ``build_kl_controller_from_config`` to get
    either the P-controller or the new PID variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Re-export for backward compatibility.
from hermes_agentic_rl.trainers.kl_controller import (  # noqa: F401
    AdaptiveKLController,
    PIDKLController,
    build_kl_controller,
    build_kl_controller_from_config,
)


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
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot
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


# AdaptiveKLController is now defined in kl_controller.py and imported above.
# Kept as re-export for backward compatibility — existing imports of
# ``from hermes_agentic_rl.trainers.ppo_utils import AdaptiveKLController``
# continue to work without modification.
