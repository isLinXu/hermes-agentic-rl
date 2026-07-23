"""Staleness-adaptive TIS controller — dynamically adjust IS clipping
based on observed policy staleness.

The existing ``TISConfig`` uses a fixed ``rho_clip`` (c̄) for all iterations.
In asynchronous / pipelined RL, staleness varies: some batches arrive with
0-step staleness (synchronous), others with 3+ steps (heavily pipelined).
A fixed clip is suboptimal:

  - Too high c̄ when staleness is large → high-variance gradients, instability.
  - Too low c̄ when staleness is zero → unnecessarily conservative, slow
    convergence (basically throwing away signal from fresh on-policy data).

This module provides a **StalenessAdaptiveTIS** controller that:

  1. Tracks a rolling average of observed staleness (learner_version −
     behavior_version).
  2. Maps staleness → rho_clip via a configurable schedule:
       - staleness = 0  → rho_clip = max_clip (trust fully, ≈ on-policy)
       - staleness = S_max → rho_clip = min_clip (conservative)
       - linear or exponential interpolation in between.
  3. Produces a ``TISConfig`` per training step, which is passed to
     ``tis_corrected_advantage``.

Usage::

    from hermes_agentic_rl.algos.common.staleness_adaptive_tis import (
        StalenessAdaptiveTIS,
        StalenessSchedule,
    )

    controller = StalenessAdaptiveTIS(
        schedule=StalenessSchedule(
            max_rho_clip=2.0,
            min_rho_clip=1.0,
            max_staleness=10,
            interpolation="linear",
        ),
    )

    # Each training step:
    controller.observe_staleness(current_staleness)
    tis_cfg = controller.get_config()  # pass to tis_corrected_advantage
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from hermes_agentic_rl.algos.common.vtrace import TISConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StalenessSchedule:
    """Maps staleness values to rho_clip values.

    Parameters
    ----------
    max_rho_clip : float
        rho_clip when staleness = 0 (fully on-policy). Default 2.0.
    min_rho_clip : float
        rho_clip when staleness >= max_staleness. Default 1.0.
    max_staleness : int
        Staleness level at which rho_clip reaches min_rho_clip. Default 10.
    interpolation : str
        "linear" or "exp" (exponential decay). Default "linear".
    rho_floor : float
        Lower clamp on the IS weight (passed through to TISConfig).
    """

    max_rho_clip: float = 2.0
    min_rho_clip: float = 1.0
    max_staleness: int = 10
    interpolation: str = "linear"
    rho_floor: float = 0.0

    def clip_at(self, staleness: float) -> float:
        """Compute rho_clip for a given staleness value.

        Parameters
        ----------
        staleness : float
            Observed staleness (learner_version − behavior_version).

        Returns
        -------
        float
            The rho_clip value in [min_rho_clip, max_rho_clip].
        """
        if staleness <= 0:
            return self.max_rho_clip
        if staleness >= self.max_staleness:
            return self.min_rho_clip

        # Interpolate between max and min.
        if self.interpolation == "exp":
            # Exponential decay: rho_clip = min + (max - min) * exp(-k * s)
            # k is chosen so that at s = max_staleness, the decay reaches min.
            import math
            k = 3.0 / self.max_staleness  # ~95% decay at max_staleness
            delta = self.max_rho_clip - self.min_rho_clip
            return self.min_rho_clip + delta * math.exp(-k * staleness)
        else:
            # Linear interpolation.
            frac = staleness / self.max_staleness
            return self.max_rho_clip - frac * (self.max_rho_clip - self.min_rho_clip)


class StalenessAdaptiveTIS:
    """Adaptive TIS controller that adjusts rho_clip based on staleness.

    Parameters
    ----------
    schedule : StalenessSchedule
        The staleness → clip mapping configuration.
    window_size : int
        Number of recent staleness observations to average over. A larger
        window gives smoother adjustments; a smaller window reacts faster.
        Default: 10.
    enabled : bool
        Master switch. When False, always returns a TISConfig with
        rho_clip = max_rho_clip and enabled=False (identity).
    """

    def __init__(
        self,
        schedule: StalenessSchedule | None = None,
        *,
        window_size: int = 10,
        enabled: bool = True,
    ) -> None:
        self.schedule = schedule or StalenessSchedule()
        self.window_size = max(1, window_size)
        self.enabled = enabled
        self._staleness_history: deque[float] = deque(maxlen=self.window_size)
        self._current_clip: float = self.schedule.max_rho_clip
        self._step_count: int = 0

    def observe_staleness(self, staleness: int | float) -> None:
        """Record an observed staleness value.

        Parameters
        ----------
        staleness : int | float
            The staleness of the current batch (learner_version −
            behavior_version). Non-negative.
        """
        s = max(0.0, float(staleness))
        self._staleness_history.append(s)
        self._update_clip()
        self._step_count += 1

    def _update_clip(self) -> None:
        """Recompute the current rho_clip from the rolling average staleness."""
        if not self._staleness_history:
            self._current_clip = self.schedule.max_rho_clip
            return
        avg_staleness = sum(self._staleness_history) / len(self._staleness_history)
        self._current_clip = self.schedule.clip_at(avg_staleness)

    def get_config(self) -> TISConfig:
        """Return a TISConfig with the current adaptive rho_clip."""
        if not self.enabled:
            return TISConfig(
                rho_clip=self.schedule.max_rho_clip,
                enabled=False,
                rho_floor=self.schedule.rho_floor,
            )
        return TISConfig(
            rho_clip=self._current_clip,
            enabled=True,
            rho_floor=self.schedule.rho_floor,
        )

    @property
    def current_clip(self) -> float:
        """The current rho_clip value."""
        return self._current_clip

    @property
    def mean_staleness(self) -> float:
        """Rolling average of observed staleness."""
        if not self._staleness_history:
            return 0.0
        return sum(self._staleness_history) / len(self._staleness_history)

    @property
    def last_staleness(self) -> float:
        """Most recent staleness observation."""
        return self._staleness_history[-1] if self._staleness_history else 0.0

    @property
    def step_count(self) -> int:
        return self._step_count

    def stats(self) -> dict[str, float]:
        """Return observability stats for logging."""
        return {
            "tis_adaptive_clip": self._current_clip,
            "tis_adaptive_mean_staleness": self.mean_staleness,
            "tis_adaptive_last_staleness": self.last_staleness,
            "tis_adaptive_steps": float(self._step_count),
            "tis_adaptive_enabled": float(self.enabled),
        }

    def reset(self) -> None:
        """Clear history and reset to initial state."""
        self._staleness_history.clear()
        self._current_clip = self.schedule.max_rho_clip
        self._step_count = 0
