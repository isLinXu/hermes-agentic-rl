"""KL coefficient controllers for adaptive trust-region training.

Provides two variants, both following the same interface (``update`` + 
``state_dict`` / ``load_state_dict``):

``AdaptiveKLController`` (P-controller)
    InstructGPT (Ouyang 2022, Appendix A.2) proportional controller.
    Simple, stable, widely used in PPO/GRPO pipelines.

``PIDKLController`` (PID controller)
    Proportional-Integral-Derivative controller over KL divergence.
    Eliminates steady-state error (I-term) and reacts to trend changes
    (D-term), which is important when KL oscillates. Used in DeepSeek-R1
    and DAPO (Yu 2024) training pipelines.

Factory function:
    ``build_kl_controller(cfg)`` reads the ``adaptive_kl_type`` config key
    ("p" or "pid") and returns the appropriate controller.

Both classes are pure-Python, stateless w.r.t. torch, and fully serializable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# P controller — InstructGPT (default, backward-compatible)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveKLController:
    """InstructGPT β-adaptive KL coefficient (proportional controller).

    Reference: Ouyang et al., "Training language models to follow
    instructions with human feedback" (2022), Appendix A.2::

        proportional_error = clip(kl / target_kl - 1, -0.2, 0.2)
        beta_next = beta * (1 + proportional_error * n_steps / horizon)

    ``horizon`` controls the timescale: with ``horizon=10000`` and one call
    per iter, β doubles/halves over ~10k sustained over/under-shoot iters.

    Set ``target_kl <= 0`` to freeze β (fixed-coefficient mode).
    """

    init_kl_coef: float = 0.2
    target_kl: float = 0.1
    horizon: float = 10000.0
    min_coef: float = 1e-4
    max_coef: float = 10.0
    value: float = field(init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.value = float(self.init_kl_coef)

    def update(self, current_kl: float, n_steps: int = 1) -> float:
        """Update β given observed KL; returns new β."""
        self._step_count += int(max(1, n_steps))
        if self.target_kl <= 0:
            return self.value
        pe = current_kl / max(self.target_kl, 1e-8) - 1.0
        pe = max(-0.2, min(0.2, pe))
        mult = 1.0 + pe * (n_steps / max(1.0, self.horizon))
        self.value = float(max(self.min_coef, min(self.max_coef, self.value * mult)))
        return self.value

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "p",
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


# ---------------------------------------------------------------------------
# PID controller — DeepSeek / DAPO variant
# ---------------------------------------------------------------------------


@dataclass
class PIDKLController:
    """Proportional-Integral-Derivative KL coefficient controller.

    Eliminates the steady-state error of the simple P-controller (I-term)
    and reacts to trend changes (D-term).

    Update rule::

        error_t    = kl_t / target_kl - 1
        integral_t = clip(integral_{t-1} + error_t * dt, -I_max, I_max)
        d_error    = (error_t - error_{t-1}) / dt
        delta_log_beta = Kp * error_t + Ki * integral_t + Kd * d_error
        beta_t = clip(beta_{t-1} * exp(delta_log_beta), min_coef, max_coef)

    Using a log-domain update (``exp``) keeps β positive and makes up/down
    adjustments symmetric in log-space (which matches the log-normal prior
    on β used in InstructGPT).

    Default gains ``Kp=0.1, Ki=0.01, Kd=0.005`` are tuned for ~1k iters of
    warm-up with ``target_kl`` in [0.05, 0.2]. Increase ``Kp`` for faster
    response; increase ``Ki`` to correct persistent bias; reduce ``Kd`` if
    the controller oscillates.

    References:
        - DeepSeek-R1 training notes (2025)
        - DAPO: Yu et al. (2024), §4.3 adaptive KL schedule
    """

    init_kl_coef: float = 0.2
    target_kl: float = 0.1
    Kp: float = 0.1
    Ki: float = 0.01
    Kd: float = 0.005
    I_max: float = 2.0     # integral anti-windup clamp
    min_coef: float = 1e-4
    max_coef: float = 10.0
    value: float = field(init=False)
    _integral: float = field(default=0.0, init=False)
    _prev_error: float = field(default=0.0, init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.value = float(self.init_kl_coef)

    def update(self, current_kl: float, n_steps: int = 1) -> float:
        """Update β given observed KL; returns new β."""
        self._step_count += int(max(1, n_steps))
        if self.target_kl <= 0:
            return self.value

        dt = float(max(1, n_steps))
        error = current_kl / max(self.target_kl, 1e-8) - 1.0

        # Integral with anti-windup
        self._integral = max(
            -self.I_max,
            min(self.I_max, self._integral + error * dt),
        )

        # Derivative (first-difference / dt)
        d_error = (error - self._prev_error) / dt
        self._prev_error = error

        delta_log = self.Kp * error + self.Ki * self._integral + self.Kd * d_error
        # Use exp to stay in log-domain; clamp delta to ±0.5 per step for safety
        delta_log = max(-0.5, min(0.5, delta_log))
        new_value = self.value * math.exp(delta_log)
        self.value = float(max(self.min_coef, min(self.max_coef, new_value)))
        return self.value

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "pid",
            "init_kl_coef": self.init_kl_coef,
            "target_kl": self.target_kl,
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
            "I_max": self.I_max,
            "min_coef": self.min_coef,
            "max_coef": self.max_coef,
            "value": self.value,
            "_integral": self._integral,
            "_prev_error": self._prev_error,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.init_kl_coef = float(state.get("init_kl_coef", self.init_kl_coef))
        self.target_kl = float(state.get("target_kl", self.target_kl))
        self.Kp = float(state.get("Kp", self.Kp))
        self.Ki = float(state.get("Ki", self.Ki))
        self.Kd = float(state.get("Kd", self.Kd))
        self.I_max = float(state.get("I_max", self.I_max))
        self.min_coef = float(state.get("min_coef", self.min_coef))
        self.max_coef = float(state.get("max_coef", self.max_coef))
        self.value = float(state.get("value", self.value))
        self._integral = float(state.get("_integral", 0.0))
        self._prev_error = float(state.get("_prev_error", 0.0))
        self._step_count = int(state.get("step_count", 0))


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

KLControllerType = AdaptiveKLController | PIDKLController
KLControllerKind = Literal["p", "pid"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_kl_controller(
    *,
    kind: KLControllerKind = "p",
    init_kl_coef: float = 0.2,
    target_kl: float = 0.1,
    horizon: float = 10000.0,
    min_coef: float = 1e-4,
    max_coef: float = 10.0,
    # PID-only
    Kp: float = 0.1,
    Ki: float = 0.01,
    Kd: float = 0.005,
    I_max: float = 2.0,
) -> KLControllerType:
    """Construct a KL controller from keyword arguments.

    Args:
        kind: ``"p"`` for InstructGPT proportional controller (default),
              ``"pid"`` for PID controller (DeepSeek/DAPO style).
        init_kl_coef: initial β value.
        target_kl: desired KL divergence setpoint.
        horizon: P-controller horizon (ignored for PID).
        min_coef, max_coef: β clamp bounds.
        Kp, Ki, Kd, I_max: PID gains (ignored for P controller).

    Returns:
        An :class:`AdaptiveKLController` or :class:`PIDKLController`.
    """
    if kind == "pid":
        return PIDKLController(
            init_kl_coef=init_kl_coef,
            target_kl=target_kl,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            I_max=I_max,
            min_coef=min_coef,
            max_coef=max_coef,
        )
    return AdaptiveKLController(
        init_kl_coef=init_kl_coef,
        target_kl=target_kl,
        horizon=horizon,
        min_coef=min_coef,
        max_coef=max_coef,
    )


def build_kl_controller_from_config(cfg: Any) -> KLControllerType | None:
    """Construct a KL controller from an ``OnPolicyTrainerConfig``.

    Returns ``None`` when adaptive KL is disabled or ``target_kl <= 0``.
    """
    if not getattr(cfg, "adaptive_kl", False):
        return None
    target_kl = float(getattr(cfg, "target_kl", 0.0) or 0.0)
    if target_kl <= 0:
        return None

    # Read init_kl_coef from the algo's config when available.
    init_beta = 0.02  # safe default
    # Caller is responsible for passing the resolved init value.

    kind: KLControllerKind = getattr(cfg, "adaptive_kl_type", "p") or "p"
    return build_kl_controller(
        kind=kind,
        init_kl_coef=float(getattr(cfg, "_resolved_init_kl_coef", init_beta)),
        target_kl=target_kl,
        horizon=float(getattr(cfg, "adaptive_kl_horizon", 10000.0)),
        min_coef=float(getattr(cfg, "adaptive_kl_min", 1e-4)),
        max_coef=float(getattr(cfg, "adaptive_kl_max", 10.0)),
        Kp=float(getattr(cfg, "adaptive_kl_Kp", 0.1)),
        Ki=float(getattr(cfg, "adaptive_kl_Ki", 0.01)),
        Kd=float(getattr(cfg, "adaptive_kl_Kd", 0.005)),
        I_max=float(getattr(cfg, "adaptive_kl_I_max", 2.0)),
    )
