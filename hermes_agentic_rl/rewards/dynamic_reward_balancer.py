"""Dynamic Reward Balancer — adaptive component weight scheduling.

Differentiator over OpenClaw-RL
--------------------------------
OpenClaw-RL uses *static* reward component weights (e.g. prm_weight=1.0,
outcome_weight=1.0). These weights are set once and never change, so:
  • An uninformative component (constant reward) wastes gradient capacity.
  • A high-variance component dominates early training before the policy
    understands it.

``DynamicRewardBalancer`` continuously estimates per-component variance and
re-scales weights so that the gradient contribution from each component is
proportional to its *informative variance* — i.e. components that actually
discriminate between good and bad actions get more weight.

Algorithm
~~~~~~~~~
At each RL iteration the balancer observes the per-component reward
distribution (mean, std) from the batch.  It then reweights:

    effective_variance(c) = EMA( std(c, batch)^2 )
    normalized_weight(c)  = base_weight(c) × sqrt( effective_variance(c) )
                            / sum_c'( base_weight(c') × sqrt(eff_var(c')) )
                            × n_components    # renormalize to preserve total scale

This is related to the inverse-variance weighting in ensemble learning but
applied to the *reward signal* rather than predictions.

Additionally, the balancer supports *curriculum scheduling*: components can
be phased in after a warmup period (e.g. enable PRM-based step reward only
after the outcome reward has converged enough to bootstrap).

Usage::

    balancer = DynamicRewardBalancer(
        base_weights={"outcome": 1.0, "prm": 0.5, "tool_fidelity": 0.3},
        cfg=DynamicBalancerConfig(warmup_iters=10, variance_floor=0.01),
    )
    # In training loop, after batch stats are computed:
    scaled = balancer.scale_batch_rewards(it, batch_stats_dict)
    # scaled["outcome_weight"] etc. are the adjusted weights for the next iter.

Integration with RewardComposer::

    weights = balancer.current_weights()
    for component in reward_manager.components:
        component.weight = weights.get(component.name, component.weight)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DynamicBalancerConfig:
    """Configuration for ``DynamicRewardBalancer``."""

    # EMA decay for variance estimates. Closer to 1.0 → slower adaptation.
    alpha: float = 0.95

    # Floor on variance to prevent division by zero / zero-variance collapse.
    variance_floor: float = 0.01

    # How many RL iters to wait before applying dynamic reweighting.
    # Before warmup, base_weights are used unchanged.
    warmup_iters: int = 10

    # Maximum ratio by which any single component's weight can change per iter.
    # E.g. 2.0 → weight can at most double or halve per step.
    max_weight_change_ratio: float = 2.0

    # When True, re-normalize so that sum(effective_weights) == sum(base_weights).
    # This prevents the total reward scale from drifting.
    preserve_scale: bool = True

    # Per-component warmup: {component_name: iter_offset}.
    # Component is not used until iter >= iter_offset.
    component_warmup: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Balancer
# ---------------------------------------------------------------------------


class DynamicRewardBalancer:
    """Adaptive reward component weight scheduler.

    Tracks per-component variance with EMA and adjusts weights so that
    higher-variance (more informative) components contribute more to the
    gradient signal.
    """

    def __init__(
        self,
        base_weights: dict[str, float],
        cfg: DynamicBalancerConfig | None = None,
    ) -> None:
        if not base_weights:
            raise ValueError("base_weights must be non-empty")
        self.base_weights: dict[str, float] = dict(base_weights)
        self.cfg = cfg or DynamicBalancerConfig()
        # EMA variance estimates: component → running_var
        self._ema_var: dict[str, float] = {k: 1.0 for k in base_weights}
        # Current effective weights (lazily updated).
        self._current_weights: dict[str, float] = dict(base_weights)
        self._iter = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def observe_batch(
        self,
        iter_idx: int,
        batch_stats: dict[str, Any],
    ) -> dict[str, float]:
        """Update variance estimates from batch stats and return new weights.

        Args:
            iter_idx: current training iteration.
            batch_stats: dict produced by ``_summarize_batch_metadata`` or
                ``batch_stats.py``. We look for keys like
                ``"reward_components.<name>.std"`` or flat
                ``"reward_<name>_std"`` (both conventions are supported).

        Returns:
            Updated effective weight dict {component_name: weight}.
        """
        self._iter = iter_idx

        # Extract per-component std from batch_stats.
        observed_std = self._extract_component_stds(batch_stats)

        # Update EMA variance estimates.
        alpha = self.cfg.alpha
        for name in self.base_weights:
            if name in observed_std:
                variance = float(observed_std[name]) ** 2
                old_var = self._ema_var.get(name, 1.0)
                self._ema_var[name] = alpha * old_var + (1.0 - alpha) * variance
            # else: keep previous estimate

        # Compute effective weights.
        if iter_idx >= self.cfg.warmup_iters:
            self._current_weights = self._compute_weights(iter_idx)
        else:
            # Before warmup: use base weights but still update EMA.
            self._current_weights = dict(self.base_weights)

        return dict(self._current_weights)

    def current_weights(self) -> dict[str, float]:
        """Return the last computed effective weights."""
        return dict(self._current_weights)

    def snapshot(self) -> dict[str, Any]:
        """Return a logging-friendly summary."""
        out: dict[str, Any] = {"balancer_iter": self._iter}
        for name, w in self._current_weights.items():
            out[f"dyn_weight/{name}"] = w
        for name, v in self._ema_var.items():
            out[f"dyn_var/{name}"] = v
        return out

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema_var": dict(self._ema_var),
            "current_weights": dict(self._current_weights),
            "iter": self._iter,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._ema_var.update(d.get("ema_var", {}))
        self._current_weights.update(d.get("current_weights", {}))
        self._iter = int(d.get("iter", 0))

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _compute_weights(self, iter_idx: int) -> dict[str, float]:
        cfg = self.cfg
        names = list(self.base_weights.keys())

        # Apply component warmup (phase-in after offset).
        active = {n for n in names if iter_idx >= cfg.component_warmup.get(n, 0)}
        if not active:
            return dict(self.base_weights)

        # Effective weight = base × sqrt(ema_var) (info-theoretic proxy).
        raw: dict[str, float] = {}
        for n in names:
            if n not in active:
                raw[n] = 0.0
                continue
            var = max(self._ema_var.get(n, 1.0), cfg.variance_floor)
            raw[n] = float(self.base_weights[n]) * math.sqrt(var)

        total_raw = sum(raw.values())
        if total_raw <= 0.0:
            return dict(self.base_weights)

        # Re-normalize.
        if cfg.preserve_scale:
            base_total = sum(self.base_weights.values())
            scale = base_total / total_raw
        else:
            scale = 1.0

        new_weights = {n: raw[n] * scale for n in names}

        # Clamp change ratio to prevent sudden shifts.
        max_ratio = cfg.max_weight_change_ratio
        if max_ratio > 0:
            prev = self._current_weights
            clamped: dict[str, float] = {}
            for n, w_new in new_weights.items():
                w_prev = prev.get(n, self.base_weights[n])
                if w_prev > 0:
                    ratio = w_new / w_prev
                    if ratio > max_ratio:
                        w_new = w_prev * max_ratio
                    elif ratio < 1.0 / max_ratio:
                        w_new = w_prev / max_ratio
                clamped[n] = max(0.0, w_new)
            new_weights = clamped

        return new_weights

    @staticmethod
    def _extract_component_stds(batch_stats: dict[str, Any]) -> dict[str, float]:
        """Extract per-component std values from batch_stats dict.

        Supports two conventions:
          1. Nested: batch_stats["reward_components"][name]["std"]
          2. Flat:   batch_stats["reward_<name>_std"]
        """
        out: dict[str, float] = {}

        # Convention 1: nested reward_components dict
        nested = batch_stats.get("reward_components")
        if isinstance(nested, dict):
            for name, comp_stats in nested.items():
                if isinstance(comp_stats, dict) and "std" in comp_stats:
                    out[str(name)] = float(comp_stats["std"])

        # Convention 2: flat keys "reward_<name>_std"
        prefix = "reward_"
        suffix = "_std"
        for key, val in batch_stats.items():
            if isinstance(key, str) and key.startswith(prefix) and key.endswith(suffix):
                name = key[len(prefix) : -len(suffix)]
                if name and isinstance(val, int | float):
                    out.setdefault(name, float(val))

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_dynamic_balancer_from_config(
    cfg_dict: dict[str, Any],
    base_weights: dict[str, float],
) -> DynamicRewardBalancer:
    """Build a DynamicRewardBalancer from a raw config dict.

    Expected keys (all optional):
      alpha, variance_floor, warmup_iters, max_weight_change_ratio,
      preserve_scale, component_warmup (dict).
    """
    dcfg = DynamicBalancerConfig(
        alpha=float(cfg_dict.get("alpha", 0.95)),
        variance_floor=float(cfg_dict.get("variance_floor", 0.01)),
        warmup_iters=int(cfg_dict.get("warmup_iters", 10)),
        max_weight_change_ratio=float(cfg_dict.get("max_weight_change_ratio", 2.0)),
        preserve_scale=bool(cfg_dict.get("preserve_scale", True)),
        component_warmup={
            str(k): int(v) for k, v in (cfg_dict.get("component_warmup") or {}).items()
        },
    )
    return DynamicRewardBalancer(base_weights=base_weights, cfg=dcfg)
