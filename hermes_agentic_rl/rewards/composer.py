"""Composable reward shaping with normalization, conditional activation, and shaping.

Background
----------
The default :class:`hermes_agentic_rl.core.reward_manager.RewardManager`
treats reward components as a flat list passed through ``weighted_sum``.
That covers the common case but lacks three primitives that surface as
soon as the reward stack grows beyond 2-3 components:

1. **Per-component running normalization** — reward magnitudes from
   different signals (filesystem verifier scores in {0, 1}, LLM judge
   scores in [0, 5], length penalty in [-3, 0]) drown each other when
   weighted-summed. Whitening each component to zero mean / unit std
   keeps the gradient signal balanced.

2. **Conditional activation** — some rewards are only meaningful at
   certain curriculum levels (e.g. tool-call schema fidelity is only
   relevant once the agent has learned to emit a tool call at all) or
   when an item carries certain metadata (multi-turn rewards only fire
   when ``trajectory.turns_used > 1``).

3. **Distance-based discount** — long trajectories should weight
   late-turn rewards more heavily; short ones less so. A simple
   per-component "turn discount" provides this without touching the
   credit-assignment logic in the trainer.

This module implements those three primitives as :class:`RewardComposer`,
a drop-in replacement for ``RewardManager``. It accepts a list of
``BaseReward`` instances plus a configuration dict that tells it how to
normalize and gate each component. Components that opt out of
normalization or gating are forwarded unchanged.

Example::

    composer = RewardComposer(
        components=[OutcomeReward(), ToolcallReward(), LLMJudgeReward()],
        config={
            "normalize": {"toolcall_reward": True, "llm_judge": True},
            "conditions": {
                "toolcall_reward": lambda item, traj: bool(traj.steps),
                "llm_judge": lambda item, traj: traj.turns_used > 1,
            },
            "turn_discount": {"toolcall_reward": 0.95},
        },
    )
    summary = await composer.evaluate(item, trajectory, tool_context=None)
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, RewardSummary, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RunningMeanStd:
    """Per-component running statistics for whitening.

    Implements Welford's online mean/variance update — numerically stable
    and identical to ``hermes_agentic_rl.trainers.ppo_utils.RunningMeanStd``
    but inlined here to avoid a circular-ish dependency on the torch
    stack from a pure-Python module.
    """

    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1.0
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2.0:
            return 0.0
        return self.m2 / (self.count - 1.0)

    @property
    def std(self) -> float:
        v = self.variance
        return math.sqrt(v) if v > 0.0 else 0.0

    def whiten(self, value: float, *, eps: float = 1e-6) -> float:
        if self.count < 2.0 or self.std < eps:
            return value
        return (value - self.mean) / (self.std + eps)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


ConditionFn = Callable[[dict[str, Any], Trajectory], bool]


@dataclass(slots=True)
class RewardComposerConfig:
    """Declarative configuration for :class:`RewardComposer`.

    Attributes
    ----------
    normalize:
        Mapping ``component.name -> bool``. Components flagged True are
        whitened against their own running statistics before aggregation.
        Defaults to False per component (raw scores).

    conditions:
        Mapping ``component.name -> Callable(item, trajectory) -> bool``.
        Components whose condition returns False are *skipped*: they do
        not contribute to the aggregate AND their running stats are not
        updated. Useful for curriculum-gated components.

    turn_discount:
        Mapping ``component.name -> float in (0, 1]``. The component
        score is multiplied by ``γ ** trajectory.turns_used`` before
        aggregation. ``γ = 1.0`` (the default) is a no-op.

    aggregator:
        Currently only ``"weighted_sum"`` is supported (the default);
        wired into :class:`RewardComposer.evaluate` so future variants
        can plug in without breaking callers.

    parallel:
        When True (default), individual reward components run via
        :func:`asyncio.gather`; otherwise sequentially. Sequential mode
        is occasionally useful for deterministic test ordering.
    """

    normalize: dict[str, bool] = field(default_factory=dict)
    conditions: dict[str, ConditionFn] = field(default_factory=dict)
    turn_discount: dict[str, float] = field(default_factory=dict)
    aggregator: str = "weighted_sum"
    parallel: bool = True


# ---------------------------------------------------------------------------
# RewardComposer
# ---------------------------------------------------------------------------


class RewardComposer:
    """Aggregator with running normalization + conditional gating + shaping.

    Drop-in replacement for ``RewardManager``: it exposes
    ``async evaluate(item, trajectory, tool_context) -> RewardSummary``
    and can be passed to :class:`hermes_agentic_rl.trainers.OnPolicyTrainer`
    in place of the default reward manager.

    The composer is stateful (running stats are maintained across calls).
    For deterministic test fixtures, instantiate a fresh composer per
    test or call :meth:`reset_stats`.
    """

    def __init__(
        self,
        components: list[BaseReward],
        config: RewardComposerConfig | dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            cfg = RewardComposerConfig()
        elif isinstance(config, dict):
            cfg = RewardComposerConfig(
                normalize=dict(config.get("normalize", {})),
                conditions=dict(config.get("conditions", {})),
                turn_discount=dict(config.get("turn_discount", {})),
                aggregator=str(config.get("aggregator", "weighted_sum")),
                parallel=bool(config.get("parallel", True)),
            )
        else:
            cfg = config
        if cfg.aggregator != "weighted_sum":
            raise ValueError(
                f"RewardComposer.aggregator={cfg.aggregator!r} is not yet "
                "supported (only 'weighted_sum')"
            )
        self.components = components
        self.cfg = cfg
        self._stats: dict[str, _RunningMeanStd] = {
            comp.name: _RunningMeanStd() for comp in components
        }
        # Track consecutive condition failures per component. After
        # ``_max_condition_failures`` raises in a row a condition is deemed
        # broken and the component is *disabled* (skipped) instead of being
        # silently force-activated forever — a silently-activated component
        # can prematurely inject curriculum-gated reward signals.
        self._condition_failures: dict[str, int] = {}
        self._max_condition_failures = 5

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardSummary:
        active: list[BaseReward] = []
        skipped: list[str] = []
        for comp in self.components:
            cond = self.cfg.conditions.get(comp.name)
            if cond is None:
                active.append(comp)
                continue
            if self._condition_failures.get(comp.name, 0) >= self._max_condition_failures:
                # Condition deemed broken — skip the component entirely.
                skipped.append(comp.name)
                continue
            if self._safe_condition(comp.name, cond, item, trajectory):
                active.append(comp)
            else:
                skipped.append(comp.name)

        if not active:
            return RewardSummary(
                final_score=0.0,
                components=[],
                metadata={
                    "aggregator": "weighted_sum",
                    "composer": "reward_composer",
                    "skipped_components": skipped,
                },
            )

        if self.cfg.parallel:
            raw_results = await asyncio.gather(
                *(comp.evaluate(item, trajectory, tool_context) for comp in active),
            )
        else:
            raw_results = []
            for comp in active:
                raw_results.append(await comp.evaluate(item, trajectory, tool_context))

        shaped: list[RewardResult] = []
        for comp, raw in zip(active, list(raw_results), strict=False):
            shaped.append(self._shape_one(comp, raw, trajectory))

        return self._aggregate(shaped, skipped)

    def reset_stats(self) -> None:
        """Clear all per-component running statistics.

        Useful when restarting an evaluation run with the same composer
        instance, or in unit tests that need a clean slate.
        """
        for stats in self._stats.values():
            stats.count = 0.0
            stats.mean = 0.0
            stats.m2 = 0.0

    def snapshot_stats(self) -> dict[str, dict[str, float]]:
        """Return the running mean/std per component, for logging."""
        return {
            name: {
                "count": stats.count,
                "mean": stats.mean,
                "std": stats.std,
            }
            for name, stats in self._stats.items()
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _safe_condition(
        self,
        comp_name: str,
        cond: ConditionFn,
        item: dict[str, Any],
        trajectory: Trajectory,
    ) -> bool:
        try:
            result = bool(cond(item, trajectory))
        except Exception as exc:
            import warnings

            n = self._condition_failures.get(comp_name, 0) + 1
            self._condition_failures[comp_name] = n
            disabled = n >= self._max_condition_failures
            msg = (
                f"RewardComposer: condition raised for {comp_name!r}: "
                f"{type(exc).__name__}: {exc}; "
                + (
                    f"disabling component after {n} consecutive failures."
                    if disabled
                    else "activating component anyway "
                    f"(failure {n}/{self._max_condition_failures})."
                )
            )
            logger.warning(msg, exc_info=True)
            warnings.warn(msg, stacklevel=2)
            return not disabled
        # Successful evaluation resets the failure streak.
        self._condition_failures[comp_name] = 0
        return result

    def _shape_one(
        self,
        comp: BaseReward,
        raw: RewardResult,
        trajectory: Trajectory,
    ) -> RewardResult:
        score = float(raw.score)
        meta = dict(raw.metadata)

        # Update running stats BEFORE whitening (so the current sample is
        # included in its own statistics — matches PPO's running-reward
        # convention).
        if self.cfg.normalize.get(comp.name, False):
            stats = self._stats.setdefault(comp.name, _RunningMeanStd())
            stats.update(score)
            whitened = stats.whiten(score)
            meta["raw_score"] = float(raw.score)
            meta["normalized_score"] = whitened
            score = whitened

        gamma = float(self.cfg.turn_discount.get(comp.name, 1.0))
        if gamma != 1.0:
            discount = gamma ** max(0, int(trajectory.turns_used))
            meta["turn_discount_gamma"] = gamma
            meta["turn_discount_applied"] = discount
            score = score * discount

        return RewardResult(
            name=raw.name,
            score=score,
            reason=raw.reason,
            weight=raw.weight,
            metadata=meta,
        )

    def _aggregate(self, shaped: list[RewardResult], skipped: list[str]) -> RewardSummary:
        total_weight = sum(r.weight for r in shaped)
        if total_weight <= 0:
            # Degenerate weighting (all components gated to zero weight, or a
            # misconfiguration) yields no usable signal. Return a zero-score
            # summary instead of raising — this mirrors
            # ``RewardManager.weighted_sum`` so the two aggregators are
            # interchangeable and a curriculum that temporarily zeroes every
            # component does not crash the rollout loop.
            return RewardSummary(
                final_score=0.0,
                components=shaped,
                metadata={
                    "aggregator": "weighted_sum",
                    "composer": "reward_composer",
                    "total_weight": total_weight,
                    "skipped_components": skipped,
                    "degenerate_zero_weight": True,
                    "running_stats": self.snapshot_stats(),
                },
            )
        final = sum(r.score * r.weight for r in shaped) / total_weight
        return RewardSummary(
            final_score=final,
            components=shaped,
            metadata={
                "aggregator": "weighted_sum",
                "composer": "reward_composer",
                "total_weight": total_weight,
                "skipped_components": skipped,
                "running_stats": self.snapshot_stats(),
            },
        )
