"""Process-reward aggregation — OpenClaw-RL's ``final = o + (1/m)·Σ rᵢ``.

OpenClaw-RL §3 (General Agentic RL): for long-horizon tasks, outcome-only
rewards leave most steps unsupervised. The fix is to combine the terminal
*outcome* reward ``o`` with the **mean process reward** across the per-step
next-state PRM votes::

    final = o + (1/m) · Σᵢ rᵢ ,   rᵢ ∈ {+1, 0, −1}

where each ``rᵢ`` is the majority-voted PRM score for step ``i`` derived from
its next-state signal (tool output, test verdict, …).

``ProcessRewardAggregator`` implements this as a drop-in RewardManager
component. It reads per-step next-state signals from
``trajectory.metadata["runtime"]["step_next_states"]`` (a list of strings,
one per assistant step). When that list is absent it falls back to the single
``runtime["next_state"]`` signal so the component degrades gracefully to a
one-step process reward.

The outcome reward ``o`` is supplied either as a constant, a metadata key, or
a callable over the trajectory, keeping this component composable with the
existing reward stack instead of duplicating outcome logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.next_state_prm import NextStatePRM


@dataclass(slots=True)
class ProcessRewardConfig:
    """Configuration for :class:`ProcessRewardAggregator`.

    process_weight: scale applied to the mean process reward term.
    outcome_weight: scale applied to the outcome term ``o``.
    outcome_metadata_key: when set, the outcome ``o`` is read from
        ``trajectory.metadata[key]`` (float). Falls back to
        ``outcome_default`` when absent.
    outcome_default: outcome used when no key / callable supplies one.
    clip: optional symmetric clip on the final aggregated reward.
    """

    process_weight: float = 1.0
    outcome_weight: float = 1.0
    outcome_metadata_key: str | None = "outcome_reward"
    outcome_default: float = 0.0
    clip: float | None = None


class ProcessRewardAggregator:
    """RewardManager component: ``final = o + mean(step PRM votes)``.

    Parameters
    ----------
    prm:
        A :class:`NextStatePRM` used to majority-vote each step's next-state
        signal into ``{+1, 0, −1}``.
    cfg:
        Aggregation weights / clipping.
    outcome_fn:
        Optional ``callable(trajectory) -> float`` overriding the metadata-key
        outcome lookup (takes precedence when provided).
    weight:
        Component weight inside the RewardManager's weighted sum.
    """

    name = "process_reward"

    def __init__(
        self,
        prm: NextStatePRM,
        cfg: ProcessRewardConfig | None = None,
        *,
        outcome_fn: Callable[[Trajectory], float] | None = None,
        weight: float = 1.0,
    ) -> None:
        self.prm = prm
        self.cfg = cfg or ProcessRewardConfig()
        self._outcome_fn = outcome_fn
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del item, tool_context
        cfg = self.cfg
        runtime = trajectory.metadata.get("runtime") or {}
        if not isinstance(runtime, dict):
            runtime = {}

        step_states = _coerce_step_states(runtime)
        response = trajectory.final_output or ""

        # Score every step's next-state signal via the m-vote PRM, in parallel.
        process_mean = 0.0
        n_steps = 0
        if step_states:
            scores = await asyncio.gather(
                *(self.prm.score(response, ns) for ns in step_states)
            )
            votes = [float(score) for score, _hint in scores]
            n_steps = len(votes)
            process_mean = sum(votes) / n_steps if n_steps else 0.0

        outcome = self._resolve_outcome(trajectory)
        final = cfg.outcome_weight * outcome + cfg.process_weight * process_mean
        if cfg.clip is not None:
            c = abs(float(cfg.clip))
            final = max(-c, min(c, final))

        return RewardResult(
            name=self.name,
            score=float(final) * self.weight,
            weight=self.weight,
            reason=(
                f"o={outcome:.3f} process_mean={process_mean:.3f} "
                f"n_steps={n_steps}"
            ),
        )

    def _resolve_outcome(self, trajectory: Trajectory) -> float:
        if self._outcome_fn is not None:
            try:
                return float(self._outcome_fn(trajectory))
            except Exception:
                return float(self.cfg.outcome_default)
        key = self.cfg.outcome_metadata_key
        if key:
            val = trajectory.metadata.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
        return float(self.cfg.outcome_default)


def _coerce_step_states(runtime: dict[str, Any]) -> list[str]:
    """Pull per-step next-state strings, falling back to the single signal."""
    raw = runtime.get("step_next_states")
    if isinstance(raw, (list, tuple)):
        out = [str(s) for s in raw if isinstance(s, str) and s.strip()]
        if out:
            return out
    single = runtime.get("next_state")
    if isinstance(single, str) and single.strip():
        return [single]
    return []
