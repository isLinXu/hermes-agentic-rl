from __future__ import annotations

import math

from hermes_agentic_rl.algos.base import RolloutRecord


def rollout_score_temperature(records: list[RolloutRecord]) -> float:
    """Return the shared rollout temperature for a training batch.

    Rollout logprobs are recorded under the sampling distribution. When
    temperature differs from 1.0, PPO/GRPO must score the same sampled tokens
    under the same temperature-scaled policy before forming ratios.
    """
    temperatures: list[float] = []
    for record in records:
        raw = record.metadata.get("rollout_temperature")
        if raw is None:
            raw = record.metadata.get("temperature")
        if raw is None or isinstance(raw, bool):
            continue
        if isinstance(raw, int | float):
            value = float(raw)
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite rollout temperature: {raw!r}")
            temperatures.append(value)

    if not temperatures:
        return 1.0

    first = temperatures[0]
    for value in temperatures[1:]:
        if abs(value - first) > 1e-8:
            raise RuntimeError(
                "mixed rollout temperatures in one update batch are not supported "
                f"({first} vs {value}); split batches by temperature first"
            )
    return first
