from __future__ import annotations

from hermes_agentic_rl.core.types import RewardResult, RewardSummary


def weighted_sum(results: list[RewardResult]) -> RewardSummary:
    if not results:
        return RewardSummary(
            final_score=0.0,
            components=[],
            metadata={"aggregator": "weighted_sum"},
        )

    total_weight = sum(result.weight for result in results)
    if total_weight <= 0:
        raise ValueError("total reward weight must be positive")

    final_score = sum(result.score * result.weight for result in results) / total_weight
    return RewardSummary(
        final_score=final_score,
        components=results,
        metadata={"aggregator": "weighted_sum", "total_weight": total_weight},
    )
