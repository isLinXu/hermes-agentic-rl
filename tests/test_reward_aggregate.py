from hermes_agentic_rl.core.types import RewardResult
from hermes_agentic_rl.rewards.aggregate import weighted_sum


def test_weighted_sum_uses_component_weights():
    summary = weighted_sum(
        [
            RewardResult(name="a", score=1.0, reason="ok", weight=0.75),
            RewardResult(name="b", score=0.5, reason="partial", weight=0.25),
        ]
    )

    assert round(summary.final_score, 4) == 0.875
    assert len(summary.components) == 2
