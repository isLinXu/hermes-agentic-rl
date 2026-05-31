from __future__ import annotations

import asyncio

import pytest

from hermes_agentic_rl.cli.train_rl import _build_env_and_rewards
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.length_penalty import (
    LengthPenaltyConfig,
    LengthPenaltyReward,
)


def _traj(*, prompt_tokens: int = 2, response_tokens: int = 6) -> Trajectory:
    return Trajectory(
        task_id="t",
        prompt="prompt",
        steps=[],
        final_output="fallback text",
        finished_naturally=True,
        turns_used=1,
        metadata={
            "runtime": {
                "rl": {
                    "prompt_ids": list(range(prompt_tokens)),
                    "response_ids": list(range(response_tokens)),
                }
            }
        },
    )


def test_length_penalty_linear_response_tokens():
    reward = LengthPenaltyReward(
        LengthPenaltyConfig(target_len=4, alpha=0.2, mode="linear")
    )

    result = asyncio.run(reward.evaluate({}, _traj(response_tokens=6), None))

    assert result.score == pytest.approx(-0.1)
    assert result.metadata["response_tokens"] == 6
    assert result.metadata["excess_ratio"] == pytest.approx(0.5)


def test_length_penalty_quadratic_total_tokens():
    reward = LengthPenaltyReward(
        LengthPenaltyConfig(
            target_len=4,
            alpha=0.2,
            mode="quadratic",
            apply_on="total",
        )
    )

    result = asyncio.run(
        reward.evaluate({}, _traj(prompt_tokens=2, response_tokens=4), None)
    )

    assert result.score == pytest.approx(-0.05)
    assert result.metadata["length"] == 6


def test_build_env_and_rewards_adds_length_penalty_component():
    _env, rewards = _build_env_and_rewards(
        {
            "environment": {"type": "echo"},
            "reward": {
                "components": [
                    {
                        "type": "LengthPenaltyReward",
                        "target_len": 4,
                        "alpha": 0.2,
                        "weight": 0.5,
                    }
                ]
            },
        }
    )

    assert any(isinstance(component, LengthPenaltyReward) for component in rewards.rewards)
