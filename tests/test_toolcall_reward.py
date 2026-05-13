import asyncio

from hermes_agentic_rl.core.types import RolloutStep, Trajectory
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward


def test_outcome_reward_from_expected_output():
    reward = OutcomeReward(weight=0.7)
    trajectory = Trajectory(
        task_id="task-1",
        prompt="do x",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )

    result = asyncio.run(
        reward.evaluate({"expected_output": "done"}, trajectory, tool_context=None)
    )

    assert result.score == 1.0
    assert result.weight == 0.7


def test_toolcall_reward_penalizes_empty_calls():
    reward = ToolcallReward(weight=0.3)
    trajectory = Trajectory(
        task_id="task-1",
        prompt="do x",
        steps=[RolloutStep(turn_index=0, tool_calls=[])],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )

    result = asyncio.run(reward.evaluate({}, trajectory, tool_context=None))

    assert result.score == 0.0
    assert "no tool calls" in result.reason
