import asyncio

from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward


def test_reward_manager_runs_components_and_aggregates():
    trajectory = Trajectory(
        task_id="task-1",
        prompt="create file",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )

    manager = RewardManager(
        rewards=[OutcomeReward(weight=0.7), ToolcallReward(weight=0.3)]
    )

    summary = asyncio.run(
        manager.evaluate({"expected_output": "done"}, trajectory, tool_context=None)
    )

    assert round(summary.final_score, 4) == 0.7
    assert len(summary.components) == 2
