import asyncio

from hermes_agentic_rl.core.types import RolloutStep, Trajectory
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward


def _trajectory_with_calls(calls: list[dict]) -> Trajectory:
    return Trajectory(
        task_id="t",
        prompt="p",
        steps=[
            RolloutStep(
                turn_index=0,
                assistant_message=None,
                tool_calls=calls,
                tool_results=[],
            )
        ],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )


def test_toolcall_reward_accepts_openai_function_name_format():
    reward = ToolcallReward(weight=1.0)
    traj = _trajectory_with_calls(
        [
            {
                "type": "function",
                "function": {"name": "write_file", "arguments": "{}"},
            }
        ]
    )
    result = asyncio.run(reward.evaluate(item={}, trajectory=traj, tool_context=None))
    assert result.score == 1.0


def test_toolcall_reward_rejects_call_with_no_name_anywhere():
    reward = ToolcallReward(weight=1.0)
    traj = _trajectory_with_calls([{"type": "function", "function": {"arguments": "{}"}}])
    result = asyncio.run(reward.evaluate(item={}, trajectory=traj, tool_context=None))
    assert result.score == 0.0


def test_outcome_reward_accepts_successful_natural_language_when_expected_done():
    reward = OutcomeReward(weight=1.0)
    traj = Trajectory(
        task_id="t",
        prompt="p",
        steps=[],
        final_output="I've successfully created hello.txt with content hello.",
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )
    item = {"expected_output": "done", "instruction": "Create hello.txt and write hello"}
    result = asyncio.run(reward.evaluate(item=item, trajectory=traj, tool_context=None))
    assert result.score == 1.0
