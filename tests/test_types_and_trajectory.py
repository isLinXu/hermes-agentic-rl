from hermes_agentic_rl.core.trajectory import trajectory_from_dict, trajectory_to_dict
from hermes_agentic_rl.core.types import RewardResult, RolloutStep, Trajectory


def test_trajectory_roundtrip():
    trajectory = Trajectory(
        task_id="task-1",
        prompt="create file",
        steps=[
            RolloutStep(
                turn_index=0,
                assistant_message="I will create the file",
                tool_calls=[{"name": "write_file", "arguments": {"path": "a.txt"}}],
                tool_results=[{"ok": True}],
            )
        ],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
        metadata={"source": "test"},
    )

    payload = trajectory_to_dict(trajectory)
    restored = trajectory_from_dict(payload)

    assert restored.task_id == "task-1"
    assert restored.steps[0].tool_calls[0]["name"] == "write_file"
    assert restored.metadata["source"] == "test"


def test_reward_result_defaults():
    result = RewardResult(name="outcome_reward", score=1.0, reason="ok")
    assert result.metadata == {}
