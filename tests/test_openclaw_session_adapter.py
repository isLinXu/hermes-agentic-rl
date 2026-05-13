from hermes_agentic_rl.backends.tiny import TinyTokenizer
from hermes_agentic_rl.collectors import (
    build_session_reward_results,
    collect_session_turn_samples,
    judge_session_turn_sample,
    render_messages_for_training,
    session_turn_sample_to_train_sample,
    session_turn_samples_to_replay_buffer,
    trajectory_to_session_turn_samples,
)
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.next_turn_feedback import (
    NextTurnFeedbackReward,
    score_feedback_messages,
)


def test_collect_session_turn_samples_extracts_feedback_until_next_assistant():
    messages = [
        {"role": "user", "content": "Fix the file"},
        {"role": "assistant", "content": "I'll inspect it."},
        {"role": "tool", "content": "ok: opened file"},
        {"role": "user", "content": "Great, now update line 2."},
        {"role": "assistant", "content": "Updated."},
        {"role": "tool", "content": "error: permission denied"},
    ]

    samples = collect_session_turn_samples(messages, session_id="sess-1", task_id="task-1")

    assert len(samples) == 2
    assert samples[0].prompt_messages[0]["role"] == "user"
    assert samples[0].assistant_message["content"] == "I'll inspect it."
    assert [m["role"] for m in samples[0].feedback_messages] == ["tool", "user"]
    assert samples[1].feedback_messages[0]["content"] == "error: permission denied"


def test_trajectory_adapter_uses_runtime_task_id_as_session_id():
    trajectory = Trajectory(
        task_id="task-9",
        prompt="Do something",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
        metadata={
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": "Done"},
                {"role": "user", "content": "thanks"},
            ],
            "runtime": {"task_id": "hermes-run-123"},
        },
    )

    samples = trajectory_to_session_turn_samples(trajectory)

    assert len(samples) == 1
    assert samples[0].session_id == "hermes-run-123"
    assert samples[0].task_id == "task-9"


def test_score_feedback_messages_distinguishes_positive_and_negative_followups():
    positive_score, positive_reason = score_feedback_messages(
        [
            {"role": "tool", "content": "success: patch applied"},
            {"role": "user", "content": "great, this works"},
        ]
    )
    negative_score, negative_reason = score_feedback_messages(
        [
            {"role": "tool", "content": "error: command failed"},
            {"role": "user", "content": "this is wrong, try again"},
        ]
    )

    assert positive_score > 0
    assert "tool_success" in positive_reason
    assert negative_score < 0
    assert "user_negative" in negative_reason


async def _evaluate_reward(messages):
    reward = NextTurnFeedbackReward(weight=1.0)
    trajectory = Trajectory(
        task_id="task-r",
        prompt="prompt",
        steps=[],
        final_output="out",
        finished_naturally=True,
        turns_used=1,
        metadata={"messages": messages, "runtime": {"task_id": "sess-r"}},
    )
    return await reward.evaluate({}, trajectory, tool_context=None)


def test_next_turn_feedback_reward_scores_average_over_assistant_turns():
    import asyncio

    result = asyncio.run(
        _evaluate_reward(
            [
                {"role": "user", "content": "step 1"},
                {"role": "assistant", "content": "doing step 1"},
                {"role": "tool", "content": "ok"},
                {"role": "user", "content": "great"},
                {"role": "assistant", "content": "doing step 2"},
                {"role": "tool", "content": "error: failed"},
            ]
        )
    )

    assert result.name == "next_turn_feedback_reward"
    assert result.metadata["n_turns"] == 2
    assert -1.0 <= result.score <= 1.0
    assert "turn0:" in result.reason


def test_session_turn_samples_can_export_to_replay_buffer():
    tokenizer = TinyTokenizer()
    samples = collect_session_turn_samples(
        [
            {"role": "user", "content": "Fix it"},
            {"role": "assistant", "content": "I fixed it"},
            {"role": "user", "content": "great, thanks"},
        ],
        session_id="sess-export",
        task_id="task-export",
    )

    prompt_text = render_messages_for_training(samples[0].prompt_messages)
    train_sample = session_turn_sample_to_train_sample(samples[0], tokenizer=tokenizer)
    buffer = session_turn_samples_to_replay_buffer(samples, tokenizer=tokenizer)

    assert "<|user|>" in prompt_text
    assert train_sample.reward > 0
    assert train_sample.metadata["session_id"] == "sess-export"
    assert len(buffer.samples) == 1


def test_session_judge_produces_component_breakdown():
    sample = collect_session_turn_samples(
        [
            {"role": "user", "content": "Fix it"},
            {
                "role": "assistant",
                "content": "I fixed it",
                "tool_calls": [{"name": "write_file"}],
            },
            {"role": "tool", "content": "success: file written"},
            {"role": "user", "content": "great, thanks"},
        ],
        session_id="sess-judge",
        task_id="task-judge",
    )[0]

    results = build_session_reward_results(sample)
    summary = judge_session_turn_sample(sample)
    tokenizer = TinyTokenizer()
    train_sample = session_turn_sample_to_train_sample(
        sample,
        tokenizer=tokenizer,
        reward_summary=summary,
    )

    assert len(results) == 3
    assert summary.final_score > 0
    assert len(train_sample.metadata["reward_components"]) == 3
    assert train_sample.metadata["reward_summary"]["aggregator"] == "weighted_sum"
