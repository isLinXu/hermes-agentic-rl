from __future__ import annotations

from typing import Any

from hermes_agentic_rl.backends.base import TokenizerProtocol
from hermes_agentic_rl.collectors.conversation_collector import (
    SessionTurnSample,
    collect_session_turn_samples,
)
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.offline.replay_buffer import ReplayBuffer, TrainSample


def render_messages_for_training(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def trajectory_to_session_turn_samples(trajectory: Trajectory) -> list[SessionTurnSample]:
    runtime = trajectory.metadata.get("runtime", {})
    session_id = None
    if isinstance(runtime, dict):
        session_id = runtime.get("task_id") or runtime.get("session_id")
    if not session_id:
        session_id = trajectory.task_id

    messages = trajectory.metadata.get("messages")
    if not isinstance(messages, list):
        return []

    safe_messages: list[dict[str, Any]] = [dict(m) for m in messages if isinstance(m, dict)]
    return collect_session_turn_samples(
        safe_messages,
        session_id=str(session_id),
        task_id=trajectory.task_id,
    )


def session_turn_sample_to_train_sample(
    sample: SessionTurnSample,
    *,
    tokenizer: TokenizerProtocol,
    reward: float | None = None,
    reward_summary: Any | None = None,
) -> TrainSample:
    prompt_text = render_messages_for_training(
        [*sample.prompt_messages, {"role": "assistant", "content": ""}]
    )
    response_content = sample.assistant_message.get("content", "")
    if not isinstance(response_content, str):
        response_content = str(response_content)
    summary = reward_summary
    sample_reward = reward
    if summary is None and sample_reward is None:
        from hermes_agentic_rl.collectors.session_judge import judge_session_turn_sample

        summary = judge_session_turn_sample(sample)
    if sample_reward is None:
        if summary is not None:
            sample_reward = float(summary.final_score)
        else:
            from hermes_agentic_rl.rewards.next_turn_feedback import score_feedback_messages

            sample_reward, _reason = score_feedback_messages(sample.feedback_messages)
    return TrainSample(
        prompt_ids=tokenizer.encode(prompt_text),
        response_ids=tokenizer.encode(response_content),
        reward=float(sample_reward),
        metadata={
            "session_id": sample.session_id,
            "task_id": sample.task_id,
            "turn_index": sample.turn_index,
            "source_turn": {
                "prompt_messages": [dict(m) for m in sample.prompt_messages],
                "assistant_message": dict(sample.assistant_message),
                "feedback_messages": [dict(m) for m in sample.feedback_messages],
            },
            "feedback_messages": [dict(m) for m in sample.feedback_messages],
            "reward_components": [
                {
                    "name": component.name,
                    "score": component.score,
                    "weight": component.weight,
                    "reason": component.reason,
                }
                for component in (summary.components if summary is not None else [])
            ],
            "reward_summary": (summary.metadata if summary is not None else {}),
        },
    )


def session_turn_samples_to_replay_buffer(
    samples: list[SessionTurnSample],
    *,
    tokenizer: TokenizerProtocol,
) -> ReplayBuffer:
    buffer = ReplayBuffer()
    for sample in samples:
        buffer.add_sample(session_turn_sample_to_train_sample(sample, tokenizer=tokenizer))
    return buffer
