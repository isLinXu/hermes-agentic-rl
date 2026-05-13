from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    out = dict(message)
    out.setdefault("role", "user")
    if out.get("content") is None:
        out["content"] = ""
    return out


@dataclass(slots=True)
class SessionTurnSample:
    session_id: str
    task_id: str | None
    turn_index: int
    prompt_messages: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    feedback_messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


def collect_session_turn_samples(
    messages: list[dict[str, Any]],
    *,
    session_id: str,
    task_id: str | None = None,
) -> list[SessionTurnSample]:
    """Split a chat transcript into Hermes session turn samples.

    Each sample captures:
    - all messages before an assistant turn as the prompt context
    - the assistant turn itself as the action
    - the following user/tool feedback until the next assistant turn
    """
    normalized = [_normalize_message(m) for m in messages]
    samples: list[SessionTurnSample] = []

    for idx, message in enumerate(normalized):
        if message.get("role") != "assistant":
            continue

        feedback_messages: list[dict[str, Any]] = []
        for follower in normalized[idx + 1 :]:
            role = follower.get("role")
            if role == "assistant":
                break
            if role in {"tool", "user", "system"}:
                feedback_messages.append(follower)

        samples.append(
            SessionTurnSample(
                session_id=session_id,
                task_id=task_id,
                turn_index=len(samples),
                prompt_messages=[dict(m) for m in normalized[:idx]],
                assistant_message=dict(message),
                feedback_messages=[dict(m) for m in feedback_messages],
                metadata={
                    "assistant_index": idx,
                    "feedback_roles": [str(m.get("role")) for m in feedback_messages],
                },
            )
        )

    return samples
