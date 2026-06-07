from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RolloutStep, Trajectory


def _extract_assistant_messages_by_turn(
    messages: list[dict[str, Any]],
    *,
    turn_count: int,
) -> list[str | None]:
    """Extract assistant message content for each turn.

    If a message has an explicit ``turn_index`` key, use that for routing.
    Otherwise, assign assistant messages to turns in positional order (skipping
    non-assistant messages).

    Returns a list of length *turn_count* where each element is the assistant
    content string for that turn, or ``None`` if no assistant message was found
    for that turn or the content is not a string.
    """
    result: list[str | None] = [None] * turn_count

    # First pass: messages with explicit turn_index take priority.
    positional_queue: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        ti = msg.get("turn_index")
        if isinstance(ti, int) and 0 <= ti < turn_count:
            content = msg.get("content")
            result[ti] = content if isinstance(content, str) else None
        else:
            positional_queue.append(msg)

    # Second pass: fill remaining None slots from positional assistant messages.
    pos_idx = 0
    for i in range(turn_count):
        if result[i] is not None:
            continue
        while pos_idx < len(positional_queue):
            content = positional_queue[pos_idx].get("content")
            pos_idx += 1
            if isinstance(content, str):
                result[i] = content
                break

    return result


class RolloutManager:
    def __init__(self, agent_loop: Any) -> None:
        self.agent_loop = agent_loop

    async def collect(self, item: dict[str, Any], prompt: str) -> Trajectory:
        raw = await self.agent_loop.run(prompt)
        tool_calls_per_turn = raw.get("tool_calls", [])
        tool_results_per_turn = raw.get("tool_results", [])
        messages = raw.get("messages", [])
        runtime_metadata = raw.get("metadata", {})

        steps: list[RolloutStep] = []
        turn_count = max(
            len(tool_calls_per_turn),
            len(tool_results_per_turn),
            raw.get("turns_used", 0),
        )
        for index in range(turn_count):
            steps.append(
                RolloutStep(
                    turn_index=index,
                    assistant_message=None,
                    tool_calls=(
                        tool_calls_per_turn[index]
                        if index < len(tool_calls_per_turn)
                        else []
                    ),
                    tool_results=(
                        tool_results_per_turn[index]
                        if index < len(tool_results_per_turn)
                        else []
                    ),
                )
            )

        return Trajectory(
            task_id=item["task_id"],
            prompt=prompt,
            steps=steps,
            final_output=raw.get("final_output"),
            finished_naturally=raw.get("finished_naturally", False),
            turns_used=raw.get("turns_used", 0),
            metadata={"messages": messages, "runtime": runtime_metadata},
        )
