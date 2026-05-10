from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RolloutStep, Trajectory


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
