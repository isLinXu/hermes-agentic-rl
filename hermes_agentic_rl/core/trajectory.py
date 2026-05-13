from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RolloutStep, Trajectory


def trajectory_to_dict(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "task_id": trajectory.task_id,
        "prompt": trajectory.prompt,
        "steps": [
            {
                "turn_index": step.turn_index,
                "assistant_message": step.assistant_message,
                "tool_calls": step.tool_calls,
                "tool_results": step.tool_results,
                "reasoning": step.reasoning,
                "metadata": step.metadata,
            }
            for step in trajectory.steps
        ],
        "final_output": trajectory.final_output,
        "finished_naturally": trajectory.finished_naturally,
        "turns_used": trajectory.turns_used,
        "metadata": trajectory.metadata,
    }


def trajectory_from_dict(payload: dict[str, Any]) -> Trajectory:
    return Trajectory(
        task_id=payload["task_id"],
        prompt=payload["prompt"],
        steps=[
            RolloutStep(
                turn_index=step["turn_index"],
                assistant_message=step.get("assistant_message"),
                tool_calls=step.get("tool_calls", []),
                tool_results=step.get("tool_results", []),
                reasoning=step.get("reasoning"),
                metadata=step.get("metadata", {}),
            )
            for step in payload.get("steps", [])
        ],
        final_output=payload.get("final_output"),
        finished_naturally=payload["finished_naturally"],
        turns_used=payload["turns_used"],
        metadata=payload.get("metadata", {}),
    )
