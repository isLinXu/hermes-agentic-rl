"""Atropos-compatible JSONL exporter.

This writes each (item, trajectory, reward_summary) as one JSONL record.
It is NOT a GRPO trainer — it performs no gradient updates. For real RL
training, use `hermes_agentic_rl.trainers.grpo_trainer.GRPOTrainer`.

The previous class name `AtroposGrpoTrainer` was misleading and is kept as a
deprecated alias below.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from hermes_agentic_rl.core.trajectory import trajectory_to_dict
from hermes_agentic_rl.core.types import RewardSummary, Trajectory
from hermes_agentic_rl.exporters.base import BaseExporter


class AtroposJsonlExporter(BaseExporter):
    """Append-only JSONL writer for Atropos-style downstream consumption."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_payload(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        reward_components = [
            {
                "name": component.name,
                "score": component.score,
                "reason": component.reason,
                "weight": component.weight,
                "metadata": component.metadata,
            }
            for component in reward_summary.components
        ]

        verifier_component = next(
            (
                component
                for component in reward_summary.components
                if component.name == "filesystem_verifier_reward"
            ),
            None,
        )
        verifier_metadata = (
            verifier_component.metadata
            if verifier_component and isinstance(verifier_component.metadata, dict)
            else {}
        )

        payload: dict[str, Any] = {
            "task_id": item.get("task_id", trajectory.task_id),
            "prompt": trajectory.prompt,
            "final_output": trajectory.final_output,
            "reward": reward_summary.final_score,
            "reward_components": reward_components,
            "trajectory": trajectory_to_dict(trajectory),
            "metadata": reward_summary.metadata,
        }

        if verifier_component is not None:
            payload.update(
                {
                    "verifier_passed": bool(verifier_component.score >= 1.0),
                    "verifier_checked_files": verifier_metadata.get("checked_files"),
                    "verifier_passed_files": verifier_metadata.get("passed_files"),
                    "verifier_failures": verifier_metadata.get("failures", []),
                }
            )

        return payload

    def _write_payload(self, payload: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def submit(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        payload = self._build_payload(item, trajectory, reward_summary)
        self._write_payload(payload)
        return payload

    def submit_sync(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        payload = self._build_payload(item, trajectory, reward_summary)
        self._write_payload(payload)
        return payload


class AtroposGrpoTrainer(AtroposJsonlExporter):
    """Deprecated alias. Use `AtroposJsonlExporter` or a real trainer from
    `hermes_agentic_rl.trainers`. This class performs NO gradient updates
    despite its historical name.
    """

    def __init__(self, output_path: Path) -> None:
        warnings.warn(
            "AtroposGrpoTrainer is a JSONL exporter, not a GRPO trainer. "
            "Use `hermes_agentic_rl.exporters.AtroposJsonlExporter` for exporting, "
            "or `hermes_agentic_rl.trainers.GRPOTrainer` for real RL training.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(output_path)
