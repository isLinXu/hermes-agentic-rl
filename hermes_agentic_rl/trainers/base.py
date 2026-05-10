from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hermes_agentic_rl.core.types import RewardSummary, Trajectory


class BaseTrainer(ABC):
    """Legacy trainer interface (exporter-style).

    This interface matches `exporters.BaseExporter.submit(item, traj, reward)`
    and is kept for backward compatibility with `AtroposGrpoTrainer`.

    A real RL trainer (owning a policy + optimizer) implements a richer
    interface — see `hermes_agentic_rl.trainers.grpo_trainer.GRPOTrainer`.
    """

    @abstractmethod
    async def submit(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        raise NotImplementedError
