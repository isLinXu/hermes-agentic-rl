from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hermes_agentic_rl.core.types import RewardSummary, Trajectory


class BaseExporter(ABC):
    """Exporter: serialize (item, trajectory, reward) → downstream trainer format.

    Exporters are side-effectful writers (to disk, stream, or queue). They do NOT
    train a policy. Naming them `Trainer` is incorrect and will be removed in 0.2.
    """

    @abstractmethod
    async def submit(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def submit_sync(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        raise NotImplementedError
