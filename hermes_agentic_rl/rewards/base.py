from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory


class BaseReward(ABC):
    name: str

    @abstractmethod
    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        raise NotImplementedError
