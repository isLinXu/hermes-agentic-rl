from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory


class BaseEnv(ABC):
    @abstractmethod
    async def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_next_item(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def format_prompt(self, item: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        raise NotImplementedError
