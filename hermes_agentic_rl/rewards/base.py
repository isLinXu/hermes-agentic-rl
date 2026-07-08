from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hermes_agentic_rl._compat import stable
from hermes_agentic_rl.core.types import RewardResult, Trajectory


@stable
class BaseReward(ABC):
    name: str

    def __init__(self, **kwargs: Any) -> None:
        """Optional hook for subclasses; kwargs are ignored by default."""
        del kwargs

    @abstractmethod
    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        raise NotImplementedError
