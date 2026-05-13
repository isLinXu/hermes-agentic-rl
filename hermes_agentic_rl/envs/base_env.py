from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory


@dataclass(slots=True)
class SupervisedSample:
    """A teacher-forced sample for optional interleaved SFT.

    `instruction` is encoded through PromptStateEncoder. `prompt_suffix` is
    appended verbatim after that encoded instruction prefix before scoring the
    supervised `response`. This lets multi-turn environments teach later turns
    under their true rollout context.
    """

    instruction: str
    response: str
    prompt_suffix: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


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

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        """Optional teacher samples used by interleaved SFT warm-start."""
        return []
