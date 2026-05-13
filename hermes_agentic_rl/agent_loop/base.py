from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgentLoop(ABC):
    """Common shape for any agent loop consumed by RolloutManager.

    `run(prompt)` must return a dict with the keys:
      - messages, tool_calls, tool_results, final_output,
        finished_naturally, turns_used, metadata
    matching the contract in `runtime.hermes_wrapper`.
    """

    @abstractmethod
    async def run(self, prompt: str) -> dict[str, Any]:
        raise NotImplementedError
