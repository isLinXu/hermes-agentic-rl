from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRuntimeAdapter(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def describe_unavailable_reason(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_agent_loop(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError
