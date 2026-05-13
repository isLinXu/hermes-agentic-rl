from __future__ import annotations

from collections import defaultdict
from typing import Any


class Registry:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = defaultdict(dict)

    def register(self, namespace: str, name: str, value: Any) -> None:
        if name in self._store[namespace]:
            raise ValueError(f"{namespace}:{name} already registered")
        self._store[namespace][name] = value

    def get(self, namespace: str, name: str) -> Any:
        try:
            return self._store[namespace][name]
        except KeyError as exc:
            raise KeyError(f"{namespace}:{name} is not registered") from exc

    def list_names(self, namespace: str) -> list[str]:
        return sorted(self._store[namespace].keys())
