"""Lightweight per-iteration profiling helpers for trainer hot paths."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _metric_name(name: str) -> str:
    return "time_" + "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") + "_ms"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


class StepProfiler:
    """Accumulates named wall-clock timings for one training iteration."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._durations_ms: dict[str, float] = {}

    def reset(self) -> None:
        self._durations_ms.clear()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            key = _metric_name(name)
            self._durations_ms[key] = self._durations_ms.get(key, 0.0) + elapsed_ms

    def as_metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return {k: float(v) for k, v in self._durations_ms.items()}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON-safe record to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(record)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def format_json_record(record: dict[str, Any]) -> str:
    """Return a JSON-line formatted training record."""

    return json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True)
