from __future__ import annotations

from typing import Any


def coerce_float(value: Any, *, default: float = 0.0) -> float:
    """Best-effort float conversion with a deterministic fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
