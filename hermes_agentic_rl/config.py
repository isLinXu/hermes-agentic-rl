from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Known integration names for config validation.
# ---------------------------------------------------------------------------
_KNOWN_INTEGRATIONS = {"fake", "hf", "vllm", "sglang", "openai", "anthropic"}


class ConfigValidationError(ValueError):
    """Raised when a config dict fails structural validation."""


def validate_config(
    cfg: dict[str, Any],
    *,
    strict: bool = True,
) -> list[str]:
    """Validate a config dict and return a list of error messages.

    If *strict* is True (default), raise :class:`ConfigValidationError` when
    any errors are found. If *strict* is False, return the error list silently
    so callers can inspect individual issues.

    Validated fields:

    * ``runtime.integration`` — must be one of the known integrations.
    * ``runtime.max_agent_turns`` — must be a positive integer.
    * ``trainer.n_iters`` — must be a positive integer (if present).
    * ``trainer.lr`` — must be a positive float (if present).
    """
    errors: list[str] = []

    runtime = cfg.get("runtime", {})
    integration = runtime.get("integration", "fake")
    if integration not in _KNOWN_INTEGRATIONS:
        errors.append(
            f"Unknown runtime.integration: {integration!r} "
            f"(expected one of {sorted(_KNOWN_INTEGRATIONS)})"
        )

    max_turns = runtime.get("max_agent_turns", 20)
    if not isinstance(max_turns, int) or max_turns < 1:
        errors.append(
            f"runtime.max_agent_turns must be a positive integer, got {max_turns!r}"
        )

    trainer = cfg.get("trainer", {})
    n_iters = trainer.get("n_iters")
    if n_iters is not None and (not isinstance(n_iters, int) or n_iters < 1):
        errors.append(f"trainer.n_iters must be a positive integer, got {n_iters!r}")

    lr = trainer.get("lr")
    if lr is not None and (not isinstance(lr, (int, float)) or lr <= 0):
        errors.append(f"trainer.lr must be a positive number, got {lr!r}")

    if strict and errors:
        raise ConfigValidationError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return errors


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    runtime = config.setdefault("runtime", {})
    runtime.setdefault("integration", "fake")
    runtime.setdefault("max_agent_turns", 20)
    runtime.setdefault("enabled_toolsets", ["terminal", "file"])

    environment = config.setdefault("environment", {})
    trainer = config.setdefault("trainer", {})
    reward = config.setdefault("reward", {})
    reward.setdefault("aggregator", "weighted_sum")

    return {
        "runtime": runtime,
        "environment": environment,
        "reward": reward,
        "trainer": trainer,
    }
