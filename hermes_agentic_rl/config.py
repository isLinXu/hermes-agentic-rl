from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


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
