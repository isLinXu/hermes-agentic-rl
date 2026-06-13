"""TrainStats — lightweight per-iteration metric accumulator.

Extracted from ``on_policy.py`` (v0.10 → v0.11 refactor) to reduce that
module's line count and make the stats object importable without pulling in
the full trainer dependency tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainStats:
    iters: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def add(self, record: dict[str, Any]) -> None:
        self.iters.append(record)

    def best_reward(self) -> float:
        return max((r["mean_reward"] for r in self.iters), default=0.0)

    def last_reward(self) -> float:
        return self.iters[-1]["mean_reward"] if self.iters else 0.0

    def mean_reward_delta(self) -> float:
        if len(self.iters) < 2:
            return 0.0
        return self.iters[-1]["mean_reward"] - self.iters[0]["mean_reward"]

    def reward_curve(self) -> list[float]:
        """Return the per-iteration mean_reward series."""
        return [float(r.get("mean_reward", 0.0)) for r in self.iters]

    def mean_kl(self) -> float:
        """Average KL divergence across all iterations."""
        kls = [float(r.get("kl", 0.0)) for r in self.iters]
        return sum(kls) / len(kls) if kls else 0.0

    def loss_curve(self) -> list[float]:
        """Return the per-iteration loss series."""
        return [float(r.get("loss", 0.0)) for r in self.iters]

    def get_column(self, key: str) -> list[Any]:
        """Extract a single column from all iteration records."""
        return [r.get(key) for r in self.iters]

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with key training statistics."""
        return {
            "n_iters": len(self.iters),
            "best_reward": self.best_reward(),
            "last_reward": self.last_reward(),
            "mean_kl": self.mean_kl(),
        }

    def to_dataframe(self):
        """Convert to a pandas DataFrame (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(self.iters)
