"""Training statistics container.

Extracted from ``on_policy.py`` to reduce coupling and allow reuse
across trainer variants (GRPO, PPO, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainStats:
    """Accumulates per-iteration records and exposes convenience views."""

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

    # v1.0 additions ──────────────────────────────────────────────────────────

    def reward_curve(self) -> list[float]:
        """Return per-iteration mean_reward as a list (chronological order)."""
        return [float(r.get("mean_reward", 0.0)) for r in self.iters]

    def mean_kl(self) -> float:
        """Average KL divergence over all logged iterations (0.0 if none)."""
        kls = [float(r.get("kl", 0.0)) for r in self.iters if "kl" in r]
        return sum(kls) / len(kls) if kls else 0.0

    def loss_curve(self) -> list[float]:
        """Return per-iteration loss as a list."""
        return [float(r.get("loss", 0.0)) for r in self.iters]

    def get_column(self, key: str, default: float = 0.0) -> list[float]:
        """Extract any scalar column across iterations.

        Example::

            lrs = stats.get_column("lr")
            grad_norms = stats.get_column("grad_norm")
        """
        return [float(r.get(key, default)) for r in self.iters]

    def summary(self) -> dict[str, Any]:
        """Return a compact summary dict for quick inspection."""
        return {
            "n_iters": len(self.iters),
            "best_reward": self.best_reward(),
            "last_reward": self.last_reward(),
            "mean_reward_delta": self.mean_reward_delta(),
            "mean_kl": self.mean_kl(),
        }

    def to_dataframe(self) -> Any:
        """Convert iteration records to a pandas DataFrame.

        Raises ``ImportError`` if pandas is not installed. This is an
        optional dependency — call sites should handle the exception.

        Example::

            df = stats.to_dataframe()
            df[["iter", "mean_reward", "loss", "kl"]].plot()
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pandas is required for TrainStats.to_dataframe(). "
                "Install it with: pip install pandas"
            ) from exc
        return pd.DataFrame(self.iters)
