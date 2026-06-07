from __future__ import annotations

import asyncio
import warnings
from typing import Any

from hermes_agentic_rl.core.types import RewardSummary, Trajectory
from hermes_agentic_rl.trainers.base import BaseTrainer


class TrainerBridge:
    """Async wrapper around a :class:`BaseTrainer` with retry and backoff.

    Parameters
    ----------
    trainer:
        The underlying trainer (must have an ``async submit`` method).
    max_retries:
        Maximum number of retry attempts (must be >= 1). The initial call
        counts as the first attempt, so ``max_retries=1`` means "try once,
        no retries".
    backoff_base:
        Base delay in seconds for exponential backoff between retries.
        0.0 disables backoff (immediate retry).
    backoff_max:
        Maximum backoff delay in seconds.
    """

    def __init__(
        self,
        trainer: BaseTrainer,
        *,
        max_retries: int = 1,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.trainer = trainer
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._submitted = 0
        self._failed = 0
        self._retried = 0
        self._last_error: str | None = None

    async def submit(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        reward_summary: RewardSummary,
    ) -> dict[str, Any]:
        """Submit with retry and exponential backoff.

        Cancellation errors (``asyncio.CancelledError``) are never retried
        and always propagate immediately.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = await self.trainer.submit(item, trajectory, reward_summary)
                self._submitted += 1
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                self._last_error = str(exc)
                if attempt < self.max_retries - 1:
                    self._retried += 1
                    delay = min(
                        self.backoff_base * (2 ** attempt),
                        self.backoff_max,
                    )
                    if delay > 0:
                        warnings.warn(
                            f"TrainerBridge retry {attempt + 1}/{self.max_retries}: "
                            f"{exc!r}; sleeping {delay:.2f}s",
                            stacklevel=2,
                        )
                        await asyncio.sleep(delay)

        self._failed += 1
        assert last_exc is not None
        raise last_exc

    def metrics(self) -> dict[str, Any]:
        """Return a snapshot of retry / error counters."""
        return {
            "submitted": self._submitted,
            "failed": self._failed,
            "retried": self._retried,
            "last_error": self._last_error,
        }
