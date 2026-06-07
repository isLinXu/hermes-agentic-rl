"""ExperienceQueue — the decoupling primitive for asynchronous RL.

OpenClaw-RL's moat is a *fully decoupled* pipeline: Policy Serving, Environment
Hosting, Judge, and Training run as independent components with no coordination
barrier ("train while serving, zero interruption"). hermes-agentic-rl is
synchronous BSP (collect → update, in lockstep). This module is the first piece
of the async path: a thread/process-safe queue that lets *producers* (rollout +
judge workers) push scored experience while a *consumer* (the learner) pulls and
updates at its own pace.

Each item carries the **behavior policy version** it was sampled under so the
learner can compute *staleness* = ``learner_version − behavior_version`` and
apply an off-policy correction (TIS / V-trace, see ``algos/common/vtrace.py``).

Design
------
* stdlib ``queue.Queue`` backing → testable in-process; a Ray/mp queue can be
  swapped behind the same ``put`` / ``get_batch`` surface later.
* **bounded** with a *drop-oldest* policy: when full, the staleest item is
  evicted so the learner never trains on arbitrarily old data (staleness is
  bounded by ``maxsize`` × throughput). Drops are counted, not silent.
* ``get_batch`` drains up to ``n`` items (blocking for at least one, then
  non-blocking for the rest) so the learner forms a minibatch from whatever has
  arrived — the natural async batching.
* sentinel-based clean shutdown.

This is intentionally infrastructure-only: it does not import torch and has no
knowledge of the algorithm, so it stays cheap to test and reuse.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_SHUTDOWN = object()  # sentinel pushed to wake a blocked consumer on close()


@dataclass(slots=True)
class ExperienceItem(Generic[T]):
    """One unit of experience produced asynchronously.

    payload: the actual experience (e.g. a list[RolloutRecord]).
    behavior_version: learner policy version the rollout was sampled under.
    produced_at: monotonic timestamp (for lag observability).
    meta: free-form producer annotations (worker id, env tag, …).
    """

    payload: T
    behavior_version: int
    produced_at: float = field(default_factory=time.monotonic)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueueStats:
    produced: int = 0
    consumed: int = 0
    dropped_oldest: int = 0
    max_depth_seen: int = 0
    last_staleness: int = 0
    staleness_sum: int = 0
    staleness_count: int = 0

    @property
    def mean_staleness(self) -> float:
        return (
            float(self.staleness_sum) / float(self.staleness_count)
            if self.staleness_count
            else 0.0
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "queue_produced": float(self.produced),
            "queue_consumed": float(self.consumed),
            "queue_dropped_oldest": float(self.dropped_oldest),
            "queue_max_depth_seen": float(self.max_depth_seen),
            "queue_last_staleness": float(self.last_staleness),
            "queue_mean_staleness": self.mean_staleness,
        }


class ExperienceQueue(Generic[T]):
    """Bounded, thread-safe, drop-oldest experience queue with staleness.

    Producers call :meth:`put`; the consumer calls :meth:`get_batch` and reads
    :meth:`learner_version` / :meth:`set_learner_version` to track its own
    policy version for staleness accounting.
    """

    def __init__(self, maxsize: int = 64) -> None:
        self.maxsize = max(1, int(maxsize))
        self._q: queue.Queue[Any] = queue.Queue(maxsize=self.maxsize)
        self._lock = threading.Lock()
        self._learner_version = 0
        self._closed = False
        self.stats = QueueStats()

    # -- learner version bookkeeping --------------------------------------

    def set_learner_version(self, version: int) -> None:
        with self._lock:
            self._learner_version = int(version)

    @property
    def learner_version(self) -> int:
        with self._lock:
            return self._learner_version

    # -- producer side ----------------------------------------------------

    def put(self, item: ExperienceItem[T]) -> None:
        """Enqueue an item; evict the oldest on overflow (bounded staleness)."""
        if self._closed:
            raise RuntimeError("put() on a closed ExperienceQueue")
        while True:
            try:
                self._q.put_nowait(item)
                break
            except queue.Full:
                # Drop the oldest item to make room — keeps the consumer on
                # fresher data instead of blocking the producer.
                try:
                    self._q.get_nowait()
                    with self._lock:
                        self.stats.dropped_oldest += 1
                except queue.Empty:  # pragma: no cover — race with consumer
                    pass
        with self._lock:
            self.stats.produced += 1
            self.stats.max_depth_seen = max(
                self.stats.max_depth_seen, self._q.qsize()
            )

    # -- consumer side ----------------------------------------------------

    def get_batch(
        self,
        n: int = 1,
        *,
        timeout: float | None = 1.0,
    ) -> list[ExperienceItem[T]]:
        """Drain up to ``n`` items.

        Blocks up to ``timeout`` for the *first* item, then takes whatever else
        is already available without blocking. Returns ``[]`` on timeout. A
        shutdown sentinel ends the batch early (and is not returned).
        """
        out: list[ExperienceItem[T]] = []
        try:
            first = self._q.get(timeout=timeout)
        except queue.Empty:
            return out
        if first is _SHUTDOWN:
            return out
        out.append(first)
        while len(out) < n:
            try:
                nxt = self._q.get_nowait()
            except queue.Empty:
                break
            if nxt is _SHUTDOWN:
                break
            out.append(nxt)
        self._record_consumed(out)
        return out

    def _record_consumed(self, items: list[ExperienceItem[T]]) -> None:
        with self._lock:
            lv = self._learner_version
            for it in items:
                staleness = max(0, lv - int(it.behavior_version))
                self.stats.consumed += 1
                self.stats.last_staleness = staleness
                self.stats.staleness_sum += staleness
                self.stats.staleness_count += 1

    @staticmethod
    def staleness_of(item: ExperienceItem[T], learner_version: int) -> int:
        return max(0, int(learner_version) - int(item.behavior_version))

    # -- lifecycle --------------------------------------------------------

    def depth(self) -> int:
        return self._q.qsize()

    def close(self) -> None:
        """Signal consumers to stop. Idempotent."""
        self._closed = True
        try:
            self._q.put_nowait(_SHUTDOWN)
        except queue.Full:  # pragma: no cover
            try:
                self._q.get_nowait()
                self._q.put_nowait(_SHUTDOWN)
            except queue.Empty:
                pass
