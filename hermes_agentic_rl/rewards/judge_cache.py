"""Judge result cache — cost control for PRM / OPD judge calls.

OpenClaw-RL's PRM does ``m`` independent judge calls per (response, next_state)
pair, and the OPD hint extractor adds one more per record. On repetitive
agentic tasks (same tool error, same terminal failure) many of these calls are
*identical*, so a content-addressed cache cuts judge cost — the dominant cost
when the judge is a hosted LLM — by 30%+ on typical workloads.

The cache is intentionally simple and dependency-free:
  * key = sha1(judge_id ⊕ response ⊕ next_state)
  * value = whatever the wrapped judge returned (hint string, PRMVote, …)
  * bounded LRU eviction (``max_entries``)
  * thread-safe (a single lock; judge calls are I/O-bound)

It wraps *any* judge callable, sync or async, via :func:`cached_judge`.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class JudgeCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return float(self.hits) / float(self.total) if self.total else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "judge_cache_hits": float(self.hits),
            "judge_cache_misses": float(self.misses),
            "judge_cache_evictions": float(self.evictions),
            "judge_cache_hit_rate": self.hit_rate,
        }


class JudgeCache:
    """Bounded, thread-safe, content-addressed cache for judge results."""

    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = max(1, int(max_entries))
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = JudgeCacheStats()

    @staticmethod
    def make_key(judge_id: str, response: str, next_state: str) -> str:
        h = hashlib.sha1()
        h.update(judge_id.encode("utf-8"))
        h.update(b"\x00")
        h.update((response or "").encode("utf-8"))
        h.update(b"\x00")
        h.update((next_state or "").encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.stats.hits += 1
                return True, self._store[key]
            self.stats.misses += 1
            return False, None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def cached_judge(
    judge_fn: Callable[[str, str], Any | Awaitable[Any]],
    cache: JudgeCache,
    *,
    judge_id: str | None = None,
) -> Callable[[str, str], Awaitable[Any]]:
    """Wrap a judge callable so identical (response, next_state) pairs hit the
    cache. The returned wrapper is always async (awaits the inner judge on a
    miss); sync judges are supported transparently.

    ``judge_id`` namespaces the cache so two different judges (e.g. a hint
    extractor and a vote judge) never collide on the same key. Defaults to the
    judge's qualified name.
    """
    jid = judge_id or getattr(judge_fn, "__qualname__", repr(judge_fn))

    async def _wrapped(response: str, next_state: str) -> Any:
        key = cache.make_key(jid, response, next_state)
        found, value = cache.get(key)
        if found:
            return value
        result = judge_fn(response, next_state)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        cache.put(key, result)
        return result

    return _wrapped
