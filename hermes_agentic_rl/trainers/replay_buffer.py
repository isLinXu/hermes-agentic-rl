"""ReplayBuffer — bounded ring buffer for on-policy RL with off-policy mixing.

Why this exists
---------------
Pure on-policy RL (GRPO/PPO) discards rollouts after one update, wasting
samples. With TIS/V-trace off-policy correction already in place
(``algos/common/vtrace.py``), mixing a small fraction of recent-but-not-current
experience into each update is safe and improves sample efficiency — especially
for GRPO where group diversity matters.

Design
------
* Ring buffer with configurable capacity (``max_records``). Oldest records are
  overwritten when capacity is reached.
* **Recency-weighted sampling**: newer records are sampled with higher probability
  controlled by ``recency_alpha`` (0 = uniform, higher = exponentially favor
  recent). This works with TIS: the staleness of each record is known
  (``policy_version`` at insertion vs current), so the off-policy correction
  compensates for distribution shift.
* Thread-safe: ``push`` / ``sample`` are guarded by a lock so async producers
  can push while the learner samples.
* Zero-dependency: no torch, no numpy — works with plain ``RolloutRecord``.
* Stats: ``buffer_size``, ``buffer_used``, ``replay_mix_ratio`` emitted per iter.

Integration
-----------
In ``OnPolicyTrainer._update_on_records``, after collecting the current batch:

    if self._replay_buffer is not None:
        batch_records = self._replay_buffer.mix_with_current(
            current=batch_records,
            mix_ratio=self.cfg.replay_mix_ratio,
            policy_version=self._update_version,
        )

The mixed batch then flows through the normal reward normalization → OPD →
update path. TIS correction in GRPO/PPO automatically adjusts for staleness.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ReplayBufferStats:
    """Observable stats emitted into the training metrics record."""

    push_count: int = 0
    sample_count: int = 0
    overflow_count: int = 0
    current_capacity: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "replay_push_count": float(self.push_count),
            "replay_sample_count": float(self.sample_count),
            "replay_overflow_count": float(self.overflow_count),
            "replay_capacity_used": float(self.current_capacity),
        }


class ReplayBuffer:
    """Bounded ring buffer supporting recency-weighted sampling.

    Parameters
    ----------
    capacity:
        Maximum number of records stored. When full, the oldest record is
        overwritten (FIFO ring).
    recency_alpha:
        Controls how much newer records are favored during sampling.
        * 0.0 → uniform sampling (pure FIFO replay)
        * >0 → exponential recency bias: weight ∝ exp(alpha * recency_rank)
        Typical values: 0.0–2.0. Higher alpha trusts recent data more.
    seed:
        Optional RNG seed for reproducibility. None → nondeterministic.
    """

    def __init__(
        self,
        capacity: int = 2048,
        recency_alpha: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.recency_alpha = float(recency_alpha)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._buffer: list[Any] = []
        self._insert_idx: int = 0  # ring pointer
        self._insert_order: int = 0  # monotonically increasing insertion counter
        self.stats = ReplayBufferStats()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def push(self, records: list[Any], *, policy_version: int = 0) -> None:
        """Push a batch of records into the ring buffer.

        Each record gets stamped with ``_replay_insert_order`` and
        ``_replay_policy_version`` metadata so sampling can compute recency
        and staleness.
        """
        if not records:
            return
        with self._lock:
            for rec in records:
                meta = getattr(rec, "metadata", None)
                if isinstance(meta, dict):
                    meta["_replay_insert_order"] = self._insert_order
                    meta["_replay_policy_version"] = int(policy_version)

                if len(self._buffer) < self.capacity:
                    self._buffer.append(rec)
                else:
                    # Ring overwrite
                    self._buffer[self._insert_idx] = rec
                    self.stats.overflow_count += 1

                self._insert_idx = (self._insert_idx + 1) % self.capacity
                self._insert_order += 1
                self.stats.push_count += 1

            self.stats.current_capacity = len(self._buffer)

    def sample(self, n: int) -> list[Any]:
        """Sample ``n`` records using recency-weighted probabilities.

        Returns fewer than ``n`` if the buffer has fewer records.
        """
        with self._lock:
            buf = list(self._buffer)
            n = min(n, len(buf))

        if n <= 0:
            return []

        if self.recency_alpha <= 0.0 or len(buf) <= 1:
            sampled = self._rng.sample(buf, n)
        else:
            # Compute recency weights: newer records (higher insert_order) get
            # exponentially more weight.
            orders = []
            for rec in buf:
                meta = getattr(rec, "metadata", None)
                if isinstance(meta, dict):
                    orders.append(meta.get("_replay_insert_order", 0))
                else:
                    orders.append(0)
            max_order = max(orders) if orders else 0
            # Normalized recency ∈ [0, 1]; newest = 1.0
            recencies = [(o / max_order) if max_order > 0 else 1.0 for o in orders]
            weights = [r ** self.recency_alpha for r in recencies]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]

            # Weighted sampling without replacement
            sampled = []
            remaining_indices = list(range(len(buf)))
            remaining_probs = list(probs)
            for _ in range(n):
                if not remaining_indices:
                    break
                idx = self._rng.choices(remaining_indices, weights=remaining_probs, k=1)[0]
                pos = remaining_indices.index(idx)
                sampled.append(buf[idx])
                remaining_indices.pop(pos)
                remaining_probs.pop(pos)
                # Renormalize
                psum = sum(remaining_probs)
                if psum > 0:
                    remaining_probs = [p / psum for p in remaining_probs]

        self.stats.sample_count += len(sampled)
        return sampled

    def mix_with_current(
        self,
        current: list[Any],
        *,
        mix_ratio: float = 0.25,
        policy_version: int = 0,
    ) -> list[Any]:
        """Mix replay records into the current batch.

        Pushes ``current`` into the buffer, then samples
        ``int(len(current) * mix_ratio)`` replay records and prepends them.
        The resulting list has ``len(current) + n_replay`` records.

        Args:
            current: this iteration's freshly-collected records.
            mix_ratio: fraction of replay records relative to current batch size.
                0.0 = pure on-policy (no replay), 1.0 = equal replay/current.
            policy_version: current learner version for staleness tagging.

        Returns:
            Combined list: ``current + replay_sampled``.
        """
        # Push current records into the buffer first (they become available
        # for *future* iterations, not this one — avoids double-use).
        self.push(current, policy_version=policy_version)

        if mix_ratio <= 0.0 or self.size == 0:
            return list(current)

        n_replay = max(0, int(len(current) * mix_ratio))
        if n_replay == 0:
            return list(current)

        replay_records = self.sample(n_replay)
        if not replay_records:
            return list(current)

        # Tag replay records so downstream code knows they came from the buffer.
        for rec in replay_records:
            meta = getattr(rec, "metadata", None)
            if isinstance(meta, dict):
                meta["_replay_sampled"] = True
                staleness = policy_version - meta.get("_replay_policy_version", policy_version)
                meta["_replay_staleness"] = max(0, staleness)

        return list(current) + replay_records

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._insert_idx = 0
            self._insert_order = 0
            self.stats.current_capacity = 0

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of buffer state (for checkpointing)."""
        with self._lock:
            return {
                "capacity": self.capacity,
                "recency_alpha": self.recency_alpha,
                "insert_idx": self._insert_idx,
                "insert_order": self._insert_order,
                "size": len(self._buffer),
                "stats": self.stats.as_dict(),
            }


def build_replay_buffer_from_config(
    cfg: dict[str, Any] | None,
) -> ReplayBuffer | None:
    """Construct a ``ReplayBuffer`` from a YAML-friendly config dict.

    Expected keys (all optional)::

        enabled: true
        capacity: 2048
        recency_alpha: 0.5
        seed: 42

    Returns None when ``enabled`` is falsy or ``cfg`` is None.
    """
    if not cfg:
        return None
    cfg = dict(cfg) if isinstance(cfg, dict) else {}
    if not cfg.get("enabled", False):
        return None
    capacity = int(cfg.get("capacity", 2048))
    alpha = float(cfg.get("recency_alpha", 0.0))
    seed = cfg.get("seed")
    seed = int(seed) if seed is not None else None
    return ReplayBuffer(capacity=capacity, recency_alpha=alpha, seed=seed)
