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
* **Pluggable sampling**: uniform, recency-weighted, and prioritized sampling.
  Recency bias is controlled by ``recency_alpha``. Prioritized replay uses a
  YAML-selected priority signal (reward magnitude, KL, reward variance, or an
  explicit metadata key) and stamps importance-sampling weights into metadata.
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
from typing import Any, Literal

ReplaySampler = Literal["uniform", "recency", "prioritized"]
PriorityMode = Literal["proportional", "rank"]


@dataclass(slots=True)
class ReplayBufferStats:
    """Observable stats emitted into the training metrics record."""

    push_count: int = 0
    sample_count: int = 0
    overflow_count: int = 0
    current_capacity: int = 0
    last_sampled_priority_mean: float = 0.0
    last_sampled_is_weight_mean: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "replay_push_count": float(self.push_count),
            "replay_sample_count": float(self.sample_count),
            "replay_overflow_count": float(self.overflow_count),
            "replay_capacity_used": float(self.current_capacity),
            "replay_sampled_priority_mean": float(self.last_sampled_priority_mean),
            "replay_sampled_is_weight_mean": float(self.last_sampled_is_weight_mean),
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
        * >0 → recency bias: weight ∝ recency_rank ** recency_alpha
        Typical values: 0.0–2.0. Higher alpha trusts recent data more.
    seed:
        Optional RNG seed for reproducibility. None → nondeterministic.
    """

    def __init__(
        self,
        capacity: int = 2048,
        recency_alpha: float = 0.0,
        seed: int | None = None,
        sampler: ReplaySampler | str | None = None,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        priority_eps: float = 1e-6,
        priority_key: str | None = None,
        priority_mode: PriorityMode | str = "proportional",
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.recency_alpha = float(recency_alpha)
        inferred_sampler = "recency" if self.recency_alpha > 0.0 else "uniform"
        sampler_value = str(sampler or inferred_sampler).lower()
        if sampler_value not in {"uniform", "recency", "prioritized"}:
            raise ValueError("ReplayBuffer sampler must be one of: uniform, recency, prioritized")
        mode_value = str(priority_mode).lower()
        if mode_value not in {"proportional", "rank"}:
            raise ValueError("ReplayBuffer priority_mode must be 'proportional' or 'rank'")
        self.sampler: ReplaySampler = sampler_value  # type: ignore[assignment]
        self.priority_alpha = max(0.0, float(priority_alpha))
        self.priority_beta = max(0.0, float(priority_beta))
        self.priority_eps = max(0.0, float(priority_eps))
        self.priority_key = priority_key
        self.priority_mode: PriorityMode = mode_value  # type: ignore[assignment]
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

    def _record_priority(self, rec: Any) -> float:
        meta = getattr(rec, "metadata", None)
        if isinstance(meta, dict):
            candidates: list[Any] = []
            if self.priority_key:
                candidates.append(meta.get(self.priority_key))
            candidates.extend(
                [
                    meta.get("_replay_priority"),
                    meta.get("replay_priority"),
                    meta.get("reward_variance"),
                    meta.get("kl"),
                    meta.get("approx_kl"),
                ]
            )
            for value in candidates:
                if isinstance(value, int | float):
                    return max(self.priority_eps, abs(float(value)))
        reward = getattr(rec, "reward", 0.0)
        if isinstance(reward, int | float):
            return max(self.priority_eps, abs(float(reward)))
        return max(self.priority_eps, 1.0)

    def _sampling_weights(self, buf: list[Any]) -> tuple[list[float], list[float]]:
        if not buf:
            return [], []
        priorities = [self._record_priority(rec) for rec in buf]

        if self.sampler == "uniform":
            return [1.0 for _ in buf], priorities

        if self.sampler == "recency":
            orders = []
            for rec in buf:
                meta = getattr(rec, "metadata", None)
                if isinstance(meta, dict):
                    orders.append(meta.get("_replay_insert_order", 0))
                else:
                    orders.append(0)
            min_order = min(orders) if orders else 0
            shifted = [max(0.0, float(o - min_order + 1)) for o in orders]
            return [max(self.priority_eps, r**self.recency_alpha) for r in shifted], priorities

        if self.priority_mode == "rank":
            ranked = sorted(enumerate(priorities), key=lambda item: item[1], reverse=True)
            ranks = [1 for _ in priorities]
            for rank, (idx, _priority) in enumerate(ranked, start=1):
                ranks[idx] = rank
            return [
                max(self.priority_eps, 1.0 / (float(rank) ** self.priority_alpha)) for rank in ranks
            ], priorities

        return [
            max(self.priority_eps, priority**self.priority_alpha) for priority in priorities
        ], priorities

    def _sample_weighted_without_replacement(
        self,
        buf: list[Any],
        weights: list[float],
        n: int,
    ) -> tuple[list[Any], list[int]]:
        sampled: list[Any] = []
        sampled_indices: list[int] = []
        remaining_indices = list(range(len(buf)))
        remaining_weights = list(weights)
        for _ in range(n):
            if not remaining_indices:
                break
            if sum(remaining_weights) <= 0:
                pos = self._rng.randrange(len(remaining_indices))
            else:
                chosen_idx = self._rng.choices(remaining_indices, weights=remaining_weights, k=1)[0]
                pos = remaining_indices.index(chosen_idx)
            idx = remaining_indices.pop(pos)
            remaining_weights.pop(pos)
            sampled.append(buf[idx])
            sampled_indices.append(idx)
        return sampled, sampled_indices

    def sample(self, n: int) -> list[Any]:
        """Sample ``n`` records using the configured replay sampler.

        Returns fewer than ``n`` if the buffer has fewer records.
        """
        with self._lock:
            buf = list(self._buffer)
            n = min(n, len(buf))

        if n <= 0:
            return []

        weights, priorities = self._sampling_weights(buf)
        if self.sampler == "uniform" or len(buf) <= 1:
            sampled_indices = self._rng.sample(range(len(buf)), n)
            sampled = [buf[idx] for idx in sampled_indices]
        else:
            sampled, sampled_indices = self._sample_weighted_without_replacement(buf, weights, n)

        total_w = sum(weights)
        sample_probs = [
            (weights[idx] / total_w) if total_w > 0 else (1.0 / len(buf)) for idx in sampled_indices
        ]
        raw_is_weights = [
            (len(buf) * max(p, self.priority_eps)) ** (-self.priority_beta) for p in sample_probs
        ]
        max_is = max(raw_is_weights) if raw_is_weights else 1.0
        is_weights = [w / max_is for w in raw_is_weights] if max_is > 0 else raw_is_weights

        sampled_priorities = [priorities[idx] for idx in sampled_indices]
        for rec, prob, is_weight, priority in zip(
            sampled, sample_probs, is_weights, sampled_priorities, strict=True
        ):
            meta = getattr(rec, "metadata", None)
            if isinstance(meta, dict):
                meta["_replay_sample_prob"] = float(prob)
                meta["_replay_is_weight"] = float(is_weight)
                meta["_replay_priority"] = float(priority)

        self.stats.sample_count += len(sampled)
        if sampled_priorities:
            self.stats.last_sampled_priority_mean = sum(sampled_priorities) / len(
                sampled_priorities
            )
        if is_weights:
            self.stats.last_sampled_is_weight_mean = sum(is_weights) / len(is_weights)
        return sampled

    def mix_with_current(
        self,
        current: list[Any],
        *,
        mix_ratio: float = 0.25,
        policy_version: int = 0,
    ) -> list[Any]:
        """Mix replay records into the current batch.

        Samples ``int(len(current) * mix_ratio)`` replay records from previous
        iterations, then pushes ``current`` into the buffer for future use.
        The resulting list has ``len(current) + n_replay`` records.

        Args:
            current: this iteration's freshly-collected records.
            mix_ratio: fraction of replay records relative to current batch size.
                0.0 = pure on-policy (no replay), 1.0 = equal replay/current.
            policy_version: current learner version for staleness tagging.

        Returns:
            Combined list: ``current + replay_sampled``.
        """
        if mix_ratio <= 0.0 or self.size == 0:
            self.push(current, policy_version=policy_version)
            return list(current)

        n_replay = max(0, int(len(current) * mix_ratio))
        if n_replay == 0:
            self.push(current, policy_version=policy_version)
            return list(current)

        replay_records = self.sample(n_replay)
        self.push(current, policy_version=policy_version)
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
                "sampler": self.sampler,
                "priority_alpha": self.priority_alpha,
                "priority_beta": self.priority_beta,
                "priority_eps": self.priority_eps,
                "priority_key": self.priority_key,
                "priority_mode": self.priority_mode,
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
        sampler: prioritized
        recency_alpha: 0.5
        priority_alpha: 0.6
        priority_beta: 0.4
        priority_key: approx_kl
        priority_mode: proportional
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
    sampler = cfg.get("sampler")
    priority_key = cfg.get("priority_key")
    return ReplayBuffer(
        capacity=capacity,
        recency_alpha=alpha,
        seed=seed,
        sampler=str(sampler) if sampler is not None else None,
        priority_alpha=float(cfg.get("priority_alpha", 0.6)),
        priority_beta=float(cfg.get("priority_beta", 0.4)),
        priority_eps=float(cfg.get("priority_eps", 1e-6)),
        priority_key=str(priority_key) if priority_key is not None else None,
        priority_mode=str(cfg.get("priority_mode", "proportional")),
    )
