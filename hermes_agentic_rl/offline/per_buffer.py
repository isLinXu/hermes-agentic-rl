"""Prioritized Experience Replay (PER) buffer.

Extends the base ``ReplayBuffer`` (append-only JSONL container) with:

1. **Capacity cap** — ring-buffer eviction (oldest first) when ``maxlen``
   is exceeded.
2. **Priority weighting** — each ``TrainSample`` is assigned a priority
   ``p_i = |advantage| + eps``.  Sampling probability is
   ``P(i) = p_i^α / Σ p_j^α``.
3. **Importance-sampling (IS) weights** — ``w_i = (1/N · 1/P(i))^β``
   normalised by ``max(w)``.  Callers can use ``w_i`` to rescale the
   gradient (multiply loss by ``w_i``) to correct for the non-uniform
   sampling bias.
4. **Priority update** — after computing the loss, trainers should call
   ``buffer.update_priorities(indices, new_td_errors)`` to keep
   priorities fresh (standard PER closed loop).

The ``PrioritizedReplayBuffer`` is backward-compatible with ``ReplayBuffer``
for JSONL IO (save/load).  ``DPOPair``s are stored without priorities and
are always sampled uniformly.

Usage::

    buf = PrioritizedReplayBuffer(maxlen=10_000, alpha=0.6, beta=0.4)

    # add samples with explicit priority (or let it default to max_p)
    buf.add_sample(sample, priority=2.5)
    buf.add_sample(sample2)          # priority = current max_priority

    # sample a mini-batch:
    indices, samples, weights = buf.sample_prioritized(batch_size=32)

    # after a gradient step, update priorities:
    buf.update_priorities(indices, td_errors=[0.3, 1.2, ...])

    # vanilla uniform sample (ignores priorities):
    random_samples = buf.sample(batch_size=32)
"""

from __future__ import annotations

import json
import random
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Re-export the base data types so callers only import from this module.
@dataclass(slots=True)
class TrainSample:
    prompt_ids: list[int]
    response_ids: list[int]
    reward: float = 0.0
    advantage: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class DPOPair:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Plain replay buffer (backward-compatible with offline/replay_buffer.py)
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Append-only buffer with JSONL round-trip (capacity-limited ring buffer).

    When ``maxlen`` is set, the oldest samples are evicted once the limit
    is reached (deque semantics).  ``maxlen=None`` → unlimited (original
    behaviour).
    """

    def __init__(self, maxlen: int | None = None) -> None:
        self.maxlen = maxlen
        if maxlen is not None:
            self.samples: deque[TrainSample] | list[TrainSample] = deque(maxlen=maxlen)
            self.pairs: deque[DPOPair] | list[DPOPair] = deque(maxlen=maxlen)
        else:
            self.samples = []
            self.pairs = []

    def add_sample(self, sample: TrainSample) -> None:
        self.samples.append(sample)

    def add_pair(self, pair: DPOPair) -> None:
        self.pairs.append(pair)

    def __len__(self) -> int:
        return len(self.samples) + len(self.pairs)

    def iter_samples(self) -> Iterator[TrainSample]:
        yield from self.samples

    def iter_dpo_pairs(self) -> Iterator[DPOPair]:
        yield from self.pairs

    def sample(self, batch_size: int) -> list[TrainSample]:
        """Uniform random sample (without replacement if possible)."""
        pool = list(self.samples)
        if not pool:
            return []
        k = min(batch_size, len(pool))
        return random.sample(pool, k)

    # --- JSONL IO ---

    def save_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as h:
            for s in self.samples:
                h.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            for pr in self.pairs:
                h.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> ReplayBuffer:
        buf = cls()
        with Path(path).open("r", encoding="utf-8") as h:
            for line in h:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "chosen_ids" in obj and "rejected_ids" in obj:
                    buf.add_pair(
                        DPOPair(
                            prompt_ids=list(obj["prompt_ids"]),
                            chosen_ids=list(obj["chosen_ids"]),
                            rejected_ids=list(obj["rejected_ids"]),
                            metadata=dict(obj.get("metadata", {})),
                        )
                    )
                else:
                    buf.add_sample(
                        TrainSample(
                            prompt_ids=list(obj["prompt_ids"]),
                            response_ids=list(obj["response_ids"]),
                            reward=float(obj.get("reward", 0.0)),
                            advantage=obj.get("advantage"),
                            metadata=dict(obj.get("metadata", {})),
                        )
                    )
        return buf

    @classmethod
    def from_samples(cls, samples: Iterable[TrainSample]) -> ReplayBuffer:
        b = cls()
        for s in samples:
            b.add_sample(s)
        return b

    @classmethod
    def from_pairs(cls, pairs: Iterable[DPOPair]) -> ReplayBuffer:
        b = cls()
        for p in pairs:
            b.add_pair(p)
        return b


# ---------------------------------------------------------------------------
# Prioritized Experience Replay
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized sampling (Schaul et al., 2015).

    Priority of sample i:  p_i = (|td_error| + eps_priority)^alpha
    Sampling probability:  P(i) = p_i / sum(p)
    IS weight:             w_i  = (1/N * 1/P(i))^beta / max(w)

    Args:
        maxlen: ring-buffer capacity.  None = unlimited.
        alpha: exponent controlling how much priority affects sampling.
            0 = uniform; 1 = fully prioritized.
        beta: IS correction exponent.  Typically annealed from 0.4 → 1.0
            over training.
        eps_priority: small constant added to |td_error| to ensure
            every sample has non-zero probability.
        default_priority: priority assigned to newly added samples when
            no explicit priority is given.  ``None`` → use current max.
    """

    def __init__(
        self,
        maxlen: int | None = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps_priority: float = 1e-6,
        default_priority: float | None = None,
    ) -> None:
        super().__init__(maxlen=maxlen)
        self.alpha = alpha
        self.beta = beta
        self.eps_priority = eps_priority
        self._default_priority = default_priority  # None → use current max

        # parallel deque of priorities for samples — deque for O(1) popleft.
        self._priorities: deque[float] = deque(maxlen=maxlen)
        self._max_priority: float = 1.0

    # --- overrides ---

    def add_sample(self, sample: TrainSample, priority: float | None = None) -> None:  # type: ignore[override]
        """Add a sample with an explicit (or default-max) priority."""
        p = (
            priority
            if priority is not None
            else (
                self._default_priority if self._default_priority is not None else self._max_priority
            )
        )
        p = max(self.eps_priority, float(p))

        # The priorities deque has maxlen set, so appending past capacity
        # automatically pops the oldest element — O(1), no manual eviction.
        super().add_sample(sample)
        self._priorities.append(p)
        self._max_priority = max(self._max_priority, p)

    def _recompute_weights(self) -> tuple[list[float], list[float]]:
        """Return (probs, IS-weights) for all stored samples."""
        prio_list = list(self._priorities)  # snapshot deque → list once
        n = len(prio_list)
        if n == 0:
            return [], []

        # P(i) = p_i^alpha / sum
        powered = [p**self.alpha for p in prio_list]
        total = sum(powered)
        probs = [pw / total for pw in powered]

        # w_i = (1/N * 1/P(i))^beta
        raw_weights = [(1.0 / max(n * prob, 1e-12)) ** self.beta for prob in probs]
        max_w = max(raw_weights)
        is_weights = [w / max_w for w in raw_weights]
        return probs, is_weights

    def sample_prioritized(
        self,
        batch_size: int,
    ) -> tuple[list[int], list[TrainSample], list[float]]:
        """Sample a mini-batch with prioritized probabilities.

        Returns:
            (indices, samples, is_weights):
              - ``indices``: positions in the internal sample list, used for
                ``update_priorities``.
              - ``samples``: the sampled ``TrainSample`` objects.
              - ``is_weights``: importance-sampling correction weights
                (same length as samples), normalised to [0, 1].
        """
        samples_list = list(self.samples)
        n = len(samples_list)
        if n == 0:
            return [], [], []

        probs, is_weights = self._recompute_weights()
        k = min(batch_size, n)

        # Weighted sampling without replacement via alias method fallback.
        indices = random.choices(range(n), weights=probs, k=k)
        # De-duplicate if sampling with replacement is not desired:
        # (keep duplicates for simplicity — standard PER uses replacement)

        chosen_samples = [samples_list[i] for i in indices]
        chosen_weights = [is_weights[i] for i in indices]
        return indices, chosen_samples, chosen_weights

    def update_priorities(
        self,
        indices: list[int],
        td_errors: list[float],
    ) -> None:
        """Update priorities after a gradient step.

        Args:
            indices: indices returned by ``sample_prioritized``.
            td_errors: per-sample TD errors (or absolute advantages).
                Length must match ``indices``.
        """
        prio_list = list(self._priorities)  # materialise for indexed write
        changed = False
        for idx, err in zip(indices, td_errors, strict=False):
            if 0 <= idx < len(prio_list):
                new_p = abs(float(err)) + self.eps_priority
                prio_list[idx] = new_p
                self._max_priority = max(self._max_priority, new_p)
                changed = True
        if changed:
            # Rebuild deque from updated list — preserves order & maxlen.
            self._priorities = deque(prio_list, maxlen=self.maxlen)

    def anneal_beta(self, step: int, total_steps: int, beta_end: float = 1.0) -> float:
        """Linearly anneal beta from ``self.beta`` to ``beta_end``.

        Call once per training step.  Returns the new beta value.
        """
        frac = min(1.0, step / max(1, total_steps))
        self.beta = self.beta + frac * (beta_end - self.beta)
        return self.beta

    def snapshot_stats(self) -> dict[str, float]:
        """Return a dict of buffer statistics for logging."""
        n = len(self._priorities)
        if n == 0:
            return {
                "size": 0,
                "max_priority": 0.0,
                "min_priority": 0.0,
                "mean_priority": 0.0,
                "beta": self.beta,
                "alpha": self.alpha,
            }
        return {
            "size": float(n),
            "max_priority": max(self._priorities),
            "min_priority": min(self._priorities),
            "mean_priority": sum(self._priorities) / n,
            "beta": self.beta,
            "alpha": self.alpha,
        }
