"""Minibatch construction and update-stats aggregation utilities.

Extracted from ``on_policy.py`` (v0.10 → v0.11 refactor).  These functions
are pure computations with no side-effects and no trainer-level state, so they
live here as module-level helpers + a thin ``MinibatchBuilder`` facade that
``OnPolicyTrainer`` can delegate to.

The ``MinibatchBuilder`` holds just the three config values it needs
(``update_epochs``, ``minibatch_size``, ``shuffle_minibatches``, ``seed``)
so tests can exercise batch splitting without instantiating a full trainer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from hermes_agentic_rl.algos.base import AlgoUpdateStats, RolloutBatch

# ---------------------------------------------------------------------------
# Pure helpers (used by OnPolicyTrainer methods)
# ---------------------------------------------------------------------------


def build_update_batches(
    batch: RolloutBatch,
    *,
    update_epochs: int,
    minibatch_size: int,
    shuffle_minibatches: bool,
    preserve_group_boundaries: bool,
    seed: int | None,
    iter_idx: int,
) -> list[RolloutBatch]:
    """Expand a single :class:`RolloutBatch` into minibatch-epoch slices.

    Args:
        batch: the full rollout batch for this iteration.
        update_epochs: how many times to iterate over the data.
        minibatch_size: target batch size per gradient step. 0 = full batch.
        shuffle_minibatches: whether to shuffle records before splitting.
        preserve_group_boundaries: keep group rollouts together.
        seed: RNG seed base (None = random).
        iter_idx: current training iteration (used to derive RNG seed).

    Returns:
        A flat list of :class:`RolloutBatch` objects in update order.
    """
    n_records = len(batch.records)
    effective_bs = max(1, minibatch_size) if minibatch_size > 0 else n_records

    batches: list[RolloutBatch] = []
    for epoch_idx in range(max(1, update_epochs)):
        if effective_bs >= n_records or n_records <= 1:
            batches.append(RolloutBatch(records=list(batch.records)))
            continue
        epoch_batches = split_minibatches(
            batch,
            minibatch_size=effective_bs,
            shuffle=shuffle_minibatches,
            preserve_group_boundaries=preserve_group_boundaries,
            rng=_make_rng(seed=seed, iter_idx=iter_idx, epoch_idx=epoch_idx),
        )
        batches.extend(epoch_batches or [RolloutBatch(records=list(batch.records))])
    return batches


def split_minibatches(
    batch: RolloutBatch,
    *,
    minibatch_size: int,
    shuffle: bool,
    preserve_group_boundaries: bool,
    rng: random.Random,
) -> list[RolloutBatch]:
    """Split *batch* into minibatches of at most *minibatch_size* records."""
    if preserve_group_boundaries:
        groups = [list(g) for g in batch.by_group().values()]
        if shuffle:
            rng.shuffle(groups)
        out: list[RolloutBatch] = []
        current: list[Any] = []
        current_size = 0
        for group in groups:
            g_size = len(group)
            if current and current_size + g_size > minibatch_size:
                out.append(RolloutBatch(records=list(current)))
                current = []
                current_size = 0
            current.extend(group)
            current_size += g_size
            if current_size >= minibatch_size:
                out.append(RolloutBatch(records=list(current)))
                current = []
                current_size = 0
        if current:
            out.append(RolloutBatch(records=list(current)))
        return out

    records = list(batch.records)
    if shuffle:
        rng.shuffle(records)
    return [
        RolloutBatch(records=records[start : start + minibatch_size])
        for start in range(0, len(records), minibatch_size)
    ]


def aggregate_update_stats(
    *,
    batch: RolloutBatch,
    step_stats: list[AlgoUpdateStats],
    n_update_batches: int,
    update_epochs: int,
    minibatch_size: int,
    summarize_batch_meta_fn: Any = None,
) -> AlgoUpdateStats:
    """Weighted-average per-minibatch stats into a single iteration stat.

    Args:
        batch: the full rollout batch (for ``n_records``).
        step_stats: one :class:`AlgoUpdateStats` per gradient step.
        n_update_batches: total number of minibatches dispatched.
        update_epochs: configured ``update_epochs`` (for logging).
        minibatch_size: configured ``minibatch_size`` (for logging).
        summarize_batch_meta_fn: optional ``_summarize_batch_metadata``
            callable; when provided its output is merged into ``extra``.

    Returns:
        Aggregated :class:`AlgoUpdateStats`.
    """
    total_records = len(batch.records)
    if not step_stats:
        return AlgoUpdateStats(
            loss=0.0,
            policy_loss=0.0,
            kl=0.0,
            entropy=0.0,
            mean_reward=0.0,
            mean_advantage=0.0,
            clip_frac=0.0,
            n_records=total_records,
            extra={
                "n_updated": 0,
                "n_optimizer_steps": 0,
                "update_epochs": max(1, update_epochs),
                "n_minibatches": n_update_batches,
            },
        )

    def _weight(stat: AlgoUpdateStats) -> int:
        return max(1, int(stat.n_records))

    total_weight = sum(_weight(s) for s in step_stats)

    def _weighted(attr: str) -> float:
        return sum(float(getattr(s, attr)) * _weight(s) for s in step_stats) / max(1, total_weight)

    extras: dict[str, Any] = {}
    first_extra = step_stats[0].extra
    if "algo" in first_extra:
        extras["algo"] = first_extra["algo"]
    extras["n_updated"] = sum(int(s.extra.get("n_updated", 0)) for s in step_stats)
    extras["n_optimizer_steps"] = len(step_stats)
    extras["update_epochs"] = max(1, update_epochs)
    extras["n_minibatches"] = n_update_batches
    extras["minibatch_size"] = total_records if minibatch_size <= 0 else minibatch_size

    numeric_means: dict[str, list[tuple[float, int]]] = {}
    for stat in step_stats:
        for key, value in stat.extra.items():
            if key in {"n_updated", "algo"}:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                numeric_means.setdefault(key, []).append((float(value), _weight(stat)))
    for key, values in numeric_means.items():
        denom = sum(w for _, w in values)
        extras[key] = sum(v * w for v, w in values) / max(1, denom)

    if summarize_batch_meta_fn is not None:
        extras.update(summarize_batch_meta_fn(batch))

    return AlgoUpdateStats(
        loss=_weighted("loss"),
        policy_loss=_weighted("policy_loss"),
        kl=_weighted("kl"),
        entropy=_weighted("entropy"),
        mean_reward=_weighted("mean_reward"),
        mean_advantage=_weighted("mean_advantage"),
        clip_frac=_weighted("clip_frac"),
        n_records=total_records,
        extra=extras,
    )


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _make_rng(*, seed: int | None, iter_idx: int, epoch_idx: int) -> random.Random:
    if seed is None:
        return random.Random()
    return random.Random(int(seed) + (iter_idx * 1009) + (epoch_idx * 9173))


# ---------------------------------------------------------------------------
# Facade dataclass (thin wrapper for OnPolicyTrainer delegation)
# ---------------------------------------------------------------------------


@dataclass
class MinibatchBuilder:
    """Thin façade exposing :func:`build_update_batches` as a stateful object.

    ``OnPolicyTrainer`` instantiates one of these and delegates its
    ``_build_update_batches`` / ``_split_minibatches`` calls here so the
    trainer class shrinks without breaking callers that override those methods.
    """

    update_epochs: int = 1
    minibatch_size: int = 0
    shuffle_minibatches: bool = False
    seed: int | None = None

    def build(
        self,
        batch: RolloutBatch,
        *,
        iter_idx: int,
        preserve_group_boundaries: bool = False,
    ) -> list[RolloutBatch]:
        return build_update_batches(
            batch,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            shuffle_minibatches=self.shuffle_minibatches,
            preserve_group_boundaries=preserve_group_boundaries,
            seed=self.seed,
            iter_idx=iter_idx,
        )
