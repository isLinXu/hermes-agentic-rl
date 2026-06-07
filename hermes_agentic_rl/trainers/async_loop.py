"""Asynchronous training loop — consume an ExperienceQueue at the learner's pace.

This is the async counterpart to ``OnPolicyTrainer.train()``. Instead of the
synchronous "collect → update" lockstep, the learner pulls already-scored
experience from an :class:`ExperienceQueue` that producers (rollout + judge
workers) fill independently — the OpenClaw-RL "train while serving" pattern.

Key responsibilities beyond the sync loop:

* **staleness accounting** — each consumed batch carries the behavior policy
  version; staleness = ``learner_version − behavior_version``.
* **staleness-gated off-policy correction** — when a batch is stale, enable
  Truncated Importance Sampling on the algorithm (``tis_rho_clip``) so the
  update is bias-corrected. Fresh (staleness 0) batches train on-policy, so a
  degenerate single-producer/single-consumer setup reproduces the sync result.
* **observability** — emits ``queue_*`` depth/lag/drop stats and an
  off-policy-ratio summary every step so async degradation is never silent.

The loop is deliberately transport-agnostic: it only needs ``get_batch`` /
``learner_version`` / ``set_learner_version`` / ``depth`` from the queue, so the
same code works with the stdlib queue (tests) or a Ray/mp queue (production).

It reuses the trainer's existing ``_update_on_records`` so the OPD hint
extraction, teacher fill, reward normalisation, AMP, and grad-accum paths are
all shared with the sync trainer — no logic is duplicated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hermes_agentic_rl.distributed.experience_queue import (
    ExperienceItem,
    ExperienceQueue,
)


@dataclass(slots=True)
class AsyncLoopConfig:
    """Configuration for :func:`run_async_training`.

    max_updates: number of learner updates to perform before stopping.
    batch_size: max experience items drained per update (``get_batch`` n).
    get_timeout: seconds to block for the first item before counting an idle
        poll. The loop keeps polling until ``max_updates`` updates land or
        ``idle_timeout`` elapses with an empty queue.
    idle_timeout: stop early if the queue stays empty this long (producers
        finished). <= 0 disables the idle cutoff.
    stale_tis_rho_clip: TIS clip applied when a batch's staleness exceeds
        ``stale_threshold``. <= 0 disables off-policy correction entirely.
    stale_threshold: staleness (in policy versions) above which TIS kicks in.
        0 means "any staleness triggers correction".
    """

    max_updates: int = 50
    batch_size: int = 8
    get_timeout: float = 1.0
    idle_timeout: float = 30.0
    stale_tis_rho_clip: float = 1.0
    stale_threshold: int = 0


def _flatten(items: list[ExperienceItem[list["RolloutRecord"]]]) -> list["RolloutRecord"]:
    out: list[RolloutRecord] = []
    for it in items:
        payload = it.payload
        if isinstance(payload, list):
            out.extend(payload)
    return out


def _max_staleness(
    items: list[ExperienceItem[Any]], learner_version: int
) -> int:
    if not items:
        return 0
    return max(
        ExperienceQueue.staleness_of(it, learner_version) for it in items
    )


def run_async_training(
    trainer: "OnPolicyTrainer",
    exp_queue: "ExperienceQueue[list[RolloutRecord]]",
    cfg: AsyncLoopConfig | None = None,
) -> "TrainStats":
    """Drive ``trainer`` from ``exp_queue`` until ``max_updates`` updates land.

    Returns the trainer's ``TrainStats``. The trainer's policy/optimizer/algo
    are reused as-is; the queue supplies the rollouts the sync loop would have
    collected itself.
    """
    cfg = cfg or AsyncLoopConfig()
    algo_cfg = getattr(trainer.algo, "cfg", None)
    has_tis = algo_cfg is not None and hasattr(algo_cfg, "tis_rho_clip")
    base_tis = float(getattr(algo_cfg, "tis_rho_clip", 0.0)) if has_tis else 0.0

    updates = 0
    last_progress = time.monotonic()
    exp_queue.set_learner_version(int(getattr(trainer, "_update_version", 0)))

    while updates < cfg.max_updates:
        items = exp_queue.get_batch(cfg.batch_size, timeout=cfg.get_timeout)
        if not items:
            if (
                cfg.idle_timeout > 0
                and time.monotonic() - last_progress > cfg.idle_timeout
            ):
                break
            continue
        last_progress = time.monotonic()

        learner_version = int(getattr(trainer, "_update_version", 0))
        exp_queue.set_learner_version(learner_version)
        staleness = _max_staleness(items, learner_version)
        records = _flatten(items)
        if not records:
            continue

        # Staleness-gated off-policy correction: enable TIS on the algo for
        # stale batches, restore the baseline afterwards so sync/fresh batches
        # are unaffected.
        applied_tis = base_tis
        if (
            has_tis
            and cfg.stale_tis_rho_clip > 0
            and staleness > cfg.stale_threshold
        ):
            applied_tis = cfg.stale_tis_rho_clip
            algo_cfg.tis_rho_clip = applied_tis  # type: ignore[union-attr]

        try:
            agg = trainer._update_on_records(records, updates)
        finally:
            if has_tis:
                algo_cfg.tis_rho_clip = base_tis  # type: ignore[union-attr]

        record = {
            "update": updates,
            "algo": trainer.algo_name,
            **agg.as_dict(),
            "async_staleness": float(staleness),
            "async_applied_tis_rho_clip": float(applied_tis),
            "async_batch_items": float(len(items)),
            "async_batch_records": float(len(records)),
            **exp_queue.stats.as_dict(),
            "queue_depth": float(exp_queue.depth()),
        }
        trainer.stats.add(record)
        if trainer.cfg.log_every and (updates % trainer.cfg.log_every == 0):
            trainer.logger(record)
        if trainer.cfg.metrics_sink is not None:
            try:
                trainer.cfg.metrics_sink(record)
            except Exception:
                pass
        updates += 1

    return trainer.stats
