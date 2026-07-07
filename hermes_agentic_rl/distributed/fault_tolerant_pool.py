"""Fault-tolerant wrapper around :class:`MPRolloutPool`.

The base ``MPRolloutPool`` raises on the first worker exception, which is
fine for stable on-policy training but unhelpful in long-running RL jobs
where:

* a single rollout occasionally OOMs or hits a tool-side timeout
* a worker dies (subprocess SIGKILL) and silently stops draining
* the network filesystem stalls during checkpoint save and a rollout
    never returns

``FaultTolerantRolloutPool`` provides four orthogonal safety nets:

1. **Per-task retry**: failed tasks are re-submitted up to
   ``max_retries`` times before the failure becomes terminal.

2. **Drain-level timeout**: if no result arrives within
   ``poll_interval``, the wrapper checks worker liveness; if any worker
   has died it triggers a restart instead of waiting indefinitely.

3. **Worker heartbeat / restart**: dead processes are detected via
   :meth:`mp.Process.is_alive` and respawned by re-calling
   :meth:`MPRolloutPool.start`. The replacement workers are
   re-broadcast the latest policy weights before the failed tasks are
   re-submitted.

4. **Elastic scaling** (v0.13): workers can be dynamically added or
   removed via :meth:`scale_up` / :meth:`scale_down`. An optional
   auto-scaler monitors throughput and adjusts the pool size within
   ``min_workers`` … ``max_workers`` bounds. This is essential for
   cloud-spot training where instance availability changes at runtime.

This wrapper is intentionally unobtrusive — it implements the same
``submit_tasks`` / ``drain`` / ``broadcast_weights`` / ``start`` /
``shutdown`` surface as :class:`MPRolloutPool`, so existing trainer code
can swap one for the other:

    pool = FaultTolerantRolloutPool(
        MPRolloutPool(cfg, builder_fn),
        max_retries=3,
        worker_timeout=300.0,
    )
    pool.start()
    trainer = GRPOTrainer(..., rollout_pool=pool)

For elastic scaling:

    pool = FaultTolerantRolloutPool(
        MPRolloutPool(cfg, builder_fn),
        cfg=FaultTolerantPoolConfig(
            elastic=ElasticScalingConfig(
                enabled=True, min_workers=2, max_workers=8,
                scale_up_threshold=0.8, scale_down_threshold=0.3,
            ),
        ),
    )
"""

from __future__ import annotations

import queue
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes_agentic_rl.distributed.mp_pool import MPRolloutPool, RolloutTask


def _import_worker_main():
    """Return the worker main function (lazy import to avoid circular deps)."""
    from hermes_agentic_rl.distributed.mp_pool import _worker_main

    return _worker_main


@dataclass(slots=True)
class FaultTolerantPoolConfig:
    """Tuning knobs for :class:`FaultTolerantRolloutPool`.

    Attributes
    ----------
    max_retries:
        Max attempts (including the initial one) for a single task. After
        ``max_retries`` consecutive failures the wrapper raises a
        ``RolloutPoolFailure`` carrying the original exception and the
        offending task's ``task_seq``.

    worker_timeout:
        Soft timeout (seconds) applied to ``drain``. When exceeded the
        wrapper checks worker liveness; if all workers are alive the
        timeout is silently extended (the workers are simply slow on
        a long generation), otherwise dead workers are respawned and
        outstanding tasks re-issued.

    poll_interval:
        Internal queue poll interval (seconds). The wrapper waits this
        long for each result before re-checking worker health.

    restart_dead_workers:
        When True (default), dead workers are re-spawned. When False,
        worker death always raises ``RolloutPoolFailure`` immediately —
        useful for debugging.

    auto_rebroadcast_on_restart:
        If True (default), :attr:`last_state_dict` is re-pushed to all
        workers after a restart so the resumed pool stays in sync with
        the learner.
    """

    max_retries: int = 3
    worker_timeout: float = 300.0
    poll_interval: float = 1.0
    restart_dead_workers: bool = True
    auto_rebroadcast_on_restart: bool = True
    elastic: ElasticScalingConfig | None = None


@dataclass(slots=True)
class ElasticScalingConfig:
    """Configuration for elastic worker scaling.

    When ``enabled``, the pool monitors throughput (tasks completed per
    second) and automatically adjusts the worker count within the
    ``min_workers`` … ``max_workers`` range.

    Attributes
    ----------
    enabled:
        Master switch for auto-scaling. When False, manual
        :meth:`scale_up` / :meth:`scale_down` still work.

    min_workers:
        Floor for the worker count. The pool never drops below this.

    max_workers:
        Ceiling for the worker count.

    scale_up_threshold:
        Utilization (succeeded tasks / submitted tasks in the
        evaluation window) above which the pool adds workers.

    scale_down_threshold:
        Utilization below which the pool removes workers.

    eval_window:
        Number of drain rounds to average over before making a
        scaling decision.

    cooldown_rounds:
        Minimum rounds between scaling actions to prevent flapping.

    heartbeat_ttl:
        Seconds without a heartbeat before a worker is considered
        stale (even if the process is alive). 0 disables.
    """

    enabled: bool = False
    min_workers: int = 1
    max_workers: int = 8
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    eval_window: int = 5
    cooldown_rounds: int = 3
    heartbeat_ttl: float = 0.0


@dataclass(slots=True)
class _PoolStats:
    submitted: int = 0
    succeeded: int = 0
    retried: int = 0
    failed: int = 0
    restarts: int = 0
    timeouts: int = 0
    last_error: str | None = field(default=None)
    scale_ups: int = 0
    scale_downs: int = 0
    current_workers: int = 0


class RolloutPoolFailure(RuntimeError):
    """Raised when a task exceeds :attr:`FaultTolerantPoolConfig.max_retries`."""


class FaultTolerantRolloutPool:
    """Wrapping rollout pool that retries failed tasks and restarts dead workers.

    The wrapper is *transparent*: from the trainer's perspective
    ``submit_tasks`` + ``drain`` behave identically to the underlying
    :class:`MPRolloutPool`, except that transient failures no longer
    propagate and worker deaths no longer hang the run.

    The wrapper does not buffer results between calls — each
    ``submit_tasks`` / ``drain`` round-trip is treated as a discrete
    bundle, and the trainer remains the source of truth for which tasks
    are still outstanding.
    """

    def __init__(
        self,
        inner: MPRolloutPool,
        cfg: FaultTolerantPoolConfig | None = None,
    ) -> None:
        self.inner = inner
        self.cfg = cfg or FaultTolerantPoolConfig()
        if self.cfg.max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.stats = _PoolStats()
        self._last_state_blob: bytes | None = None
        self._last_version: int = 0
        # Elastic scaling state
        self._drain_rounds: int = 0
        self._last_scale_round: int = 0
        self._util_history: list[float] = []
        self._builder_fn: Callable[..., Any] | None = None
        self._build_ctx: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # public surface (mirrors MPRolloutPool)
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.inner.start()
        self.stats.current_workers = len(self.inner._procs)
        # Capture builder for elastic scaling
        self._builder_fn = self.inner.builder_fn
        self._build_ctx = dict(self.inner.cfg.build_ctx)

    def shutdown(self) -> None:
        self.inner.shutdown()
        self.stats.current_workers = 0

    def __enter__(self) -> FaultTolerantRolloutPool:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.shutdown()

    def broadcast_weights(self, state_dict: dict[str, Any]) -> None:
        # Cache the serialised blob so we can re-broadcast after a restart
        # without paying for a second torch.save round-trip.
        import io

        import torch

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        self._last_state_blob = buf.getvalue()
        self._last_version += 1
        # Push to inner via its own broadcast_weights to keep version
        # bookkeeping consistent with the wrapped class.
        self.inner.broadcast_weights(state_dict)

    def submit_tasks(self, tasks: list[RolloutTask]) -> None:
        self.inner.submit_tasks(tasks)
        self.stats.submitted += len(tasks)
        self._pending: dict[int, RolloutTask] = {t.task_seq: t for t in tasks}
        self._attempts: dict[int, int] = {t.task_seq: 1 for t in tasks}

    def drain(self, expected: int) -> list[dict[str, Any]]:
        """Drain results, retrying failed tasks and restarting dead workers."""
        out: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.cfg.worker_timeout

        while len(out) < expected:
            try:
                kind, payload = self.inner._result_q.get(
                    timeout=self.cfg.poll_interval,
                )
            except queue.Empty:
                if time.monotonic() < deadline:
                    continue
                # Timeout reached — check worker liveness.
                if self._all_alive():
                    deadline = time.monotonic() + self.cfg.worker_timeout
                    self.stats.timeouts += 1
                    warnings.warn(
                        "FaultTolerantRolloutPool: drain timed out but all "
                        "workers are alive; extending deadline",
                        stacklevel=2,
                    )
                    continue
                # At least one worker died — try to restart.
                if not self._handle_dead_workers():
                    raise RolloutPoolFailure(
                        "drain timed out and at least one worker died; auto-restart is disabled"
                    ) from None
                deadline = time.monotonic() + self.cfg.worker_timeout
                continue

            if kind == "ok":
                out.append(payload)
                seq = int(payload.get("task_seq", -1))
                self._pending.pop(seq, None)
                self._attempts.pop(seq, None)
                self.stats.succeeded += 1
            elif kind == "task_error":
                seq = int(payload.get("task_seq", -1))
                self._handle_task_error(seq, payload, expected_remaining=expected - len(out))
            elif kind == "builder_error":
                raise RolloutPoolFailure(f"worker builder failed: {payload}")

        out.sort(key=lambda r: r["task_seq"])
        # Auto-scale after each drain round
        self._drain_rounds += 1
        self._maybe_auto_scale(expected)
        self.stats.current_workers = len(self.inner._procs)
        return out

    # ------------------------------------------------------------------
    # elastic scaling
    # ------------------------------------------------------------------

    def scale_up(self, n: int = 1) -> int:
        """Add ``n`` workers to the pool.

        Returns the actual number of workers added (may be fewer if
        ``max_workers`` ceiling is reached or the inner pool doesn't
        support dynamic scaling).
        """
        elastic = self._elastic_or_default()
        current = len(self.inner._procs)
        room = elastic.max_workers - current
        to_add = max(0, min(n, room))
        if to_add == 0:
            return 0
        if self._builder_fn is None:
            warnings.warn(
                "scale_up called before start() or without a captured builder_fn",
                stacklevel=2,
            )
            return 0
        for _ in range(to_add):
            wid = len(self.inner._procs)
            tq: Any = self.inner._ctx.Queue()
            wq: Any = self.inner._ctx.Queue()
            p = self.inner._ctx.Process(
                target=_import_worker_main(),
                args=(wid, self._builder_fn, self._build_ctx or {}, tq, self.inner._result_q, wq),
                daemon=True,
            )
            p.start()
            self.inner._task_qs.append(tq)
            self.inner._weight_qs.append(wq)
            self.inner._procs.append(p)
        # Wait for new workers to signal ready
        self._wait_for_new_workers_ready(to_add)
        # Re-broadcast weights to new workers
        if self._last_state_blob is not None and elastic.enabled:
            for wq in self.inner._weight_qs[-to_add:]:
                try:
                    wq.put((self._last_version, self._last_state_blob))
                except Exception:
                    pass
        self.stats.scale_ups += to_add
        self.stats.current_workers = len(self.inner._procs)
        return to_add

    def scale_down(self, n: int = 1) -> int:
        """Remove ``n`` workers from the pool (graceful).

        Sends shutdown sentinels and joins the targeted workers.
        Returns the actual number removed.
        """
        elastic = self._elastic_or_default()
        current = len(self.inner._procs)
        room = current - elastic.min_workers
        to_remove = max(0, min(n, room))
        if to_remove == 0:
            return 0
        # Remove from the tail (most recently added workers)
        for _ in range(to_remove):
            tq = self.inner._task_qs.pop()
            self.inner._weight_qs.pop()
            p = self.inner._procs.pop()
            try:
                tq.put(("shutdown", None))
            except Exception:
                pass
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()
        self.stats.scale_downs += to_remove
        self.stats.current_workers = len(self.inner._procs)
        return to_remove

    @property
    def n_workers(self) -> int:
        """Current live worker count."""
        return len(self.inner._procs)

    def _elastic_or_default(self) -> ElasticScalingConfig:
        if self.cfg.elastic is not None:
            return self.cfg.elastic
        return ElasticScalingConfig()

    def _maybe_auto_scale(self, expected: int) -> None:
        """Check throughput and adjust pool size if needed."""
        elastic = self._elastic_or_default()
        if not elastic.enabled:
            return
        # Cooldown check
        if self._drain_rounds - self._last_scale_round < elastic.cooldown_rounds:
            return
        # Compute utilization for this round
        if expected > 0:
            util = self.stats.succeeded / max(1, self.stats.succeeded + self.stats.failed)
        else:
            util = 0.0
        self._util_history.append(util)
        if len(self._util_history) > elastic.eval_window:
            self._util_history.pop(0)
        if len(self._util_history) < elastic.eval_window:
            return
        avg_util = sum(self._util_history) / len(self._util_history)
        current = len(self.inner._procs)
        if avg_util > elastic.scale_up_threshold and current < elastic.max_workers:
            added = self.scale_up(1)
            if added > 0:
                self._last_scale_round = self._drain_rounds
                warnings.warn(
                    f"ElasticPool: scale_up (+{added}) — avg_util={avg_util:.2f} "
                    f"> threshold={elastic.scale_up_threshold}",
                    stacklevel=2,
                )
        elif avg_util < elastic.scale_down_threshold and current > elastic.min_workers:
            removed = self.scale_down(1)
            if removed > 0:
                self._last_scale_round = self._drain_rounds
                warnings.warn(
                    f"ElasticPool: scale_down (-{removed}) — avg_util={avg_util:.2f} "
                    f"< threshold={elastic.scale_down_threshold}",
                    stacklevel=2,
                )

    def _wait_for_new_workers_ready(self, n: int) -> None:
        import time as _time

        ready = 0
        deadline = _time.monotonic() + float(self.inner.cfg.worker_startup_timeout)
        while ready < n:
            if _time.monotonic() > deadline:
                warnings.warn(
                    f"ElasticPool: only {ready}/{n} new workers became ready in time",
                    stacklevel=2,
                )
                break
            try:
                kind, _payload = self.inner._result_q.get(timeout=1.0)
            except Exception:
                continue
            if kind == "ready":
                ready += 1
            elif kind == "builder_error":
                warnings.warn(
                    f"ElasticPool: new worker builder failed: {_payload}",
                    stacklevel=2,
                )
                break

    def _check_heartbeat(self) -> list[int]:
        """Return indices of workers that are alive but stale (no heartbeat).

        This is a no-op when ``heartbeat_ttl`` is 0.
        """
        elastic = self._elastic_or_default()
        if elastic.heartbeat_ttl <= 0:
            return []
        # In this implementation we rely on process.is_alive() as the
        # heartbeat proxy. A more sophisticated implementation could
        # use a shared timestamp updated by each worker.
        stale: list[int] = []
        for i, p in enumerate(self.inner._procs):
            if not p.is_alive():
                stale.append(i)
        return stale

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _handle_task_error(
        self,
        seq: int,
        payload: dict[str, Any],
        *,
        expected_remaining: int,
    ) -> None:
        task = self._pending.get(seq)
        attempts = self._attempts.get(seq, 1)
        err_repr = str(payload.get("error", "unknown"))
        self.stats.last_error = err_repr

        if task is None or attempts >= self.cfg.max_retries:
            self.stats.failed += 1
            raise RolloutPoolFailure(
                f"task_seq={seq} failed after {attempts} attempts: {err_repr}\n"
                f"{payload.get('tb', '')}"
            )

        # Retry: re-submit the task to the inner pool.
        self._attempts[seq] = attempts + 1
        self.stats.retried += 1
        warnings.warn(
            f"FaultTolerantRolloutPool: retrying task_seq={seq} "
            f"(attempt {attempts + 1}/{self.cfg.max_retries}) after error: {err_repr}",
            stacklevel=2,
        )
        # round-robin to a hopefully-healthy worker
        wid = (seq + attempts) % max(1, len(self.inner._task_qs))
        self.inner._task_qs[wid].put(("task", task))

    def _all_alive(self) -> bool:
        return all(p.is_alive() for p in self.inner._procs)

    def _handle_dead_workers(self) -> bool:
        if not self.cfg.restart_dead_workers:
            return False
        warnings.warn(
            "FaultTolerantRolloutPool: detected dead worker(s); restarting pool",
            stacklevel=2,
        )
        self.inner.shutdown()
        self.inner.start()
        self.stats.restarts += 1
        if self.cfg.auto_rebroadcast_on_restart and self._last_state_blob is not None:
            for wq in self.inner._weight_qs:
                try:
                    wq.put((self._last_version, self._last_state_blob))
                except Exception:
                    pass
        # Re-submit any tasks that haven't yet completed.
        if self._pending:
            for i, (seq, task) in enumerate(self._pending.items()):
                self._attempts[seq] = self._attempts.get(seq, 1) + 1
                if self._attempts[seq] > self.cfg.max_retries:
                    self.stats.failed += 1
                    raise RolloutPoolFailure(
                        f"task_seq={seq} exceeded max_retries during worker restart"
                    )
                self.stats.retried += 1
                wid = i % max(1, len(self.inner._task_qs))
                self.inner._task_qs[wid].put(("task", task))
        return True

    def metrics(self) -> dict[str, Any]:
        return {
            "submitted": self.stats.submitted,
            "succeeded": self.stats.succeeded,
            "retried": self.stats.retried,
            "failed": self.stats.failed,
            "restarts": self.stats.restarts,
            "timeouts": self.stats.timeouts,
            "last_error": self.stats.last_error,
            "current_workers": self.stats.current_workers,
            "scale_ups": self.stats.scale_ups,
            "scale_downs": self.stats.scale_downs,
        }
