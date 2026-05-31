"""Fault-tolerant wrapper around :class:`MPRolloutPool`.

The base ``MPRolloutPool`` raises on the first worker exception, which is
fine for stable on-policy training but unhelpful in long-running RL jobs
where:

* a single rollout occasionally OOMs or hits a tool-side timeout
* a worker dies (subprocess SIGKILL) and silently stops draining
* the network filesystem stalls during checkpoint save and a rollout
  never returns

``FaultTolerantRolloutPool`` provides three orthogonal safety nets:

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
"""

from __future__ import annotations

import queue
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes_agentic_rl.distributed.mp_pool import MPRolloutPool, RolloutTask


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


@dataclass(slots=True)
class _PoolStats:
    submitted: int = 0
    succeeded: int = 0
    retried: int = 0
    failed: int = 0
    restarts: int = 0
    timeouts: int = 0
    last_error: str | None = field(default=None)


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

    # ------------------------------------------------------------------
    # public surface (mirrors MPRolloutPool)
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.inner.start()

    def shutdown(self) -> None:
        self.inner.shutdown()

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
                kind, payload = self.inner._result_q.get(  # noqa: SLF001
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
                        "drain timed out and at least one worker died; "
                        "auto-restart is disabled"
                    )
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
                raise RolloutPoolFailure(
                    f"worker builder failed: {payload}"
                )

        out.sort(key=lambda r: r["task_seq"])
        return out

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
        wid = (seq + attempts) % max(1, len(self.inner._task_qs))  # noqa: SLF001
        self.inner._task_qs[wid].put(("task", task))  # noqa: SLF001

    def _all_alive(self) -> bool:
        return all(p.is_alive() for p in self.inner._procs)  # noqa: SLF001

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
            for wq in self.inner._weight_qs:  # noqa: SLF001
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
                wid = i % max(1, len(self.inner._task_qs))  # noqa: SLF001
                self.inner._task_qs[wid].put(("task", task))  # noqa: SLF001
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
        }
