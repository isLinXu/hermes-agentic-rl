from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from hermes_agentic_rl.collectors.replay_export import append_jsonl
from hermes_agentic_rl.framework import SessionTrainingPipeline


@dataclass(slots=True)
class LocalSessionSidecarConfig:
    session_log_path: str | None = None
    replay_output_path: str | None = None
    backend: dict[str, Any] = field(default_factory=dict, repr=False)
    judge: dict[str, Any] = field(default_factory=dict, repr=False)
    flush_interval_sec: float = 0.5
    max_batch_size: int = 16
    queue_maxsize: int = 0
    drop_when_full: bool = False


class SidecarStats(TypedDict):
    submitted: int
    dropped: int
    written_sessions: int
    written_samples: int
    flushes: int
    flush_errors: int
    last_batch_size: int
    last_flush_ts: float
    last_submit_ts: float
    last_error_ts: float
    last_error: str
    max_queue_depth: int


class LocalSessionSidecar:
    def __init__(self, cfg: LocalSessionSidecarConfig) -> None:
        if not cfg.session_log_path and not cfg.replay_output_path:
            raise ValueError("sidecar requires session_log_path and/or replay_output_path")
        if cfg.replay_output_path and not cfg.backend:
            raise ValueError("sidecar with replay_output_path requires backend tokenizer config")

        self.cfg = cfg
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=max(0, cfg.queue_maxsize)
        )
        self._stop_sentinel = object()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._stats: SidecarStats = {
            "submitted": 0,
            "dropped": 0,
            "written_sessions": 0,
            "written_samples": 0,
            "flushes": 0,
            "flush_errors": 0,
            "last_batch_size": 0,
            "last_flush_ts": 0.0,
            "last_submit_ts": 0.0,
            "last_error_ts": 0.0,
            "last_error": "",
            "max_queue_depth": 0,
        }
        self._session_api = (
            SessionTrainingPipeline.from_config(
                {
                    "runtime": {"integration": "fake"},
                    "backend": cfg.backend,
                    "judge": cfg.judge,
                },
                build_sidecar=False,
            )
            if cfg.replay_output_path
            else None
        )
        self._thread = threading.Thread(
            target=self._worker, name="hermes-session-sidecar", daemon=True
        )
        self._thread.start()

    def submit(self, payload: dict[str, Any]) -> bool:
        if self._closed:
            raise RuntimeError("sidecar is closed")
        try:
            if self.cfg.drop_when_full:
                self._queue.put_nowait(dict(payload))
            else:
                self._queue.put(dict(payload), timeout=max(1.0, self.cfg.flush_interval_sec))
        except queue.Full:
            with self._stats_lock:
                self._stats["dropped"] += 1
            return False
        with self._stats_lock:
            self._stats["submitted"] += 1
            self._stats["last_submit_ts"] = time.time()
            self._stats["max_queue_depth"] = max(
                self._stats["max_queue_depth"],
                int(self._queue.qsize()),
            )
        return True

    def flush(self, timeout: float | None = None) -> None:
        del timeout
        self._queue.join()

    def snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            snap = cast(dict[str, Any], self._stats.copy())
        queue_maxsize = int(self.cfg.queue_maxsize)
        queue_size = int(self._queue.qsize())
        snap["queue_size"] = queue_size
        snap["queue_maxsize"] = queue_maxsize
        snap["queue_utilization"] = (
            float(queue_size) / float(queue_maxsize) if queue_maxsize > 0 else 0.0
        )
        snap["worker_alive"] = bool(self._thread.is_alive())
        snap["closed"] = bool(self._closed)
        return snap

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._stop_sentinel)
        self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        batch: list[dict[str, Any]] = []
        while True:
            try:
                item = self._queue.get(timeout=self.cfg.flush_interval_sec)
            except queue.Empty:
                if batch:
                    self._flush_batch(batch)
                    batch = []
                continue

            if item is self._stop_sentinel:
                if batch:
                    self._flush_batch(batch)
                    batch = []
                self._queue.task_done()
                break

            if not isinstance(item, dict):
                self._queue.task_done()
                continue
            batch.append(item)
            if len(batch) >= max(1, self.cfg.max_batch_size):
                self._flush_batch(batch)
                batch = []

    def _flush_batch(self, batch: list[dict[str, Any]]) -> None:
        try:
            if self.cfg.session_log_path:
                append_jsonl(self.cfg.session_log_path, batch)
            if self.cfg.replay_output_path and self._session_api is not None:
                replay_payloads: list[dict[str, Any]] = []
                for record in batch:
                    replay_payloads.extend(self._session_api.replay_payloads_from_record(record))
                if replay_payloads:
                    append_jsonl(self.cfg.replay_output_path, replay_payloads)
                with self._stats_lock:
                    self._stats["written_samples"] += len(replay_payloads)
            with self._stats_lock:
                self._stats["written_sessions"] += len(batch)
                self._stats["flushes"] += 1
                self._stats["last_batch_size"] = len(batch)
                self._stats["last_flush_ts"] = time.time()
                self._stats["last_error"] = ""
        except Exception as exc:
            with self._stats_lock:
                self._stats["flush_errors"] += 1
                self._stats["dropped"] += len(batch)
                self._stats["last_batch_size"] = len(batch)
                self._stats["last_error_ts"] = time.time()
                self._stats["last_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for _ in batch:
                self._queue.task_done()


_SIDECARS: dict[str, LocalSessionSidecar] = {}
_SIDECARS_LOCK = threading.Lock()


def _sidecar_key(cfg: LocalSessionSidecarConfig) -> str:
    return json.dumps(
        {
            "session_log_path": cfg.session_log_path,
            "replay_output_path": cfg.replay_output_path,
            "backend": cfg.backend,
            "judge": cfg.judge,
            "flush_interval_sec": cfg.flush_interval_sec,
            "max_batch_size": cfg.max_batch_size,
            "queue_maxsize": cfg.queue_maxsize,
            "drop_when_full": cfg.drop_when_full,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def get_or_create_sidecar(cfg: LocalSessionSidecarConfig) -> LocalSessionSidecar:
    key = _sidecar_key(cfg)
    with _SIDECARS_LOCK:
        sidecar = _SIDECARS.get(key)
        if sidecar is None:
            sidecar = LocalSessionSidecar(cfg)
            _SIDECARS[key] = sidecar
        return sidecar


def build_sidecar_from_runtime_config(runtime_cfg: dict[str, Any]) -> LocalSessionSidecar | None:
    sidecar_cfg = runtime_cfg.get("session_sidecar")
    if not isinstance(sidecar_cfg, dict) or not bool(sidecar_cfg.get("enabled", False)):
        return None

    cfg = LocalSessionSidecarConfig(
        session_log_path=sidecar_cfg.get("session_log_path"),
        replay_output_path=sidecar_cfg.get("replay_output_path"),
        backend=dict(sidecar_cfg.get("backend") or {}),
        judge=dict(sidecar_cfg.get("judge") or {}),
        flush_interval_sec=float(sidecar_cfg.get("flush_interval_sec", 0.5)),
        max_batch_size=int(sidecar_cfg.get("max_batch_size", 16)),
        queue_maxsize=int(sidecar_cfg.get("queue_maxsize", 0)),
        drop_when_full=bool(sidecar_cfg.get("drop_when_full", False)),
    )
    return get_or_create_sidecar(cfg)


def close_all_sidecars() -> None:
    with _SIDECARS_LOCK:
        sidecars = list(_SIDECARS.values())
        _SIDECARS.clear()
    for sidecar in sidecars:
        sidecar.close()
