"""Pluggable metrics writers for OnPolicyTrainer.

A writer receives a flat dict of metrics per iteration (via
``metrics_sink``) and fans them out to one or more backends. This module
provides four concrete implementations, each with graceful-degradation:

- ``JsonlMetricsWriter``: zero-dep, appends one JSON line per iter.
- ``StdoutMetricsWriter``: prints to stdout (same format as trainer's
  default logger). Useful for running a headless sink alongside the
  dashboard.
- ``TensorBoardMetricsWriter``: uses ``torch.utils.tensorboard.SummaryWriter``.
  Silently becomes a no-op if TB isn't importable.
- ``WandbMetricsWriter``: uses ``wandb.log`` if wandb is installed and
  ``wandb.init`` has been called; no-op otherwise.

``MultiMetricsWriter`` composes N writers with fail-safe dispatch —
one writer crashing doesn't break the trainer.

Usage::

    writer = MultiMetricsWriter([
        JsonlMetricsWriter(output_dir / "metrics.jsonl"),
        StdoutMetricsWriter(),
        TensorBoardMetricsWriter(output_dir / "tb"),
    ])
    trainer_cfg.metrics_sink = writer

All writers are callable (``writer(record)``) to match the
``Callable[[dict[str, Any]], None]`` contract on ``OnPolicyTrainerConfig``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class MetricsWriter(Protocol):
    def __call__(self, record: dict[str, Any]) -> None: ...


class JsonlMetricsWriter:
    """Append one JSON object per call to a file. Create parent dirs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file so an empty run produces a visible artifact.
        if not self.path.exists():
            self.path.touch()

    def __call__(self, record: dict[str, Any]) -> None:
        try:
            serializable = _to_json_safe(record)
            with self.path.open("a", encoding="utf-8") as h:
                h.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        except Exception:
            # never fail training because of metric IO
            pass

    def close(self) -> None:  # symmetry with TB/W&B writers
        pass


class StdoutMetricsWriter:
    """Format-and-print a record. Mirrors trainer default logger."""

    _KEYS = (
        "iter", "algo", "mean_reward", "loss", "policy_loss",
        "value_loss", "mean_advantage", "kl", "clip_frac", "n_updated",
    )

    def __call__(self, record: dict[str, Any]) -> None:
        parts: list[str] = []
        for k in self._KEYS:
            v = record.get(k)
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif v is not None:
                parts.append(f"{k}={v}")
        print("[metrics] " + " ".join(parts))


class TensorBoardMetricsWriter:
    """Use ``torch.utils.tensorboard.SummaryWriter`` if available.

    Gracefully becomes a no-op if tensorboard isn't installed. The iter
    step is taken from ``record['iter']`` if present, else a monotonic
    internal counter.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self._step = 0
        self._sw: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._sw = SummaryWriter(log_dir=str(log_dir))
        except Exception:
            self._sw = None  # disable silently

    def __call__(self, record: dict[str, Any]) -> None:
        if self._sw is None:
            return
        step = int(record.get("iter", self._step))
        self._step = step + 1
        try:
            for k, v in record.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self._sw.add_scalar(f"train/{k}", float(v), step)
        except Exception:
            pass

    def close(self) -> None:
        if self._sw is not None:
            try:
                self._sw.close()
            except Exception:
                pass


class WandbMetricsWriter:
    """Forward each record to ``wandb.log`` if wandb is active.

    Assumes the caller has already called ``wandb.init()`` (we do NOT init
    here — that's a user-facing decision). If wandb is missing or no run
    is active, this is a no-op.
    """

    def __init__(self, *, prefix: str = "train") -> None:
        self._prefix = prefix.rstrip("/")
        self._wandb: Any = None
        try:
            import wandb  # type: ignore

            # only enable if a run is already active
            if getattr(wandb, "run", None) is not None:
                self._wandb = wandb
        except Exception:
            self._wandb = None

    def __call__(self, record: dict[str, Any]) -> None:
        if self._wandb is None:
            return
        try:
            step = int(record.get("iter", 0))
            payload = {
                f"{self._prefix}/{k}": v
                for k, v in record.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            self._wandb.log(payload, step=step)
        except Exception:
            pass

    def close(self) -> None:
        pass


class MultiMetricsWriter:
    """Dispatch each record to N writers, isolating failures.

    Also forwards a pre-existing ``Callable`` (e.g. the live dashboard's
    ``record`` method), so users can chain the new writers alongside
    legacy sinks.
    """

    def __init__(
        self,
        writers: list[Callable[[dict[str, Any]], None]] | None = None,
    ) -> None:
        self._writers: list[Callable[[dict[str, Any]], None]] = list(writers or [])

    def add(self, writer: Callable[[dict[str, Any]], None]) -> None:
        self._writers.append(writer)

    def __call__(self, record: dict[str, Any]) -> None:
        for w in self._writers:
            try:
                w(record)
            except Exception:
                # metrics must never break training
                pass

    def close(self) -> None:
        for w in self._writers:
            close = getattr(w, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------


def _to_json_safe(obj: Any) -> Any:
    """Best-effort: convert an arbitrary record to JSON-serializable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return repr(obj)


def build_writer_from_config(
    cfg: dict[str, Any] | None,
    *,
    output_dir: str | Path | None = None,
    extra: list[Callable[[dict[str, Any]], None]] | None = None,
) -> MultiMetricsWriter | None:
    """Build a ``MultiMetricsWriter`` from a small YAML-friendly dict.

    Accepted keys (all optional)::

        jsonl: <path>     # or true to default to <output_dir>/metrics.jsonl
        stdout: true
        tensorboard: <dir>  # or true to default to <output_dir>/tb
        wandb: true | {prefix: "..."}

    Returns None if cfg is falsy and no extras are provided.
    """
    cfg = cfg or {}
    writers: list[Callable[[dict[str, Any]], None]] = []

    jsonl = cfg.get("jsonl")
    if jsonl:
        if jsonl is True:
            if output_dir is None:
                raise RuntimeError("metrics.jsonl=true requires an output_dir")
            path = Path(output_dir) / "metrics.jsonl"
        else:
            path = Path(str(jsonl))
        writers.append(JsonlMetricsWriter(path))

    if cfg.get("stdout"):
        writers.append(StdoutMetricsWriter())

    tb = cfg.get("tensorboard")
    if tb:
        if tb is True:
            if output_dir is None:
                raise RuntimeError("metrics.tensorboard=true requires an output_dir")
            tb_dir = Path(output_dir) / "tb"
        else:
            tb_dir = Path(str(tb))
        writers.append(TensorBoardMetricsWriter(tb_dir))

    wb = cfg.get("wandb")
    if wb:
        prefix = "train"
        if isinstance(wb, dict):
            prefix = str(wb.get("prefix", prefix))
        writers.append(WandbMetricsWriter(prefix=prefix))

    if extra:
        writers.extend(extra)

    if not writers:
        return None
    return MultiMetricsWriter(writers)
