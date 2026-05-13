"""Pluggable metrics writers for on-policy and offline trainers.

A writer receives a metrics record and fans it out to one or more
destinations. This module provides four concrete implementations, each with
graceful degradation:

- ``JsonlMetricsWriter``: zero-dep, appends one JSON line per call.
- ``StdoutMetricsWriter``: prints the usual training summary line.
- ``TensorBoardMetricsWriter``: uses ``torch.utils.tensorboard.SummaryWriter``.
  Silently becomes a no-op if tensorboard is unavailable.
- ``WandbMetricsWriter``: auto-initializes a W&B run when configured, logs all
  scalar metrics (including nested numeric sub-fields), and becomes a no-op if
  wandb is unavailable or init fails.

``MultiMetricsWriter`` composes N writers with fail-safe dispatch, so one
metrics backend crashing never breaks training.
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
        if not self.path.exists():
            self.path.touch()

    def __call__(self, record: dict[str, Any]) -> None:
        try:
            serializable = _to_json_safe(record)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        pass


class StdoutMetricsWriter:
    """Format-and-print a record. Mirrors the trainer default logger."""

    _KEYS = (
        "iter",
        "algo",
        "mean_reward",
        "loss",
        "policy_loss",
        "value_loss",
        "mean_advantage",
        "kl",
        "clip_frac",
        "n_updated",
    )

    def __call__(self, record: dict[str, Any]) -> None:
        parts: list[str] = []
        for key in self._KEYS:
            value = record.get(key)
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            elif value is not None:
                parts.append(f"{key}={value}")
        print("[metrics] " + " ".join(parts))


class TensorBoardMetricsWriter:
    """Use ``torch.utils.tensorboard.SummaryWriter`` if available."""

    def __init__(self, log_dir: str | Path) -> None:
        self._step = 0
        self._sw: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._sw = SummaryWriter(log_dir=str(log_dir))
        except Exception:
            self._sw = None

    def __call__(self, record: dict[str, Any]) -> None:
        if self._sw is None:
            return
        step = int(record.get("iter", self._step))
        self._step = step + 1
        try:
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._sw.add_scalar(f"train/{key}", float(value), step)
        except Exception:
            pass

    def close(self) -> None:
        if self._sw is None:
            return
        try:
            self._sw.close()
        except Exception:
            pass


class WandbMetricsWriter:
    """Forward each record to W&B, auto-initializing when configured."""

    def __init__(
        self,
        *,
        prefix: str = "train",
        init_kwargs: dict[str, Any] | None = None,
        finish_on_close: bool = True,
    ) -> None:
        self._prefix = prefix.rstrip("/")
        self._finish_on_close = finish_on_close
        self._wandb: Any = None
        self._run: Any = None
        self._owns_run = False
        try:
            import wandb  # type: ignore

            self._wandb = wandb
            run = getattr(wandb, "run", None)
            if run is None and init_kwargs is not None:
                self._run = wandb.init(**init_kwargs)
                self._owns_run = self._run is not None
            else:
                self._run = run
            if self._run is not None:
                try:
                    wandb.define_metric(f"{self._prefix}/iter")
                    wandb.define_metric(
                        f"{self._prefix}/*",
                        step_metric=f"{self._prefix}/iter",
                    )
                except Exception:
                    pass
        except Exception:
            self._wandb = None
            self._run = None

    def __call__(self, record: dict[str, Any]) -> None:
        if self._wandb is None or self._run is None:
            return
        try:
            step_value = record.get("iter", 0)
            if isinstance(step_value, bool) or step_value is None:
                step_value = 0
            step = int(step_value)
            wandb_step = max(0, step)
            payload = _flatten_numeric_fields(record, prefix=self._prefix)
            if f"{self._prefix}/iter" not in payload:
                payload[f"{self._prefix}/iter"] = step
            self._wandb.log(payload, step=wandb_step)
        except Exception:
            pass

    def update_summary(self, summary: dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            flat = _flatten_numeric_fields(summary)
            for key, value in flat.items():
                self._run.summary[key] = value
        except Exception:
            pass

    def close(self) -> None:
        if self._wandb is None or self._run is None:
            return
        if not self._finish_on_close or not self._owns_run:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass


class MultiMetricsWriter:
    """Dispatch each record to N writers, isolating failures."""

    def __init__(
        self,
        writers: list[Callable[[dict[str, Any]], None]] | None = None,
    ) -> None:
        self._writers: list[Callable[[dict[str, Any]], None]] = list(writers or [])

    def add(self, writer: Callable[[dict[str, Any]], None]) -> None:
        self._writers.append(writer)

    def __call__(self, record: dict[str, Any]) -> None:
        for writer in self._writers:
            try:
                writer(record)
            except Exception:
                pass

    def close(self) -> None:
        for writer in self._writers:
            close = getattr(writer, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def update_summary(self, summary: dict[str, Any]) -> None:
        for writer in self._writers:
            update_summary = getattr(writer, "update_summary", None)
            if callable(update_summary):
                try:
                    update_summary(summary)
                except Exception:
                    pass


def _to_json_safe(obj: Any) -> Any:
    """Best-effort: convert an arbitrary object to JSON-serializable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(key): _to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _flatten_numeric_fields(obj: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts, keeping only numeric/bool scalar leaves."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            key_str = str(key)
            child_prefix = f"{prefix}/{key_str}" if prefix else key_str
            out.update(_flatten_numeric_fields(value, prefix=child_prefix))
        return out
    if isinstance(obj, bool):
        return {prefix: obj} if prefix else {}
    if isinstance(obj, (int, float)):
        return {prefix: float(obj)} if prefix else {}
    return {}


def _build_wandb_init_kwargs(
    wb_cfg: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate YAML config + runtime metadata into ``wandb.init`` kwargs."""
    init_kwargs = {
        key: value
        for key, value in wb_cfg.items()
        if key not in {"enabled", "prefix", "finish_on_close", "config"}
    }

    if output_dir is not None and "dir" not in init_kwargs:
        init_kwargs["dir"] = str(Path(output_dir) / "wandb")
    if "project" not in init_kwargs:
        init_kwargs["project"] = "hermes-agentic-rl"

    config_payload: dict[str, Any] = {}
    if context:
        for key, value in context.items():
            if key == "config" and isinstance(value, dict):
                config_payload.update(_to_json_safe(value))
                continue
            if key in {"name", "group", "job_type", "entity", "mode", "notes"}:
                if key not in init_kwargs and value is not None:
                    init_kwargs[key] = value
                continue
            if key == "tags":
                if "tags" not in init_kwargs and value is not None:
                    init_kwargs["tags"] = list(value)
                continue
            config_payload[key] = _to_json_safe(value)

    wb_config = wb_cfg.get("config")
    if isinstance(wb_config, dict):
        config_payload.update(_to_json_safe(wb_config))
    elif wb_config is not None:
        config_payload["config"] = _to_json_safe(wb_config)

    if config_payload:
        init_kwargs["config"] = config_payload

    return init_kwargs


def build_writer_from_config(
    cfg: dict[str, Any] | None,
    *,
    output_dir: str | Path | None = None,
    extra: list[Callable[[dict[str, Any]], None]] | None = None,
    wandb_context: dict[str, Any] | None = None,
) -> MultiMetricsWriter | None:
    """Build a ``MultiMetricsWriter`` from a small YAML-friendly dict.

    Accepted keys (all optional)::

        jsonl: <path>     # or true to default to <output_dir>/metrics.jsonl
        stdout: true
        tensorboard: <dir>  # or true to default to <output_dir>/tb
        wandb: true | {
          prefix: train,
          project: hermes-agentic-rl,
          name: my-run,
          group: exp-a,
          mode: online | offline | disabled,
          tags: [grpo, echo],
        }

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
        if isinstance(wb, dict) and str(wb.get("mode", "")).lower() == "disabled":
            wb = None
        if isinstance(wb, dict) and not bool(wb.get("enabled", True)):
            wb = None
        if wb:
            wb_cfg = dict(wb) if isinstance(wb, dict) else {}
            writers.append(
                WandbMetricsWriter(
                    prefix=str(wb_cfg.get("prefix", "train")),
                    init_kwargs=_build_wandb_init_kwargs(
                        wb_cfg,
                        output_dir=output_dir,
                        context=wandb_context,
                    ),
                    finish_on_close=bool(wb_cfg.get("finish_on_close", True)),
                )
            )

    if extra:
        writers.extend(extra)

    if not writers:
        return None
    return MultiMetricsWriter(writers)
