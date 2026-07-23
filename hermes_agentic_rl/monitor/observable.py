from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from hermes_agentic_rl.monitor.writers import MultiMetricsWriter, build_writer_from_config


class UnifiedObservable:
    """Small facade over JSONL/stdout/TensorBoard/W&B metrics writers."""

    def __init__(
        self,
        output_dir: str | Path | None,
        config: dict[str, Any] | None = None,
        *,
        extra: list[Callable[[dict[str, Any]], None]] | None = None,
        wandb_context: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None
        writer_config = {"jsonl": True} if config is None else dict(config)
        self.writer: MultiMetricsWriter | None = build_writer_from_config(
            writer_config,
            output_dir=self.output_dir,
            extra=extra,
            wandb_context=wandb_context,
        )

    def log(self, record: dict[str, Any] | None = None, **fields: Any) -> None:
        payload: dict[str, Any] = dict(record or {})
        payload.update(fields)
        if self.writer is not None:
            self.writer(payload)

    def __call__(self, record: dict[str, Any]) -> None:
        self.log(record)

    def update_summary(self, summary: dict[str, Any]) -> None:
        if self.writer is not None:
            self.writer.update_summary(summary)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> UnifiedObservable:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
