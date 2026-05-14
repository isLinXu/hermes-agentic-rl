from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.collectors.skill_export import export_skill_candidates


def _load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def run_skill_export_config(
    cfg: dict[str, Any],
    *,
    input_path: str | None = None,
    output_path: str | None = None,
) -> int:
    raw_export_cfg = cfg.get("skill_export")
    export_cfg = dict(raw_export_cfg) if isinstance(raw_export_cfg, dict) else dict(cfg)
    effective_input = input_path or export_cfg.get("input_path") or cfg.get("input_path")
    if not effective_input:
        raise ValueError("skill-export requires `skill_export.input_path` or --input")
    effective_output = (
        output_path
        or export_cfg.get("output_dir")
        or export_cfg.get("output_path")
        or cfg.get("output_dir")
        or cfg.get("output_path")
    )
    if not effective_output:
        raise ValueError("skill-export requires `skill_export.output_dir` or --output")

    summary = export_skill_candidates(
        input_path=str(effective_input),
        output_dir=str(effective_output),
        config=export_cfg,
    )
    print(
        "[skill-export] "
        f"records={summary['records_read']} candidates={summary['candidate_records']} "
        f"skills={summary['skills_exported']} output={summary['output_dir']}"
    )
    return 0


def run_skill_export(
    config_path: str,
    *,
    input_path: str | None = None,
    output_path: str | None = None,
) -> int:
    return run_skill_export_config(
        _load_config(config_path),
        input_path=input_path,
        output_path=output_path,
    )
