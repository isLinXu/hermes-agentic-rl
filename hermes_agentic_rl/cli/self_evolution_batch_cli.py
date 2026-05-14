from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.cli.session_eval_export_cli import run_session_eval_export_config
from hermes_agentic_rl.cli.session_replay_cli import run_session_replay_config
from hermes_agentic_rl.cli.session_train_worker_cli import run_session_train_worker_config
from hermes_agentic_rl.eval.capability_axes import infer_objective_axes


def _load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _slugify(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return slug or fallback


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _root_stage_config(cfg: dict[str, Any]) -> dict[str, Any]:
    root = copy.deepcopy(cfg)
    root.pop("self_evolution_batch", None)
    return root


def _stage_enabled(
    batch_cfg: dict[str, Any],
    direction_cfg: dict[str, Any],
    key: str,
    *,
    default: bool = True,
) -> bool:
    batch_stages = batch_cfg.get("stages", {}) if isinstance(batch_cfg.get("stages"), dict) else {}
    direction_stages = (
        direction_cfg.get("stages", {}) if isinstance(direction_cfg.get("stages"), dict) else {}
    )
    if key in direction_stages:
        return bool(direction_stages[key])
    if key in batch_stages:
        return bool(batch_stages[key])
    return default


def _compose_stage_config(
    root_cfg: dict[str, Any],
    base_stage_cfg: dict[str, Any],
    direction_stage_cfg: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    cfg = _deep_merge(root_cfg, base_stage_cfg)
    cfg = _deep_merge(cfg, direction_stage_cfg)
    return _deep_merge(defaults, cfg)


def _direction_config(direction: Any, index: int) -> dict[str, Any]:
    if isinstance(direction, str):
        return {"name": direction}
    if isinstance(direction, dict):
        return dict(direction)
    raise ValueError(f"self_evolution_batch.directions[{index}] must be a string or mapping")


def _manifest_samples(path: Path) -> dict[str, Any]:
    manifest = _json_load(path / "manifest.json")
    samples = manifest.get("samples")
    return dict(samples) if isinstance(samples, dict) else {}


def run_self_evolution_batch_config(cfg: dict[str, Any]) -> int:
    batch_cfg = cfg.get("self_evolution_batch")
    if not isinstance(batch_cfg, dict):
        batch_cfg = cfg

    input_path = batch_cfg.get("input_path") or cfg.get("input_path")
    if not input_path:
        raise ValueError("self-evolution-batch requires `self_evolution_batch.input_path`")

    output_dir = Path(str(batch_cfg.get("output_dir") or cfg.get("output_dir") or "outputs/self_evolution_batch"))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_directions = batch_cfg.get("directions")
    if not isinstance(raw_directions, list) or not raw_directions:
        raise ValueError("self-evolution-batch requires at least one direction")

    root_cfg = _root_stage_config(cfg)
    base_replay_cfg = cfg.get("session_replay", {}) if isinstance(cfg.get("session_replay"), dict) else {}
    base_worker_cfg = (
        cfg.get("session_train_worker", {})
        if isinstance(cfg.get("session_train_worker"), dict)
        else {}
    )
    base_export_cfg = (
        cfg.get("session_eval_export", {})
        if isinstance(cfg.get("session_eval_export"), dict)
        else {}
    )

    direction_summaries: list[dict[str, Any]] = []
    exit_code = 0

    for index, raw_direction in enumerate(raw_directions):
        direction_cfg = _direction_config(raw_direction, index)
        name = str(direction_cfg.get("name") or f"direction-{index + 1}")
        slug = _slugify(name, fallback=f"direction-{index + 1}")
        direction_dir = output_dir / slug
        direction_dir.mkdir(parents=True, exist_ok=True)

        replay_path = direction_dir / "replay.jsonl"
        replay_quality_path = direction_dir / "replay_quality_report.json"
        replay_quarantine_path = direction_dir / "replay_quarantine.jsonl"
        worker_policy_path = direction_dir / "policy.pt"
        worker_state_path = direction_dir / "worker_state.json"
        worker_quarantine_path = direction_dir / "worker_quarantine.jsonl"
        worker_metrics_path = direction_dir / "worker_metrics.jsonl"
        dataset_dir = direction_dir / "self_evolution_dataset"

        replay_summary: dict[str, Any] = {}
        worker_summary: dict[str, Any] = {}
        export_summary: dict[str, Any] = {}
        worker_algo = str(direction_cfg.get("algo") or "bc")

        if _stage_enabled(batch_cfg, direction_cfg, "session_replay", default=True):
            replay_defaults = {
                "input_path": str(input_path),
                "output_path": str(replay_path),
                "quality_report_path": str(replay_quality_path),
                "quarantine_path": str(replay_quarantine_path),
            }
            replay_cfg = _compose_stage_config(
                root_cfg,
                base_replay_cfg,
                direction_cfg.get("session_replay", {})
                if isinstance(direction_cfg.get("session_replay"), dict)
                else {},
                replay_defaults,
            )
            code = run_session_replay_config(replay_cfg)
            exit_code = exit_code or code
            replay_summary = _json_load(replay_quality_path)

        if _stage_enabled(batch_cfg, direction_cfg, "session_train_worker", default=True):
            worker_stage = direction_cfg.get("session_train_worker")
            if not isinstance(worker_stage, dict):
                worker_stage = direction_cfg.get("worker", {})
            if not isinstance(worker_stage, dict):
                worker_stage = {}
            worker_defaults = {
                "algo": str(direction_cfg.get("algo") or worker_stage.get("algo") or "bc"),
                "input_path": str(replay_path),
                "save_path": str(worker_policy_path),
                "state_path": str(worker_state_path),
                "quarantine_path": str(worker_quarantine_path),
                "metrics": {"jsonl": str(worker_metrics_path)},
            }
            worker_cfg = _compose_stage_config(
                root_cfg,
                base_worker_cfg,
                worker_stage,
                worker_defaults,
            )
            worker_algo = str(worker_cfg.get("algo", worker_algo))
            code = run_session_train_worker_config(worker_cfg, once=True)
            exit_code = exit_code or code
            worker_summary = _json_load(worker_state_path)

        if _stage_enabled(batch_cfg, direction_cfg, "session_eval_export", default=True):
            export_stage = direction_cfg.get("session_eval_export")
            if not isinstance(export_stage, dict):
                export_stage = direction_cfg.get("export", {})
            if not isinstance(export_stage, dict):
                export_stage = {}
            export_defaults = {
                "input_path": str(input_path),
                "output_path": str(dataset_dir),
                "source": f"hermes-self-evolution:{slug}",
            }
            export_cfg = _compose_stage_config(
                root_cfg,
                base_export_cfg,
                export_stage,
                export_defaults,
            )
            code = run_session_eval_export_config(export_cfg)
            exit_code = exit_code or code
            export_summary = _json_load(dataset_dir / "manifest.json")

        direction_summary = {
            "name": name,
            "slug": slug,
            "description": direction_cfg.get("description", ""),
            "objective": direction_cfg.get("objective", {}),
            "capability_plan": {
                "target_metrics": (
                    direction_cfg.get("objective", {}).get("target_metrics", [])
                    if isinstance(direction_cfg.get("objective"), dict)
                    else []
                ),
                "axes": infer_objective_axes(
                    direction_cfg.get("objective", {}).get("target_metrics", [])
                    if isinstance(direction_cfg.get("objective"), dict)
                    else []
                ),
            },
            "paths": {
                "directory": str(direction_dir),
                "replay": str(replay_path),
                "replay_quality_report": str(replay_quality_path),
                "policy": str(worker_policy_path),
                "worker_state": str(worker_state_path),
                "worker_metrics": str(worker_metrics_path),
                "self_evolution_dataset": str(dataset_dir),
            },
            "replay": {
                "samples_written": int(replay_summary.get("samples_written", _count_jsonl(replay_path))),
                "reward_distribution": replay_summary.get("reward_distribution", {}),
                "quality_filtered": replay_summary.get("quality_filtered", {}),
                "quarantined": replay_summary.get("quarantined", {}),
            },
            "worker": {
                "algo": worker_algo,
                "updates": int(worker_summary.get("updates", 0) or 0),
                "trained_samples": int(worker_summary.get("trained_samples", 0) or 0),
                "trained_pairs": int(worker_summary.get("trained_pairs", 0) or 0),
                "candidate_pairs": int(worker_summary.get("candidate_pairs", 0) or 0),
                "last_loss": worker_summary.get("last_loss"),
            },
            "self_evolution_dataset": {
                "samples": _manifest_samples(dataset_dir),
                "manifest": export_summary,
            },
        }
        _json_dump(direction_dir / "direction_summary.json", direction_summary)
        direction_summaries.append(direction_summary)

        samples = direction_summary["self_evolution_dataset"]["samples"]
        print(
            "[self-evolution-batch] "
            f"direction={slug} replay_samples={direction_summary['replay']['samples_written']} "
            f"updates={direction_summary['worker']['updates']} "
            f"train={samples.get('train', 0)} val={samples.get('val', 0)} "
            f"holdout={samples.get('holdout', 0)}"
        )

    batch_summary: dict[str, Any] = {
        "command": "self-evolution-batch",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "directions": direction_summaries,
        "totals": {
            "directions": len(direction_summaries),
            "replay_samples": sum(item["replay"]["samples_written"] for item in direction_summaries),
            "worker_updates": sum(item["worker"]["updates"] for item in direction_summaries),
        },
        "capability_axes": sorted(
            {
                axis
                for direction in direction_summaries
                for axis in direction.get("capability_plan", {}).get("axes", [])
            }
        ),
    }
    _json_dump(output_dir / "batch_summary.json", batch_summary)
    print(
        "[self-evolution-batch] "
        f"summary={output_dir / 'batch_summary.json'} "
        f"directions={len(direction_summaries)} "
        f"updates={batch_summary['totals']['worker_updates']}"
    )
    return int(exit_code)


def run_self_evolution_batch(config_path: str | Path) -> int:
    return run_self_evolution_batch_config(_load_config(config_path))
