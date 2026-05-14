from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.cli.online_cycle_cli import run_online_cycle_config
from hermes_agentic_rl.collectors.skill_export import export_skill_candidates


def _load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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


def _stage_enabled(
    stage_cfg: dict[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    raw_stages = stage_cfg.get("stages")
    stages = raw_stages if isinstance(raw_stages, dict) else {}
    if key in stages:
        return bool(stages[key])
    raw_stage = stage_cfg.get(key)
    if isinstance(raw_stage, dict) and "enabled" in raw_stage:
        return bool(raw_stage["enabled"])
    return default


def _runtime_session_log_path(cfg: dict[str, Any]) -> str | None:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    sidecar_cfg = runtime_cfg.get("session_sidecar")
    if isinstance(sidecar_cfg, dict) and sidecar_cfg.get("session_log_path"):
        return str(sidecar_cfg["session_log_path"])
    if runtime_cfg.get("session_log_path"):
        return str(runtime_cfg["session_log_path"])
    return None


def _runtime_replay_path(cfg: dict[str, Any]) -> str | None:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    sidecar_cfg = runtime_cfg.get("session_sidecar")
    if isinstance(sidecar_cfg, dict) and sidecar_cfg.get("replay_output_path"):
        return str(sidecar_cfg["replay_output_path"])
    return None


def _configured_replay_path(cfg: dict[str, Any], stage_cfg: dict[str, Any]) -> str | None:
    raw_skill_export_cfg = stage_cfg.get("skill_export")
    skill_export_cfg: dict[str, Any] = (
        dict(raw_skill_export_cfg) if isinstance(raw_skill_export_cfg, dict) else {}
    )
    if skill_export_cfg.get("input_path"):
        return str(skill_export_cfg["input_path"])

    raw_online_cycle_cfg = cfg.get("online_cycle")
    online_cycle_cfg: dict[str, Any] = (
        dict(raw_online_cycle_cfg) if isinstance(raw_online_cycle_cfg, dict) else {}
    )
    raw_replay_stage = online_cycle_cfg.get("session_replay")
    replay_stage: dict[str, Any] = (
        dict(raw_replay_stage) if isinstance(raw_replay_stage, dict) else {}
    )
    if replay_stage.get("output_path"):
        return str(replay_stage["output_path"])
    return _runtime_replay_path(cfg)


def _count_jsonl(path: str | Path | None) -> int:
    if not path:
        return 0
    target = Path(path)
    if not target.exists():
        return 0
    return sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())


def _stage_options(stage_cfg: dict[str, Any], key: str) -> dict[str, Any]:
    raw = stage_cfg.get(key)
    return dict(raw) if isinstance(raw, dict) else {}


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    signals = summary.get("signals", {})
    skill_quality = signals.get("skill_quality", {}) if isinstance(signals, dict) else {}
    status_counts = (
        skill_quality.get("status_counts", {})
        if isinstance(skill_quality, dict)
        else {}
    )
    lines = [
        "# Online Self-Evolution Report",
        "",
        f"- Output directory: `{summary.get('output_dir')}`",
        f"- Exit code: `{summary.get('exit_code')}`",
        "",
        "## Stages",
        "",
        "| stage | enabled | exit_code | key output |",
        "|---|---|---|---|",
    ]
    stages = summary.get("stages", {})
    if isinstance(stages, dict):
        for name, payload in stages.items():
            if not isinstance(payload, dict):
                continue
            key_output = (
                payload.get("summary_path")
                or payload.get("output_dir")
                or payload.get("replay_path")
                or payload.get("config_path")
                or ""
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(name),
                        str(payload.get("enabled", False)),
                        str(payload.get("exit_code", "")),
                        str(key_output),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Signals",
            "",
            f"- Sessions: `{summary.get('signals', {}).get('sessions', 0)}`",
            f"- Replay records: `{summary.get('signals', {}).get('replay_records', 0)}`",
            f"- Skill candidates exported: `{summary.get('signals', {}).get('skills_exported', 0)}`",
            f"- Skill ready_for_review: `{status_counts.get('ready_for_review', 0)}`",
            f"- Skill draft: `{status_counts.get('draft', 0)}`",
            f"- Skill blocked: `{status_counts.get('blocked', 0)}`",
            f"- Eval recommendation: `{summary.get('signals', {}).get('eval_recommendation', '')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_online_self_evolve_config(
    cfg: dict[str, Any],
    *,
    limit: int | None = None,
    seed: int | None = None,
    once: bool = False,
) -> int:
    raw_stage_cfg = cfg.get("online_self_evolve")
    stage_cfg = dict(raw_stage_cfg) if isinstance(raw_stage_cfg, dict) else {}
    output_dir = Path(str(stage_cfg.get("output_dir") or "outputs/hermes_online_self_evolve"))
    output_dir.mkdir(parents=True, exist_ok=True)

    stages: dict[str, Any] = {}
    exit_code = 0

    if _stage_enabled(stage_cfg, "online_cycle", default=True):
        cycle_options = _stage_options(stage_cfg, "online_cycle")
        cycle_code = run_online_cycle_config(
            cfg,
            limit=limit if limit is not None else cycle_options.get("limit"),
            seed=seed if seed is not None else cycle_options.get("seed"),
            once=once or bool(cycle_options.get("once", False)),
        )
        exit_code = exit_code or cycle_code
        stages["online_cycle"] = {
            "enabled": True,
            "exit_code": cycle_code,
            "session_log_path": _runtime_session_log_path(cfg),
            "replay_path": _configured_replay_path(cfg, stage_cfg),
        }
    else:
        stages["online_cycle"] = {"enabled": False}

    replay_path = _configured_replay_path(cfg, stage_cfg)
    skill_summary: dict[str, Any] = {}
    if _stage_enabled(stage_cfg, "skill_export", default=True):
        if not replay_path:
            raise ValueError(
                "online-self-evolve skill_export requires a replay input path "
                "from online_self_evolve.skill_export.input_path, "
                "online_cycle.session_replay.output_path, or "
                "runtime.session_sidecar.replay_output_path"
            )
        skill_cfg = _deep_merge(
            {
                "input_path": replay_path,
                "output_dir": str(output_dir / "skill_candidates"),
            },
            _stage_options(stage_cfg, "skill_export"),
        )
        skill_summary = export_skill_candidates(
            input_path=str(skill_cfg["input_path"]),
            output_dir=str(skill_cfg["output_dir"]),
            config=skill_cfg,
        )
        stages["skill_export"] = {
            "enabled": True,
            "exit_code": 0,
            "input_path": str(skill_cfg["input_path"]),
            "output_dir": str(skill_cfg["output_dir"]),
            "summary_path": str(Path(str(skill_cfg["output_dir"])) / "summary.json"),
            "quality_report_path": str(Path(str(skill_cfg["output_dir"])) / "quality_report.json"),
            "skills_exported": int(skill_summary.get("skills_exported", 0) or 0),
            "status_counts": (
                skill_summary.get("quality", {}).get("status_counts", {})
                if isinstance(skill_summary.get("quality"), dict)
                else {}
            ),
        }
    else:
        stages["skill_export"] = {"enabled": False}

    eval_summary: dict[str, Any] = {}
    if _stage_enabled(stage_cfg, "eval_gate", default=False):
        eval_cfg = _stage_options(stage_cfg, "eval_gate")
        config_path = eval_cfg.get("config_path") or eval_cfg.get("config")
        if not config_path:
            raise ValueError("online-self-evolve eval_gate requires `config_path`")
        eval_output_dir = str(eval_cfg.get("output_dir") or output_dir / "eval_gate")
        from hermes_agentic_rl.eval.rl_eval import run_eval_gate

        eval_code = run_eval_gate(str(config_path), output_dir=eval_output_dir)
        exit_code = exit_code or eval_code
        eval_summary = _json_load(Path(eval_output_dir) / "eval_summary.json")
        stages["eval_gate"] = {
            "enabled": True,
            "exit_code": eval_code,
            "config_path": str(config_path),
            "output_dir": eval_output_dir,
            "summary_path": str(Path(eval_output_dir) / "eval_summary.json"),
            "recommendation": (
                eval_summary.get("promotion_readout", {}).get("recommendation")
                if isinstance(eval_summary.get("promotion_readout"), dict)
                else None
            ),
        }
    else:
        stages["eval_gate"] = {"enabled": False}

    session_log_path = _runtime_session_log_path(cfg)
    summary = {
        "command": "online-self-evolve",
        "output_dir": str(output_dir),
        "exit_code": int(exit_code),
        "stages": stages,
        "signals": {
            "sessions": _count_jsonl(session_log_path),
            "replay_records": _count_jsonl(replay_path),
            "skills_exported": int(skill_summary.get("skills_exported", 0) or 0),
            "skill_quality": (
                skill_summary.get("quality", {})
                if isinstance(skill_summary.get("quality"), dict)
                else {}
            ),
            "eval_recommendation": (
                stages.get("eval_gate", {}).get("recommendation")
                if isinstance(stages.get("eval_gate"), dict)
                else None
            ),
        },
        "artifacts": {
            "summary": str(output_dir / "online_self_evolve_summary.json"),
            "report": str(output_dir / "online_self_evolve_report.md"),
        },
    }
    _json_dump(output_dir / "online_self_evolve_summary.json", summary)
    _write_report(output_dir / "online_self_evolve_report.md", summary)
    signals = summary["signals"]
    assert isinstance(signals, dict)
    print(
        "[online-self-evolve] "
        f"summary={output_dir / 'online_self_evolve_summary.json'} "
        f"replay_records={signals['replay_records']} "
        f"skills={signals['skills_exported']} "
        f"exit_code={exit_code}"
    )
    return int(exit_code)


def run_online_self_evolve(
    config_path: str | Path,
    *,
    limit: int | None = None,
    seed: int | None = None,
    once: bool = False,
) -> int:
    return run_online_self_evolve_config(
        _load_config(config_path),
        limit=limit,
        seed=seed,
        once=once,
    )
