from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SKILL_EXPORT_SCHEMA_VERSION = 1
EXCLUDED_GROUP_AXES = {"self_evolution_signal", "skill_learning"}

AXIS_PROCEDURES = {
    "task_success": (
        "Clarify the target outcome, preserve the user's success criteria, "
        "and finish with an answer or artifact that directly satisfies the task."
    ),
    "tool_use_reliability": (
        "Select tools only when they are necessary, keep tool names and "
        "arguments schema-valid, and verify tool feedback before finalizing."
    ),
    "interaction_control": (
        "Use feedback as control signal: recover from explicit failures, avoid "
        "looping, and stop cleanly once the task is resolved."
    ),
    "prompt_context": (
        "Keep the relevant long-context facts visible, summarize tool results "
        "compactly, and avoid dropping constraints that affect the final action."
    ),
    "skill_learning": (
        "Extract reusable procedures from repeated successful turns and keep "
        "validation cases beside the procedure."
    ),
}


def export_skill_candidates(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _normalize_skill_export_config(config)
    records = _load_replay_records(input_path)
    candidates, rejected = _select_candidate_records(records, cfg)
    grouped = _group_candidates(candidates, cfg)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skill_summaries: list[dict[str, Any]] = []
    for group_key, group_records in sorted(grouped.items()):
        if len(group_records) < int(cfg["min_examples_per_skill"]):
            continue
        skill_summary = _write_skill_candidate(
            output_dir=out_dir,
            group_key=group_key,
            records=group_records,
            config=cfg,
        )
        skill_summaries.append(skill_summary)

    quality_report = _build_quality_report(
        skill_summaries=skill_summaries,
        candidates=candidates,
        rejected=rejected,
        config=cfg,
    )
    summary = {
        "schema_version": SKILL_EXPORT_SCHEMA_VERSION,
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "records_read": len(records),
        "candidate_records": len(candidates),
        "rejected_records": len(rejected),
        "skills_exported": len(skill_summaries),
        "filters": {
            "require_skill_candidate": cfg["require_skill_candidate"],
            "min_reward": cfg["min_reward"],
            "require_any_capability_axes": list(cfg["require_any_capability_axes"]),
            "require_any_recommended_uses": list(cfg["require_any_recommended_uses"]),
            "min_examples_per_skill": cfg["min_examples_per_skill"],
            "max_examples_per_skill": cfg["max_examples_per_skill"],
            "group_by": cfg["group_by"],
        },
        "rejection_reasons": dict(sorted(Counter(item["reason"] for item in rejected).items())),
        "quality": quality_report,
        "skills": skill_summaries,
    }
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "quality_report.json", quality_report)
    return summary


def _normalize_skill_export_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw or {})
    return {
        "name_prefix": str(cfg.get("name_prefix", "hermes")),
        "description_prefix": str(
            cfg.get(
                "description_prefix",
                "Candidate Hermes agent procedure mined from replay data.",
            )
        ),
        "require_skill_candidate": bool(cfg.get("require_skill_candidate", True)),
        "min_reward": float(cfg.get("min_reward", 0.0)),
        "require_any_capability_axes": _string_set(cfg.get("require_any_capability_axes")),
        "require_any_recommended_uses": _string_set(cfg.get("require_any_recommended_uses")),
        "group_by": str(cfg.get("group_by", "primary_axis")),
        "min_examples_per_skill": max(1, int(cfg.get("min_examples_per_skill", 1))),
        "max_examples_per_skill": max(1, int(cfg.get("max_examples_per_skill", 8))),
        "quality_min_examples": max(1, int(cfg.get("quality_min_examples", 2))),
        "quality_min_mean_reward": float(
            cfg.get("quality_min_mean_reward", cfg.get("min_reward", 0.0))
        ),
        "quality_min_mean_usefulness": float(cfg.get("quality_min_mean_usefulness", 0.5)),
        "quality_min_axis_consistency": float(cfg.get("quality_min_axis_consistency", 0.6)),
        "quality_min_validation_examples": max(
            1, int(cfg.get("quality_min_validation_examples", 1))
        ),
        "quality_max_negative_signal_ratio": float(
            cfg.get("quality_max_negative_signal_ratio", 0.25)
        ),
        "quality_ready_min_score": float(cfg.get("quality_ready_min_score", 0.75)),
        "quality_blocked_max_score": float(cfg.get("quality_blocked_max_score", 0.35)),
        "include_full_records": bool(cfg.get("include_full_records", False)),
    }


def _load_replay_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.suffix in {".jsonl", ".jl"}:
        records: list[dict[str, Any]] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    payload = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return [item for item in payload["samples"] if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def _select_candidate_records(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    required_axes = set(cfg["require_any_capability_axes"])
    required_uses = set(cfg["require_any_recommended_uses"])

    for idx, record in enumerate(records):
        metadata = _metadata(record)
        mining = _mining(metadata)
        reward = _record_reward(record)
        axes = _axes(metadata, mining)
        uses = _uses(mining)

        reason = None
        if cfg["require_skill_candidate"] and not bool(mining.get("skill_candidate")):
            reason = "not_skill_candidate"
        elif reward < float(cfg["min_reward"]):
            reason = "reward_below_min"
        elif required_axes and not (axes & required_axes):
            reason = "missing_required_axis"
        elif required_uses and not (uses & required_uses):
            reason = "missing_required_use"

        if reason is not None:
            rejected.append({"index": idx, "reason": reason})
            continue
        candidates.append(record)
    return candidates, rejected


def _group_candidates(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_by = str(cfg.get("group_by", "primary_axis"))
    for record in records:
        metadata = _metadata(record)
        mining = _mining(metadata)
        if group_by == "single":
            key = "agentic-replay-skill"
        elif group_by == "recommended_use":
            key = next(iter(sorted(_uses(mining))), "review")
        else:
            key = _primary_axis(_axes(metadata, mining))
        grouped[key].append(record)
    return dict(grouped)


def _write_skill_candidate(
    *,
    output_dir: Path,
    group_key: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = sorted(
        records,
        key=lambda record: _usefulness(_mining(_metadata(record)), _record_reward(record)),
        reverse=True,
    )[: int(config["max_examples_per_skill"])]
    axis_counts = Counter(
        axis for record in selected for axis in _axes(_metadata(record), _mining(_metadata(record)))
    )
    use_counts = Counter(use for record in selected for use in _uses(_mining(_metadata(record))))
    skill_name = _slugify(f"{config['name_prefix']}-{group_key}-candidate")
    skill_dir = output_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    examples = [_record_to_example(record) for record in selected]
    quality = _skill_quality(
        records=selected,
        examples=examples,
        axis_counts=axis_counts,
        config=config,
    )

    skill_md = _skill_markdown(
        skill_name=skill_name,
        group_key=group_key,
        examples=examples,
        axis_counts=axis_counts,
        use_counts=use_counts,
        description_prefix=str(config["description_prefix"]),
        quality=quality,
    )
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    _write_jsonl(skill_dir / "validation.jsonl", [_validation_example(item) for item in examples])
    manifest: dict[str, Any] = {
        "schema_version": SKILL_EXPORT_SCHEMA_VERSION,
        "name": skill_name,
        "group_key": group_key,
        "source_records": len(records),
        "examples_written": len(examples),
        "quality": quality,
        "axis_counts": dict(sorted(axis_counts.items())),
        "recommended_use_counts": dict(sorted(use_counts.items())),
        "artifacts": {
            "skill": str(skill_dir / "SKILL.md"),
            "validation": str(skill_dir / "validation.jsonl"),
            "manifest": str(skill_dir / "manifest.json"),
        },
        "examples": examples,
    }
    if bool(config["include_full_records"]):
        manifest["records"] = selected
    _write_json(skill_dir / "manifest.json", manifest)
    return {
        "name": skill_name,
        "group_key": group_key,
        "source_records": len(records),
        "examples_written": len(examples),
        "quality": quality,
        "axis_counts": dict(sorted(axis_counts.items())),
        "recommended_use_counts": dict(sorted(use_counts.items())),
        "path": str(skill_dir),
    }


def _skill_markdown(
    *,
    skill_name: str,
    group_key: str,
    examples: list[dict[str, Any]],
    axis_counts: Counter[str],
    use_counts: Counter[str],
    description_prefix: str,
    quality: dict[str, Any],
) -> str:
    title = _title(skill_name)
    description = (
        f"{description_prefix} Primary replay group: {group_key}. "
        "Review and harden before installing as a production Skill."
    )
    axes = [axis for axis, _count in axis_counts.most_common()]
    procedures = [_procedure_for_axis(axis) for axis in axes]
    if not procedures:
        procedures = [_procedure_for_axis(group_key)]

    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {description}",
        "---",
        "",
        f"# {title}",
        "",
        "## Status",
        "",
        f"Quality status: `{quality.get('status', 'draft')}`. "
        "This is an auto-mined Skill candidate generated from Hermes replay turns. "
        "Review and harden it before installing as a production Skill.",
        "",
        "## Quality Gate",
        "",
        f"- Score: `{float(quality.get('score', 0.0)):.4f}`",
        f"- Status: `{quality.get('status', 'draft')}`",
        f"- Reasons: `{', '.join(str(item) for item in quality.get('reasons', []))}`",
        f"- Blockers: `{', '.join(str(item) for item in quality.get('blockers', []))}`",
        "",
        "## When To Use",
        "",
        f"Use this candidate when a Hermes agent task matches `{group_key}` "
        "behavior and needs a reusable procedure backed by replay evidence.",
        "",
        "## Procedure",
        "",
    ]
    lines.extend(f"- {procedure}" for procedure in procedures)
    lines.extend(
        [
            "",
            "## Replay Evidence",
            "",
            "| reward | axes | task | observed assistant behavior | feedback |",
            "|---|---|---|---|---|",
        ]
    )
    for example in examples[:5]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_reward(example.get("reward")),
                    ", ".join(str(axis) for axis in example.get("axes", [])),
                    _md_cell(str(example.get("task_input", ""))),
                    _md_cell(str(example.get("assistant_response", ""))),
                    _md_cell(str(example.get("feedback", ""))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "Use `validation.jsonl` beside this file as the first benchmark "
            "slice before promoting this candidate into the active Hermes Skill set.",
            "",
            "## Mining Metadata",
            "",
            f"- Source examples: `{len(examples)}`",
            f"- Capability axes: `{dict(sorted(axis_counts.items()))}`",
            f"- Recommended uses: `{dict(sorted(use_counts.items()))}`",
            f"- Quality metrics: `{quality.get('metrics', {})}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_quality_report(
    *,
    skill_summaries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    statuses = Counter(
        str(skill.get("quality", {}).get("status", "draft"))
        for skill in skill_summaries
        if isinstance(skill.get("quality"), dict)
    )
    scores = [
        float(skill.get("quality", {}).get("score", 0.0))
        for skill in skill_summaries
        if isinstance(skill.get("quality"), dict)
    ]
    blocked_reasons = Counter(
        reason
        for skill in skill_summaries
        if isinstance(skill.get("quality"), dict)
        for reason in skill["quality"].get("blockers", [])
    )
    return {
        "schema_version": SKILL_EXPORT_SCHEMA_VERSION,
        "skills_evaluated": len(skill_summaries),
        "candidate_records": len(candidates),
        "rejected_records": len(rejected),
        "status_counts": {
            "ready_for_review": statuses.get("ready_for_review", 0),
            "draft": statuses.get("draft", 0),
            "blocked": statuses.get("blocked", 0),
        },
        "mean_quality_score": (sum(scores) / len(scores)) if scores else 0.0,
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "thresholds": {
            "quality_min_examples": config["quality_min_examples"],
            "quality_min_mean_reward": config["quality_min_mean_reward"],
            "quality_min_mean_usefulness": config["quality_min_mean_usefulness"],
            "quality_min_axis_consistency": config["quality_min_axis_consistency"],
            "quality_min_validation_examples": config["quality_min_validation_examples"],
            "quality_max_negative_signal_ratio": config["quality_max_negative_signal_ratio"],
            "quality_ready_min_score": config["quality_ready_min_score"],
            "quality_blocked_max_score": config["quality_blocked_max_score"],
        },
        "skills": [
            {
                "name": skill.get("name"),
                "group_key": skill.get("group_key"),
                "path": skill.get("path"),
                "quality": skill.get("quality", {}),
            }
            for skill in skill_summaries
        ],
    }


def _skill_quality(
    *,
    records: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    axis_counts: Counter[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    sample_count = len(records)
    rewards = [_record_reward(record) for record in records]
    usefulness_scores = [
        _usefulness(_mining(_metadata(record)), _record_reward(record)) for record in records
    ]
    negative_signals = sum(
        1
        for record in records
        if _has_negative_signal(_mining(_metadata(record)), _metadata(record))
    )
    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    mean_usefulness = sum(usefulness_scores) / len(usefulness_scores) if usefulness_scores else 0.0
    # Capability traces are multi-label. A good candidate may consistently include
    # one dominant axis while also carrying skill-learning/self-evolution signals.
    axis_consistency = (
        min(1.0, max(axis_counts.values()) / sample_count) if sample_count and axis_counts else 0.0
    )
    validation_examples = len(examples)
    negative_signal_ratio = negative_signals / sample_count if sample_count else 1.0

    checks: dict[str, dict[str, Any]] = {
        "min_examples": {
            "required": int(config["quality_min_examples"]),
            "actual": sample_count,
            "passed": sample_count >= int(config["quality_min_examples"]),
        },
        "min_mean_reward": {
            "required": float(config["quality_min_mean_reward"]),
            "actual": mean_reward,
            "passed": mean_reward >= float(config["quality_min_mean_reward"]),
        },
        "min_mean_usefulness": {
            "required": float(config["quality_min_mean_usefulness"]),
            "actual": mean_usefulness,
            "passed": mean_usefulness >= float(config["quality_min_mean_usefulness"]),
        },
        "min_axis_consistency": {
            "required": float(config["quality_min_axis_consistency"]),
            "actual": axis_consistency,
            "passed": axis_consistency >= float(config["quality_min_axis_consistency"]),
        },
        "min_validation_examples": {
            "required": int(config["quality_min_validation_examples"]),
            "actual": validation_examples,
            "passed": validation_examples >= int(config["quality_min_validation_examples"]),
        },
        "max_negative_signal_ratio": {
            "required": float(config["quality_max_negative_signal_ratio"]),
            "actual": negative_signal_ratio,
            "passed": negative_signal_ratio <= float(config["quality_max_negative_signal_ratio"]),
        },
    }
    passed = sum(1 for item in checks.values() if bool(item["passed"]))
    score = passed / len(checks) if checks else 0.0
    blockers = [name for name, payload in checks.items() if not bool(payload["passed"])]
    ready_min_score = float(config["quality_ready_min_score"])
    blocked_max_score = float(config["quality_blocked_max_score"])
    if blockers and score <= blocked_max_score:
        status = "blocked"
    elif not blockers and score >= ready_min_score:
        status = "ready_for_review"
    else:
        status = "draft"
    reasons = [name for name, payload in checks.items() if bool(payload["passed"])]
    return {
        "status": status,
        "score": round(score, 6),
        "checks": checks,
        "blockers": blockers,
        "reasons": reasons,
        "metrics": {
            "sample_count": sample_count,
            "mean_reward": round(mean_reward, 6),
            "mean_usefulness_score": round(mean_usefulness, 6),
            "axis_consistency": round(axis_consistency, 6),
            "validation_examples": validation_examples,
            "negative_signal_ratio": round(negative_signal_ratio, 6),
        },
    }


def _has_negative_signal(mining: dict[str, Any], metadata: dict[str, Any]) -> bool:
    signals = mining.get("signals")
    if isinstance(signals, dict) and bool(signals.get("negative_feedback", False)):
        return True
    reasons = mining.get("reasons")
    if isinstance(reasons, list) and any(str(reason) == "negative_feedback" for reason in reasons):
        return True
    source_turn = metadata.get("source_turn", {})
    if isinstance(source_turn, dict) and "feedback_messages" in source_turn:
        feedback_messages = source_turn.get("feedback_messages")
    else:
        feedback_messages = metadata.get("feedback_messages")
    if isinstance(feedback_messages, list):
        joined = " ".join(
            _message_content(message) for message in feedback_messages if isinstance(message, dict)
        ).lower()
        return any(
            keyword in joined for keyword in ("wrong", "error", "failed", "不对", "错误", "失败")
        )
    return False


def _record_to_example(record: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(record)
    mining = _mining(metadata)
    source_turn = (
        metadata.get("source_turn", {}) if isinstance(metadata.get("source_turn"), dict) else {}
    )
    prompt_messages = _message_list(source_turn.get("prompt_messages"))
    assistant_message = source_turn.get("assistant_message")
    feedback_messages = _message_list(
        source_turn.get("feedback_messages")
        if "feedback_messages" in source_turn
        else metadata.get("feedback_messages")
    )
    if not isinstance(assistant_message, dict):
        assistant_message = {}

    task_input = _first_message_content(prompt_messages, role="user")
    if not task_input:
        task_input = str(
            metadata.get("task_id") or metadata.get("session_id") or "Hermes replay task"
        )

    assistant_response = _message_content(assistant_message)
    feedback = " ".join(_message_content(message) for message in feedback_messages).strip()
    return {
        "session_id": metadata.get("session_id"),
        "task_id": metadata.get("task_id"),
        "turn_index": metadata.get("turn_index"),
        "reward": _record_reward(record),
        "usefulness_score": _usefulness(mining, _record_reward(record)),
        "axes": sorted(_axes(metadata, mining)),
        "reasons": list(mining.get("reasons", []))
        if isinstance(mining.get("reasons"), list)
        else [],
        "recommended_uses": sorted(_uses(mining)),
        "task_input": _truncate(task_input, 800),
        "assistant_response": _truncate(assistant_response, 1000),
        "feedback": _truncate(feedback, 800),
    }


def _validation_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_input": example.get("task_input", ""),
        "expected_behavior": example.get("assistant_response", ""),
        "metadata": {
            "session_id": example.get("session_id"),
            "task_id": example.get("task_id"),
            "turn_index": example.get("turn_index"),
            "reward": example.get("reward"),
            "axes": example.get("axes", []),
            "recommended_uses": example.get("recommended_uses", []),
        },
    }


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _mining(metadata: dict[str, Any]) -> dict[str, Any]:
    mining = metadata.get("replay_mining")
    return dict(mining) if isinstance(mining, dict) else {}


def _axes(metadata: dict[str, Any], mining: dict[str, Any]) -> set[str]:
    axes: set[str] = set()
    for raw in (metadata.get("capability_axes"), mining.get("axes")):
        if isinstance(raw, list):
            axes.update(str(item) for item in raw if str(item))
    return axes


def _uses(mining: dict[str, Any]) -> set[str]:
    raw = mining.get("recommended_uses")
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if str(item)}


def _primary_axis(axes: set[str]) -> str:
    preferred = sorted(axis for axis in axes if axis not in EXCLUDED_GROUP_AXES)
    if preferred:
        return preferred[0]
    return next(iter(sorted(axes)), "general-agentic-pattern")


def _record_reward(record: dict[str, Any]) -> float:
    try:
        return float(record.get("reward", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _usefulness(mining: dict[str, Any], reward: float) -> float:
    try:
        return float(mining.get("usefulness_score", reward))
    except (TypeError, ValueError):
        return reward


def _message_list(value: Any) -> list[dict[str, Any]]:
    return (
        [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    )


def _first_message_content(messages: list[dict[str, Any]], *, role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            content = _message_content(message)
            if content:
                return content
    return ""


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _procedure_for_axis(axis: str) -> str:
    return AXIS_PROCEDURES.get(
        axis,
        "Follow the successful replay pattern, preserve the user's "
        "constraints, and verify the result before finalizing.",
    )


def _string_set(value: Any) -> set[str]:
    if value is None or value is False:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, list):
        return {str(item) for item in value if str(item)}
    return set()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return slug or "hermes-skill-candidate"


def _title(value: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[-_]+", value) if part)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _md_cell(value: str) -> str:
    return _truncate(value.replace("|", "\\|").replace("\n", "<br>"), 220)


def _format_reward(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)
