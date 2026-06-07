"""distill-skills: one-command skill distillation from raw agent traces.

This is a thin orchestration layer over the existing, battle-tested
``collectors.skill_export.export_skill_candidates``. Its single job is to
remove the hidden 4-step pipeline that previously blocked casual use:

    raw chat log  ->  SessionTurnSample  ->  replay_mining  ->  metadata  ->  export

Before this module, ``skill-export`` required every record to *already* carry
``metadata.replay_mining`` (axes / skill_candidate / usefulness_score). That
field is produced upstream by ``replay_mining.mine_session_turn_sample`` and is
almost never present in a user's raw logs. ``distill-skills`` runs that mining
step automatically, so a user can point the command at ordinary OpenAI-style
``messages`` logs and get installable Skill packages out.

Supported input shapes (auto-detected, all JSON / JSONL):
  1. Already-mined replay records (``metadata.replay_mining`` present)
     -> passed straight through to the exporter.
  2. Chat sessions::
         {"session_id": "...", "task_id": "...", "reward": 0.8,
          "messages": [{"role": "user"|"assistant"|"tool", "content": "..."}]}
     -> split into turn samples, mined per turn, reward propagated.
  3. A bare list of OpenAI messages (single anonymous session).

The heavy lifting (candidate selection, grouping by capability axis, 6-gate
quality scoring, SKILL.md + validation.jsonl + manifest.json emission) is
unchanged and reused verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_agentic_rl.collectors.conversation_collector import (
    collect_session_turn_samples,
)
from hermes_agentic_rl.collectors.replay_mining import mine_session_turn_sample
from hermes_agentic_rl.collectors.skill_export import export_skill_candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def distill_skills(
    *,
    trace_paths: list[str | Path],
    output_dir: str | Path,
    min_reward: float = 0.0,
    status_filter: str | None = None,
    mining_config: dict[str, Any] | None = None,
    export_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Distill installable Skill candidates from raw agent traces.

    Args:
        trace_paths: one or more ``.json`` / ``.jsonl`` files of raw sessions.
        output_dir: where the Skill packages + reports are written.
        min_reward: minimum per-turn reward to keep a candidate.
        status_filter: if set (``ready_for_review`` / ``draft`` / ``blocked``),
            the human report highlights only skills with that quality status.
        mining_config: forwarded to ``replay_mining``.
        export_config: forwarded to ``skill_export.export_skill_candidates``.

    Returns:
        The exporter ``summary`` dict, augmented with ``mining`` stats.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [1] Load + auto-mine into the record shape the exporter expects.
    records, mining_stats = _load_and_mine_traces(
        trace_paths, mining_config=mining_config
    )

    # Persist the intermediate mined replay so the run is reproducible /
    # debuggable and so power users can re-run the exporter alone.
    mined_path = out_dir / "_mined_replay.jsonl"
    _write_jsonl(mined_path, records)

    # [2] Reuse the existing exporter verbatim.
    cfg: dict[str, Any] = dict(export_config or {})
    cfg.setdefault("min_reward", float(min_reward))
    summary = export_skill_candidates(
        input_path=str(mined_path),
        output_dir=str(out_dir),
        config=cfg,
    )
    summary["mining"] = mining_stats

    # [3] Render the human-readable report.
    report_path = out_dir / "DISTILL_REPORT.md"
    report_path.write_text(
        _render_report(summary, status_filter=status_filter), encoding="utf-8"
    )
    summary["report_path"] = str(report_path)
    return summary


# ---------------------------------------------------------------------------
# [1] Load + auto-mine
# ---------------------------------------------------------------------------


def _load_and_mine_traces(
    trace_paths: list[str | Path],
    *,
    mining_config: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_records: list[dict[str, Any]] = []
    for path in trace_paths:
        raw_records.extend(_load_one(path))

    mined: list[dict[str, Any]] = []
    n_passthrough = 0
    n_turns_mined = 0
    n_sessions = 0
    for raw in raw_records:
        # Shape 1: already mined -> pass through untouched.
        if _already_mined(raw):
            mined.append(raw)
            n_passthrough += 1
            continue

        # Shape 2/3: a chat session -> split + mine per assistant turn.
        session_records = _mine_session(raw, mining_config=mining_config)
        if session_records:
            n_sessions += 1
            n_turns_mined += len(session_records)
            mined.extend(session_records)

    stats = {
        "trace_files": len(trace_paths),
        "raw_records": len(raw_records),
        "passthrough_records": n_passthrough,
        "sessions_mined": n_sessions,
        "turns_mined": n_turns_mined,
        "total_mined_records": len(mined),
    }
    return mined, stats


def _load_one(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix in {".jsonl", ".jl"}:
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                out.append(payload)
            elif isinstance(payload, list):
                # A JSONL line that is itself a list of messages.
                out.append({"messages": payload})
        return out

    payload = json.loads(text)
    if isinstance(payload, list):
        # Either a list of session dicts, or a bare list of messages.
        if payload and all(
            isinstance(m, dict) and "role" in m for m in payload
        ):
            return [{"messages": payload}]
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return [it for it in payload["samples"] if isinstance(it, dict)]
        return [payload]
    return []


def _already_mined(record: dict[str, Any]) -> bool:
    meta = record.get("metadata")
    return isinstance(meta, dict) and isinstance(meta.get("replay_mining"), dict)


def _mine_session(
    raw: dict[str, Any],
    *,
    mining_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        return []

    session_id = str(raw.get("session_id") or raw.get("id") or "session")
    task_id = raw.get("task_id")
    session_reward = _coerce_float(raw.get("reward"), default=None)
    # Optional per-turn rewards aligned to assistant turns.
    turn_rewards = raw.get("turn_rewards")
    if not isinstance(turn_rewards, list):
        turn_rewards = None

    samples = collect_session_turn_samples(
        messages, session_id=session_id, task_id=task_id
    )

    out: list[dict[str, Any]] = []
    for sample in samples:
        reward = _turn_reward(
            sample.turn_index, turn_rewards, session_reward
        )
        mining = mine_session_turn_sample(
            sample, reward=reward, metadata=sample.metadata, config=mining_config
        )
        if mining is None:
            continue
        record = {
            "reward": reward,
            "metadata": {
                "session_id": session_id,
                "task_id": task_id,
                "turn_index": sample.turn_index,
                "capability_axes": list(mining.get("axes", [])),
                "replay_mining": mining,
                "source_turn": {
                    "prompt_messages": sample.prompt_messages,
                    "assistant_message": sample.assistant_message,
                    "feedback_messages": sample.feedback_messages,
                },
            },
        }
        out.append(record)
    return out


def _turn_reward(
    turn_index: int,
    turn_rewards: list[Any] | None,
    session_reward: float | None,
) -> float:
    if turn_rewards is not None and 0 <= turn_index < len(turn_rewards):
        return _coerce_float(turn_rewards[turn_index], default=0.0)
    if session_reward is not None:
        return session_reward
    return 0.0


def _coerce_float(value: Any, *, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# [3] Human-readable report
# ---------------------------------------------------------------------------


def _render_report(summary: dict[str, Any], *, status_filter: str | None) -> str:
    mining = summary.get("mining", {})
    quality = summary.get("quality", {})
    status_counts = quality.get("status_counts", {})
    skills = summary.get("skills", [])

    lines = [
        "# Skill Distillation Report",
        "",
        "> Auto-generated by `hermes-rl distill-skills`. Each skill below is a "
        "**candidate** mined from real agent traces — review and harden before "
        "installing as a production Skill.",
        "",
        "## Funnel",
        "",
        f"- Trace files read: **{mining.get('trace_files', 0)}**",
        f"- Raw records: **{mining.get('raw_records', 0)}** "
        f"({mining.get('sessions_mined', 0)} sessions + "
        f"{mining.get('passthrough_records', 0)} pre-mined)",
        f"- Turns mined: **{mining.get('turns_mined', 0)}**",
        f"- Candidate turns (passed filters): **{summary.get('candidate_records', 0)}**",
        f"- Rejected turns: **{summary.get('rejected_records', 0)}**",
        f"- **Skills exported: {summary.get('skills_exported', 0)}**",
        "",
        "## Quality Status",
        "",
        "| status | count | meaning |",
        "|---|---|---|",
        f"| `ready_for_review` | {status_counts.get('ready_for_review', 0)} | "
        "passed all quality gates — review first |",
        f"| `draft` | {status_counts.get('draft', 0)} | "
        "partial signal — needs more examples / evidence |",
        f"| `blocked` | {status_counts.get('blocked', 0)} | "
        "failed gates (low reward / inconsistent / negative-heavy) |",
        "",
        f"- Mean quality score: **{float(quality.get('mean_quality_score', 0.0)):.3f}**",
    ]

    rejection = summary.get("rejection_reasons", {})
    if rejection:
        lines += ["", "### Why turns were rejected", ""]
        for reason, count in sorted(rejection.items()):
            lines.append(f"- `{reason}`: {count}")

    lines += ["", "## Exported Skills", ""]
    if not skills:
        lines.append(
            "_No skills met the export threshold. Lower `--min-reward`, "
            "supply more successful traces, or check that your logs contain "
            "assistant turns with positive feedback._"
        )
    else:
        shown = skills
        if status_filter:
            shown = [
                s
                for s in skills
                if str(s.get("quality", {}).get("status")) == status_filter
            ]
            lines.append(f"_Filtered to status = `{status_filter}`._")
            lines.append("")
        lines += [
            "| skill | status | score | examples | primary axis | path |",
            "|---|---|---|---|---|---|",
        ]
        for s in shown:
            q = s.get("quality", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{s.get('name', '')}`",
                        f"`{q.get('status', 'draft')}`",
                        f"{float(q.get('score', 0.0)):.2f}",
                        str(s.get("examples_written", 0)),
                        str(s.get("group_key", "")),
                        f"`{s.get('path', '')}`",
                    ]
                )
                + " |"
            )

    lines += [
        "",
        "## Next Steps",
        "",
        "1. Open a `ready_for_review` skill's `SKILL.md` and sanity-check the procedure.",
        "2. Run its `validation.jsonl` as the first benchmark slice.",
        "3. Install the package directory as a Skill once it passes review.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
