from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.collectors.replay_export import append_jsonl
from hermes_agentic_rl.collectors.replay_mining import (
    mine_session_turn_sample,
    summarize_replay_mining,
)
from hermes_agentic_rl.collectors.replay_quality import (
    apply_record_quality_filters,
    load_jsonl_records_best_effort,
    normalize_replay_records,
    summarize_rejections,
)
from hermes_agentic_rl.collectors.trajectory_adapter import (
    session_turn_sample_to_train_sample,
)
from hermes_agentic_rl.framework import build_framework
from hermes_agentic_rl.offline.replay_buffer import ReplayBuffer, TrainSample


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _load_session_records(
    path: str | Path,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    p = Path(path)
    if p.suffix in {".jsonl", ".jl"}:
        return load_jsonl_records_best_effort(p)

    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
        records = list(payload["sessions"])
    else:
        records = [payload]
    return records, [], {"loaded_records": len(records)}


def _train_sample_to_replay_record(sample: TrainSample) -> dict[str, Any]:
    return {
        "prompt_ids": list(sample.prompt_ids),
        "response_ids": list(sample.response_ids),
        "reward": float(sample.reward),
        "advantage": sample.advantage,
        "metadata": dict(sample.metadata),
    }


def _reward_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        reward = float(record.get("reward", 0.0))
        if reward > 0:
            counter["positive"] += 1
        elif reward < 0:
            counter["negative"] += 1
        else:
            counter["zero"] += 1
    return {
        "positive": counter["positive"],
        "negative": counter["negative"],
        "zero": counter["zero"],
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(target)


def _session_rejection(
    *,
    kind: str,
    index: int,
    error: Exception,
    record: Any | None = None,
    turn_index: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "index": index,
        "error": f"{type(error).__name__}: {error}",
    }
    if turn_index is not None:
        payload["turn_index"] = turn_index
    if record is not None:
        payload["record"] = record
    return payload


def _buffer_from_replay_records(records: list[dict[str, Any]]) -> ReplayBuffer:
    buffer = ReplayBuffer()
    for record in records:
        buffer.add_sample(
            TrainSample(
                prompt_ids=list(record["prompt_ids"]),
                response_ids=list(record["response_ids"]),
                reward=float(record.get("reward", 0.0)),
                advantage=record.get("advantage"),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return buffer


def _build_quality_report(
    *,
    input_path: str | Path,
    output_path: str | Path,
    quarantine_path: str | Path | None,
    quality_report_path: str | Path | None,
    input_stats: dict[str, Any],
    sessions_loaded: int,
    sessions_exported: int,
    turns_seen: int,
    raw_samples_generated: int,
    written_records: list[dict[str, Any]],
    input_rejections: list[dict[str, Any]],
    session_rejections: list[dict[str, Any]],
    replay_invalid: list[dict[str, Any]],
    quality_filtered: list[dict[str, Any]],
    quality_cfg: dict[str, Any],
    replay_mining_cfg: Any,
) -> dict[str, Any]:
    quarantined = input_rejections + session_rejections + replay_invalid + quality_filtered
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "quarantine_path": str(quarantine_path) if quarantine_path else "",
        "quality_report_path": str(quality_report_path) if quality_report_path else "",
        "input_stats": dict(input_stats),
        "sessions_loaded": sessions_loaded,
        "sessions_exported": sessions_exported,
        "turns_seen": turns_seen,
        "raw_samples_generated": raw_samples_generated,
        "samples_written": len(written_records),
        "reward_distribution": _reward_distribution(written_records),
        "input_rejections": summarize_rejections(input_rejections),
        "session_rejections": summarize_rejections(session_rejections),
        "replay_invalid": summarize_rejections(replay_invalid),
        "quality_filtered": summarize_rejections(quality_filtered),
        "quarantined": summarize_rejections(quarantined),
        "data_quality": dict(quality_cfg),
        "replay_mining": summarize_replay_mining(
            written_records,
            config=replay_mining_cfg,
        ),
    }


def run_session_replay_config(
    cfg: dict[str, Any],
    *,
    output_path: str | None = None,
    input_path: str | None = None,
) -> int:
    framework = build_framework(cfg, build_sidecar=False)
    session_api = framework.session
    effective_input = input_path or cfg.get("input_path")
    if not effective_input:
        raise ValueError("session-replay requires `input_path` in config or --input")
    effective_output = output_path or cfg.get("output_path")
    if not effective_output:
        raise ValueError("session-replay requires `output_path` in config or --output")

    records, input_rejections, input_stats = _load_session_records(effective_input)
    quarantine_path = cfg.get("quarantine_path")
    quality_report_path = cfg.get("quality_report_path")
    quality_cfg = cfg.get("data_quality", {}) or {}
    replay_mining_cfg = cfg.get("replay_mining", {})
    exported_records: list[dict[str, Any]] = []
    session_rejections: list[dict[str, Any]] = []
    session_count = 0
    turn_count = 0

    for idx, record in enumerate(records):
        try:
            turn_samples = session_api.samples_from_record(record)
        except Exception as exc:
            session_rejections.append(
                _session_rejection(
                    kind="invalid_session_record",
                    index=idx,
                    error=exc,
                    record=record,
                )
            )
            continue
        if not turn_samples:
            continue
        session_count += 1
        turn_count += len(turn_samples)
        for turn_idx, sample in enumerate(turn_samples):
            try:
                reward_summary = None
                if session_api.judge_config:
                    from hermes_agentic_rl.collectors.session_judge import (
                        judge_session_turn_sample,
                    )

                    reward_summary = judge_session_turn_sample(
                        sample,
                        session_api.judge_config,
                    )
                train_sample = session_turn_sample_to_train_sample(
                    sample,
                    tokenizer=session_api.tokenizer,
                    reward_summary=reward_summary,
                )
                mining = mine_session_turn_sample(
                    sample,
                    reward=train_sample.reward,
                    metadata=train_sample.metadata,
                    config=replay_mining_cfg,
                )
                if mining is not None:
                    train_sample.metadata["replay_mining"] = mining
                    train_sample.metadata["capability_axes"] = list(mining.get("axes", []))
            except Exception as exc:
                session_rejections.append(
                    _session_rejection(
                        kind="invalid_session_turn_sample",
                        index=idx,
                        turn_index=turn_idx,
                        error=exc,
                        record={
                            "session_id": getattr(sample, "session_id", None),
                            "task_id": getattr(sample, "task_id", None),
                        },
                    )
                )
                continue
            exported_records.append(_train_sample_to_replay_record(train_sample))

    valid_records, replay_invalid = normalize_replay_records(exported_records)
    valid_records, quality_filtered = apply_record_quality_filters(valid_records, quality_cfg)
    quarantined = input_rejections + session_rejections + replay_invalid + quality_filtered
    if quarantine_path and quarantined:
        append_jsonl(quarantine_path, quarantined)

    buffer = _buffer_from_replay_records(valid_records)
    buffer.save_jsonl(effective_output)

    report = _build_quality_report(
        input_path=effective_input,
        output_path=effective_output,
        quarantine_path=quarantine_path,
        quality_report_path=quality_report_path,
        input_stats=input_stats,
        sessions_loaded=len(records),
        sessions_exported=session_count,
        turns_seen=turn_count,
        raw_samples_generated=len(exported_records),
        written_records=valid_records,
        input_rejections=input_rejections,
        session_rejections=session_rejections,
        replay_invalid=replay_invalid,
        quality_filtered=quality_filtered,
        quality_cfg=quality_cfg,
        replay_mining_cfg=replay_mining_cfg,
    )
    if quality_report_path:
        _write_json(quality_report_path, report)

    reward_distribution = report["reward_distribution"]
    print(
        "[session-replay] "
        f"sessions={session_count}/{len(records)} turns={turn_count} "
        f"raw_samples={len(exported_records)} samples={len(buffer.samples)} "
        f"positive={reward_distribution['positive']} negative={reward_distribution['negative']} "
        f"zero={reward_distribution['zero']} input_rejected={len(input_rejections)} "
        f"session_rejected={len(session_rejections)} invalid_replay={len(replay_invalid)} "
        f"quality_filtered={len(quality_filtered)} quarantined={len(quarantined)} "
        f"report={quality_report_path or ''} output={effective_output}"
    )
    return 0


def run_session_replay(
    config_path: str | Path,
    *,
    output_path: str | None = None,
    input_path: str | None = None,
) -> int:
    cfg = _load_config(config_path)
    return run_session_replay_config(
        cfg,
        output_path=output_path,
        input_path=input_path,
    )
