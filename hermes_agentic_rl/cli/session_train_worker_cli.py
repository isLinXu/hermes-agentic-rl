from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.collectors.replay_export import append_jsonl
from hermes_agentic_rl.collectors.replay_quality import (
    apply_record_quality_filters as _apply_record_quality_filters,
)
from hermes_agentic_rl.collectors.replay_quality import (
    normalize_replay_records as _normalize_replay_records,
)
from hermes_agentic_rl.framework import build_framework
from hermes_agentic_rl.monitor.dashboard import LiveDashboard
from hermes_agentic_rl.monitor.writers import build_writer_from_config
from hermes_agentic_rl.offline.replay_buffer import ReplayBuffer


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _build_backend(cfg: dict[str, Any]) -> Any:
    backend_cfg = cfg.get("backend", {}) or {}
    name = str(backend_cfg.get("name", "tiny"))
    if name == "tiny":
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

        tiny_cfg = dict(backend_cfg)
        tiny_cfg.pop("name", None)
        return TinyCausalLMBackend(TinyBackendConfig(**tiny_cfg))
    if name == "hf":
        from hermes_agentic_rl.backends.hf import HFBackendConfig, HFCausalLMBackend

        hf_cfg = dict(backend_cfg)
        hf_cfg.pop("name", None)
        return HFCausalLMBackend(HFBackendConfig(**hf_cfg))
    raise ValueError(f"unsupported backend.name for session train worker: {name!r}")


def _load_policy_if_present(backend: Any, path: str | Path) -> None:
    target = Path(path)
    if not target.exists() or not hasattr(backend, "model"):
        return

    import torch

    try:
        state = torch.load(target, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(target, map_location="cpu")
    backend.model.load_state_dict(state)  # type: ignore[attr-defined]


def _load_rm_head_if_present(reward_model: Any, path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        return

    import torch

    try:
        state = torch.load(target, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(target, map_location="cpu")
    reward_model.head.load_state_dict(state)


def _default_worker_state() -> dict[str, Any]:
    return {
        "offset": 0,
        "records_seen": 0,
        "trained_samples": 0,
        "trained_pairs": 0,
        "candidate_pairs": 0,
        "pending_pairs": 0,
        "seen_pairs": 0,
        "updates": 0,
        "invalid_json_lines": 0,
        "invalid_records": 0,
        "quality_filtered_records": 0,
        "quarantined_records": 0,
        "seen_pair_fingerprints": [],
    }


def _load_worker_state(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return _default_worker_state()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return _default_worker_state()
    state = _default_worker_state()
    state.update(loaded)
    raw_fingerprints = state.get("seen_pair_fingerprints")
    if isinstance(raw_fingerprints, list):
        state["seen_pair_fingerprints"] = [str(item) for item in raw_fingerprints if item]
    else:
        state["seen_pair_fingerprints"] = []
    state["seen_pairs"] = max(
        int(state.get("seen_pairs", 0)),
        len(state["seen_pair_fingerprints"]),
    )
    return state


def _save_worker_state(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(state)
    raw_fingerprints = serializable.get("seen_pair_fingerprints")
    if isinstance(raw_fingerprints, set):
        serializable["seen_pair_fingerprints"] = sorted(str(item) for item in raw_fingerprints if item)
    elif isinstance(raw_fingerprints, list):
        serializable["seen_pair_fingerprints"] = [
            str(item) for item in raw_fingerprints if item
        ]
    else:
        serializable["seen_pair_fingerprints"] = []
    serializable["seen_pairs"] = max(
        int(serializable.get("seen_pairs", 0)),
        len(serializable["seen_pair_fingerprints"]),
    )
    tmp_target = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp_target.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_target.replace(target)


def _read_new_jsonl_records(
    path: str | Path,
    offset: int,
    *,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return [], offset, {"invalid_json_lines": 0, "rejected": []}

    records: list[dict[str, Any]] = []
    safe_offset = max(0, int(offset))
    rejected: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        while True:
            if max_records is not None and len(records) >= max_records:
                break
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.endswith("\n"):
                handle.seek(line_start)
                break
            safe_offset = handle.tell()
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append(
                    {
                        "kind": "invalid_json_line",
                        "path": str(target),
                        "offset": line_start,
                        "raw_line": line,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if isinstance(payload, dict):
                records.append(payload)
            else:
                rejected.append(
                    {
                        "kind": "non_dict_json_line",
                        "path": str(target),
                        "offset": line_start,
                        "raw_payload": payload,
                        "error": f"expected dict, got {type(payload).__name__}",
                    }
                )
    return records, safe_offset, {"invalid_json_lines": len(rejected), "rejected": rejected}


def _source_state(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"signature": "missing", "size": 0, "inode": 0}
    stat = target.stat()
    inode = int(getattr(stat, "st_ino", 0) or 0)
    return {
        "signature": f"{inode}:{stat.st_size}:{stat.st_mtime_ns}",
        "size": int(stat.st_size),
        "inode": inode,
    }


def _pair_fingerprint(pair: Any) -> str:
    payload = json.dumps(
        {
            "prompt_ids": list(pair.prompt_ids),
            "chosen_ids": list(pair.chosen_ids),
            "rejected_ids": list(pair.rejected_ids),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _should_reset_for_missing_checkpoint(
    *,
    algo: str,
    state: dict[str, Any],
    save_path: str | Path,
) -> bool:
    if Path(save_path).exists():
        return False
    if int(state.get("updates", 0)) > 0:
        return True
    if algo == "bc":
        return int(state.get("offset", 0)) > 0
    return bool(state.get("seen_pair_fingerprints"))


def _resolve_bc_offset(
    state: dict[str, Any],
    current_source: dict[str, Any],
) -> tuple[int, str | None]:
    previous_offset = max(0, int(state.get("offset", 0)))
    current_size = int(current_source.get("size", 0))
    previous_signature = str(state.get("source_signature", "") or "")
    current_signature = str(current_source.get("signature", "") or "")
    previous_inode = int(state.get("source_inode", 0) or 0)
    current_inode = int(current_source.get("inode", 0) or 0)

    if previous_offset > current_size:
        return 0, "truncated"
    if previous_offset > 0 and previous_inode and current_inode and previous_inode != current_inode:
        return 0, "rotated"
    if (
        previous_offset > 0
        and previous_signature
        and current_signature
        and previous_signature != current_signature
        and current_size <= previous_offset
    ):
        return 0, "rewritten"
    return previous_offset, None


def _emit_worker_metrics(
    sink: Any,
    *,
    algo: str,
    state: dict[str, Any],
    event: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if sink is None:
        return
    record = {
        "ts": time.time(),
        "command": "session-train-worker",
        "event": event,
        "algo": algo,
        "updates": int(state.get("updates", 0)),
        "records_seen": int(state.get("records_seen", 0)),
        "trained_samples": int(state.get("trained_samples", 0)),
        "trained_pairs": int(state.get("trained_pairs", 0)),
        "candidate_pairs": int(state.get("candidate_pairs", 0)),
        "pending_pairs": int(state.get("pending_pairs", 0)),
        "seen_pairs": int(state.get("seen_pairs", 0)),
        "invalid_json_lines": int(state.get("invalid_json_lines", 0)),
        "invalid_records": int(state.get("invalid_records", 0)),
        "quality_filtered_records": int(state.get("quality_filtered_records", 0)),
        "quarantined_records": int(state.get("quarantined_records", 0)),
    }
    if "last_loss" in state:
        record["last_loss"] = float(state["last_loss"])
    if extra:
        record.update(extra)
    sink(record)


def run_session_train_worker_config(
    cfg: dict[str, Any],
    *,
    once: bool = False,
) -> int:
    framework = build_framework(cfg, build_sidecar=False)
    session_api = framework.session
    input_path = cfg.get("input_path")
    save_path = cfg.get("save_path")
    if not input_path:
        raise ValueError("session-train-worker requires `input_path`")
    if not save_path:
        raise ValueError("session-train-worker requires `save_path`")

    train_cfg = cfg.get("train", {}) or {}
    algo = str(cfg.get("algo", "bc")).lower()
    if algo not in {"bc", "dpo", "rm"}:
        raise ValueError(
            f"session-train-worker currently supports algo in {{'bc','dpo','rm'}}, got {algo!r}"
        )

    state_path = cfg.get("state_path") or f"{save_path}.state.json"
    quarantine_path = cfg.get("quarantine_path")
    poll_interval_sec = float(train_cfg.get("poll_interval_sec", 1.0))
    min_reward = float(train_cfg.get("min_reward", 0.0))
    max_idle_polls = int(train_cfg.get("max_idle_polls", 1 if (once or cfg.get('run_once')) else 0))
    max_records_per_update = int(train_cfg.get("max_records_per_update", 0))
    max_pairs_per_update = int(train_cfg.get("max_pairs_per_update", 0))
    run_once = bool(cfg.get("run_once", False)) or once
    train_only_new_pairs = bool(train_cfg.get("train_only_new_pairs", True))
    quality_cfg = train_cfg.get("data_quality", {}) or {}
    dashboard_cfg = cfg.get("dashboard", {}) or {}
    dashboard: LiveDashboard | None = None
    extra_sinks: list[Any] = []
    if bool(dashboard_cfg.get("enabled", False)):
        dashboard = LiveDashboard(
            host=str(dashboard_cfg.get("host", "127.0.0.1")),
            port=int(dashboard_cfg.get("port", 8765)),
        )
        url = dashboard.start()
        print(f"[session-train-worker] dashboard live at {url}")
        extra_sinks.append(dashboard.record)
    metrics_writer = build_writer_from_config(
        cfg.get("metrics") if isinstance(cfg, dict) else None,
        output_dir=Path(save_path).parent,
        extra=extra_sinks or None,
    )
    metrics_sink = metrics_writer
    if metrics_sink is None and extra_sinks:
        metrics_sink = extra_sinks[0]

    backend = _build_backend(cfg)
    _load_policy_if_present(backend, save_path)

    from hermes_agentic_rl.offline.bc import BCConfig, BCTrainer
    from hermes_agentic_rl.offline.dpo import DPOConfig, DPOTrainer

    bc_cfg = BCConfig(
        n_epochs=int(train_cfg.get("n_epochs", 1)),
        batch_size=int(train_cfg.get("batch_size", 4)),
        lr=float(train_cfg.get("lr", 5e-3)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        shuffle=bool(train_cfg.get("shuffle", True)),
        seed=train_cfg.get("seed", 0),
        log_every=int(train_cfg.get("log_every", 0)),
        min_response_tokens=int(train_cfg.get("min_response_tokens", 1)),
    )
    dpo_cfg = DPOConfig(
        n_epochs=int(train_cfg.get("n_epochs", 1)),
        batch_size=int(train_cfg.get("batch_size", 4)),
        lr=float(train_cfg.get("lr", 5e-4)),
        beta=float(train_cfg.get("beta", 0.1)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        shuffle=bool(train_cfg.get("shuffle", True)),
        seed=train_cfg.get("seed", 0),
        log_every=int(train_cfg.get("log_every", 0)),
    )
    pref_cfg = train_cfg.get("preference", {}) or {}
    min_reward_gap = float(pref_cfg.get("min_reward_gap", 0.25))
    max_pairs_per_prompt = int(pref_cfg.get("max_pairs_per_prompt", 1))

    reward_model = None
    rm_cfg = None
    if algo == "rm":
        from hermes_agentic_rl.rewards.reward_model import (
            RewardModel,
            RewardModelConfig,
        )

        freeze_base = bool(train_cfg.get("freeze_base", True))
        reward_model = RewardModel(backend, freeze_base=freeze_base)
        _load_rm_head_if_present(reward_model, save_path)
        rm_cfg = RewardModelConfig(
            lr=float(train_cfg.get("lr", 1e-3)),
            n_epochs=int(train_cfg.get("n_epochs", 1)),
            batch_size=int(train_cfg.get("batch_size", 4)),
            freeze_base=freeze_base,
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            shuffle=bool(train_cfg.get("shuffle", True)),
            seed=train_cfg.get("seed", 0),
            log_every=int(train_cfg.get("log_every", 0)),
        )

    state = _load_worker_state(state_path)
    if _should_reset_for_missing_checkpoint(algo=algo, state=state, save_path=save_path):
        state = _default_worker_state()
        print(
            "[session-train-worker] "
            f"algo={algo} checkpoint_missing=true restart_from_replay=true save_path={save_path}"
        )
        _emit_worker_metrics(
            metrics_sink,
            algo=algo,
            state=state,
            event="checkpoint_reset",
            extra={"save_path": str(save_path)},
        )
    idle_polls = 0
    try:
        while True:
            scan_rejected: list[dict[str, Any]] = []
            if algo == "bc":
                current_source = _source_state(input_path)
                read_offset, reset_reason = _resolve_bc_offset(state, current_source)
                if reset_reason is not None:
                    print(
                        "[session-train-worker] "
                        f"algo=bc source_reset={reset_reason} offset={state.get('offset', 0)}"
                    )
                records, new_offset, read_stats = _read_new_jsonl_records(
                    input_path,
                    read_offset,
                    max_records=(max_records_per_update or None),
                )
                scan_rejected = list(read_stats.get("rejected", []))
                has_update = bool(records)
                current_signature = str(current_source.get("signature"))
                wrote_rejects_this_scan = True
            else:
                current_source = _source_state(input_path)
                current_signature = str(current_source.get("signature"))
                pending_pairs = int(state.get("pending_pairs", 0))
                backlog_pending = train_only_new_pairs and pending_pairs > 0
                wrote_rejects_this_scan = current_signature != state.get("source_signature")
                if current_signature == state.get("source_signature") and not backlog_pending:
                    records = []
                    has_update = False
                    new_offset = int(state.get("offset", 0))
                else:
                    all_records, _new_offset, read_stats = _read_new_jsonl_records(input_path, 0)
                    records = all_records
                    scan_rejected = list(read_stats.get("rejected", []))
                    has_update = bool(records)
                    new_offset = _new_offset

            valid_records, invalid_records = _normalize_replay_records(records)
            valid_records, quality_rejected = _apply_record_quality_filters(valid_records, quality_cfg)
            total_rejected = scan_rejected + invalid_records + quality_rejected
            if total_rejected and wrote_rejects_this_scan:
                state["invalid_json_lines"] = int(state.get("invalid_json_lines", 0)) + len(scan_rejected)
                state["invalid_records"] = int(state.get("invalid_records", 0)) + len(invalid_records)
                state["quality_filtered_records"] = int(
                    state.get("quality_filtered_records", 0)
                ) + len(quality_rejected)
                if quarantine_path:
                    append_jsonl(quarantine_path, total_rejected)
                    state["quarantined_records"] = int(state.get("quarantined_records", 0)) + len(total_rejected)
            records = valid_records
            has_update = bool(records)

            if not has_update:
                _save_worker_state(state_path, state)
                idle_polls += 1
                _emit_worker_metrics(
                    metrics_sink,
                    algo=algo,
                    state=state,
                    event="idle",
                    extra={
                        "source_signature": current_signature,
                        "scan_rejected": len(scan_rejected),
                        "invalid_records_this_scan": len(invalid_records),
                        "quality_rejected_this_scan": len(quality_rejected),
                    },
                )
                if run_once or (max_idle_polls > 0 and idle_polls >= max_idle_polls):
                    break
                time.sleep(poll_interval_sec)
                continue

            idle_polls = 0
            state["offset"] = new_offset
            state["source_signature"] = current_signature
            state["source_inode"] = int(current_source.get("inode", 0))
            if algo != "bc":
                state["records_seen"] = len(records)
            else:
                state["records_seen"] = int(state.get("records_seen", 0)) + len(records)
                state["pending_pairs"] = 0

            metric_extra = {
                "source_signature": current_signature,
                "scan_rejected": len(scan_rejected),
                "invalid_records_this_scan": len(invalid_records),
                "quality_rejected_this_scan": len(quality_rejected),
                "quarantine_path": str(quarantine_path) if quarantine_path else "",
            }

            if algo == "bc":
                buffer = session_api.replay_buffer_from_replay_records(
                    records,
                    min_reward=min_reward,
                )
                if buffer.samples:
                    bc_trainer = BCTrainer(backend, buffer, cfg=bc_cfg)
                    bc_stats = bc_trainer.train()
                    bc_trainer.save_policy(save_path)
                    state["trained_samples"] = int(state.get("trained_samples", 0)) + len(buffer.samples)
                    state["updates"] = int(state.get("updates", 0)) + 1
                    if bc_stats.steps:
                        state["last_loss"] = float(bc_stats.steps[-1]["nll"])
                    print(
                        "[session-train-worker] "
                        f"algo=bc update={state['updates']} records={len(records)} "
                        f"trained={len(buffer.samples)} save_path={save_path}"
                    )
                    _emit_worker_metrics(
                        metrics_sink,
                        algo=algo,
                        state=state,
                        event="update",
                        extra={
                            **metric_extra,
                            "records": len(records),
                            "trained_now": len(buffer.samples),
                        },
                    )
                else:
                    print(
                        "[session-train-worker] "
                        f"algo=bc records={len(records)} trained=0 min_reward={min_reward}"
                    )
                    _emit_worker_metrics(
                        metrics_sink,
                        algo=algo,
                        state=state,
                        event="filtered",
                        extra={
                            **metric_extra,
                            "records": len(records),
                            "trained_now": 0,
                            "min_reward": min_reward,
                        },
                    )
            else:
                samples = session_api.replay_train_samples_from_records(
                    records,
                    min_reward=min_reward,
                )
                all_pairs = session_api.preference_pairs_from_train_samples(
                    samples,
                    min_reward_gap=min_reward_gap,
                    max_pairs_per_prompt=max_pairs_per_prompt,
                )
                seen_pair_fingerprints = {
                    str(item) for item in state.get("seen_pair_fingerprints", []) if item
                }
                if train_only_new_pairs:
                    unseen_pairs = []
                    unseen_pair_fingerprints: list[str] = []
                    for pair in all_pairs:
                        fingerprint = _pair_fingerprint(pair)
                        if fingerprint in seen_pair_fingerprints:
                            continue
                        unseen_pairs.append(pair)
                        unseen_pair_fingerprints.append(fingerprint)
                    if max_pairs_per_update > 0:
                        pairs = unseen_pairs[:max_pairs_per_update]
                        new_pair_fingerprints = unseen_pair_fingerprints[:max_pairs_per_update]
                    else:
                        pairs = unseen_pairs
                        new_pair_fingerprints = unseen_pair_fingerprints
                else:
                    pairs = list(all_pairs[:max_pairs_per_update] if max_pairs_per_update > 0 else all_pairs)
                    new_pair_fingerprints = [_pair_fingerprint(pair) for pair in pairs]
                buffer = ReplayBuffer.from_pairs(pairs)
                state["trained_samples"] = len(samples)
                state["candidate_pairs"] = len(all_pairs)
                state["pending_pairs"] = max(
                    0,
                    len(all_pairs) - len(seen_pair_fingerprints) - len(buffer.pairs),
                ) if train_only_new_pairs else max(0, len(all_pairs) - len(buffer.pairs))
                if buffer.pairs:
                    if algo == "dpo":
                        dpo_trainer = DPOTrainer(backend, buffer, cfg=dpo_cfg)
                        dpo_stats = dpo_trainer.train()
                        dpo_trainer.save_policy(save_path)
                        if dpo_stats.steps:
                            state["last_loss"] = float(dpo_stats.steps[-1]["loss"])
                    else:
                        from hermes_agentic_rl.rewards.reward_model import RewardModelTrainer

                        rm_trainer = RewardModelTrainer(reward_model, buffer, cfg=rm_cfg)  # type: ignore[arg-type]
                        rm_stats = rm_trainer.train()
                        rm_trainer.save_head(save_path)
                        if rm_stats.steps:
                            state["last_loss"] = float(rm_stats.steps[-1]["loss"])
                    state["trained_pairs"] = len(buffer.pairs)
                    state["updates"] = int(state.get("updates", 0)) + 1
                    if train_only_new_pairs:
                        seen_pair_fingerprints.update(new_pair_fingerprints)
                        state["seen_pair_fingerprints"] = sorted(seen_pair_fingerprints)
                    state["seen_pairs"] = max(
                        int(state.get("seen_pairs", 0)),
                        len(state.get("seen_pair_fingerprints", [])),
                    )
                    state["pending_pairs"] = max(
                        0,
                        len(all_pairs) - len(state.get("seen_pair_fingerprints", [])),
                    ) if train_only_new_pairs else max(0, len(all_pairs) - len(buffer.pairs))
                    print(
                        "[session-train-worker] "
                        f"algo={algo} update={state['updates']} records={len(records)} "
                        f"samples={len(samples)} candidate_pairs={len(all_pairs)} "
                        f"new_pairs={len(buffer.pairs)} pending_pairs={state['pending_pairs']} "
                        f"save_path={save_path}"
                    )
                    _emit_worker_metrics(
                        metrics_sink,
                        algo=algo,
                        state=state,
                        event="update",
                        extra={
                            **metric_extra,
                            "records": len(records),
                            "pairs_now": len(buffer.pairs),
                        },
                    )
                else:
                    state["trained_pairs"] = 0
                    state["seen_pairs"] = max(
                        int(state.get("seen_pairs", 0)),
                        len(state.get("seen_pair_fingerprints", [])),
                    )
                    print(
                        "[session-train-worker] "
                        f"algo={algo} records={len(records)} samples={len(samples)} "
                        f"candidate_pairs={len(all_pairs)} new_pairs=0 "
                        f"pending_pairs={state['pending_pairs']} min_gap={min_reward_gap}"
                    )
                    _emit_worker_metrics(
                        metrics_sink,
                        algo=algo,
                        state=state,
                        event="filtered",
                        extra={
                            **metric_extra,
                            "records": len(records),
                            "pairs_now": 0,
                            "min_reward_gap": min_reward_gap,
                        },
                    )

            _save_worker_state(state_path, state)
            if run_once:
                break

        _save_worker_state(state_path, state)
        return 0
    finally:
        if dashboard is not None:
            dashboard.stop()
        if metrics_sink is not None:
            close = getattr(metrics_sink, "close", None)
            if callable(close):
                close()


def run_session_train_worker(
    config_path: str | Path,
    *,
    once: bool = False,
) -> int:
    cfg = _load_config(config_path)
    return run_session_train_worker_config(cfg, once=once)
