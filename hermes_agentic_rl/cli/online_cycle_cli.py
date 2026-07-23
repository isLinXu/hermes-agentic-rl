from __future__ import annotations

import asyncio
import copy
import json
import random
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.cli.session_eval_export_cli import (
    run_session_eval_export_config,
)
from hermes_agentic_rl.cli.session_replay_cli import run_session_replay_config
from hermes_agentic_rl.collectors.sidecar import close_all_sidecars
from hermes_agentic_rl.datasets.jsonl_loader import load_jsonl_dataset
from hermes_agentic_rl.framework import build_framework


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _compose_stage_config(
    root_cfg: dict[str, Any],
    stage_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = copy.deepcopy(root_cfg)
    merged.pop("online_cycle", None)
    for key, value in (stage_cfg or {}).items():
        merged[key] = copy.deepcopy(value)
    return merged


def _runtime_session_log_path(cfg: dict[str, Any]) -> str | None:
    runtime_cfg = cfg.get("runtime", {}) or {}
    sidecar_cfg = runtime_cfg.get("session_sidecar")
    if isinstance(sidecar_cfg, dict):
        session_log_path = sidecar_cfg.get("session_log_path")
        if session_log_path:
            return str(session_log_path)
    session_log_path = runtime_cfg.get("session_log_path")
    if session_log_path:
        return str(session_log_path)
    return None


def _runtime_replay_output_path(cfg: dict[str, Any]) -> str | None:
    runtime_cfg = cfg.get("runtime", {}) or {}
    sidecar_cfg = runtime_cfg.get("session_sidecar")
    if not isinstance(sidecar_cfg, dict):
        return None
    replay_output_path = sidecar_cfg.get("replay_output_path")
    if replay_output_path:
        return str(replay_output_path)
    return None


def _prepare_root_config(cfg: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(cfg)
    runtime_cfg = prepared.get("runtime", {}) or {}
    sidecar_cfg = runtime_cfg.get("session_sidecar")
    if isinstance(sidecar_cfg, dict):
        if sidecar_cfg.get("replay_output_path") and not sidecar_cfg.get("backend"):
            backend_cfg = prepared.get("backend")
            if isinstance(backend_cfg, dict) and backend_cfg:
                sidecar_cfg["backend"] = copy.deepcopy(backend_cfg)
        if not sidecar_cfg.get("judge") and isinstance(prepared.get("judge"), dict):
            sidecar_cfg["judge"] = copy.deepcopy(prepared["judge"])
    return prepared


def _prompt_from_item(item: dict[str, Any]) -> str:
    for key in ("instruction", "prompt", "task", "input"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(item.get("task_id", "")).strip()


def _load_rollout_items(
    cfg: dict[str, Any],
    *,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    cycle_cfg = cfg.get("online_cycle", {}) or {}
    inline_items = cycle_cfg.get("items")
    if isinstance(inline_items, list):
        items = [dict(item) for item in inline_items if isinstance(item, dict)]
    else:
        env_cfg = cfg.get("environment", {}) or {}
        dataset_path = env_cfg.get("dataset_path")
        if not dataset_path:
            raise ValueError(
                "online-cycle requires `environment.dataset_path` or `online_cycle.items`"
            )
        items = load_jsonl_dataset(dataset_path)

    effective_seed = seed if seed is not None else cycle_cfg.get("seed")
    if effective_seed is not None:
        random.Random(int(effective_seed)).shuffle(items)
    return items


def _select_cycle_items(
    items: list[dict[str, Any]],
    *,
    limit: int | None,
    cycle_index: int,
) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(items):
        return list(items)
    start = (cycle_index * limit) % len(items)
    end = start + limit
    if end <= len(items):
        return list(items[start:end])
    overflow = end - len(items)
    return list(items[start:]) + list(items[:overflow])


async def _collect_cycle(
    cfg: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    framework = build_framework(cfg, build_sidecar=True)
    rewards: list[float] = []
    task_ids: list[str] = []
    try:
        for item in items:
            prompt = _prompt_from_item(item)
            trajectory, summary = await framework.env.collect_and_judge(
                item,
                prompt=prompt,
            )
            del trajectory
            rewards.append(float(summary.final_score))
            task_ids.append(str(item.get("task_id", "")))
        if framework.session.session_sidecar is not None:
            framework.session.flush_sidecar()
    finally:
        framework.session.close_sidecar()
        close_all_sidecars()

    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    positive = sum(1 for reward in rewards if reward > 0)
    return {
        "items": len(items),
        "task_ids": task_ids,
        "mean_reward": mean_reward,
        "positive_rewards": positive,
        "nonzero_ratio": (positive / len(rewards)) if rewards else 0.0,
    }


def run_online_cycle_config(
    cfg: dict[str, Any],
    *,
    limit: int | None = None,
    seed: int | None = None,
    once: bool = False,
) -> int:
    prepared_cfg = _prepare_root_config(cfg)
    cycle_cfg = prepared_cfg.get("online_cycle", {}) or {}
    total_cycles = max(1, int(cycle_cfg.get("cycles", 1)))
    if once:
        total_cycles = 1

    rollout_items = _load_rollout_items(prepared_cfg, seed=seed)
    if not rollout_items:
        raise ValueError("online-cycle found no rollout items to process")

    limit_value = limit if limit is not None else cycle_cfg.get("limit")
    rollout_limit = int(limit_value) if limit_value is not None else None

    session_log_path = _runtime_session_log_path(prepared_cfg)
    raw_replay_output_path = _runtime_replay_output_path(prepared_cfg)
    if not session_log_path and not raw_replay_output_path:
        raise ValueError(
            "online-cycle requires runtime.session_log_path or "
            "runtime.session_sidecar.session_log_path/replay_output_path"
        )

    replay_stage_cfg = cycle_cfg.get("session_replay") or prepared_cfg.get("session_replay")
    worker_stage_cfg = cycle_cfg.get("session_train_worker") or prepared_cfg.get(
        "session_train_worker"
    )
    eval_stage_cfg = cycle_cfg.get("session_eval_export") or prepared_cfg.get("session_eval_export")

    total_items = 0
    total_positive = 0
    mean_rewards: list[float] = []
    worker_input_path = raw_replay_output_path

    for cycle_index in range(total_cycles):
        cycle_items = _select_cycle_items(
            rollout_items,
            limit=rollout_limit,
            cycle_index=cycle_index,
        )
        if not cycle_items:
            break

        collect_stats = asyncio.run(_collect_cycle(prepared_cfg, cycle_items))
        total_items += int(collect_stats["items"])
        total_positive += int(collect_stats["positive_rewards"])
        mean_rewards.append(float(collect_stats["mean_reward"]))

        if isinstance(replay_stage_cfg, dict):
            replay_cfg = _compose_stage_config(prepared_cfg, replay_stage_cfg)
            if not replay_cfg.get("input_path"):
                if not session_log_path:
                    raise ValueError(
                        "online-cycle session_replay stage requires a session log path"
                    )
                replay_cfg["input_path"] = session_log_path
            if not replay_cfg.get("output_path"):
                if raw_replay_output_path:
                    replay_cfg["output_path"] = raw_replay_output_path
                else:
                    raise ValueError(
                        "online-cycle session_replay stage requires `output_path` "
                        "when runtime.session_sidecar.replay_output_path is unset"
                    )
            exit_code = run_session_replay_config(replay_cfg)
            if exit_code != 0:
                return exit_code
            worker_input_path = str(replay_cfg["output_path"])

        if isinstance(worker_stage_cfg, dict):
            from hermes_agentic_rl.cli.session_train_worker_cli import (
                run_session_train_worker_config,
            )

            worker_cfg = _compose_stage_config(prepared_cfg, worker_stage_cfg)
            if not worker_cfg.get("input_path"):
                if not worker_input_path:
                    raise ValueError(
                        "online-cycle session_train_worker stage requires `input_path` "
                        "or runtime.session_sidecar.replay_output_path/session_replay output"
                    )
                worker_cfg["input_path"] = worker_input_path
            exit_code = run_session_train_worker_config(worker_cfg, once=True)
            if exit_code != 0:
                return exit_code

        if isinstance(eval_stage_cfg, dict):
            eval_cfg = _compose_stage_config(prepared_cfg, eval_stage_cfg)
            if not eval_cfg.get("input_path"):
                if not session_log_path:
                    raise ValueError(
                        "online-cycle session_eval_export stage requires a session log path"
                    )
                eval_cfg["input_path"] = session_log_path
            exit_code = run_session_eval_export_config(eval_cfg)
            if exit_code != 0:
                return exit_code

        worker_save_path = ""
        if isinstance(worker_stage_cfg, dict):
            worker_save_path = str(
                worker_stage_cfg.get("save_path") or worker_stage_cfg.get("head_path") or ""
            )
        eval_output_path = ""
        if isinstance(eval_stage_cfg, dict):
            eval_output_path = str(
                eval_stage_cfg.get("output_path") or eval_stage_cfg.get("output_dir") or ""
            )
        print(
            "[online-cycle] "
            f"cycle={cycle_index + 1}/{total_cycles} "
            f"items={collect_stats['items']} "
            f"mean_reward={collect_stats['mean_reward']:.4f} "
            f"replay_input={worker_input_path or ''} "
            f"policy={worker_save_path} "
            f"eval_output={eval_output_path}"
        )

    final_mean_reward = (sum(mean_rewards) / len(mean_rewards)) if mean_rewards else 0.0
    nonzero_ratio = (total_positive / total_items) if total_items else 0.0
    print(
        "[online-cycle] "
        f"completed cycles={len(mean_rewards)} "
        f"items={total_items} "
        f"mean_reward={final_mean_reward:.4f} "
        f"nonzero_ratio={nonzero_ratio:.4f}"
    )
    return 0


def run_online_cycle(
    config_path: str | Path,
    *,
    limit: int | None = None,
    seed: int | None = None,
    once: bool = False,
) -> int:
    cfg = _load_config(config_path)
    return run_online_cycle_config(
        cfg,
        limit=limit,
        seed=seed,
        once=once,
    )
