"""RL checkpoint evaluation on held-out environment items.

This module is intentionally benchmark-shaped rather than trainer-shaped:
it compares one or more policies on the same selected samples, records every
rollout, aggregates structured reward metadata, and emits paired A/B deltas.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.cli.train_rl import (
    _backend_name,
    _build_backend,
    _build_env_and_rewards,
    _make_agent_loop_factory,
)
from hermes_agentic_rl.core.dataset import (
    select_items_by_group as _select_items_by_group,
)
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.trajectory import trajectory_to_dict
from hermes_agentic_rl.core.types import RewardSummary, Trajectory
from hermes_agentic_rl.eval.ab_test import paired_welch_t
from hermes_agentic_rl.eval.capability_axes import (
    build_capability_report,
    capability_report_markdown,
    normalize_capability_axes,
)
from hermes_agentic_rl.monitor.writers import MultiMetricsWriter, build_writer_from_config

STRUCTURED_METADATA_KEYS = (
    "exact_match",
    "similarity",
    "tool_call_present",
    "tool_call_parse_ok",
    "tool_name_match",
    "argument_key_overlap",
    "argument_value_similarity",
    "partial_tool_call_score",
    "partial_open_tag",
    "partial_close_tag",
    "partial_json_braces",
    "partial_name_key",
    "partial_arguments_key",
    "prediction_tool_call_count",
    "target_tool_call_count",
    "prediction_chars",
    "target_chars",
    "context_required_fact_recall",
    "context_constraint_satisfaction",
    "context_tool_summary_retention",
    "context_distractor_avoidance",
    "context_precision",
    "context_compression_ok",
    "context_response_chars",
    "context_prompt_chars",
    "context_noise_blocks",
    "context_required_facts",
    "context_forbidden_hits",
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def select_items_by_group(
    items: list[dict[str, Any]],
    *,
    split: str = "val",
    group_key: str = "source_trace_id",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select train/val/test/all items while keeping trace groups intact."""
    try:
        return _select_items_by_group(
            items,
            split=split,
            group_key=group_key,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    except ValueError as exc:
        if "split must be one of" in str(exc):
            raise ValueError("eval split must be one of: train, val, test, all") from exc
        raise


def split_items_by_source_trace_id(
    items: list[dict[str, Any]],
    *,
    split: str = "val",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    selected, _info = select_items_by_group(
        items,
        split=split,
        group_key="source_trace_id",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return selected


def _component_scores(summary: RewardSummary) -> dict[str, float]:
    return {component.name: float(component.score) for component in summary.components}


def _component_metadata(summary: RewardSummary) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for component in summary.components:
        if component.metadata:
            out[component.name] = dict(component.metadata)
    return out


def _extract_response_len(trajectory: Trajectory) -> int:
    runtime = trajectory.metadata.get("runtime")
    if isinstance(runtime, dict):
        rl = runtime.get("rl")
        if isinstance(rl, dict):
            response_ids = rl.get("response_ids")
            if isinstance(response_ids, list):
                return len(response_ids)
    return len(str(trajectory.final_output or ""))


def _metric_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(baseline) | set(candidate))
    return {key: float(candidate.get(key, 0.0) - baseline.get(key, 0.0)) for key in keys}


def _success_metric_score(
    *,
    reward: float,
    component_scores: dict[str, float],
    component_metadata: dict[str, dict[str, Any]],
    metric: str,
) -> tuple[float, bool]:
    """Return the per-rollout score used for success-rate thresholding."""
    metric = str(metric or "reward")
    if metric in {"reward", "final_score", "mean_reward"}:
        return reward, True
    if metric.startswith("component/"):
        key = metric.split("/", 1)[1]
        value = _as_float(component_scores.get(key))
        return (value if value is not None else 0.0), value is not None
    if metric.startswith("metadata/"):
        key = metric.split("/", 1)[1]
    elif metric.startswith("metadata_"):
        key = metric[len("metadata_") :]
    else:
        key = metric
    for metadata in component_metadata.values():
        value = _as_float(metadata.get(key))
        if value is not None:
            return value, True
    return 0.0, False


def _flatten_policy_metrics(
    rewards: list[float],
    successes: list[float],
    success_scores: list[float],
    success_metric_found: list[float],
    turns: list[float],
    response_lens: list[float],
    finished: list[float],
    component_values: dict[str, list[float]],
    metadata_values: dict[str, list[float]],
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_rollouts": float(len(rewards)),
        "mean_reward": _mean(rewards),
        "std_reward": _std(rewards),
        "success_rate": _mean(successes),
        "success_score_mean": _mean(success_scores),
        "success_metric_found_rate": _mean(success_metric_found),
        "mean_turns": _mean(turns),
        "mean_response_len": _mean(response_lens),
        "finished_naturally_rate": _mean(finished),
    }
    for key, values in component_values.items():
        metrics[f"component/{key}"] = _mean(values)
    for key, values in metadata_values.items():
        metrics[f"metadata/{key}"] = _mean(values)
    return metrics


def _leaderboard_markdown(policy_reports: list[dict[str, Any]]) -> str:
    columns = [
        ("name", "name"),
        ("n", "n_rollouts"),
        ("reward", "mean_reward"),
        ("std", "std_reward"),
        ("success", "success_rate"),
        ("finish", "finished_naturally_rate"),
        ("len", "mean_response_len"),
        ("parse", "metadata/tool_call_parse_ok"),
        ("name_match", "metadata/tool_name_match"),
        ("arg_keys", "metadata/argument_key_overlap"),
        ("arg_vals", "metadata/argument_value_similarity"),
        ("sim", "metadata/similarity"),
    ]
    lines = ["| " + " | ".join(label for label, _ in columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for report in policy_reports:
        metrics = report["metrics"]
        row: list[str] = []
        for label, key in columns:
            if key == "name":
                row.append(str(report["name"]))
                continue
            value = metrics.get(key, 0.0)
            if label == "n":
                row.append(str(int(value)))
            else:
                row.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _ranking_markdown(ranking: list[dict[str, Any]], *, metric: str) -> str:
    lines = [f"| rank | name | {metric} | mean_reward | success_rate | checkpoint |"]
    lines.append("|---|---|---|---|---|---|")
    for entry in ranking:
        checkpoint = entry.get("checkpoint_path") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entry["rank"]),
                    str(entry["name"]),
                    f"{float(entry['score']):.4f}",
                    f"{float(entry['mean_reward']):.4f}",
                    f"{float(entry['success_rate']):.4f}",
                    str(checkpoint),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _rank_policy_reports(
    policy_reports: list[dict[str, Any]],
    *,
    metric: str = "mean_reward",
) -> list[dict[str, Any]]:
    return sorted(
        policy_reports,
        key=lambda report: float(report["metrics"].get(metric, float("-inf"))),
        reverse=True,
    )


def _normalize_promotion_gate(
    eval_cfg: dict[str, Any],
    *,
    rank_metric: str,
) -> dict[str, Any]:
    raw = eval_cfg.get("promotion_gate")
    if raw is False:
        return {"enabled": False, "rank_metric": rank_metric}
    raw_dict = dict(raw) if isinstance(raw, dict) else {}
    max_p_value_raw = raw_dict.get("max_p_value")
    max_p_value = None
    if max_p_value_raw is not None and max_p_value_raw != "":
        max_p_value = float(max_p_value_raw)
    max_capability_regression_raw = raw_dict.get("max_capability_regression")
    max_capability_regression = None
    if max_capability_regression_raw is not None and max_capability_regression_raw != "":
        max_capability_regression = float(max_capability_regression_raw)
    raw_capability_thresholds = raw_dict.get("capability_thresholds")
    capability_thresholds = (
        {str(axis): float(threshold) for axis, threshold in raw_capability_thresholds.items()}
        if isinstance(raw_capability_thresholds, dict)
        else {}
    )
    return {
        "enabled": bool(raw_dict.get("enabled", True)),
        "candidate": str(raw_dict.get("candidate", "best_non_baseline")),
        "rank_metric": rank_metric,
        "fail_on_hold": bool(raw_dict.get("fail_on_hold", False)),
        "min_reward_delta": float(raw_dict.get("min_reward_delta", 0.0)),
        "min_success_rate_delta": float(raw_dict.get("min_success_rate_delta", 0.0)),
        "min_rank_metric_delta": float(raw_dict.get("min_rank_metric_delta", 0.0)),
        "require_any_improvement": bool(raw_dict.get("require_any_improvement", True)),
        "require_paired_winner": bool(raw_dict.get("require_paired_winner", False)),
        "max_p_value": max_p_value,
        "required_capability_axes": _string_list(raw_dict.get("required_capability_axes")),
        "min_capability_delta": float(
            raw_dict.get("min_capability_delta", raw_dict.get("min_axis_delta", 0.0))
        ),
        "capability_thresholds": capability_thresholds,
        "max_capability_regression": max_capability_regression,
    }


def _string_list(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _capability_delta_map(capability_report: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(capability_report, dict):
        return {}
    raw_deltas = capability_report.get("deltas")
    if not isinstance(raw_deltas, dict):
        return {}
    deltas: dict[str, float] = {}
    for axis, payload in raw_deltas.items():
        if not isinstance(payload, dict):
            continue
        delta = _as_float(payload.get("delta"))
        if delta is not None:
            deltas[str(axis)] = delta
    return deltas


def _capability_gate_checks(
    *,
    gate: dict[str, Any],
    capability_report: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    required_axes = _string_list(gate.get("required_capability_axes"))
    thresholds = (
        dict(gate.get("capability_thresholds", {}))
        if isinstance(gate.get("capability_thresholds"), dict)
        else {}
    )
    max_regression = gate.get("max_capability_regression")
    if not required_axes and not thresholds and max_regression is None:
        return checks

    deltas = _capability_delta_map(capability_report)
    if (required_axes or thresholds) and not deltas:
        checks["capability_report_present"] = {
            "required": True,
            "actual": False,
            "passed": False,
        }

    axes_to_check = list(dict.fromkeys(required_axes + [str(axis) for axis in thresholds]))
    default_delta = float(gate.get("min_capability_delta", 0.0))
    for axis in axes_to_check:
        required = float(thresholds.get(axis, default_delta))
        actual = deltas.get(axis)
        checks[f"capability_axis/{axis}/min_delta"] = {
            "required": required,
            "actual": actual,
            "passed": actual is not None and actual >= required,
        }

    if max_regression is not None:
        allowed_regression = float(max_regression)
        if not deltas:
            checks["capability_axis/max_regression"] = {
                "required": f">= {-allowed_regression}",
                "actual": None,
                "passed": False,
            }
        for axis, delta in sorted(deltas.items()):
            checks[f"capability_axis/{axis}/max_regression"] = {
                "required": f">= {-allowed_regression}",
                "actual": delta,
                "passed": delta >= -allowed_regression,
            }
    return checks


def _promotion_readout(
    *,
    policy_reports: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    rank_metric: str,
    eval_cfg: dict[str, Any],
    capability_report: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    gate = _normalize_promotion_gate(eval_cfg, rank_metric=rank_metric)
    if not gate["enabled"]:
        return {
            "enabled": False,
            "rank_metric": rank_metric,
            "recommendation": "disabled",
            "passed": False,
        }

    if not policy_reports:
        return None

    baseline = policy_reports[0]
    candidate_mode = str(gate.get("candidate", "best_non_baseline"))
    if candidate_mode != "best_non_baseline":
        raise RuntimeError(
            "eval_rl.promotion_gate.candidate currently supports only `best_non_baseline`"
        )

    best_non_baseline = next(
        iter(_rank_policy_reports(policy_reports[1:], metric=rank_metric)),
        None,
    )
    if best_non_baseline is None:
        return {
            "enabled": True,
            "gate": gate,
            "baseline": baseline["name"],
            "candidate": baseline["name"],
            "rank_metric": rank_metric,
            "recommendation": "baseline_only",
            "reason": "no candidate checkpoint was evaluated",
            "passed": False,
        }

    comparison = next(
        (item for item in comparisons if item["candidate"] == best_non_baseline["name"]),
        None,
    )
    metric_delta = dict(comparison["metric_delta"]) if comparison else {}
    reward_ab = dict(comparison["reward_ab"]) if comparison else {}
    reward_delta = float(metric_delta.get("mean_reward", 0.0))
    success_rate_delta = float(metric_delta.get("success_rate", 0.0))
    best_score = float(best_non_baseline["metrics"].get(rank_metric, 0.0))
    baseline_score = float(baseline["metrics"].get(rank_metric, 0.0))
    rank_metric_delta = best_score - baseline_score
    capability_deltas = _capability_delta_map(capability_report)
    any_improvement = (
        reward_delta > 0.0
        or success_rate_delta > 0.0
        or rank_metric_delta > 0.0
        or reward_ab.get("winner") == "candidate"
        or any(delta > 0.0 for delta in capability_deltas.values())
    )
    checks: dict[str, dict[str, Any]] = {
        "min_reward_delta": {
            "required": float(gate["min_reward_delta"]),
            "actual": reward_delta,
            "passed": reward_delta >= float(gate["min_reward_delta"]),
        },
        "min_success_rate_delta": {
            "required": float(gate["min_success_rate_delta"]),
            "actual": success_rate_delta,
            "passed": success_rate_delta >= float(gate["min_success_rate_delta"]),
        },
        "min_rank_metric_delta": {
            "required": float(gate["min_rank_metric_delta"]),
            "actual": rank_metric_delta,
            "passed": rank_metric_delta >= float(gate["min_rank_metric_delta"]),
        },
    }
    if bool(gate.get("require_any_improvement", True)):
        checks["require_any_improvement"] = {
            "required": True,
            "actual": any_improvement,
            "passed": any_improvement,
        }
    if bool(gate.get("require_paired_winner", False)):
        checks["require_paired_winner"] = {
            "required": "candidate",
            "actual": reward_ab.get("winner"),
            "passed": reward_ab.get("winner") == "candidate",
        }
    max_p_value = gate.get("max_p_value")
    if max_p_value is not None:
        approx_p = _as_float(reward_ab.get("approx_p"))
        checks["max_p_value"] = {
            "required": float(max_p_value),
            "actual": approx_p,
            "passed": approx_p is not None and approx_p <= float(max_p_value),
        }
    checks.update(
        _capability_gate_checks(
            gate=gate,
            capability_report=capability_report,
        )
    )

    failed_checks = [name for name, payload in checks.items() if not payload["passed"]]
    passed = best_non_baseline["name"] != baseline["name"] and not failed_checks

    return {
        "enabled": True,
        "gate": gate,
        "baseline": baseline["name"],
        "candidate": best_non_baseline["name"],
        "rank_metric": rank_metric,
        "baseline_score": baseline_score,
        "candidate_score": best_score,
        "rank_metric_delta": rank_metric_delta,
        "reward_delta": reward_delta,
        "success_rate_delta": success_rate_delta,
        "capability_deltas": capability_deltas,
        "reward_ab": reward_ab,
        "metric_delta": metric_delta,
        "checks": checks,
        "failed_checks": failed_checks,
        "passed": passed,
        "recommendation": "promote" if passed else "hold",
        "reasons": failed_checks,
    }


def _promotion_markdown(promotion_readout: dict[str, Any] | None) -> str:
    if not promotion_readout:
        return "# Promotion Readout\n\nNo promotion data available.\n"

    lines = ["# Promotion Readout", ""]
    recommendation = str(promotion_readout.get("recommendation", "unknown"))
    lines.append(f"- Recommendation: `{recommendation}`")
    lines.append(f"- Passed: `{promotion_readout.get('passed', False)}`")

    if not promotion_readout.get("enabled", True):
        lines.append("- Gate: disabled")
        return "\n".join(lines) + "\n"

    baseline = promotion_readout.get("baseline")
    candidate = promotion_readout.get("candidate")
    if baseline is not None:
        lines.append(f"- Baseline: `{baseline}`")
    if candidate is not None:
        lines.append(f"- Candidate: `{candidate}`")
    lines.append(f"- Rank metric: `{promotion_readout.get('rank_metric', 'mean_reward')}`")

    numeric_keys = (
        "baseline_score",
        "candidate_score",
        "rank_metric_delta",
        "reward_delta",
        "success_rate_delta",
    )
    for key in numeric_keys:
        value = _as_float(promotion_readout.get(key))
        if value is not None:
            lines.append(f"- {key}: `{value:+.4f}`")

    reward_ab = promotion_readout.get("reward_ab")
    if isinstance(reward_ab, dict) and reward_ab:
        winner = reward_ab.get("winner", "unknown")
        mean_diff = _as_float(reward_ab.get("mean_diff"))
        approx_p = _as_float(reward_ab.get("approx_p"))
        payload = f"- Paired A/B winner: `{winner}`"
        if mean_diff is not None:
            payload += f", mean_diff=`{mean_diff:+.4f}`"
        if approx_p is not None:
            payload += f", approx_p=`{approx_p:.4f}`"
        lines.append(payload)

    checks = promotion_readout.get("checks")
    if isinstance(checks, dict) and checks:
        lines.append("")
        lines.append("## Checks")
        lines.append("")
        lines.append("| name | required | actual | passed |")
        lines.append("|---|---|---|---|")
        for name, payload in checks.items():
            if not isinstance(payload, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(name),
                        str(payload.get("required")),
                        str(payload.get("actual")),
                        str(payload.get("passed")),
                    ]
                )
                + " |"
            )
    failed_checks = promotion_readout.get("failed_checks")
    if isinstance(failed_checks, list) and failed_checks:
        lines.append("")
        lines.append(f"- Failed checks: `{', '.join(str(item) for item in failed_checks)}`")

    return "\n".join(lines) + "\n"


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (base_dir / resolved).resolve()
    return resolved


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort checkpoints by iteration when available, then by name."""
    parts = [path.parent.name, path.stem, path.name]
    for part in parts:
        match = re.search(r"(?:iter|step|policy_iter)[_-]?(\d+)", part)
        if match:
            return (0, int(match.group(1)), str(path))
    return (1, 0, str(path))


def _discover_checkpoint_paths(checkpoint_dir: str | Path, *, base_dir: Path) -> list[Path]:
    resolved = _resolve_path(checkpoint_dir, base_dir=base_dir)
    if resolved.is_file():
        return [resolved]
    if not resolved.exists():
        raise RuntimeError(f"checkpoint_dir does not exist: {resolved}")
    candidates: dict[str, Path] = {}
    for pattern in ("**/model.pt", "**/policy.pt", "policy_*.pt", "**/policy_*.pt"):
        for path in resolved.glob(pattern):
            if path.is_file():
                candidates[str(path.resolve())] = path.resolve()
    ordered = sorted(candidates.values(), key=_checkpoint_sort_key)
    if not ordered:
        raise RuntimeError(f"no checkpoints found under {resolved}")
    return ordered


def _expand_policy_specs(
    policies: list[dict[str, Any]],
    *,
    base_dir: Path,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for policy_index, policy_spec in enumerate(policies):
        checkpoint_dir = policy_spec.get("checkpoint_dir")
        if checkpoint_dir:
            discovered = _discover_checkpoint_paths(checkpoint_dir, base_dir=base_dir)
            max_checkpoints = policy_spec.get("max_checkpoints")
            if max_checkpoints not in {None, ""}:
                discovered = discovered[: max(1, int(str(max_checkpoints)))]
            prefix = str(policy_spec.get("name") or "checkpoint")
            for path in discovered:
                suffix = path.parent.name if path.name in {"model.pt", "policy.pt"} else path.stem
                expanded.append(
                    {
                        **policy_spec,
                        "name": f"{prefix}:{suffix}" if prefix else suffix,
                        "checkpoint_path": str(path),
                        "policy_index": policy_index,
                    }
                )
            continue
        expanded.append({**policy_spec, "policy_index": policy_index})
    return expanded


def _load_checkpoint_into_backend(
    backend: Any, checkpoint_path: str | Path, *, base_dir: Path
) -> Path:
    if not hasattr(backend, "model"):
        raise RuntimeError("checkpoint loading requires backend.model")
    path = _resolve_path(checkpoint_path, base_dir=base_dir)
    if path.is_dir():
        if (path / "model.pt").exists():
            path = path / "model.pt"
        elif (path / "policy.pt").exists():
            path = path / "policy.pt"
        else:
            raise RuntimeError(f"checkpoint directory has no model.pt or policy.pt: {path}")
    if not path.exists():
        raise RuntimeError(f"checkpoint path does not exist: {path}")

    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model_state"), dict):
        state = state["model_state"]
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint did not contain a state_dict: {path}")
    backend.model.load_state_dict(state)  # type: ignore[attr-defined]
    return path


def _cfg_for_eval_loop(cfg: dict[str, Any], eval_cfg: dict[str, Any]) -> dict[str, Any]:
    loop_cfg = copy.deepcopy(cfg)
    train_block = dict(loop_cfg.get("train_rl") or {})
    if "temperature" in eval_cfg:
        train_block["temperature"] = float(eval_cfg["temperature"])
    if "max_new_tokens" in eval_cfg:
        train_block["max_new_tokens"] = int(eval_cfg["max_new_tokens"])
    loop_cfg["train_rl"] = train_block
    return loop_cfg


async def _run_policy_eval(
    *,
    policy_name: str,
    policy_index: int,
    backend: Any,
    env: Any,
    reward_manager: Any,
    items: list[dict[str, Any]],
    eval_cfg: dict[str, Any],
    loop_factory: Any,
    rollouts_path: Path,
    metrics_writer: MultiMetricsWriter | None,
) -> dict[str, Any]:
    success_threshold = float(eval_cfg.get("success_threshold", 0.5))
    success_metric = str(eval_cfg.get("success_metric", "reward"))
    seed_base = eval_cfg.get("seed", 0)
    include_text = bool(eval_cfg.get("include_rollout_text", True))
    include_trajectories = bool(eval_cfg.get("include_trajectories", False))

    rewards: list[float] = []
    successes: list[float] = []
    success_scores: list[float] = []
    success_metric_found: list[float] = []
    turns: list[float] = []
    response_lens: list[float] = []
    finished: list[float] = []
    component_values: dict[str, list[float]] = defaultdict(list)
    metadata_values: dict[str, list[float]] = defaultdict(list)
    step_stride = max(1, len(items)) + 1

    await env.setup()
    for sample_index, item in enumerate(items):
        seed = (int(seed_base) + sample_index) if seed_base is not None else None
        instruction = env.format_prompt(item)
        loop = loop_factory(backend=backend, seed=seed)
        trajectory = await RolloutManager(loop).collect(item, instruction)
        summary = await reward_manager.evaluate(item, trajectory, tool_context=None)

        reward = float(summary.final_score)
        response_len = float(_extract_response_len(trajectory))
        comp_scores = _component_scores(summary)
        comp_metadata = _component_metadata(summary)
        success_score, metric_found = _success_metric_score(
            reward=reward,
            component_scores=comp_scores,
            component_metadata=comp_metadata,
            metric=success_metric,
        )
        success = 1.0 if success_score >= success_threshold else 0.0
        rewards.append(reward)
        successes.append(success)
        success_scores.append(success_score)
        success_metric_found.append(1.0 if metric_found else 0.0)
        turns.append(float(trajectory.turns_used))
        response_lens.append(response_len)
        finished.append(1.0 if trajectory.finished_naturally else 0.0)

        for key, value in comp_scores.items():
            component_values[key].append(float(value))
        for metadata in comp_metadata.values():
            for key in STRUCTURED_METADATA_KEYS:
                metadata_value = _as_float(metadata.get(key))
                if metadata_value is not None:
                    metadata_values[key].append(metadata_value)

        record: dict[str, Any] = {
            "policy": policy_name,
            "policy_index": policy_index,
            "sample_index": sample_index,
            "task_id": item.get("task_id"),
            "source_trace_id": item.get("source_trace_id"),
            "assistant_turn_index": item.get("assistant_turn_index"),
            "reward": reward,
            "success_metric": success_metric,
            "success_score": success_score,
            "success": bool(success),
            "finished_naturally": trajectory.finished_naturally,
            "turns_used": trajectory.turns_used,
            "response_len": response_len,
            "component_scores": comp_scores,
            "component_metadata": comp_metadata,
        }
        if include_text:
            record["prediction"] = trajectory.final_output or ""
            record["target"] = item.get("target_response_full") or item.get("target_response")
        if include_trajectories:
            record["trajectory"] = trajectory_to_dict(trajectory)
        _jsonl_append(rollouts_path, record)

        if metrics_writer is not None:
            metrics_record: dict[str, Any] = {
                "iter": policy_index * step_stride + sample_index,
                "policy_index": policy_index,
                "sample_index": sample_index,
                "reward": reward,
                "success_score": success_score,
                "success_metric_found": 1.0 if metric_found else 0.0,
                "success": success,
                "finished_naturally": 1.0 if trajectory.finished_naturally else 0.0,
                "response_len": response_len,
                "turns_used": float(trajectory.turns_used),
                "policy": {policy_name: {"reward": reward, "success": success}},
            }
            for key, values in metadata_values.items():
                if values:
                    metrics_record[f"metadata_{key}"] = values[-1]
            metrics_writer(metrics_record)

    metrics = _flatten_policy_metrics(
        rewards,
        successes,
        success_scores,
        success_metric_found,
        turns,
        response_lens,
        finished,
        component_values,
        metadata_values,
    )
    return {
        "name": policy_name,
        "policy_index": policy_index,
        "n_rollouts": len(rewards),
        "metrics": metrics,
        "rewards": rewards,
    }


def _prepare_eval_items(
    env: Any,
    eval_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_rollouts = int(eval_cfg.get("n_rollouts", 32))
    split = str(eval_cfg.get("split", "val"))
    split_by = str(eval_cfg.get("split_by", "source_trace_id"))
    val_ratio = float(eval_cfg.get("val_ratio", eval_cfg.get("holdout_ratio", 0.1)))
    test_ratio = float(eval_cfg.get("test_ratio", 0.1))
    seed = int(eval_cfg.get("seed", 0))

    all_items = list(getattr(env, "items", []) or [])
    if not all_items:
        raise RuntimeError(
            "eval-rl currently requires an environment exposing an `items` list; "
            "HermesReasoningTraceEnv does this for local parquet/JSONL evaluation."
        )

    selected, split_info = select_items_by_group(
        all_items,
        split=split,
        group_key=split_by,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if n_rollouts > 0:
        selected = selected[:n_rollouts]
    split_info["requested_rollouts"] = n_rollouts
    split_info["selected_items_after_limit"] = len(selected)
    if not selected:
        raise RuntimeError(
            "eval split selected zero items; increase dataset_limit or "
            "adjust val_ratio/test_ratio/split"
        )
    return selected, split_info


def run_eval_rl(
    config_path: str,
    output_dir: str | None = None,
    *,
    force_fail_on_hold: bool = False,
    command_name: str = "eval-rl",
) -> int:
    base_dir = Path.cwd()
    cfg = _load_yaml(config_path)
    eval_cfg = dict(cfg.get("eval_rl") or {})
    if force_fail_on_hold:
        promotion_gate = dict(eval_cfg.get("promotion_gate") or {})
        promotion_gate["fail_on_hold"] = True
        eval_cfg["promotion_gate"] = promotion_gate
    out_dir = _resolve_path(
        output_dir or eval_cfg.get("output_dir") or "outputs/eval_rl",
        base_dir=base_dir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_for_env = copy.deepcopy(cfg)
    env_override = dict(cfg_for_env.get("environment") or {})
    if eval_cfg.get("dataset_limit") not in {None, ""}:
        env_override["dataset_limit"] = int(eval_cfg["dataset_limit"])
    if eval_cfg.get("shuffle") is not None:
        env_override["shuffle"] = bool(eval_cfg["shuffle"])
    cfg_for_env["environment"] = env_override

    env, reward_manager = _build_env_and_rewards(cfg_for_env)
    selected_items, split_info = _prepare_eval_items(env, eval_cfg)
    loop_factory = _make_agent_loop_factory(_cfg_for_eval_loop(cfg, eval_cfg))

    policies = eval_cfg.get("policies") or [{"name": "baseline"}]
    if not isinstance(policies, list) or not policies:
        raise RuntimeError("eval_rl.policies must be a non-empty list")
    policies = _expand_policy_specs(policies, base_dir=base_dir)

    backend_name = _backend_name(cfg)
    env_type = str((cfg.get("environment") or {}).get("type", "unknown"))
    metrics_writer = build_writer_from_config(
        cfg.get("metrics"),
        output_dir=out_dir,
        wandb_context={
            "name": str(eval_cfg.get("name") or out_dir.name),
            "job_type": command_name,
            "tags": [command_name, backend_name, env_type],
            "command": command_name,
            "config_path": str(config_path),
            "output_dir": str(out_dir),
            "config": cfg,
        },
    )

    rollouts_path = out_dir / "eval_rollouts.jsonl"
    if rollouts_path.exists():
        rollouts_path.unlink()

    policy_reports: list[dict[str, Any]] = []
    try:
        for policy_index, policy_spec in enumerate(policies):
            if not isinstance(policy_spec, dict):
                raise RuntimeError("each eval_rl.policies item must be a mapping")
            name = str(policy_spec.get("name") or f"policy_{policy_index}")
            backend = _build_backend(
                cfg, need_value_head=bool(policy_spec.get("need_value_head", False))
            )
            checkpoint_path = policy_spec.get("checkpoint_path")
            loaded_checkpoint: str | None = None
            if checkpoint_path:
                loaded_checkpoint = str(
                    _load_checkpoint_into_backend(
                        backend,
                        checkpoint_path,
                        base_dir=base_dir,
                    )
                )
            print(
                f"[{command_name}] policy={name} backend={backend_name} "
                f"device={backend.device} checkpoint={loaded_checkpoint or '<fresh baseline>'}"
            )
            report = asyncio.run(
                _run_policy_eval(
                    policy_name=name,
                    policy_index=policy_index,
                    backend=backend,
                    env=env,
                    reward_manager=reward_manager,
                    items=selected_items,
                    eval_cfg=eval_cfg,
                    loop_factory=loop_factory,
                    rollouts_path=rollouts_path,
                    metrics_writer=metrics_writer,
                )
            )
            report["checkpoint_path"] = loaded_checkpoint
            policy_reports.append(report)
            if metrics_writer is not None:
                metrics_writer(
                    {
                        "iter": policy_index * (max(1, len(selected_items)) + 1)
                        + len(selected_items),
                        "policy_index": policy_index,
                        "summary": {name: report["metrics"]},
                    }
                )

        comparisons: list[dict[str, Any]] = []
        baseline = policy_reports[0]
        baseline_metrics = baseline["metrics"]
        for candidate in policy_reports[1:]:
            n = min(len(baseline["rewards"]), len(candidate["rewards"]))
            ab = paired_welch_t(baseline["rewards"][:n], candidate["rewards"][:n])
            comparisons.append(
                {
                    "baseline": baseline["name"],
                    "candidate": candidate["name"],
                    "n": n,
                    "reward_ab": asdict(ab),
                    "metric_delta": _metric_delta(baseline_metrics, candidate["metrics"]),
                }
            )

        rank_metric = str(eval_cfg.get("rank_metric", "mean_reward"))
        ranking = _rank_policy_reports(policy_reports, metric=rank_metric)
        best_policy = ranking[0] if ranking else None
        capability_axes = normalize_capability_axes(eval_cfg.get("capability_axes"))
        capability_report = build_capability_report(
            policy_reports,
            axes=capability_axes,
        )
        promotion_readout = _promotion_readout(
            policy_reports=policy_reports,
            comparisons=comparisons,
            rank_metric=rank_metric,
            eval_cfg=eval_cfg,
            capability_report=capability_report,
        )
        ranking_summary = [
            {
                "rank": index + 1,
                "name": report["name"],
                "checkpoint_path": report.get("checkpoint_path"),
                "score": float(report["metrics"].get(rank_metric, 0.0)),
                "mean_reward": float(report["metrics"].get("mean_reward", 0.0)),
                "success_rate": float(report["metrics"].get("success_rate", 0.0)),
            }
            for index, report in enumerate(ranking)
        ]
        summary = {
            "command": command_name,
            "config_path": str(config_path),
            "output_dir": str(out_dir),
            "split": split_info,
            "rank_metric": rank_metric,
            "success_metric": str(eval_cfg.get("success_metric", "reward")),
            "best_policy": (
                {
                    "name": best_policy["name"],
                    "checkpoint_path": best_policy.get("checkpoint_path"),
                    "score": float(best_policy["metrics"].get(rank_metric, 0.0)),
                    "metrics": best_policy["metrics"],
                }
                if best_policy is not None
                else None
            ),
            "ranking": ranking_summary,
            "success_threshold": float(eval_cfg.get("success_threshold", 0.5)),
            "promotion_readout": promotion_readout,
            "capability_report": capability_report,
            "policies": policy_reports,
            "comparisons": comparisons,
            "artifacts": {
                "summary": str(out_dir / "eval_summary.json"),
                "rollouts": str(rollouts_path),
                "leaderboard": str(out_dir / "leaderboard.md"),
                "ranking": str(out_dir / "ranking.md"),
                "promotion": str(out_dir / "promotion.md"),
                "capability_report": str(out_dir / "capability_report.md"),
            },
        }
        _json_dump(out_dir / "eval_summary.json", summary)
        (out_dir / "leaderboard.md").write_text(
            _leaderboard_markdown(policy_reports),
            encoding="utf-8",
        )
        (out_dir / "ranking.md").write_text(
            _ranking_markdown(ranking_summary, metric=rank_metric),
            encoding="utf-8",
        )
        (out_dir / "promotion.md").write_text(
            _promotion_markdown(promotion_readout),
            encoding="utf-8",
        )
        (out_dir / "capability_report.md").write_text(
            capability_report_markdown(capability_report),
            encoding="utf-8",
        )
        if isinstance(metrics_writer, MultiMetricsWriter):
            metrics_writer.update_summary(
                {
                    "command": command_name,
                    "split": split_info,
                    "success_metric": str(eval_cfg.get("success_metric", "reward")),
                    "success_threshold": float(eval_cfg.get("success_threshold", 0.5)),
                    "promotion_readout": promotion_readout,
                    "capability_report": capability_report,
                    "policy_metrics": {
                        report["name"]: report["metrics"] for report in policy_reports
                    },
                    "comparison_metrics": {
                        comparison["candidate"]: comparison for comparison in comparisons
                    },
                    "best_policy": summary["best_policy"],
                    "artifacts": summary["artifacts"],
                }
            )

        print(f"[{command_name}] summary saved to {out_dir / 'eval_summary.json'}")
        print(f"[{command_name}] rollouts saved to {rollouts_path}")
        print(_leaderboard_markdown(policy_reports), end="")
        if (
            isinstance(promotion_readout, dict)
            and promotion_readout.get("enabled", True)
            and promotion_readout.get("gate", {}).get("fail_on_hold", False)
            and promotion_readout.get("recommendation") != "promote"
        ):
            print(
                f"[{command_name}] promotion gate requested failure: "
                f"recommendation={promotion_readout.get('recommendation')}"
            )
            return 3
        return 0
    finally:
        if isinstance(metrics_writer, MultiMetricsWriter):
            metrics_writer.close()


def run_eval_gate(config_path: str, output_dir: str | None = None) -> int:
    return run_eval_rl(
        config_path,
        output_dir=output_dir,
        force_fail_on_hold=True,
        command_name="eval-gate",
    )
