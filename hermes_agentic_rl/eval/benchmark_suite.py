"""Benchmark suite orchestration and scorecard generation.

The suite layer intentionally sits above ``eval-rl``. Each benchmark still uses
the regular held-out evaluator, while this module standardizes multi-benchmark
execution, threshold checks, and scorecard artifacts.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from hermes_agentic_rl.eval.rl_eval import run_eval_rl

BENCHMARK_SUITE_SCHEMA_VERSION = 1


EvalRunner = Callable[[str, str | None], int]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("benchmark suite config must be a YAML mapping")
    return payload


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (base_dir / resolved).resolve()
    return resolved


def _resolve_config_path(path: str | Path, *, config_dir: Path, cwd: Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    suite_relative = (config_dir / resolved).resolve()
    if suite_relative.exists():
        return suite_relative
    return (cwd / resolved).resolve()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned or "benchmark"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    value = _as_float(metrics.get(metric))
    if value is not None:
        return value
    if metric.startswith("metadata_"):
        return _as_float(metrics.get(f"metadata/{metric[len('metadata_') :]}"))
    if metric.startswith("component_"):
        return _as_float(metrics.get(f"component/{metric[len('component_') :]}"))
    return None


def _thresholds(spec: dict[str, Any]) -> dict[str, Any]:
    raw = spec.get("thresholds")
    if isinstance(raw, dict):
        thresholds = dict(raw)
    else:
        thresholds = {}
    for key in (
        "min_score",
        "min_mean_reward",
        "min_success_rate",
        "min_reward_delta",
        "min_success_rate_delta",
        "min_rank_metric_delta",
    ):
        if key in spec and key not in thresholds:
            thresholds[key] = spec[key]
    return thresholds


def _check_min(
    checks: dict[str, dict[str, Any]],
    *,
    name: str,
    actual: float | None,
    required: Any,
) -> None:
    required_float = float(required)
    checks[name] = {
        "required": required_float,
        "actual": actual,
        "passed": actual is not None and actual >= required_float,
    }


def _summary_path_from_output(output_dir: Path) -> Path:
    return output_dir / "eval_summary.json"


def _load_eval_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def build_benchmark_readout(
    *,
    spec: dict[str, Any],
    output_dir: Path,
    exit_code: int,
    summary: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a normalized per-benchmark readout from an ``eval-rl`` summary."""

    name = str(spec.get("name") or output_dir.name)
    required = bool(spec.get("required", True))
    weight = float(spec.get("weight", 1.0))
    checks: dict[str, dict[str, Any]] = {
        "eval_completed": {
            "required": True,
            "actual": summary is not None,
            "passed": summary is not None and exit_code in {0, 3},
        }
    }

    rank_metric = str(summary.get("rank_metric", "mean_reward")) if summary else "mean_reward"
    raw_best_policy = summary.get("best_policy") if summary else None
    best_policy: dict[str, Any] = raw_best_policy if isinstance(raw_best_policy, dict) else {}
    raw_best_metrics = best_policy.get("metrics")
    best_metrics: dict[str, Any] = raw_best_metrics if isinstance(raw_best_metrics, dict) else {}
    score_metric = str(spec.get("score_metric") or rank_metric)
    best_score = _metric_value(best_metrics, score_metric)
    if best_score is None:
        best_score = _as_float(best_policy.get("score"))

    raw_promotion = summary.get("promotion_readout") if summary else None
    promotion: dict[str, Any] = raw_promotion if isinstance(raw_promotion, dict) else {}
    promotion_recommendation = str(promotion.get("recommendation", "unknown"))

    thresholds = _thresholds(spec)
    if "min_score" in thresholds:
        _check_min(checks, name="min_score", actual=best_score, required=thresholds["min_score"])
    if "min_mean_reward" in thresholds:
        _check_min(
            checks,
            name="min_mean_reward",
            actual=_metric_value(best_metrics, "mean_reward"),
            required=thresholds["min_mean_reward"],
        )
    if "min_success_rate" in thresholds:
        _check_min(
            checks,
            name="min_success_rate",
            actual=_metric_value(best_metrics, "success_rate"),
            required=thresholds["min_success_rate"],
        )
    if "min_reward_delta" in thresholds:
        _check_min(
            checks,
            name="min_reward_delta",
            actual=_as_float(promotion.get("reward_delta")),
            required=thresholds["min_reward_delta"],
        )
    if "min_success_rate_delta" in thresholds:
        _check_min(
            checks,
            name="min_success_rate_delta",
            actual=_as_float(promotion.get("success_rate_delta")),
            required=thresholds["min_success_rate_delta"],
        )
    if "min_rank_metric_delta" in thresholds:
        _check_min(
            checks,
            name="min_rank_metric_delta",
            actual=_as_float(promotion.get("rank_metric_delta")),
            required=thresholds["min_rank_metric_delta"],
        )
    if bool(spec.get("require_promotion", False)):
        checks["require_promotion"] = {
            "required": "promote",
            "actual": promotion_recommendation,
            "passed": promotion_recommendation == "promote",
        }

    failed_checks = [key for key, payload in checks.items() if not bool(payload["passed"])]
    status = "passed" if not failed_checks else "failed"
    tags = spec.get("tags")
    tags_list = [str(item) for item in tags] if isinstance(tags, list) else []
    summary_path = _summary_path_from_output(output_dir)

    return {
        "name": name,
        "status": status,
        "required": required,
        "weight": weight,
        "tags": tags_list,
        "config_path": str(spec.get("config_path", "")),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "exit_code": exit_code,
        "error": error,
        "rank_metric": rank_metric,
        "score_metric": score_metric,
        "score": best_score,
        "best_policy": {
            "name": best_policy.get("name"),
            "checkpoint_path": best_policy.get("checkpoint_path"),
            "metrics": best_metrics,
        },
        "promotion": {
            "recommendation": promotion_recommendation,
            "passed": promotion.get("passed"),
            "baseline": promotion.get("baseline"),
            "candidate": promotion.get("candidate"),
            "reward_delta": promotion.get("reward_delta"),
            "success_rate_delta": promotion.get("success_rate_delta"),
            "rank_metric_delta": promotion.get("rank_metric_delta"),
            "capability_deltas": promotion.get("capability_deltas", {}),
        },
        "checks": checks,
        "failed_checks": failed_checks,
    }


def build_scorecard(
    *,
    config_path: str | Path,
    output_dir: Path,
    benchmark_readouts: list[dict[str, Any]],
    suite_cfg: dict[str, Any],
) -> dict[str, Any]:
    required = [item for item in benchmark_readouts if bool(item.get("required", True))]
    required_passed = [
        item for item in required if str(item.get("status")) == "passed"
    ]
    scored = [
        item
        for item in benchmark_readouts
        if _as_float(item.get("score")) is not None and float(item.get("weight", 1.0)) > 0
    ]
    total_weight = sum(float(item.get("weight", 1.0)) for item in scored)
    weighted_score = (
        sum(float(item["score"]) * float(item.get("weight", 1.0)) for item in scored)
        / total_weight
        if total_weight
        else 0.0
    )
    status_counts = Counter(str(item.get("status", "unknown")) for item in benchmark_readouts)
    suite_passed = len(required_passed) == len(required)
    return {
        "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "suite_name": str(suite_cfg.get("name") or output_dir.name),
        "passed": suite_passed,
        "benchmarks_total": len(benchmark_readouts),
        "required_total": len(required),
        "required_passed": len(required_passed),
        "required_pass_rate": (len(required_passed) / len(required)) if required else 1.0,
        "weighted_score": weighted_score,
        "status_counts": dict(sorted(status_counts.items())),
        "benchmarks": benchmark_readouts,
        "artifacts": {
            "scorecard": str(output_dir / "scorecard.json"),
            "markdown": str(output_dir / "scorecard.md"),
        },
    }


def scorecard_markdown(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Scorecard",
        "",
        f"- Suite: `{scorecard.get('suite_name', '')}`",
        f"- Passed: `{scorecard.get('passed', False)}`",
        f"- Required pass rate: `{float(scorecard.get('required_pass_rate', 0.0)):.4f}`",
        f"- Weighted score: `{float(scorecard.get('weighted_score', 0.0)):.4f}`",
        "",
        "## Benchmarks",
        "",
        "| name | status | required | score | best_policy | promotion | output |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in scorecard.get("benchmarks", []):
        if not isinstance(item, dict):
            continue
        score = _as_float(item.get("score"))
        raw_best_policy = item.get("best_policy")
        best_policy: dict[str, Any] = (
            raw_best_policy if isinstance(raw_best_policy, dict) else {}
        )
        raw_promotion = item.get("promotion")
        promotion: dict[str, Any] = raw_promotion if isinstance(raw_promotion, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("name", "")),
                    str(item.get("status", "")),
                    str(item.get("required", True)),
                    f"{score:.4f}" if score is not None else "",
                    str(best_policy.get("name") or ""),
                    str(promotion.get("recommendation") or ""),
                    str(item.get("output_dir", "")),
                ]
            )
            + " |"
        )

    failed = [
        item for item in scorecard.get("benchmarks", [])
        if isinstance(item, dict) and item.get("failed_checks")
    ]
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        lines.append("| benchmark | failed_checks |")
        lines.append("|---|---|")
        for item in failed:
            failed_checks = item.get("failed_checks")
            values = ", ".join(str(check) for check in failed_checks) if isinstance(failed_checks, list) else ""
            lines.append(f"| {item.get('name', '')} | {values} |")
    return "\n".join(lines) + "\n"


def _suite_config(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("benchmark_suite", payload)
    if not isinstance(raw, dict):
        raise ValueError("benchmark_suite must be a YAML mapping")
    return dict(raw)


def _benchmark_output_dir(
    *,
    spec: dict[str, Any],
    suite_output_dir: Path,
    name: str,
) -> Path:
    raw = spec.get("output_dir")
    if raw:
        return _resolve_path(str(raw), base_dir=Path.cwd())
    return suite_output_dir / _slugify(name)


def run_benchmark_suite(
    config_path: str,
    output_dir: str | None = None,
    *,
    eval_runner: EvalRunner | None = None,
) -> int:
    config_file = Path(config_path).resolve()
    cfg = _load_yaml(config_file)
    suite_cfg = _suite_config(cfg)
    suite_out = _resolve_path(
        output_dir or suite_cfg.get("output_dir") or "outputs/benchmark_suite",
        base_dir=Path.cwd(),
    )
    suite_out.mkdir(parents=True, exist_ok=True)
    raw_benchmarks = suite_cfg.get("benchmarks")
    if not isinstance(raw_benchmarks, list) or not raw_benchmarks:
        raise ValueError("benchmark_suite.benchmarks must be a non-empty list")

    runner = eval_runner or run_eval_rl
    readouts: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(raw_benchmarks):
        if not isinstance(raw_spec, dict):
            raise ValueError("each benchmark_suite.benchmarks item must be a mapping")
        spec = dict(raw_spec)
        config_raw = spec.get("config_path")
        if not config_raw:
            raise ValueError("each benchmark requires config_path")
        name = str(spec.get("name") or Path(str(config_raw)).stem or f"benchmark_{index}")
        spec["name"] = name
        child_config = _resolve_config_path(
            str(config_raw),
            config_dir=config_file.parent,
            cwd=Path.cwd(),
        )
        spec["config_path"] = str(child_config)
        bench_out = _benchmark_output_dir(spec=spec, suite_output_dir=suite_out, name=name)
        print(f"[benchmark-suite] running {name}: {child_config}")
        exit_code = 1
        error: str | None = None
        try:
            exit_code = runner(str(child_config), str(bench_out))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[benchmark-suite] {name} failed before scorecard aggregation: {error}")
        try:
            summary = _load_eval_summary(_summary_path_from_output(bench_out))
        except Exception as exc:
            summary = None
            summary_error = f"{type(exc).__name__}: {exc}"
            error = f"{error}; summary={summary_error}" if error else f"summary={summary_error}"
        readout = build_benchmark_readout(
            spec=spec,
            output_dir=bench_out,
            exit_code=exit_code,
            summary=summary,
            error=error,
        )
        readouts.append(readout)

    scorecard = build_scorecard(
        config_path=config_file,
        output_dir=suite_out,
        benchmark_readouts=readouts,
        suite_cfg=suite_cfg,
    )
    _json_dump(suite_out / "scorecard.json", scorecard)
    (suite_out / "scorecard.md").write_text(scorecard_markdown(scorecard), encoding="utf-8")
    print(f"[benchmark-suite] scorecard saved to {suite_out / 'scorecard.json'}")
    print(scorecard_markdown(scorecard), end="")
    fail_on_required_failure = bool(suite_cfg.get("fail_on_required_failure", True))
    return 0 if scorecard["passed"] or not fail_on_required_failure else 3
