"""Unified benchmark runner.

Usage:
    python run_benchmarks.py [--warmup N] [--iterations N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

# Make benchmarks discoverable when run from project root or benchmarks dir
_bench_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_bench_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import benchmark modules
_bench_modules = {}

for _key, _mod_name in (
    ("algorithms", "bench_algos"),
    ("backend", "bench_backend"),
    ("rollout", "bench_rollout"),
    ("memory", "bench_memory"),
):
    try:
        _bench_modules[_key] = __import__(f"benchmarks.{_mod_name}", fromlist=["run_benchmark"])
    except Exception as exc:
        _bench_modules[_key] = None
        print(f"Warning: could not import {_mod_name}: {exc}")


def _print_table(title: str, rows: list[list[str]]) -> None:
    """Print a simple formatted table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    if not rows:
        print("  (no data)")
        return
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        line = "  ".join(
            str(cell).ljust(width + 2)
            for cell, width in zip(row, col_widths)
        )
        print(f"  {line}")
    print(f"{'=' * 60}")


def _format_algo_results(data: dict[str, Any]) -> list[list[str]]:
    """Format algorithm benchmark results into table rows."""
    rows = []
    rows.append(["Benchmark", "Samples", "Avg Time (ms)", "Samples/sec", "Mem Delta (MB)"])
    for name, bench in sorted(data.get("benchmarks", {}).items()):
        if bench.get("error"):
            rows.append([name, "—", "ERROR", "—", "—"])
            continue
        rows.append(
            [
                name,
                str(bench.get("samples", "—")),
                f"{bench.get('avg_time_ms', 0.0):.3f}",
                f"{bench.get('samples_per_sec', 0.0):.2f}",
                f"{bench.get('memory_delta_mb', 0.0):.3f}",
            ]
        )
    return rows


def _format_backend_results(data: dict[str, Any]) -> list[list[str]]:
    """Format backend benchmark results into table rows."""
    rows = []
    rows.append(["Benchmark", "Seq Length", "Batch Size", "Latency (ms)", "Throughput (tok/s)"])
    for name, bench in sorted(data.get("benchmarks", {}).items()):
        if bench.get("error"):
            rows.append([name, "—", "—", "ERROR", "—"])
            continue
        rows.append(
            [
                name,
                str(bench.get("seq_length", "—")),
                str(bench.get("batch_size", "—")),
                f"{bench.get('avg_latency_ms', 0.0):.3f}",
                f"{bench.get('throughput_tokens_per_sec', 0.0):.2f}",
            ]
        )
    return rows


def _format_rollout_results(data: dict[str, Any]) -> list[list[str]]:
    """Format rollout benchmark results into table rows."""
    rows = []
    rows.append(["Benchmark", "Group Size", "Group Time (ms)", "Rollouts/sec", "Per Rollout (ms)"])
    for name, bench in sorted(data.get("benchmarks", {}).items()):
        if bench.get("error"):
            rows.append([name, "—", "ERROR", "—", "—"])
            continue
        rows.append(
            [
                name,
                str(bench.get("group_size", "—")),
                f"{bench.get('avg_time_per_group_ms', 0.0):.3f}",
                f"{bench.get('rollouts_per_sec', 0.0):.2f}",
                f"{bench.get('time_per_rollout_ms', 0.0):.3f}",
            ]
        )
    return rows


def _format_memory_results(data: dict[str, Any]) -> list[list[str]]:
    """Format memory benchmark results into table rows."""
    rows = []
    rows.append(["Benchmark", "Batch Size", "Avg Time (ms)", "Peak Mem (MB)", "Mem Delta (MB)"])
    for name, bench in sorted(data.get("benchmarks", {}).items()):
        if bench.get("error"):
            rows.append([name, "—", "ERROR", "—", "—"])
            continue
        rows.append(
            [
                name,
                str(bench.get("batch_size", "—")),
                f"{bench.get('avg_time_ms', 0.0):.3f}",
                f"{bench.get('peak_memory_mb', 0.0):.3f}",
                f"{bench.get('memory_delta_mb', 0.0):.3f}",
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Agentic RL Benchmark Suite")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per benchmark")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Measurement iterations per benchmark"
    )
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument(
        "--skip", type=str, default="", help="Comma-separated list of benchmarks to skip"
    )
    args = parser.parse_args()

    skip_set = set(s.strip().lower() for s in args.skip.split(",") if s.strip())

    print("=" * 60)
    print("  Hermes Agentic RL Benchmark Suite")
    print(f"  Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")
    print("=" * 60)

    all_results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "python_version": sys.version,
        "results": {},
    }

    # Run each benchmark module
    for name, mod in _bench_modules.items():
        if name in skip_set:
            print(f"\n>> Skipping {name} benchmark (--skip)")
            continue
        if mod is None:
            print(f"\n>> Skipping {name} benchmark (import failed)")
            all_results["results"][name] = {"error": "import failed"}
            continue

        print(f"\n>> Running {name} benchmark ...")
        t0 = time.perf_counter()
        try:
            result = mod.run_benchmark(warmup=args.warmup, iterations=args.iterations)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback

            traceback.print_exc()
            result = {"error": str(exc)}
        t1 = time.perf_counter()
        result["elapsed_sec"] = t1 - t0
        all_results["results"][name] = result
        print(f"  Done in {t1 - t0:.2f}s")

    # Print formatted tables
    print("\n\n")
    print("=" * 60)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    for name in ["algorithms", "backend", "rollout", "memory"]:
        result = all_results["results"].get(name)
        if result is None:
            continue
        if name == "algorithms":
            rows = _format_algo_results(result)
            _print_table("Algorithm Compute-Loss Throughput", rows)
        elif name == "backend":
            rows = _format_backend_results(result)
            _print_table("Backend Latency & Throughput", rows)
        elif name == "rollout":
            rows = _format_rollout_results(result)
            _print_table("Rollout Collection Throughput", rows)
        elif name == "memory":
            rows = _format_memory_results(result)
            _print_table("Memory Usage During Forward Pass", rows)

    # Save JSON report
    output_path = args.output
    if output_path is None:
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(reports_dir, f"benchmark_{ts}.json")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  JSON report saved to: {output_path}")
    except Exception as exc:
        print(f"\n  ERROR saving JSON report: {exc}")

    total = sum(
        r.get("elapsed_sec", 0) for r in all_results["results"].values()
    )
    print(f"\n  Total benchmark time: {total:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
