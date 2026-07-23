"""Benchmark rollout collection throughput."""

from __future__ import annotations

import gc
import time
import traceback
from typing import Any

_benchmark_results: dict[str, Any] = {}


def _simulate_collect_group(
    backend: Any, group_size: int, max_new_tokens: int, prompt_len: int
) -> list[Any]:
    """Simulate the core of _collect_group: generate a group of rollouts."""
    tokenizer = backend.tokenizer
    prompt_text = "a" * (prompt_len * 2)
    prompt_ids = tokenizer.encode(prompt_text)[:prompt_len]
    records = []
    for _ in range(group_size):
        try:
            gen = backend.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                seed=None,
            )
        except Exception:
            traceback.print_exc()
            continue
        # Build a minimal RolloutRecord
        from hermes_agentic_rl.algos.base import RolloutRecord

        records.append(
            RolloutRecord(
                prompt_ids=list(prompt_ids),
                response_ids=list(gen.response_ids),
                old_logprobs=list(gen.logprobs),
                reward=0.0,
                group_id="benchmark_group",
                metadata={},
            )
        )
    return records


def run_benchmark(warmup: int = 1, iterations: int = 10) -> dict[str, Any]:
    """Run rollout collection benchmarks and return structured results."""
    results: dict[str, Any] = {"module": "bench_rollout", "benchmarks": {}}
    try:
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
    except Exception as exc:
        results["error"] = f"Import failed: {exc}"
        return results

    device = "cpu"
    dtype = "float32"
    cfg = TinyBackendConfig(
        dim=32,
        n_heads=4,
        n_layers=2,
        max_len=512,
        device=device,
        dtype=dtype,
        seed=42,
    )
    backend = TinyCausalLMBackend(cfg)

    group_sizes = [1, 4, 8]
    prompt_len = 16
    max_new_tokens = 24

    for gs in group_sizes:
        # Warmup
        for _ in range(warmup):
            try:
                _simulate_collect_group(backend, gs, max_new_tokens, prompt_len)
            except Exception:
                traceback.print_exc()
                break

        gc.collect()
        times = []
        total_rollouts = 0
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                recs = _simulate_collect_group(backend, gs, max_new_tokens, prompt_len)
            except Exception:
                traceback.print_exc()
                times.append(0.0)
                continue
            t1 = time.perf_counter()
            times.append(t1 - t0)
            total_rollouts += len(recs)

        avg_time = sum(times) / len(times) if times else 0.0
        avg_rollouts = total_rollouts / iterations if iterations else 0.0
        label = f"group_size_{gs}"
        results["benchmarks"][label] = {
            "group_size": gs,
            "prompt_len": prompt_len,
            "max_new_tokens": max_new_tokens,
            "iterations": iterations,
            "avg_time_per_group_ms": avg_time * 1000.0,
            "rollouts_per_sec": avg_rollouts / avg_time if avg_time > 0 else 0.0,
            "time_per_rollout_ms": (avg_time / avg_rollouts * 1000.0) if avg_rollouts > 0 else 0.0,
        }
        print(
            f"  {label} done: {avg_time * 1000.0:.3f} ms/group, "
            f"{results['benchmarks'][label]['rollouts_per_sec']:.2f} rollouts/sec"
        )

    return results


if __name__ == "__main__":
    out = run_benchmark()
    print("\n=== Rollout Benchmark Results ===")
    for name, data in out.get("benchmarks", {}).items():
        print(
            f"{name:20s}  group_time={data['avg_time_per_group_ms']:10.3f}ms  "
            f"rollouts/sec={data['rollouts_per_sec']:8.2f}  "
            f"per_rollout={data['time_per_rollout_ms']:8.3f}ms"
        )
