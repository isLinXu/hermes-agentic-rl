"""Benchmark backend latency: forward pass and generation."""

from __future__ import annotations

import gc
import time
import traceback
from typing import Any

_benchmark_results: dict[str, Any] = {}


def run_benchmark(warmup: int = 1, iterations: int = 10) -> dict[str, Any]:
    """Run backend latency benchmarks and return structured results."""
    results: dict[str, Any] = {"module": "bench_backend", "benchmarks": {}}
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
        max_len=1024,
        device=device,
        dtype=dtype,
        seed=42,
    )
    backend = TinyCausalLMBackend(cfg)
    tokenizer = backend.tokenizer

    seq_lengths = [128, 256, 512, 1024]
    # For generation we use half the seq length as prompt, half as response cap
    max_new_tokens_list = [64, 128, 256, 512]

    # ---- Benchmark generate() ----
    for seq_len, max_new in zip(seq_lengths, max_new_tokens_list):
        prompt_ids = tokenizer.encode("a" * (seq_len // 4))[: max(seq_len // 2, 1)]
        # Warmup
        for _ in range(warmup):
            try:
                backend.generate(prompt_ids, max_new_tokens=max_new, temperature=1.0, seed=42)
            except Exception:
                traceback.print_exc()
                break

        gc.collect()
        times = []
        total_tokens = 0
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                gen = backend.generate(prompt_ids, max_new_tokens=max_new, temperature=1.0, seed=42)
            except Exception:
                traceback.print_exc()
                times.append(0.0)
                continue
            t1 = time.perf_counter()
            times.append(t1 - t0)
            total_tokens += len(gen.response_ids)

        avg_time = sum(times) / len(times) if times else 0.0
        avg_tokens = total_tokens / iterations if iterations else 0.0
        label = f"generate_seq{seq_len}"
        results["benchmarks"][label] = {
            "seq_length": seq_len,
            "max_new_tokens": max_new,
            "prompt_len": len(prompt_ids),
            "iterations": iterations,
            "avg_latency_ms": avg_time * 1000.0,
            "throughput_tokens_per_sec": avg_tokens / avg_time if avg_time > 0 else 0.0,
        }
        tp = results["benchmarks"][label]["throughput_tokens_per_sec"]
        print(f"  {label} done: {avg_time * 1000.0:.3f} ms, {tp:.2f} tok/s")

    # ---- Benchmark score_batch() (forward pass) ----
    # Varying total sequence length (prompt + response)
    vocab_size = tokenizer.vocab_size
    for seq_len in seq_lengths:
        batch_size = 4
        prompt_len = seq_len // 4
        response_len = seq_len // 4
        prompt_ids_list = [
            [(i + j) % vocab_size for j in range(prompt_len)] for i in range(batch_size)
        ]
        response_ids_list = [
            [(i + j + 10) % vocab_size for j in range(response_len)] for i in range(batch_size)
        ]

        # Warmup
        for _ in range(warmup):
            try:
                backend.score_batch(prompt_ids_list, response_ids_list, temperature=1.0)
            except Exception:
                traceback.print_exc()
                break

        gc.collect()
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                _logp, _mask = backend.score_batch(
                    prompt_ids_list, response_ids_list, temperature=1.0
                )
            except Exception:
                traceback.print_exc()
                times.append(0.0)
                continue
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_time = sum(times) / len(times) if times else 0.0
        total_seq_tokens = sum(
            len(p) + len(r)
            for p, r in zip(prompt_ids_list, response_ids_list)
        )
        label = f"score_batch_seq{seq_len}"
        results["benchmarks"][label] = {
            "seq_length": seq_len,
            "batch_size": batch_size,
            "iterations": iterations,
            "avg_latency_ms": avg_time * 1000.0,
            "throughput_tokens_per_sec": total_seq_tokens / avg_time if avg_time > 0 else 0.0,
        }
        tp = results["benchmarks"][label]["throughput_tokens_per_sec"]
        print(f"  {label} done: {avg_time * 1000.0:.3f} ms, {tp:.2f} tok/s")

    return results


if __name__ == "__main__":
    out = run_benchmark()
    print("\n=== Backend Benchmark Results ===")
    for name, data in out.get("benchmarks", {}).items():
        print(
            f"{name:25s}  latency={data['avg_latency_ms']:10.3f}ms  "
            f"throughput={data['throughput_tokens_per_sec']:10.2f} tok/s"
        )
