"""Benchmark algorithm compute_loss throughput."""

from __future__ import annotations

import gc
import time
import traceback
from typing import Any

_benchmark_results: dict[str, Any] = {}


def _mem_mb() -> float:
    try:
        import tracemalloc

        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            return peak / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _make_synthetic_batch(
    batch_size: int, prompt_len: int, response_len: int, vocab_size: int = 80
) -> Any:
    """Create a synthetic RolloutBatch with the specified sizes."""
    from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord

    records = []
    for i in range(batch_size):
        prompt_ids = [(i + j) % vocab_size for j in range(prompt_len)]
        response_ids = [(i + j + 10) % vocab_size for j in range(response_len)]
        old_logprobs = [-0.5 - (j * 0.01) for j in range(response_len)]
        reward = float(i % 5) * 0.25
        group_id = f"group_{i // 2}"
        records.append(
            RolloutRecord(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                old_logprobs=old_logprobs,
                reward=reward,
                group_id=group_id,
                metadata={},
            )
        )
    return RolloutBatch(records=records)


def _bench_algo(
    algo: Any,
    policy: Any,
    ref_policy: Any,
    batch: Any,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    """Benchmark a single algorithm's compute_loss."""
    # Warmup
    for _ in range(warmup):
        try:
            loss, _ = algo.compute_loss(policy, ref_policy, batch)
            if hasattr(loss, "backward"):
                loss.backward()
        except Exception:
            traceback.print_exc()
            return {"error": True}

    gc.collect()
    mem_before = _mem_mb()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            loss, _ = algo.compute_loss(policy, ref_policy, batch)
            if hasattr(loss, "backward"):
                loss.backward()
        except Exception:
            traceback.print_exc()
            return {"error": True}
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mem_after = _mem_mb()
    total_time = sum(times)
    avg_time = total_time / len(times)
    samples = len(batch.records)
    return {
        "samples": samples,
        "iterations": iterations,
        "total_time_sec": total_time,
        "avg_time_ms": avg_time * 1000.0,
        "samples_per_sec": samples / avg_time if avg_time > 0 else 0.0,
        "memory_delta_mb": mem_after - mem_before,
    }


def run_benchmark(warmup: int = 1, iterations: int = 10) -> dict[str, Any]:
    """Run algorithm throughput benchmarks and return structured results."""
    results: dict[str, Any] = {"module": "bench_algos", "benchmarks": {}}
    try:
        from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
        from hermes_agentic_rl.algos.ppo import PPO, PPOConfig
        from hermes_agentic_rl.algos.rloo import RLOOAlgo, RLOOConfig
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
    except Exception as exc:
        results["error"] = f"Import failed: {exc}"
        return results

    try:
        import tracemalloc

        tracemalloc.start()
    except Exception:
        pass

    device = "cpu"
    dtype = "float32"
    policy_cfg = TinyBackendConfig(
        dim=32,
        n_heads=4,
        n_layers=2,
        max_len=512,
        device=device,
        dtype=dtype,
        seed=42,
    )
    policy = TinyCausalLMBackend(policy_cfg)

    # PPO needs a value head
    ppo_policy_cfg = TinyBackendConfig(
        dim=32,
        n_heads=4,
        n_layers=2,
        max_len=512,
        device=device,
        dtype=dtype,
        seed=42,
        with_value_head=True,
    )
    ppo_policy = TinyCausalLMBackend(ppo_policy_cfg)

    ref_policy = policy.clone_frozen()

    grpo = GRPO(GRPOConfig(clip_eps=0.2, kl_coef=0.02, entropy_coef=0.0, advantage_norm="batch"))
    ppo = PPO(PPOConfig(clip_eps=0.2, vf_coef=0.5, kl_coef=0.0, entropy_coef=0.0))
    rloo = RLOOAlgo(RLOOConfig(clip_eps=0.2, kl_coef=0.02, entropy_coef=0.0, normalize=True))

    batch_sizes = [1, 4, 8, 16]
    prompt_len = 16
    response_len = 24
    vocab_size = policy.tokenizer.vocab_size

    for bs in batch_sizes:
        batch = _make_synthetic_batch(bs, prompt_len, response_len, vocab_size=vocab_size)
        key_base = f"bs_{bs}"

        # GRPO
        label = f"grpo_{key_base}"
        print(f"  Benchmarking {label} ...")
        results["benchmarks"][label] = _bench_algo(
            grpo, policy, ref_policy, batch, warmup, iterations
        )

        # PPO
        label = f"ppo_{key_base}"
        print(f"  Benchmarking {label} ...")
        results["benchmarks"][label] = _bench_algo(
            ppo, ppo_policy, ref_policy, batch, warmup, iterations
        )

        # RLOO
        label = f"rloo_{key_base}"
        print(f"  Benchmarking {label} ...")
        results["benchmarks"][label] = _bench_algo(
            rloo, policy, ref_policy, batch, warmup, iterations
        )

        # Zero gradients between algorithms to avoid accumulation
        for p in [policy, ppo_policy]:
            for param in p.trainable_parameters():
                if hasattr(param, "grad") and param.grad is not None:
                    param.grad = None

    try:
        import tracemalloc

        tracemalloc.stop()
    except Exception:
        pass

    return results


if __name__ == "__main__":
    out = run_benchmark()
    print("\n=== Algorithm Benchmark Results ===")
    for name, data in out.get("benchmarks", {}).items():
        if data.get("error"):
            print(f"{name}: ERROR")
            continue
        print(
            f"{name:20s}  samples={data['samples']:3d}  "
            f"avg_time={data['avg_time_ms']:8.3f}ms  "
            f"throughput={data['samples_per_sec']:8.2f} samples/sec  "
            f"mem_delta={data['memory_delta_mb']:7.3f}MB"
        )
