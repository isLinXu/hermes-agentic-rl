"""Benchmark memory usage during algorithm forward pass."""

from __future__ import annotations

import gc
import time
import traceback
from typing import Any

_benchmark_results: dict[str, Any] = {}


def _make_synthetic_batch(
    batch_size: int, prompt_len: int, response_len: int, vocab_size: int = 80
) -> Any:
    """Create a synthetic RolloutBatch."""
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


def run_benchmark(warmup: int = 1, iterations: int = 5) -> dict[str, Any]:
    """Run memory benchmarks and return structured results."""
    results: dict[str, Any] = {"module": "bench_memory", "benchmarks": {}}
    try:
        import torch

        from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
        from hermes_agentic_rl.algos.ppo import PPO, PPOConfig
        from hermes_agentic_rl.algos.rloo import RLOOAlgo, RLOOConfig
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
    except Exception as exc:
        results["error"] = f"Import failed: {exc}"
        return results

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

    grpo = GRPO(GRPOConfig(clip_eps=0.2, kl_coef=0.02, advantage_norm="batch"))
    ppo = PPO(PPOConfig(clip_eps=0.2, vf_coef=0.5, kl_coef=0.0))
    rloo = RLOOAlgo(RLOOConfig(clip_eps=0.2, kl_coef=0.02))

    batch_sizes = [4, 8, 16]
    prompt_len = 16
    response_len = 24
    vocab_size = policy.tokenizer.vocab_size

    algos = [
        ("grpo", grpo, policy),
        ("ppo", ppo, ppo_policy),
        ("rloo", rloo, policy),
    ]

    for algo_name, algo, pol in algos:
        for bs in batch_sizes:
            batch = _make_synthetic_batch(bs, prompt_len, response_len, vocab_size=vocab_size)
            label = f"{algo_name}_bs{bs}"

            # Warmup
            for _ in range(warmup):
                try:
                    loss, _ = algo.compute_loss(pol, ref_policy, batch)
                    if hasattr(loss, "backward"):
                        loss.backward()
                except Exception:
                    traceback.print_exc()
                    break

            # Zero grads before measurement
            for param in pol.trainable_parameters():
                if hasattr(param, "grad") and param.grad is not None:
                    param.grad = None

            gc.collect()
            peak_mb = 0.0
            mem_before = 0.0
            mem_after = 0.0

            # Try tracemalloc for peak tracking
            try:
                import tracemalloc

                tracemalloc.start()
                _, mem_before = tracemalloc.get_traced_memory()
                mem_before = mem_before / (1024 * 1024)
            except Exception:
                tracemalloc = None  # type: ignore[misc]

            # Also try torch memory if available
            torch_mem_before = 0.0
            torch_mem_after = 0.0
            if hasattr(torch, "_C") and hasattr(torch._C, "_cuda"):
                try:
                    torch_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception:
                    pass

            times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                try:
                    loss, _ = algo.compute_loss(pol, ref_policy, batch)
                    if hasattr(loss, "backward"):
                        loss.backward()
                except Exception:
                    traceback.print_exc()
                    break
                t1 = time.perf_counter()
                times.append(t1 - t0)

            if tracemalloc is not None:
                try:
                    _, mem_peak = tracemalloc.get_traced_memory()
                    peak_mb = mem_peak / (1024 * 1024)
                    _, mem_after = tracemalloc.get_traced_memory()
                    mem_after = mem_after / (1024 * 1024)
                    tracemalloc.stop()
                except Exception:
                    pass

            if hasattr(torch, "_C") and hasattr(torch._C, "_cuda"):
                try:
                    torch_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception:
                    pass

            avg_time = sum(times) / len(times) if times else 0.0
            results["benchmarks"][label] = {
                "algo": algo_name,
                "batch_size": bs,
                "iterations": iterations,
                "avg_time_ms": avg_time * 1000.0,
                "peak_memory_mb": peak_mb,
                "memory_delta_mb": mem_after - mem_before,
                "torch_cuda_delta_mb": torch_mem_after - torch_mem_before,
            }
            print(f"  {label} done: peak_mem={peak_mb:.3f}MB  avg_time={avg_time * 1000.0:.3f}ms")

            # Zero grads
            for param in pol.trainable_parameters():
                if hasattr(param, "grad") and param.grad is not None:
                    param.grad = None

    return results


if __name__ == "__main__":
    out = run_benchmark()
    print("\n=== Memory Benchmark Results ===")
    for name, data in out.get("benchmarks", {}).items():
        print(
            f"{name:20s}  peak_mem={data['peak_memory_mb']:10.3f}MB  "
            f"mem_delta={data['memory_delta_mb']:10.3f}MB  "
            f"cuda_delta={data['torch_cuda_delta_mb']:10.3f}MB  "
            f"avg_time={data['avg_time_ms']:8.3f}ms"
        )
