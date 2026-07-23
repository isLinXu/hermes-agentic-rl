"""Performance benchmark suite for hermes-agentic-rl.

Measures throughput and latency of key RL training components:
  - Rollout collection (single + batch)
  - Advantage computation (GRPO, RLOO, GAE)
  - Loss computation + backward pass
  - Full training iteration (rollout + update)
  - Weight sync (in-process)
  - Replay buffer operations

Run with::

    pytest tests/benchmarks/test_perf_benchmarks.py -v --benchmark-only

Or programmatically::

    python -m hermes_agentic_rl.benchmarks.perf_suite --quick
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv, build_sim_tool_dataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_runs: int
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "n_runs": self.n_runs,
            **self.metadata,
        }


def _run_benchmark(
    name: str,
    fn: Any,
    n_runs: int = 10,
    warmup: int = 2,
    **metadata: Any,
) -> BenchmarkResult:
    """Run a function n_runs times and collect timing statistics."""
    # Warmup runs (not measured).
    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return BenchmarkResult(
        name=name,
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        max_ms=max(times),
        n_runs=n_runs,
        metadata=metadata,
    )


@dataclass(slots=True)
class BenchmarkSuite:
    """Runs a suite of performance benchmarks and collects results."""

    dim: int = 32
    n_layers: int = 2
    max_len: int = 128
    group_size: int = 4
    n_samples: int = 16
    n_iters: int = 3
    n_runs: int = 5

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks and return results."""
        results: list[BenchmarkResult] = []
        results.extend(self._bench_rollout())
        results.extend(self._bench_advantage())
        results.extend(self._bench_loss_backward())
        results.extend(self._bench_full_iter())
        results.extend(self._bench_weight_sync())
        results.extend(self._bench_replay_buffer())
        return results

    def _make_backend(self) -> TinyCausalLMBackend:
        return TinyCausalLMBackend(
            TinyBackendConfig(
                dim=self.dim,
                n_layers=self.n_layers,
                max_len=self.max_len,
                seed=42,
            )
        )

    def _make_env(self) -> SimToolEnv:
        return SimToolEnv(build_sim_tool_dataset(n=self.n_samples, seed=42))

    # ── Rollout benchmarks ──

    def _bench_rollout(self) -> list[BenchmarkResult]:
        from hermes_agentic_rl.core.reward_manager import RewardManager
        from hermes_agentic_rl.trainers.grpo_trainer import (
            GRPOTrainer,
            GRPOTrainerConfig,
        )

        backend = self._make_backend()
        env = self._make_env()
        cfg = GRPOTrainerConfig(
            n_iters=1,
            group_size=self.group_size,
            prompts_per_iter=1,
            max_new_tokens=8,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = GRPOTrainer(
                policy=backend, env=env, reward_manager=RewardManager(), cfg=cfg,
            )

        async def _collect_once() -> Any:

            item = await env.get_next_item()
            return await trainer._collect_group(item)

        import asyncio

        def _run():
            asyncio.run(_collect_once())

        return [
            _run_benchmark(
                "rollout.single_group",
                _run,
                n_runs=self.n_runs,
                group_size=self.group_size,
                max_new_tokens=8,
            ),
        ]

    # ── Advantage computation benchmarks ──

    def _bench_advantage(self) -> list[BenchmarkResult]:
        from hermes_agentic_rl.algos.common.advantage import (
            group_normalize_advantage,
        )

        # Create synthetic rewards.
        B = self.group_size
        rewards = [float(i) * 0.5 + 0.1 for i in range(B)]

        def _run():
            group_normalize_advantage(rewards)

        return [
            _run_benchmark(
                "advantage.grpo",
                _run,
                n_runs=self.n_runs,
                batch_size=B,
            ),
        ]

    # ── Loss + backward benchmarks ──

    def _bench_loss_backward(self) -> list[BenchmarkResult]:
        backend = self._make_backend()

        # Create a dummy batch.
        B, T = 4, 16
        input_ids = torch.randint(0, backend.tokenizer.vocab_size, (B, T))
        labels = input_ids.clone()

        def _run():
            outputs = backend.model(input_ids)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
            loss.backward()

        return [
            _run_benchmark(
                "loss_backward.tiny",
                _run,
                n_runs=self.n_runs,
                batch_size=B,
                seq_len=T,
            ),
        ]

    # ── Full training iteration benchmark ──

    def _bench_full_iter(self) -> list[BenchmarkResult]:
        from hermes_agentic_rl.core.reward_manager import RewardManager
        from hermes_agentic_rl.trainers.grpo_trainer import (
            GRPOTrainer,
            GRPOTrainerConfig,
        )

        backend = self._make_backend()
        env = self._make_env()
        cfg = GRPOTrainerConfig(
            n_iters=self.n_iters,
            group_size=self.group_size,
            prompts_per_iter=1,
            max_new_tokens=8,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = GRPOTrainer(
                policy=backend, env=env, reward_manager=RewardManager(), cfg=cfg,
            )

        def _run():
            trainer.train()

        return [
            _run_benchmark(
                "full_train.tiny",
                _run,
                n_runs=self.n_runs,
                n_iters=self.n_iters,
                group_size=self.group_size,
            ),
        ]

    # ── Weight sync benchmark ──

    def _bench_weight_sync(self) -> list[BenchmarkResult]:
        from hermes_agentic_rl.client_server import ClientServerPair, SyncMode

        backend = self._make_backend()
        optimizer = torch.optim.SGD(backend.model.parameters(), lr=0.01)
        pair = ClientServerPair.create(
            model=backend.model,
            rollout_backend=backend,
            optimizer=optimizer,
            sync_mode=SyncMode.FULL,
        )

        # Simulate a train step + sync.
        loss = torch.tensor(0.01, requires_grad=True)

        def _run():
            pair.step_and_sync(loss)

        return [
            _run_benchmark(
                "weight_sync.in_process",
                _run,
                n_runs=self.n_runs,
                model_params=sum(p.numel() for p in backend.model.parameters()),
            ),
        ]

    # ── Replay buffer benchmarks ──

    def _bench_replay_buffer(self) -> list[BenchmarkResult]:
        from hermes_agentic_rl.algos.base import RolloutRecord
        from hermes_agentic_rl.trainers.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=128)

        records = [
            RolloutRecord(
                prompt_ids=[1],
                response_ids=[2, 3],
                old_logprobs=[-0.5, -0.6],
                reward=float(i),
                group_id="g0",
            )
            for i in range(32)
        ]

        def _run_push():
            buf.push(records)

        def _run_sample():
            buf.push(records)
            buf.sample(16)

        return [
            _run_benchmark(
                "replay_buffer.push_batch",
                _run_push,
                n_runs=self.n_runs,
                batch_size=len(records),
            ),
            _run_benchmark(
                "replay_buffer.sample",
                _run_sample,
                n_runs=self.n_runs,
                sample_size=16,
            ),
        ]


def run_quick_suite() -> list[dict[str, Any]]:
    """Run a quick benchmark suite (smaller sizes, fewer runs).

    Returns results as a list of dicts suitable for JSON serialization.
    """
    suite = BenchmarkSuite(
        dim=16, n_layers=1, max_len=64, group_size=2, n_samples=4,
        n_iters=1, n_runs=3,
    )
    results = suite.run_all()
    return [r.as_dict() for r in results]


def run_full_suite() -> list[dict[str, Any]]:
    """Run the full benchmark suite."""
    suite = BenchmarkSuite()
    results = suite.run_all()
    return [r.as_dict() for r in results]


if __name__ == "__main__":
    import json
    import sys

    if "--quick" in sys.argv:
        results = run_quick_suite()
    else:
        results = run_full_suite()

    print(json.dumps(results, indent=2))
