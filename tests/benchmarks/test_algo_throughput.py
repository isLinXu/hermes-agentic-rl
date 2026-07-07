"""Performance micro-benchmarks for core RL algorithms.

Run locally::

    pytest tests/benchmarks/ -m benchmark --benchmark-only

These benchmarks use TinyCausalLM only — no GPU or HF weights required.

If ``pytest-benchmark`` is not installed the tests are **skipped** (not errored)
so that the rest of the suite can still collect and run normally.
"""

from __future__ import annotations

import pytest

# If pytest-benchmark is missing, skip every test in this module instead of
# raising a collection-time ``fixture 'benchmark' not found`` error.
pytestmark: pytest.MarkDecorator = pytest.mark.benchmark

try:
    import pytest_benchmark  # noqa: F401  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    def _skip_benchmark(*_args: object, **_kwargs: object) -> None:
        pytest.skip("pytest-benchmark not installed — run pip install pytest-benchmark")

    # Replace the ``benchmark`` fixture so collection succeeds.
    @pytest.fixture  # type: ignore[misc]
    def benchmark() -> object:  # type: ignore[misc]
        return _skip_benchmark

    pytestmark = pytest.mark.skip(reason="pytest-benchmark not installed")

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.common import (
    clipped_surrogate_loss_batched,
    compute_gae_batched,
    group_normalize_advantage_tensor,
)
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.trainers.kl_controller import PIDKLController


def _tiny_backend(*, dim: int = 32, max_len: int = 64) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(dim=dim, n_heads=4, n_layers=2, max_len=max_len, seed=0)
    )


def _rollout_batch(group_size: int = 8, resp_len: int = 6) -> RolloutBatch:
    records = [
        RolloutRecord(
            prompt_ids=[1, 2, 3],
            response_ids=list(range(10, 10 + resp_len)),
            old_logprobs=[-0.1] * resp_len,
            reward=float(i % 3),
            group_id="g0",
        )
        for i in range(group_size)
    ]
    return RolloutBatch(records)


@pytest.mark.benchmark
def test_benchmark_grpo_compute_loss(benchmark) -> None:
    backend = _tiny_backend()
    algo = GRPO(GRPOConfig(kl_coef=0.0, kl_estimator="k3"))
    batch = _rollout_batch()

    def _run() -> None:
        algo.compute_loss(backend, None, batch)

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_clipped_surrogate_batched(benchmark) -> None:
    B, T = 16, 12
    new_lp = torch.randn(B, T, requires_grad=True)
    old_lp = torch.randn(B, T)
    adv = torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)

    benchmark(clipped_surrogate_loss_batched, new_lp, old_lp, adv, mask)


@pytest.mark.benchmark
def test_benchmark_gae_batched(benchmark) -> None:
    B, T = 16, 32
    rewards = torch.zeros(B, T)
    rewards[:, -1] = torch.linspace(-1.0, 1.0, B)
    values = torch.randn(B, T) * 0.1
    mask = torch.ones(B, T, dtype=torch.bool)

    benchmark(compute_gae_batched, rewards, values, mask, normalize=True)


@pytest.mark.benchmark
def test_benchmark_group_normalize_advantage_tensor(benchmark) -> None:
    rewards = torch.randn(64)

    benchmark(group_normalize_advantage_tensor, rewards)


@pytest.mark.benchmark
def test_benchmark_pid_kl_controller_update(benchmark) -> None:
    ctrl = PIDKLController(init_kl_coef=0.05, target_kl=0.1, Kp=0.1, Ki=0.01, Kd=0.005)
    kl_series = [0.2, 0.15, 0.12, 0.1, 0.09, 0.11, 0.13, 0.08]

    def _run() -> None:
        for kl in kl_series:
            ctrl.update(kl)

    benchmark(_run)
