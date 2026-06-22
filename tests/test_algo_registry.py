from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos import (
    ALGO_REGISTRY,
    GRPO,
    PPO,
    HybridAlgo,
    OPDAlgo,
    get_algo,
    list_algos,
    register_algo,
)
from hermes_agentic_rl.algos.base import AlgoUpdateStats, BaseAlgo, RolloutBatch
from hermes_agentic_rl.backends.base import LLMBackend


def test_builtin_algorithms_are_registered() -> None:
    assert get_algo("grpo") is GRPO
    assert get_algo("ppo") is PPO
    assert get_algo("opd") is OPDAlgo
    assert get_algo("hybrid") is HybridAlgo
    assert get_algo("missing-algo") is GRPO
    assert {"grpo", "ppo", "opd", "hybrid"}.issubset(set(list_algos()))


def test_register_algo_rejects_duplicate_name() -> None:
    class DemoAlgo(BaseAlgo):
        def compute_loss(
            self,
            policy: LLMBackend,
            ref_policy: LLMBackend | None,
            batch: RolloutBatch,
        ):
            del policy, ref_policy, batch
            zero = torch.zeros(())
            return zero, AlgoUpdateStats(
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                entropy=0.0,
                mean_reward=0.0,
                mean_advantage=0.0,
                clip_frac=0.0,
                n_records=0,
            )

    try:
        register_algo("demo_test_algo", DemoAlgo)
        assert ALGO_REGISTRY["demo_test_algo"] is DemoAlgo
        with pytest.raises(ValueError):
            register_algo("demo-test-algo", GRPO)
    finally:
        ALGO_REGISTRY.pop("demo_test_algo", None)
