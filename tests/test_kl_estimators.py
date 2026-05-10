"""KL estimators (k1 / k2 / k3) — Schulman 2020."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.common.kl import kl_from_logprobs


def test_kl_all_estimators_zero_on_identical_logprobs():
    new = torch.tensor([-0.5, -0.3, -0.1])
    ref = new.clone()
    for est in ("k1", "k2", "k3"):
        kl = kl_from_logprobs(new, ref, estimator=est)
        assert abs(float(kl.item())) < 1e-6, f"{est} non-zero on identical: {kl}"


def test_kl_k2_and_k3_non_negative():
    torch.manual_seed(0)
    new = torch.randn(32)
    ref = torch.randn(32)
    for est in ("k2", "k3"):
        kl = kl_from_logprobs(new, ref, estimator=est)
        assert float(kl.item()) >= -1e-6, f"{est} went negative: {kl}"


def test_kl_k1_can_be_negative():
    # Construct a case where logπ_new < logπ_ref systematically → negative K1.
    new = torch.tensor([-2.0, -2.0, -2.0])
    ref = torch.tensor([-1.0, -1.0, -1.0])
    kl = kl_from_logprobs(new, ref, estimator="k1")
    assert float(kl.item()) < 0.0


def test_kl_k3_differentiable():
    new = torch.tensor([-0.5, -0.3, -0.1], requires_grad=True)
    ref = torch.tensor([-1.0, -1.0, -1.0])
    kl = kl_from_logprobs(new, ref, estimator="k3")
    kl.backward()
    assert new.grad is not None
    assert torch.isfinite(new.grad).all()


def test_kl_empty_returns_zero():
    empty = torch.zeros(0)
    for est in ("k1", "k2", "k3"):
        kl = kl_from_logprobs(empty, empty, estimator=est)
        assert float(kl.item()) == 0.0


def test_kl_unknown_estimator_raises():
    with pytest.raises(ValueError):
        kl_from_logprobs(torch.zeros(3), torch.zeros(3), estimator="k4")  # type: ignore[arg-type]


def test_grpo_accepts_k3_estimator_end_to_end():
    """GRPOTrainer with kl_estimator=k3 should produce finite loss + finite KL."""
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.echo_task_env import (
        EchoRewardComponent,
        EchoTaskEnv,
        build_default_echo_dataset,
    )
    from hermes_agentic_rl.trainers.grpo_trainer import (
        GRPOTrainer,
        GRPOTrainerConfig,
    )

    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    cfg = GRPOTrainerConfig(
        n_iters=2,
        group_size=4,
        prompts_per_iter=1,
        lr=1e-3,
        max_new_tokens=4,
        use_reference=True,
        kl_coef=0.01,
        kl_estimator="k3",
        log_every=100,
        seed=1,
    )
    trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
    stats = trainer.train()
    assert len(stats.iters) == 2
    # KL should be finite and (for k3) >= 0
    for rec in stats.iters:
        kl = rec.get("kl", 0.0)
        assert kl == kl  # not NaN
        assert kl >= -1e-5  # k3 is non-negative (tiny float slack)
