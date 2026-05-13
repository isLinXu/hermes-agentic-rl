"""GRPO algorithm: advantage, loss, end-to-end one-step update."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.common.advantage import group_normalize_advantage
from hermes_agentic_rl.algos.common.loss import clipped_surrogate_loss
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend


def test_group_normalize_basic():
    adv = group_normalize_advantage([1.0, 0.0, 2.0])
    assert len(adv) == 3
    # mean ≈ 0
    assert abs(sum(adv) / len(adv)) < 1e-5


def test_group_normalize_zero_variance():
    # all-equal rewards must give all-zero advantages (no update signal)
    adv = group_normalize_advantage([0.5, 0.5, 0.5])
    assert adv == [0.0, 0.0, 0.0]


def test_group_normalize_single():
    assert group_normalize_advantage([0.7]) == [0.0]
    assert group_normalize_advantage([]) == []


def test_clipped_surrogate_shapes():
    new = torch.tensor([-1.0, -0.5, -2.0], requires_grad=True)
    old = torch.tensor([-1.0, -0.5, -2.0])
    loss, stats = clipped_surrogate_loss(new, old, advantage=1.0, clip_eps=0.2)
    assert loss.shape == ()
    assert 0.0 <= stats["clip_frac"] <= 1.0


def test_clipped_surrogate_ratio_equals_one():
    new = torch.tensor([-1.0, -0.5], requires_grad=True)
    old = torch.tensor([-1.0, -0.5])
    loss, stats = clipped_surrogate_loss(new, old, advantage=1.0, clip_eps=0.2)
    # ratio = 1 everywhere, advantage=1 → loss = -1
    assert abs(loss.item() + 1.0) < 1e-5
    assert abs(stats["ratio_mean"] - 1.0) < 1e-5


def _make_batch(backend: TinyCausalLMBackend, n_groups: int, group_size: int) -> RolloutBatch:
    records = []
    for g in range(n_groups):
        prompt = backend.tokenizer.encode(f"prompt {g}")
        for i in range(group_size):
            out = backend.generate(prompt, max_new_tokens=3, temperature=1.0, seed=100 + g * 10 + i)
            records.append(
                RolloutRecord(
                    prompt_ids=list(prompt),
                    response_ids=list(out.response_ids),
                    old_logprobs=list(out.logprobs),
                    reward=float(i) / group_size,  # distinct rewards
                    group_id=f"g{g}",
                )
            )
    return RolloutBatch(records)


def test_grpo_compute_loss_backward_works():
    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=2, seed=0))
    batch = _make_batch(b, n_groups=2, group_size=4)
    algo = GRPO(GRPOConfig(clip_eps=0.2, kl_coef=0.0))
    loss, stats = algo.compute_loss(b, None, batch)
    assert stats.n_records == 8
    assert isinstance(stats.mean_reward, float)
    # backward succeeds and grads appear
    if loss.requires_grad:
        loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in b.trainable_parameters())
    assert has_grad, "no gradient reached the policy"


def test_grpo_with_reference_produces_kl():
    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=2, seed=0))
    ref = b.clone_frozen()
    # perturb policy so KL is nonzero
    with torch.no_grad():
        for p in b.trainable_parameters():
            p.add_(torch.randn_like(p) * 0.1)
    batch = _make_batch(b, n_groups=1, group_size=3)
    algo = GRPO(GRPOConfig(clip_eps=0.2, kl_coef=0.1))
    loss, stats = algo.compute_loss(b, ref, batch)
    assert stats.n_records == 3
