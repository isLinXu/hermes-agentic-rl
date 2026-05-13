"""Tests for the PPO algorithm + GAE helpers + clipped value loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos import PPO, PPOConfig
from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.common import (
    clipped_value_loss,
    compute_gae,
    terminal_token_rewards,
)
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend


def _rand_backend(seed: int = 0, with_value: bool = True) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(
            seed=seed,
            with_value_head=with_value,
            dim=16,
            n_heads=2,
            n_layers=1,
            max_len=32,
        )
    )


def test_terminal_token_rewards_basic():
    assert terminal_token_rewards(0.8, 4) == [0.0, 0.0, 0.0, 0.8]
    assert terminal_token_rewards(0.0, 0) == []


def test_compute_gae_matches_bellman_for_gamma1_lam1():
    # With gamma=lam=1, GAE = returns - values
    values = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    rewards = [0.0, 0.0, 1.0]
    advs, returns = compute_gae(rewards, values, gamma=1.0, lam=1.0)
    expected_returns = torch.tensor([1.0, 1.0, 1.0])
    expected_advs = expected_returns - values
    assert torch.allclose(returns, expected_returns, atol=1e-5)
    assert torch.allclose(advs, expected_advs, atol=1e-5)


def test_clipped_value_loss_never_negative():
    v_new = torch.tensor([0.5, 1.0], requires_grad=True)
    v_old = torch.tensor([0.0, 0.0])
    returns = torch.tensor([0.3, 0.3])
    loss, stats = clipped_value_loss(v_new, v_old, returns, clip_eps=0.2)
    assert loss.item() >= 0.0
    assert "value_clip_frac" in stats


def test_ppo_requires_value_head():
    backend = _rand_backend(with_value=False)
    algo = PPO(PPOConfig())
    rec = RolloutRecord(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        old_logprobs=[-0.1, -0.2],
        reward=0.5,
        group_id="g",
    )
    with pytest.raises(RuntimeError, match="value head"):
        algo.compute_loss(backend, None, RolloutBatch([rec]))


def test_ppo_one_step_produces_gradients():
    backend = _rand_backend(with_value=True)
    algo = PPO(PPOConfig(entropy_coef=0.01, normalize_advantage=False))
    rec = RolloutRecord(
        prompt_ids=[1, 2, 3, 4],
        response_ids=[5, 6, 7],
        old_logprobs=[-0.3, -0.4, -0.5],
        reward=0.8,
        group_id="g",
    )
    batch = RolloutBatch([rec])
    # capture params snapshot
    params_before = {k: v.detach().clone() for k, v in backend.model.named_parameters()}
    optim = torch.optim.SGD(list(backend.trainable_parameters()), lr=0.1)
    optim.zero_grad()
    loss, stats = algo.compute_loss(backend, None, batch)
    assert loss.requires_grad
    loss.backward()
    optim.step()
    # at least one param must have moved
    moved = sum(
        1
        for k, v in backend.model.named_parameters()
        if not torch.allclose(v.detach(), params_before[k], atol=1e-8)
    )
    assert moved > 0
    # stats shape
    d = stats.as_dict()
    assert d["algo"] == "ppo"
    assert "value_loss" in d
    assert d["n_updated"] == 1
