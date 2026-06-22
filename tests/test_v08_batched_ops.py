"""v0.8: batched backend + vectorized GAE + adaptive KL + reward RMS."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.common import compute_gae, compute_gae_batched
from hermes_agentic_rl.algos.common.loss import (
    clipped_surrogate_loss,
    clipped_surrogate_loss_batched,
    clipped_value_loss,
    clipped_value_loss_batched,
)
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.algos.ppo import PPO, PPOConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.trainers.on_policy import _summarize_batch_metadata
from hermes_agentic_rl.trainers.ppo_utils import (
    AdaptiveKLController,
    RunningMeanStd,
)

# --------------------------------------------------------------------------
# score_batch parity: batched logprobs must equal per-record score (up to
# numerical tolerance), and the returned tensor must be differentiable.
# --------------------------------------------------------------------------


def test_score_batch_matches_per_record_tiny():
    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    prompts = [
        b.tokenizer.encode("hello"),
        b.tokenizer.encode("world stuff"),
        b.tokenizer.encode("x"),
    ]
    responses = []
    for p in prompts:
        out = b.generate(p, max_new_tokens=4, temperature=1.0, seed=123)
        responses.append(out.response_ids)

    logp_batch, mask = b.score_batch(prompts, responses)
    B, _T_max = logp_batch.shape
    assert B == 3
    assert mask.dtype == torch.bool

    for i, (p, r) in enumerate(zip(prompts, responses, strict=True)):
        per_rec = b.score(p, r)
        R = per_rec.numel()
        # backend can right-truncate if prompt+response > max_len; compare
        # the common prefix of the last R positions.
        R_b = int(mask[i].sum().item())
        n_common = min(R, R_b)
        if n_common == 0:
            continue
        assert torch.allclose(
            logp_batch[i, :n_common].detach(),
            per_rec[-n_common:].detach(),
            atol=1e-5,
        ), f"row {i}: batched logp differs from per-record"


def test_summarize_batch_metadata_includes_reward_component_metadata():
    batch = RolloutBatch(
        records=[
            RolloutRecord(
                prompt_ids=[1, 2],
                response_ids=[3],
                old_logprobs=[-0.1],
                reward=0.5,
                group_id="g",
                metadata={
                    "reward_components": [
                        {
                            "name": "hermes_reasoning_trace_match",
                            "score": 0.5,
                            "weight": 1.0,
                            "metadata": {
                                "tool_name_match": 1.0,
                                "argument_key_overlap": 0.5,
                                "exact_match": False,
                                "reward_mode": "hybrid",
                            },
                        }
                    ],
                    "finished_naturally": True,
                    "turns_used": 1,
                    "tool_calls_count": 0,
                    "tool_results_count": 0,
                    "final_output_chars": 10,
                    "prompt_tokens": 2,
                    "response_tokens": 1,
                },
            )
        ]
    )

    summary = _summarize_batch_metadata(batch)
    component_meta = summary["reward_component_metadata"]["hermes_reasoning_trace_match"]

    assert component_meta["tool_name_match"]["mean"] == 1.0
    assert component_meta["argument_key_overlap"]["mean"] == 0.5
    assert "exact_match" not in component_meta
    assert "reward_mode" not in component_meta


def test_score_batch_is_differentiable():
    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    prompts = [b.tokenizer.encode("a"), b.tokenizer.encode("bb")]
    responses = [
        b.generate(prompts[0], max_new_tokens=3, seed=1).response_ids,
        b.generate(prompts[1], max_new_tokens=3, seed=2).response_ids,
    ]
    logp, mask = b.score_batch(prompts, responses)
    loss = -(logp * mask.to(logp.dtype)).sum()
    assert loss.requires_grad
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in b.trainable_parameters()
    )
    assert has_grad


def test_score_with_value_batch_shapes():
    b = TinyCausalLMBackend(
        TinyBackendConfig(
            dim=16,
            n_heads=2,
            n_layers=1,
            seed=0,
            with_value_head=True,
        )
    )
    prompts = [b.tokenizer.encode("a"), b.tokenizer.encode("bc")]
    responses = [
        b.generate(prompts[0], max_new_tokens=2, seed=1).response_ids,
        b.generate(prompts[1], max_new_tokens=3, seed=2).response_ids,
    ]
    logp, ent, val, mask = b.score_with_value_batch(prompts, responses)
    assert logp.shape == ent.shape == val.shape == mask.shape
    assert mask.dtype == torch.bool
    # All padded positions must have mask=False and zero values.
    padded = ~mask
    if padded.any():
        assert float(logp[padded].abs().max().item()) == 0.0
        assert float(val[padded].abs().max().item()) == 0.0


# --------------------------------------------------------------------------
# compute_gae_batched: row-wise equals compute_gae per-row.
# --------------------------------------------------------------------------


def test_gae_batched_matches_per_row():
    # 3 rollouts of varying length: 5, 3, 1
    rewards_list = [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5],
        [1.0],
    ]
    values_list = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.0, 0.1, 0.2],
        [0.3],
    ]
    B = 3
    T_max = 5
    rewards = torch.zeros(B, T_max)
    values = torch.zeros(B, T_max)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, (r, v) in enumerate(zip(rewards_list, values_list, strict=True)):
        L = len(r)
        rewards[i, :L] = torch.tensor(r)
        values[i, :L] = torch.tensor(v)
        mask[i, :L] = True

    advs_b, rets_b = compute_gae_batched(
        rewards,
        values,
        mask,
        gamma=0.99,
        lam=0.95,
        normalize=False,
    )

    for i in range(B):
        L = int(mask[i].sum().item())
        advs_i, rets_i = compute_gae(
            rewards_list[i],
            torch.tensor(values_list[i]),
            gamma=0.99,
            lam=0.95,
            normalize=False,
        )
        assert torch.allclose(advs_b[i, :L], advs_i, atol=1e-5), f"advs mismatch row {i}"
        assert torch.allclose(rets_b[i, :L], rets_i, atol=1e-5), f"rets mismatch row {i}"


# --------------------------------------------------------------------------
# Batched clipped surrogate: approx_kl is nonneg, n_tokens counted right.
# --------------------------------------------------------------------------


def test_clipped_surrogate_batched_basic():
    new = torch.tensor(
        [[-1.0, -0.5, -2.0, 0.0], [-0.7, -0.7, 0.0, 0.0], [-1.5, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    old = torch.tensor(
        [[-1.0, -0.5, -2.0, 0.0], [-0.7, -0.7, 0.0, 0.0], [-1.5, 0.0, 0.0, 0.0]],
    )
    adv = torch.tensor([1.0, 0.5, -0.5]).unsqueeze(-1)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False], [True, False, False, False]],
    )
    loss, stats = clipped_surrogate_loss_batched(new, old, adv, mask, clip_eps=0.2)
    # ratio == 1 everywhere, so stats:
    assert abs(stats["ratio_mean"] - 1.0) < 1e-5
    assert stats["approx_kl"] >= 0.0
    assert stats["n_tokens"] == 3 + 2 + 1
    assert loss.requires_grad


def test_clipped_value_loss_batched_never_negative():
    B, T = 2, 3
    v_new = torch.tensor([[0.5, 1.0, 0.0], [0.2, 0.3, 0.4]], requires_grad=True)
    v_old = torch.zeros(B, T)
    returns = torch.tensor([[0.3, 0.3, 0.0], [0.1, 0.1, 0.1]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    loss, stats = clipped_value_loss_batched(v_new, v_old, returns, mask, clip_eps=0.2)
    assert loss.item() >= 0.0
    assert stats["value_clip_frac"] >= 0.0


# --------------------------------------------------------------------------
# GRPO / PPO: end-to-end single-step update still works.
# --------------------------------------------------------------------------


def _make_batch(b: TinyCausalLMBackend, n_groups: int, gsize: int) -> RolloutBatch:
    recs = []
    for g in range(n_groups):
        p = b.tokenizer.encode(f"p{g}")
        for i in range(gsize):
            o = b.generate(p, max_new_tokens=3, seed=g * 7 + i)
            recs.append(
                RolloutRecord(
                    prompt_ids=list(p),
                    response_ids=list(o.response_ids),
                    old_logprobs=list(o.logprobs),
                    reward=float(i) / gsize,
                    group_id=f"g{g}",
                )
            )
    return RolloutBatch(recs)


def test_grpo_batched_produces_gradient():
    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    batch = _make_batch(b, 2, 3)
    algo = GRPO(GRPOConfig(clip_eps=0.2, kl_coef=0.0))
    loss, stats = algo.compute_loss(b, None, batch)
    assert stats.n_records == 6
    if loss.requires_grad:
        loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in b.trainable_parameters()
    )
    assert "approx_kl" in stats.extra


def test_grpo_ratios_use_rollout_temperature_metadata():
    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    prompt = b.tokenizer.encode("temperature")
    recs = []
    for i in range(3):
        out = b.generate(prompt, max_new_tokens=4, temperature=0.7, seed=i)
        recs.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=float(i),
                group_id="g",
                metadata={"rollout_temperature": 0.7},
            )
        )
    algo = GRPO(GRPOConfig(clip_eps=0.2, kl_coef=0.0))
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(recs))
    assert stats.extra["score_temperature"] == 0.7
    assert stats.clip_frac == 0.0
    assert stats.extra["approx_kl"] < 1e-8


def test_grpo_batched_with_reference_kl():
    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    ref = b.clone_frozen()
    with torch.no_grad():
        for p in b.trainable_parameters():
            p.add_(torch.randn_like(p) * 0.1)
    batch = _make_batch(b, 1, 3)
    algo = GRPO(GRPOConfig(clip_eps=0.2, kl_coef=0.1))
    loss, stats = algo.compute_loss(b, ref, batch)
    assert stats.kl != 0.0
    assert loss.requires_grad


def test_ppo_batched_respects_old_values_metadata():
    b = TinyCausalLMBackend(
        TinyBackendConfig(
            dim=16,
            n_heads=2,
            n_layers=1,
            seed=0,
            with_value_head=True,
        )
    )
    p = [1, 2, 3, 4]
    r = [5, 6, 7]
    # Frozen old_values that DIFFER from the current V_new
    rec = RolloutRecord(
        prompt_ids=list(p),
        response_ids=list(r),
        old_logprobs=[-0.3, -0.4, -0.5],
        reward=0.8,
        group_id="g",
        metadata={"_ppo_old_values": [0.05, 0.10, 0.15]},
    )
    algo = PPO(PPOConfig(vf_clip_eps=0.001, normalize_advantage=False))
    _loss, stats = algo.compute_loss(b, None, RolloutBatch([rec]))
    # value_clip_frac should be > 0 because V_new - V_old is large vs 0.001
    assert stats.extra["value_clip_frac"] > 0.0


# --------------------------------------------------------------------------
# RunningMeanStd + AdaptiveKLController
# --------------------------------------------------------------------------


def test_running_mean_std_converges():
    rms = RunningMeanStd()
    import random

    random.seed(0)
    data = [random.gauss(3.0, 2.0) for _ in range(5000)]
    # Update in batches of 50
    for i in range(0, len(data), 50):
        rms.update(data[i : i + 50])
    assert abs(rms.mean - 3.0) < 0.15
    assert abs(rms.std - 2.0) < 0.15


def test_running_mean_std_state_roundtrip():
    rms = RunningMeanStd()
    rms.update([1.0, 2.0, 3.0, 4.0])
    state = rms.state_dict()
    rms2 = RunningMeanStd()
    rms2.load_state_dict(state)
    assert abs(rms2.mean - rms.mean) < 1e-9
    assert abs(rms2.var - rms.var) < 1e-9
    assert abs(rms2.normalize(5.0) - rms.normalize(5.0)) < 1e-9


def test_adaptive_kl_scales_up_on_overshoot():
    ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, horizon=10.0)
    # Observed KL is 10x target → proportional_error is clipped to +0.2,
    # β scales by ~1.02 per update.
    for _ in range(20):
        ctrl.update(current_kl=0.1, n_steps=1)
    assert ctrl.value > 0.1  # beta has grown
    assert ctrl.value <= 10.0  # clamped


def test_adaptive_kl_scales_down_on_undershoot():
    ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.1, horizon=10.0)
    for _ in range(20):
        ctrl.update(current_kl=0.01, n_steps=1)  # 10x below target
    assert ctrl.value < 0.1
    assert ctrl.value >= 1e-4


def test_adaptive_kl_disabled_when_target_zero():
    ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.0)
    for _ in range(10):
        ctrl.update(current_kl=5.0, n_steps=1)
    assert ctrl.value == 0.1
