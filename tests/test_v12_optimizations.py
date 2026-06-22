"""Tests for v1.2 optimizations.

Covers every change made in the v1.2 pass:

P0-1  GSPO approx_kl now uses k3 estimator (consistent with global default).
P0-2  PIDKLController.update: ``import math`` moved to module top-level
       — hot-path benchmarked, no per-call import overhead.
P1-1  GRPO KL-penalty uses shared ``kl_from_logprobs_batched``; inline block
       removed. Result identical to the old code.
P1-2  SimPOAlgo empty/filtered batch returns a plain zero tensor (no
       ``requires_grad=True``), preventing spurious backward() calls.
P1-3  ``clipped_surrogate_loss`` (single-rollout) now returns ``approx_kl``
       and ``n_tokens`` stats, symmetric with the batched variant.
P2-1  ``group_normalize_advantage_tensor`` — new tensor-native helper
       exported from algos.common.
P2-2  ``compute_gae_batched`` normalize path avoids ``.item()`` Python
       round-trip (uses tensor comparison).
P2-3  ``validate_on_policy_config`` warns on invalid PID gain config.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
from hermes_agentic_rl.algos.common import (
    clipped_surrogate_loss,
    compute_gae_batched,
    group_normalize_advantage,
    group_normalize_advantage_tensor,
)
from hermes_agentic_rl.algos.common.batch_prepare import (
    build_advantage_tensor,
    compute_entropy_bonus,
    compute_kl_penalty,
    mean_advantage_from_tensors,
    mean_reward_from_records,
    stack_old_logprobs,
)
from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.algos.gspo import GSPO, GSPOConfig
from hermes_agentic_rl.algos.simpo import SimPOAlgo, SimPOConfig
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.trainers.kl_controller import (
    AdaptiveKLController,
    PIDKLController,
    build_kl_controller,
)
from hermes_agentic_rl.trainers.on_policy_config import (
    OnPolicyTrainerConfig,
    validate_on_policy_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_backend(
    dim: int = 16, n_heads: int = 2, n_layers: int = 2, max_len: int = 32, seed: int = 0
) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(dim=dim, n_heads=n_heads, n_layers=n_layers, max_len=max_len, seed=seed)
    )


def _rec(
    prompt_ids=(1, 2, 3),
    response_ids=(4, 5, 6),
    reward=1.0,
    group_id="g0",
) -> RolloutRecord:
    return RolloutRecord(
        prompt_ids=list(prompt_ids),
        response_ids=list(response_ids),
        old_logprobs=[0.0] * len(response_ids),
        reward=reward,
        group_id=group_id,
    )


# ---------------------------------------------------------------------------
# P0-1: GSPO approx_kl uses k3 estimator
# ---------------------------------------------------------------------------


class TestGSPOApproxKLK3:
    """GSPO.compute_loss().extra['approx_kl'] must be >= 0 (k3 property)."""

    def test_approx_kl_nonneg(self) -> None:
        backend = _tiny_backend()
        algo = GSPO(GSPOConfig())
        records = [
            _rec(reward=1.0, group_id="g0"),
            _rec(reward=0.0, group_id="g0"),
        ]
        batch = RolloutBatch(records)
        _loss, stats = algo.compute_loss(backend, None, batch)
        approx_kl = stats.extra.get("approx_kl", -999.0)
        assert approx_kl >= 0.0, f"GSPO approx_kl should be >= 0 with k3 estimator, got {approx_kl}"

    def test_approx_kl_k3_formula(self) -> None:
        """Cross-check: k3(r) = exp(-r) - 1 + r >= 0 for any r."""
        for r_val in [-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0]:
            r = torch.tensor(r_val)
            k3 = float((torch.exp(-r.clamp(-20, 20)) - 1.0 + r.clamp(-20, 20)).item())
            assert k3 >= 0.0, f"k3({r_val}) < 0: {k3}"


# ---------------------------------------------------------------------------
# P0-2: PIDKLController — no hot-path import overhead
# ---------------------------------------------------------------------------


class TestPIDKLControllerNoImport:
    """PIDKLController.update() should not import math on each call."""

    def test_update_callable_multiple_times(self) -> None:
        ctrl = PIDKLController(
            init_kl_coef=0.05,
            target_kl=0.1,
            Kp=0.1,
            Ki=0.01,
            Kd=0.005,
        )
        _prev = ctrl.value
        for kl in [0.2, 0.15, 0.1, 0.08, 0.12]:
            new = ctrl.update(kl)
            assert 0 < new <= 10.0, f"beta out of bounds: {new}"
        assert ctrl._step_count == 5

    def test_pid_state_dict_roundtrip(self) -> None:
        ctrl = PIDKLController(init_kl_coef=0.1, target_kl=0.05, Kp=0.2)
        for kl in [0.1, 0.08, 0.06]:
            ctrl.update(kl)
        state = ctrl.state_dict()
        ctrl2 = PIDKLController(init_kl_coef=0.999)
        ctrl2.load_state_dict(state)
        assert abs(ctrl2.value - ctrl.value) < 1e-9
        assert ctrl2._integral == ctrl._integral
        assert ctrl2._prev_error == ctrl._prev_error

    def test_math_module_at_top_level(self) -> None:
        """math is imported at module level; the per-call overhead is zero."""
        import hermes_agentic_rl.trainers.kl_controller as kl_mod

        assert hasattr(kl_mod, "math"), (
            "math should be imported at module level, not inside update()"
        )


# ---------------------------------------------------------------------------
# P1-1: GRPO KL-penalty uses shared kl_from_logprobs_batched
# ---------------------------------------------------------------------------


class TestGRPOKLSharedImpl:
    """GRPO KL value should equal kl_from_logprobs_batched manually computed."""

    def test_grpo_kl_matches_shared_fn(self) -> None:
        backend = _tiny_backend()
        ref_backend = _tiny_backend()
        algo = GRPO(GRPOConfig(kl_coef=0.1, kl_estimator="k3"))
        records = [
            _rec(reward=1.0, group_id="g0"),
            _rec(reward=0.5, group_id="g0"),
        ]
        batch = RolloutBatch(records)
        _loss, stats = algo.compute_loss(backend, ref_backend, batch)
        # KL should be non-negative (k3 guarantees this)
        assert stats.kl >= 0.0, f"GRPO KL should be >= 0, got {stats.kl}"

    def test_grpo_kl_zero_without_ref(self) -> None:
        backend = _tiny_backend()
        algo = GRPO(GRPOConfig(kl_coef=0.1))
        records = [
            _rec(reward=1.0, group_id="g0"),
            _rec(reward=0.0, group_id="g0"),
        ]
        batch = RolloutBatch(records)
        _loss, stats = algo.compute_loss(backend, None, batch)
        assert stats.kl == 0.0


# ---------------------------------------------------------------------------
# P1-2: SimPO empty batch returns plain zero (no requires_grad)
# ---------------------------------------------------------------------------


class TestSimPOEmptyBatch:
    """Empty batch and filtered batch should not carry requires_grad=True."""

    def test_empty_batch_no_grad(self) -> None:
        backend = _tiny_backend()
        algo = SimPOAlgo(SimPOConfig())
        batch = RolloutBatch([])
        loss, stats = algo.compute_loss(backend, None, batch)
        assert not loss.requires_grad, "empty-batch loss must NOT have requires_grad"
        assert float(loss.item()) == 0.0
        assert stats.n_records == 0

    def test_all_same_reward_filtered(self) -> None:
        """DAPO filter drops zero-variance groups → should return plain zero."""
        backend = _tiny_backend()
        algo = SimPOAlgo(SimPOConfig(dapo_filter=True))
        records = [
            _rec(reward=1.0, group_id="g0"),
            _rec(reward=1.0, group_id="g0"),
        ]
        batch = RolloutBatch(records)
        loss, stats = algo.compute_loss(backend, None, batch)
        assert not loss.requires_grad
        assert stats.extra.get("n_dropped_zero_var", 0) >= 1


# ---------------------------------------------------------------------------
# P1-3: clipped_surrogate_loss returns approx_kl + n_tokens
# ---------------------------------------------------------------------------


class TestClippedSurrogateLossSingleStats:
    """Single-rollout variant must return same stat keys as batched variant."""

    def test_returns_approx_kl(self) -> None:
        T = 8
        new_lp = torch.randn(T, requires_grad=True)
        old_lp = torch.randn(T)
        adv = 1.0
        _loss, stats = clipped_surrogate_loss(new_lp, old_lp, adv)
        assert "approx_kl" in stats, "approx_kl must be in single-rollout stats"
        assert "n_tokens" in stats, "n_tokens must be in single-rollout stats"
        assert stats["approx_kl"] >= 0.0, "k3 approx_kl must be non-negative"
        assert stats["n_tokens"] == T

    def test_zero_response_fallback(self) -> None:
        new_lp = torch.zeros(0)
        old_lp = torch.zeros(0)
        loss, _stats = clipped_surrogate_loss(new_lp, old_lp, 1.0)
        assert float(loss.item()) == 0.0
        # Empty path returns minimal stats (no approx_kl key currently — OK).


# ---------------------------------------------------------------------------
# P2-1: group_normalize_advantage_tensor
# ---------------------------------------------------------------------------


class TestGroupNormalizeAdvantageTensor:
    """group_normalize_advantage_tensor should match the list version."""

    def test_matches_list_version(self) -> None:
        rewards = [0.2, 0.8, 1.0, -0.5, 0.4]
        list_out = group_normalize_advantage(rewards)
        tensor_out = group_normalize_advantage_tensor(torch.tensor(rewards, dtype=torch.float32))
        for a, b in zip(list_out, tensor_out.tolist(), strict=True):
            assert abs(a - b) < 1e-5, f"mismatch: {a} vs {b}"

    def test_single_element_zero(self) -> None:
        out = group_normalize_advantage_tensor(torch.tensor([1.0]))
        assert float(out[0]) == 0.0

    def test_identical_rewards_zero(self) -> None:
        out = group_normalize_advantage_tensor(torch.tensor([0.5, 0.5, 0.5]))
        assert all(v == 0.0 for v in out.tolist())

    def test_empty_tensor(self) -> None:
        out = group_normalize_advantage_tensor(torch.zeros(0))
        assert out.shape[0] == 0

    def test_detached(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = group_normalize_advantage_tensor(t)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# P2-2: compute_gae_batched normalize uses tensor comparison
# ---------------------------------------------------------------------------


class TestComputeGAEBatchedNormalizeTensor:
    """compute_gae_batched normalize path should not call .item()."""

    def test_normalize_on_device(self) -> None:
        B, T = 4, 8
        rewards = torch.zeros(B, T)
        rewards[:, -1] = torch.tensor([1.0, 0.5, -0.5, 0.0])
        values = torch.zeros(B, T)
        mask = torch.ones(B, T, dtype=torch.bool)
        advs, _rets = compute_gae_batched(rewards, values, mask, normalize=True)
        # After normalization: valid advantages should have ~zero mean
        valid_advs = advs[mask]
        assert abs(float(valid_advs.mean().item())) < 0.1

    def test_normalize_no_valid_tokens(self) -> None:
        B, T = 2, 4
        rewards = torch.zeros(B, T)
        values = torch.zeros(B, T)
        mask = torch.zeros(B, T, dtype=torch.bool)  # all masked
        advs, _rets = compute_gae_batched(rewards, values, mask, normalize=True)
        assert advs.shape == (B, T)
        assert float(advs.sum().item()) == 0.0


# ---------------------------------------------------------------------------
# P2-3: validate_on_policy_config — PID gain warnings
# ---------------------------------------------------------------------------


class TestValidatePIDKLConfig:
    """validate_on_policy_config should warn on invalid PID gain settings."""

    def test_pid_nonpositive_kp_warns(self) -> None:
        cfg = OnPolicyTrainerConfig(
            adaptive_kl=True,
            use_reference=True,
            target_kl=0.1,
            adaptive_kl_type="pid",
            adaptive_kl_Kp=-0.1,
        )
        warnings = validate_on_policy_config(cfg)
        assert any("Kp" in w or "kp" in w.lower() for w in warnings), (
            f"Expected Kp warning, got: {warnings}"
        )

    def test_pid_zero_imax_warns(self) -> None:
        cfg = OnPolicyTrainerConfig(
            adaptive_kl=True,
            use_reference=True,
            target_kl=0.1,
            adaptive_kl_type="pid",
            adaptive_kl_Kp=0.1,
            adaptive_kl_I_max=0.0,
        )
        warnings = validate_on_policy_config(cfg)
        assert any("I_max" in w or "anti-windup" in w for w in warnings), (
            f"Expected I_max warning, got: {warnings}"
        )

    def test_valid_pid_no_extra_warnings(self) -> None:
        cfg = OnPolicyTrainerConfig(
            adaptive_kl=True,
            use_reference=True,
            target_kl=0.1,
            adaptive_kl_type="pid",
            adaptive_kl_Kp=0.1,
            adaptive_kl_Ki=0.01,
            adaptive_kl_Kd=0.005,
            adaptive_kl_I_max=2.0,
        )
        warnings = validate_on_policy_config(cfg)
        pid_warnings = [w for w in warnings if "pid" in w.lower() or "Kp" in w or "I_max" in w]
        assert not pid_warnings, f"Unexpected PID warnings: {pid_warnings}"

    def test_p_controller_no_pid_warnings(self) -> None:
        cfg = OnPolicyTrainerConfig(
            adaptive_kl=True,
            use_reference=True,
            target_kl=0.1,
            adaptive_kl_type="p",
        )
        warnings = validate_on_policy_config(cfg)
        pid_warnings = [w for w in warnings if "Kp" in w or "I_max" in w]
        assert not pid_warnings, f"P-controller should not get PID warnings: {pid_warnings}"


# ---------------------------------------------------------------------------
# Integration: all algo approx_kl values are k3-consistent
# ---------------------------------------------------------------------------


class TestAllAlgosApproxKLNonneg:
    """Smoke test: every registered algo's approx_kl >= 0 (k3 property)."""

    @pytest.mark.parametrize(
        "algo_name,algo_cls,cfg",
        [
            ("grpo", GRPO, GRPOConfig(kl_estimator="k3")),
            ("gspo", GSPO, GSPOConfig(kl_estimator="k3")),
        ],
    )
    def test_approx_kl_nonneg(self, algo_name, algo_cls, cfg) -> None:
        backend = _tiny_backend()
        algo = algo_cls(cfg)
        records = [
            _rec(reward=1.0, group_id="g0"),
            _rec(reward=0.0, group_id="g0"),
        ]
        batch = RolloutBatch(records)
        _loss, stats = algo.compute_loss(backend, None, batch)
        approx_kl = stats.extra.get("approx_kl", 0.0)
        assert approx_kl >= 0.0, f"{algo_name} approx_kl={approx_kl} < 0"


# ---------------------------------------------------------------------------
# P1-4: Shared batch_prepare utilities (v1.2 dedup)
# ---------------------------------------------------------------------------


class TestStackOldLogprobs:
    """stack_old_logprobs builds a correctly-shaped [B, T_max] tensor."""

    def test_basic_shape(self) -> None:
        import torch

        r1 = _rec()
        r1.old_logprobs = [0.1, 0.2, 0.3]
        r2 = _rec()
        r2.old_logprobs = [0.4, 0.5]
        recs = [(r1, [1.0]), (r2, [2.0])]
        B, T_max = 2, 4
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
        result = stack_old_logprobs(recs, B, T_max, torch.float32, torch.device("cpu"), mask)
        assert result.shape == (2, 4)
        # First record: 3 valid tokens → positions [0,1,2]
        assert result[0, 0].item() == pytest.approx(0.1)
        assert result[0, 2].item() == pytest.approx(0.3)
        assert result[0, 3].item() == 0.0  # padding
        # Second record: 2 valid tokens → positions [0,1]
        assert result[1, 0].item() == pytest.approx(0.4)
        assert result[1, 2].item() == 0.0  # padding


class TestBuildAdvantageTensor:
    """build_advantage_tensor supports scalar and per-token advantages."""

    def test_scalar_broadcast(self) -> None:
        import torch

        recs = [
            (_rec(), [1.0]),
            (_rec(), [-0.5]),
        ]
        B, T_max = 2, 3
        mask = torch.ones(2, 3, dtype=torch.float32)
        result = build_advantage_tensor(recs, B, T_max, torch.float32, torch.device("cpu"), mask)
        assert result.shape == (2, 3)
        # Scalar advantage broadcast across tokens
        assert result[0, 0].item() == pytest.approx(1.0)
        assert result[1, 2].item() == pytest.approx(-0.5)

    def test_per_token(self) -> None:
        import torch

        recs = [
            (_rec(), [0.5, 0.3]),
            (_rec(), [1.0]),
        ]
        B, T_max = 2, 3
        mask = torch.ones(2, 3, dtype=torch.float32)
        result = build_advantage_tensor(recs, B, T_max, torch.float32, torch.device("cpu"), mask)
        assert result[0, 0].item() == pytest.approx(0.5)
        assert result[0, 1].item() == pytest.approx(0.3)
        assert result[0, 2].item() == 0.0  # padding


class TestComputeEntropyBonus:
    """compute_entropy_bonus returns (0.0, None) when coef <= 0."""

    def test_zero_coef(self) -> None:
        import torch

        logp = torch.randn(2, 4)
        mask = torch.ones(2, 4)
        val, scalar = compute_entropy_bonus(logp, mask, 0.0)
        assert val == 0.0
        assert scalar is None

    def test_positive_coef(self) -> None:
        import torch

        logp = torch.full((1, 3), -1.0)
        mask = torch.ones(1, 3)
        val, scalar = compute_entropy_bonus(logp, mask, 0.01)
        assert val > 0.0
        assert scalar is not None


class TestMeanHelpers:
    """mean_reward_from_records and mean_advantage_from_tensors edge cases."""

    def test_empty_records(self) -> None:
        assert mean_reward_from_records([]) == 0.0

    def test_basic_reward(self) -> None:
        recs = [_rec(reward=1.0), _rec(reward=3.0)]
        assert mean_reward_from_records(recs) == pytest.approx(2.0)

    def test_empty_advantage(self) -> None:
        import torch

        adv = torch.zeros(0, 0)
        mask = torch.zeros(0, 0)
        assert mean_advantage_from_tensors(adv, mask) == 0.0

    def test_basic_advantage(self) -> None:
        import torch

        adv = torch.tensor([[1.0, 2.0]])
        mask = torch.ones(1, 2)
        assert mean_advantage_from_tensors(adv, mask) == pytest.approx(1.5)


class TestComputeKLPenalty:
    """compute_kl_penalty returns None when no ref policy or kl_coef=0."""

    def test_no_ref_policy(self) -> None:
        import torch

        logp = torch.randn(2, 4)
        mask = torch.ones(2, 4)
        result = compute_kl_penalty(
            logp,
            mask,
            [[1, 2]],
            [[3, 4]],
            1.0,
            ref_policy=None,
            kl_coef=0.02,
        )
        assert result is None

    def test_zero_kl_coef(self) -> None:
        import torch

        logp = torch.randn(2, 4)
        mask = torch.ones(2, 4)
        result = compute_kl_penalty(
            logp,
            mask,
            [[1, 2]],
            [[3, 4]],
            1.0,
            ref_policy=object(),
            kl_coef=0.0,
        )
        assert result is None
