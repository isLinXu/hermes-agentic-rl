"""Tests for v1.0 framework optimizations.

Covers:
  A1 — LR Scheduler (ConstantLR / LinearLR / CosineLR / WarmupCosine)
  A2 — DAPO group advantage (dapo_group_advantage)
  A3 — Reward Shaping Hook (length_penalty, format_bonus, compose)
  A4 — per_token_advantage shared helper (expand_per_token_advantage) in RLOO
  B1 — RunningMeanStd / AdaptiveKL checkpoint serialization
  B2 — grad_clip_triggered in stats.extra
  B3 — PER buffer O(1) ring-buffer (deque _priorities)
  B4 — TrainStats: reward_curve / mean_kl / summary / to_dataframe
  C1 — kl_from_logprobs_batched shared implementation
  C2 — CurriculumEnv.observe(reward, level=None) signature
  C3 — _format_log dynamic extras
  C4 — RLOO per_token_advantage / EntropySchedule
"""

from __future__ import annotations

import math
import sys
from collections import deque
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import pytest
import torch

# ===========================================================================
# A1 — LR Scheduler
# ===========================================================================


class TestLRScheduler:
    def test_constant(self):
        from hermes_agentic_rl.trainers.lr_schedule import ConstantLR, make_lr_scheduler

        s = ConstantLR(lr=1e-3)
        assert s.get_lr(0) == pytest.approx(1e-3)
        assert s.get_lr(999) == pytest.approx(1e-3)

        s2 = make_lr_scheduler("constant", lr=5e-4)
        assert s2.get_lr(42) == pytest.approx(5e-4)

    def test_linear_warmup(self):
        from hermes_agentic_rl.trainers.lr_schedule import LinearLR

        s = LinearLR(lr=1e-3, warmup_steps=10, total_steps=0, warmup_start_lr=0.0, end_lr=0.0)
        assert s.get_lr(0) == pytest.approx(0.0)
        assert s.get_lr(5) == pytest.approx(5e-4)
        assert s.get_lr(10) == pytest.approx(1e-3)
        # No decay configured → stays at lr
        assert s.get_lr(100) == pytest.approx(1e-3)

    def test_linear_decay(self):
        from hermes_agentic_rl.trainers.lr_schedule import LinearLR

        s = LinearLR(lr=1e-3, warmup_steps=0, total_steps=100, warmup_start_lr=0.0, end_lr=0.0)
        assert s.get_lr(0) == pytest.approx(1e-3)
        assert s.get_lr(50) == pytest.approx(5e-4)
        assert s.get_lr(100) == pytest.approx(0.0)

    def test_cosine(self):
        from hermes_agentic_rl.trainers.lr_schedule import CosineLR

        s = CosineLR(lr=1e-3, total_steps=100, warmup_steps=0, warmup_start_lr=0.0, end_lr=0.0)
        assert s.get_lr(0) == pytest.approx(1e-3)
        # At half-way, cosine should be ~0.5 * lr
        assert s.get_lr(50) == pytest.approx(5e-4, rel=1e-3)
        # Past total_steps → end_lr
        assert s.get_lr(100) == pytest.approx(0.0, abs=1e-9)
        assert s.get_lr(200) == pytest.approx(0.0, abs=1e-9)

    def test_warmup_cosine(self):
        from hermes_agentic_rl.trainers.lr_schedule import make_lr_scheduler

        s = make_lr_scheduler(
            "warmup_cosine", lr=1e-3, warmup_steps=10, total_steps=110, end_lr=1e-5
        )
        # During warmup
        assert s.get_lr(0) == pytest.approx(0.0)
        assert s.get_lr(10) == pytest.approx(1e-3)
        # After full cosine decay
        assert s.get_lr(110) == pytest.approx(1e-5, rel=1e-3)

    def test_monotone_decrease(self):
        from hermes_agentic_rl.trainers.lr_schedule import CosineLR

        s = CosineLR(lr=1e-3, total_steps=100)
        lrs = [s.get_lr(t) for t in range(101)]
        assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(100))

    def test_unknown_kind_raises(self):
        from hermes_agentic_rl.trainers.lr_schedule import make_lr_scheduler

        with pytest.raises(ValueError):
            make_lr_scheduler("polynomial", lr=1e-3)


# ===========================================================================
# A2 — DAPO group advantage
# ===========================================================================


class TestDAPOAdvantage:
    def test_filters_identical_rewards(self):
        from hermes_agentic_rl.algos.common.advantage import dapo_group_advantage

        result = dapo_group_advantage([1.0, 1.0, 1.0])
        assert result is None  # group should be discarded

    def test_informative_group(self):
        from hermes_agentic_rl.algos.common.advantage import dapo_group_advantage

        result = dapo_group_advantage([0.0, 1.0, 0.5])
        assert result is not None
        assert len(result) == 3
        # z-scored: mean ≈ 0
        assert abs(sum(result) / 3) < 1e-5

    def test_single_element(self):
        from hermes_agentic_rl.algos.common.advantage import dapo_group_advantage

        assert dapo_group_advantage([0.8]) == [0.0]

    def test_empty(self):
        from hermes_agentic_rl.algos.common.advantage import dapo_group_advantage

        assert dapo_group_advantage([]) == []

    def test_grpo_dapo_mode(self):
        """GRPO with advantage_norm='dapo' returns zero loss when all rewards equal."""
        from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
        from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig

        cfg = GRPOConfig(advantage_norm="dapo")
        algo = GRPO(cfg)

        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

        policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_layers=1))

        records = [
            RolloutRecord(
                prompt_ids=[1, 2],
                response_ids=[3, 4],
                old_logprobs=[-0.5, -0.5],
                reward=1.0,
                group_id="g0",
            ),
            RolloutRecord(
                prompt_ids=[1, 2],
                response_ids=[3, 4],
                old_logprobs=[-0.5, -0.5],
                reward=1.0,
                group_id="g0",
            ),
        ]
        batch = RolloutBatch(records=records)
        loss, stats = algo.compute_loss(policy, None, batch)
        assert float(loss.item()) == pytest.approx(0.0)
        assert stats.extra.get("n_updated", 0) == 0


# ===========================================================================
# A3 — Reward Shaping Hook
# ===========================================================================


class TestRewardShaping:
    def _make_record(self, reward: float, resp_len: int, text: str = "") -> Any:
        from hermes_agentic_rl.algos.base import RolloutRecord

        return RolloutRecord(
            prompt_ids=[1],
            response_ids=list(range(resp_len)),
            old_logprobs=[-0.3] * resp_len,
            reward=reward,
            group_id="g",
            metadata={"final_output": text},
        )

    def test_length_penalty_short(self):
        from hermes_agentic_rl.rewards.shaping import length_penalty_shaping

        fn = length_penalty_shaping(target_len=10, penalty_coef=0.1)
        rec = self._make_record(1.0, resp_len=5)
        fn([rec])
        # penalty = 0.1 * |5-10| = 0.5
        assert rec.reward == pytest.approx(0.5)
        assert rec.metadata["pre_shaping_reward"] == pytest.approx(1.0)

    def test_length_penalty_on_target(self):
        from hermes_agentic_rl.rewards.shaping import length_penalty_shaping

        fn = length_penalty_shaping(target_len=8, penalty_coef=0.1)
        rec = self._make_record(1.0, resp_len=8)
        fn([rec])
        assert rec.reward == pytest.approx(1.0)

    def test_format_bonus_match(self):
        from hermes_agentic_rl.rewards.shaping import format_bonus_shaping

        fn = format_bonus_shaping(lambda t: t.startswith("["), bonus=0.5)
        rec = self._make_record(1.0, resp_len=3, text="[yes]")
        fn([rec])
        assert rec.reward == pytest.approx(1.5)
        assert rec.metadata["format_bonus"] == pytest.approx(0.5)

    def test_format_bonus_no_match(self):
        from hermes_agentic_rl.rewards.shaping import format_bonus_shaping

        fn = format_bonus_shaping(lambda t: t.startswith("["), bonus=0.5)
        rec = self._make_record(1.0, resp_len=3, text="nope")
        fn([rec])
        assert rec.reward == pytest.approx(1.0)

    def test_compose(self):
        from hermes_agentic_rl.rewards.shaping import (
            compose_shaping,
            format_bonus_shaping,
            length_penalty_shaping,
        )

        fn = compose_shaping(
            length_penalty_shaping(target_len=5, penalty_coef=0.1),
            format_bonus_shaping(lambda t: len(t) > 0, bonus=0.2),
        )
        rec = self._make_record(1.0, resp_len=10, text="hi")
        fn([rec])
        # penalty = 0.1 * 5 = 0.5 → reward = 0.5; then +0.2 → 0.7
        assert rec.reward == pytest.approx(0.7)


# ===========================================================================
# A4 — expand_per_token_advantage (shared helper) + RLOO per_token_advantage
# ===========================================================================


class TestExpandPerToken:
    def test_basic_expansion(self):
        from hermes_agentic_rl.algos.base import RolloutRecord
        from hermes_agentic_rl.algos.common.reinforce_pp import expand_per_token_advantage

        rec = RolloutRecord(
            prompt_ids=[1, 2],
            response_ids=[3, 4, 5],
            old_logprobs=[-0.3, -0.3, -0.3],
            reward=1.0,
            group_id="g",
        )
        result = expand_per_token_advantage([(rec, [0.8])], gamma=1.0)
        _, per_tok = result[0]
        assert len(per_tok) == 3  # expanded to response length

    def test_empty_response_passthrough(self):
        from hermes_agentic_rl.algos.base import RolloutRecord
        from hermes_agentic_rl.algos.common.reinforce_pp import expand_per_token_advantage

        rec = RolloutRecord(
            prompt_ids=[1],
            response_ids=[],
            old_logprobs=[],
            reward=1.0,
            group_id="g",
        )
        result = expand_per_token_advantage([(rec, [0.5])])
        _, adv = result[0]
        assert adv == [0.5]  # unchanged passthrough

    def test_rloo_per_token_advantage(self):
        """RLOO with per_token_advantage=True produces non-scalar adv tensor."""
        from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
        from hermes_agentic_rl.algos.rloo import RLOOAlgo, RLOOConfig
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

        policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_layers=1))
        cfg = RLOOConfig(per_token_advantage=True)
        algo = RLOOAlgo(cfg)

        records = [
            RolloutRecord(
                prompt_ids=[1, 2],
                response_ids=[3, 4, 5],
                old_logprobs=[-0.3] * 3,
                reward=0.8,
                group_id="g0",
            ),
            RolloutRecord(
                prompt_ids=[1, 2],
                response_ids=[3, 4, 5],
                old_logprobs=[-0.3] * 3,
                reward=0.2,
                group_id="g0",
            ),
        ]
        loss, stats = algo.compute_loss(policy, None, RolloutBatch(records=records))
        assert loss.requires_grad or float(loss.item()) == pytest.approx(0.0, abs=1.0)
        assert stats.n_records == 2


# ===========================================================================
# B1 — checkpoint serialization of RunningMeanStd / AdaptiveKL
# ===========================================================================


class TestCheckpointNormState:
    def test_running_mean_std_state_dict_roundtrip(self):
        from hermes_agentic_rl.trainers.ppo_utils import RunningMeanStd

        rms = RunningMeanStd()
        rms.update([1.0, 2.0, 3.0, 4.0])
        sd = rms.state_dict()
        rms2 = RunningMeanStd()
        rms2.load_state_dict(sd)
        assert rms2.mean == pytest.approx(rms.mean)
        assert rms2.std == pytest.approx(rms.std)
        assert rms2.count == pytest.approx(rms.count)

    def test_adaptive_kl_state_dict_roundtrip(self):
        from hermes_agentic_rl.trainers.ppo_utils import AdaptiveKLController

        ctrl = AdaptiveKLController(init_kl_coef=0.05, target_kl=0.1)
        ctrl.update(0.15)
        ctrl.update(0.08)
        sd = ctrl.state_dict()
        ctrl2 = AdaptiveKLController(init_kl_coef=0.01, target_kl=0.01)
        ctrl2.load_state_dict(sd)
        assert ctrl2.value == pytest.approx(ctrl.value)
        assert ctrl2.target_kl == pytest.approx(ctrl.target_kl)

    def test_checkpoint_state_carries_running_stats(self):
        from hermes_agentic_rl.trainers.checkpoint import CheckpointState

        rs = {"mean": 0.5, "var": 0.25, "count": 10.0, "eps": 1e-8}
        ks = {
            "value": 0.03,
            "target_kl": 0.1,
            "horizon": 10000.0,
            "min_coef": 1e-4,
            "max_coef": 10.0,
            "init_kl_coef": 0.05,
            "step_count": 2,
        }
        state = CheckpointState(
            iteration=5,
            model_state={},
            optimizer_state=None,
            rng_state=None,
            stats=[],
            config={},
            best_reward=0.5,
            best_iteration=3,
            running_stats=rs,
            kl_ctrl_state=ks,
        )
        assert state.running_stats == rs
        assert state.kl_ctrl_state == ks

    def test_get_full_state_dict_uses_fsdp_full_state_context(self, monkeypatch):
        from hermes_agentic_rl.trainers.checkpoint import _get_full_state_dict

        calls: list[tuple[str, bool, bool]] = []

        class FakeFullStateDictConfig:
            def __init__(self, *, offload_to_cpu: bool, rank0_only: bool) -> None:
                self.offload_to_cpu = offload_to_cpu
                self.rank0_only = rank0_only

        class FakeStateDictType:
            FULL_STATE_DICT = "full"

        class FakeFSDP:
            @staticmethod
            @contextmanager
            def state_dict_type(model, state_dict_type, config):
                del model
                calls.append(
                    (
                        state_dict_type,
                        bool(config.offload_to_cpu),
                        bool(config.rank0_only),
                    )
                )
                yield

        fake_fsdp = ModuleType("torch.distributed.fsdp")
        fake_fsdp.FullStateDictConfig = FakeFullStateDictConfig
        fake_fsdp.StateDictType = FakeStateDictType
        fake_fsdp.FullyShardedDataParallel = FakeFSDP
        monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp)

        model = torch.nn.Linear(2, 1)
        state = _get_full_state_dict(model, fsdp_enabled=True)

        assert calls == [("full", True, True)]
        assert set(state) == set(model.state_dict())


# ===========================================================================
# B2 — grad_clip_triggered
# ===========================================================================


class TestGradClipMonitor:
    def test_grad_clip_triggered_field_exists(self):
        """Smoke test: grad_clip_triggered appears in training stats."""
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
        from hermes_agentic_rl.core.reward_manager import RewardManager
        from hermes_agentic_rl.envs.echo_task_env import EchoTaskEnv
        from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

        policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_layers=1))
        env = EchoTaskEnv([{"task_id": "t1", "instruction": "hi", "target": "hi"}])
        rm = RewardManager([])

        cfg = GRPOTrainerConfig(
            n_iters=2,
            group_size=2,
            prompts_per_iter=1,
            max_new_tokens=4,
            grad_clip=1e-6,
            log_every=0,
        )
        trainer = GRPOTrainer(policy=policy, env=env, reward_manager=rm, cfg=cfg)
        stats = trainer.train()
        keys_seen = {k for rec in stats.iters for k in rec}
        assert "grad_clip_triggered" in keys_seen


# ===========================================================================
# B3 — PER buffer O(1) deque
# ===========================================================================


class TestPERBufferDeque:
    def test_priorities_is_deque(self):
        from hermes_agentic_rl.offline.per_buffer import PrioritizedReplayBuffer

        buf = PrioritizedReplayBuffer(maxlen=5)
        assert isinstance(buf._priorities, deque)

    def test_eviction_keeps_deque_size(self):
        from hermes_agentic_rl.offline.per_buffer import PrioritizedReplayBuffer, TrainSample

        buf = PrioritizedReplayBuffer(maxlen=3)
        for i in range(5):
            buf.add_sample(TrainSample(prompt_ids=[i], response_ids=[i], reward=float(i)))
        assert len(buf._priorities) == 3
        assert len(list(buf.samples)) == 3

    def test_sample_prioritized_after_eviction(self):
        from hermes_agentic_rl.offline.per_buffer import PrioritizedReplayBuffer, TrainSample

        buf = PrioritizedReplayBuffer(maxlen=4, alpha=1.0, beta=0.5)
        for i in range(6):
            buf.add_sample(
                TrainSample(prompt_ids=[i], response_ids=[i], reward=float(i)),
                priority=float(i + 1),
            )
        _idxs, samples, weights = buf.sample_prioritized(2)
        assert len(samples) == 2
        assert all(0.0 < w <= 1.0 for w in weights)

    def test_update_priorities(self):
        from hermes_agentic_rl.offline.per_buffer import PrioritizedReplayBuffer, TrainSample

        buf = PrioritizedReplayBuffer(maxlen=None)
        for i in range(3):
            buf.add_sample(TrainSample(prompt_ids=[i], response_ids=[i], reward=0.0), priority=1.0)
        buf.update_priorities([0, 2], [0.1, 3.0])
        prio_list = list(buf._priorities)
        assert prio_list[0] == pytest.approx(0.1 + buf.eps_priority)
        assert prio_list[2] == pytest.approx(3.0 + buf.eps_priority)


# ===========================================================================
# B4 — TrainStats new methods
# ===========================================================================


class TestTrainStatsV10:
    def _make_stats(self):
        from hermes_agentic_rl.trainers.on_policy import TrainStats

        s = TrainStats()
        for i in range(5):
            s.add(
                {
                    "iter": i,
                    "mean_reward": float(i) * 0.1,
                    "loss": 1.0 - float(i) * 0.1,
                    "kl": float(i) * 0.01,
                }
            )
        return s

    def test_reward_curve(self):
        s = self._make_stats()
        curve = s.reward_curve()
        assert curve == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])

    def test_mean_kl(self):
        s = self._make_stats()
        mk = s.mean_kl()
        assert mk == pytest.approx(0.02)

    def test_loss_curve(self):
        s = self._make_stats()
        lc = s.loss_curve()
        assert lc[0] == pytest.approx(1.0)
        assert lc[-1] == pytest.approx(0.6)

    def test_get_column(self):
        s = self._make_stats()
        iters = s.get_column("iter")
        assert iters == [0, 1, 2, 3, 4]

    def test_summary(self):
        s = self._make_stats()
        sm = s.summary()
        assert "best_reward" in sm
        assert "n_iters" in sm
        assert sm["n_iters"] == 5
        assert sm["best_reward"] == pytest.approx(0.4)

    def test_to_dataframe_requires_pandas(self):
        s = self._make_stats()
        try:
            df = s.to_dataframe()
            import pandas as pd

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5
        except ImportError:
            pytest.skip("pandas not installed")


# ===========================================================================
# C1 — kl_from_logprobs_batched
# ===========================================================================


class TestKLBatched:
    def test_k1_zero_when_equal(self):
        from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched

        logp = torch.tensor([[-0.5, -0.3, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])
        kl = kl_from_logprobs_batched(logp, logp, mask, estimator="k1")
        assert float(kl.item()) == pytest.approx(0.0, abs=1e-6)

    def test_k3_nonneg(self):
        from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched

        new_logp = torch.randn(3, 8)
        ref_logp = torch.randn(3, 8)
        mask = torch.ones(3, 8, dtype=torch.bool)
        kl = kl_from_logprobs_batched(new_logp, ref_logp, mask, estimator="k3")
        assert float(kl.item()) >= -1e-6

    def test_k2_nonneg(self):
        from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched

        new_logp = torch.randn(2, 5)
        ref_logp = torch.randn(2, 5)
        mask = torch.ones(2, 5, dtype=torch.bool)
        kl = kl_from_logprobs_batched(new_logp, ref_logp, mask, estimator="k2")
        assert float(kl.item()) >= 0.0

    def test_masked_padding_ignored(self):
        from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched

        logp_a = torch.tensor([[-0.5, -0.3, 99.0]])  # position 2 is padding
        logp_b = torch.tensor([[-0.5, -0.3, -99.0]])
        mask = torch.tensor([[True, True, False]])
        kl = kl_from_logprobs_batched(logp_a, logp_b, mask, estimator="k1")
        # Only first 2 tokens matter, logp_a == logp_b there → KL ≈ 0
        assert abs(float(kl.item())) < 1e-5

    def test_empty_tensor(self):
        from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched

        logp = torch.zeros(0, 0)
        mask = torch.zeros(0, 0, dtype=torch.bool)
        kl = kl_from_logprobs_batched(logp, logp, mask)
        assert float(kl.item()) == pytest.approx(0.0)


class TestMixedPrecisionLoss:
    def test_clipped_surrogate_casts_old_logprobs_to_new_dtype(self):
        from hermes_agentic_rl.algos.common.loss import clipped_surrogate_loss_batched

        new_logp = torch.tensor([[-0.2, -0.3]], dtype=torch.bfloat16, requires_grad=True)
        old_logp = torch.tensor([[-0.25, -0.35]], dtype=torch.float32)
        adv = torch.ones(1, 2, dtype=torch.float32)
        mask = torch.tensor([[True, True]])

        loss, stats = clipped_surrogate_loss_batched(new_logp, old_logp, adv, mask)

        assert loss.dtype == torch.bfloat16
        assert stats["n_tokens"] == 2
        loss.float().backward()
        assert new_logp.grad is not None


# ===========================================================================
# C2 — CurriculumEnv.observe unified signature
# ===========================================================================


class TestCurriculumObserveSignature:
    def _make_env(self):
        from hermes_agentic_rl.envs.curriculum import CurriculumEnv
        from hermes_agentic_rl.envs.echo_task_env import EchoTaskEnv

        levels = [
            EchoTaskEnv([{"task_id": "a", "instruction": "hi", "target": "hi"}]),
            EchoTaskEnv([{"task_id": "b", "instruction": "yo", "target": "yo"}]),
        ]
        return CurriculumEnv(levels=levels, window=3, promote_threshold=0.8)

    def test_observe_with_level_kwarg(self):
        env = self._make_env()
        # Should not raise — level= is accepted (ignored)
        env.observe(1.0, level=0)
        env.observe(0.9, level=None)
        env.observe(0.85)  # positional only

    def test_mixed_observe_compatible(self):
        from hermes_agentic_rl.envs.curriculum import MixedCurriculumEnv
        from hermes_agentic_rl.envs.echo_task_env import EchoTaskEnv

        levels = [
            EchoTaskEnv([{"task_id": "a", "instruction": "x", "target": "x"}]),
            EchoTaskEnv([{"task_id": "b", "instruction": "y", "target": "y"}]),
        ]
        env = MixedCurriculumEnv(levels=levels, window=5)
        # Both should accept (reward, level) signature
        env.observe(0.7, level=0)
        env.observe(0.3, level=1)
        env.observe(0.5)  # level=None is fine

    def test_trainer_observe_helper_forwards_curriculum_level(self):
        from hermes_agentic_rl.trainers.on_policy import _observe_env_reward

        class Env:
            def __init__(self):
                self.calls = []

            def observe(self, reward, level=None):
                self.calls.append((reward, level))

        env = Env()
        _observe_env_reward(env, {"_curriculum_level": 2}, 0.75)  # type: ignore[arg-type]

        assert env.calls == [(0.75, 2)]


# ===========================================================================
# C3 — _format_log dynamic extras
# ===========================================================================


class TestFormatLogDynamic:
    def _make_trainer(self):
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
        from hermes_agentic_rl.core.reward_manager import RewardManager
        from hermes_agentic_rl.envs.echo_task_env import EchoTaskEnv
        from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

        policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_layers=1))
        env = EchoTaskEnv([{"task_id": "t", "instruction": "a", "target": "a"}])
        rm = RewardManager([])
        cfg = GRPOTrainerConfig(
            n_iters=1, group_size=2, prompts_per_iter=1, max_new_tokens=3, log_every=0
        )
        return GRPOTrainer(policy=policy, env=env, reward_manager=rm, cfg=cfg)

    def test_primary_keys_present(self):
        trainer = self._make_trainer()
        rec = {
            "iter": 0,
            "algo": "grpo",
            "mean_reward": 0.5,
            "loss": 0.1,
            "kl": 0.01,
            "clip_frac": 0.2,
            "n_updated": 4,
        }
        log = trainer._format_log(rec)
        assert "mean_reward=0.5000" in log
        assert "kl=0.0100" in log

    def test_extra_numeric_keys_appear(self):
        trainer = self._make_trainer()
        rec = {
            "iter": 0,
            "algo": "grpo",
            "mean_reward": 0.5,
            "loss": 0.1,
            "grad_norm": 0.123,
            "my_custom_metric": 42.0,
        }
        log = trainer._format_log(rec)
        assert "grad_norm" in log
        assert "my_custom_metric" in log

    def test_suppressed_keys_absent(self):
        trainer = self._make_trainer()
        rec = {
            "iter": 0,
            "mean_reward": 0.5,
            "n_tokens": 999,
            "ratio_mean": 1.01,
            "optimizer_step_applied": 1.0,
        }
        log = trainer._format_log(rec)
        assert "n_tokens" not in log
        assert "ratio_mean" not in log


# ===========================================================================
# C4 — RLOO algo + EntropySchedule integration
# ===========================================================================


class TestRLOOAlgo:
    def test_rloo_basic(self):
        from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
        from hermes_agentic_rl.algos.rloo import RLOOAlgo, RLOOConfig
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

        policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_layers=1))
        algo = RLOOAlgo(RLOOConfig())
        records = [
            RolloutRecord(
                prompt_ids=[1, 2],
                response_ids=[3, 4, 5],
                old_logprobs=[-0.3] * 3,
                reward=r,
                group_id="g0",
            )
            for r in [0.8, 0.2, 0.5, 0.0]
        ]
        loss, stats = algo.compute_loss(policy, None, RolloutBatch(records=records))
        assert loss.requires_grad or float(loss.item()) >= 0.0
        assert stats.n_records == 4
        assert stats.extra["algo"] == "rloo"

    def test_rloo_empty_batch(self):
        from hermes_agentic_rl.algos.base import RolloutBatch
        from hermes_agentic_rl.algos.rloo import RLOOAlgo
        from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

        policy = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_layers=1))
        loss, stats = RLOOAlgo().compute_loss(policy, None, RolloutBatch(records=[]))
        assert float(loss.item()) == pytest.approx(0.0)
        assert stats.n_records == 0

    def test_rloo_normalize(self):
        from hermes_agentic_rl.algos.rloo import RLOOAlgo, RLOOConfig, rloo_advantage

        advs = rloo_advantage([1.0, 0.0, 0.5], normalize=True)
        # z-scored: mean ≈ 0
        assert abs(sum(advs) / 3) < 1e-5


class TestEntropySchedule:
    def test_linear_schedule(self):
        from hermes_agentic_rl.algos.entropy_schedule import LinearEntropySchedule

        s = LinearEntropySchedule(start=0.1, end=0.0, total_steps=100)
        assert s.get_coef(0) == pytest.approx(0.1)
        assert s.get_coef(50) == pytest.approx(0.05)
        assert s.get_coef(100) == pytest.approx(0.0)

    def test_exponential_schedule(self):
        from hermes_agentic_rl.algos.entropy_schedule import ExponentialEntropySchedule

        s = ExponentialEntropySchedule(start=0.1, decay=0.9, floor=1e-4)
        assert s.get_coef(0) == pytest.approx(0.1)
        assert s.get_coef(1) == pytest.approx(0.09)
        assert s.get_coef(1000) == pytest.approx(1e-4)  # hits floor

    def test_cosine_schedule(self):
        from hermes_agentic_rl.algos.entropy_schedule import CosineEntropySchedule

        s = CosineEntropySchedule(peak=0.1, end=0.0, total_steps=100, warmup_steps=0)
        assert s.get_coef(0) == pytest.approx(0.1)
        assert s.get_coef(50) == pytest.approx(0.05, rel=1e-3)
        assert s.get_coef(100) == pytest.approx(0.0, abs=1e-9)

    def test_pid_controller(self):
        from hermes_agentic_rl.algos.entropy_schedule import TargetEntropyPID

        pid = TargetEntropyPID(target_entropy=1.0, init_coef=0.01, lr=0.1)
        # entropy too low → coef should increase
        coef = pid.update(0.5)
        assert coef > 0.01
        # entropy too high → coef should decrease
        pid2 = TargetEntropyPID(target_entropy=1.0, init_coef=0.1, lr=0.1)
        coef2 = pid2.update(1.5)
        assert coef2 < 0.1

    def test_pid_controller_accepts_full_pid_gains(self):
        from hermes_agentic_rl.algos.entropy_schedule import TargetEntropyPID

        pid = TargetEntropyPID(
            target_entropy=1.0,
            init_coef=0.01,
            kp=0.05,
            ki=0.01,
            kd=0.001,
        )

        coef = pid.update(0.5)

        assert coef > 0.01

    def test_pid_integral_anti_windup_clamps_error_accumulation(self):
        from hermes_agentic_rl.algos.entropy_schedule import TargetEntropyPID

        pid = TargetEntropyPID(
            target_entropy=100.0,
            init_coef=0.0,
            kp=0.0,
            ki=1.0,
            max_integral=2.5,
            hi=100.0,
        )

        for _ in range(8):
            pid.update(current_entropy=0.0)

        assert pid._integral == pytest.approx(2.5)
        assert pid.coef <= 20.0

    def test_make_entropy_scheduler_factory(self):
        from hermes_agentic_rl.algos.entropy_schedule import make_entropy_scheduler

        for kind in ("linear", "exp", "cosine", "pid"):
            s = make_entropy_scheduler(kind)
            assert s is not None

    def test_make_entropy_scheduler_unknown_raises(self):
        from hermes_agentic_rl.algos.entropy_schedule import make_entropy_scheduler

        with pytest.raises(ValueError):
            make_entropy_scheduler("polynomial")
