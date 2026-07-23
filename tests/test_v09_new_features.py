"""Tests for the new v0.9 additions:
- RLOO (Leave-One-Out baseline)
- EntropySchedulers (linear / exp / cosine / PID)
- PPO asymmetric clip
- PrioritizedReplayBuffer
- MixedCurriculumEnv
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# RLOO — pure-Python advantage
# ---------------------------------------------------------------------------
from hermes_agentic_rl.algos.rloo import rloo_advantage


def test_rloo_advantage_basic():
    # With rewards [0.0, 1.0]:
    # A[0] = 0.0 - 1.0 = -1.0,  A[1] = 1.0 - 0.0 = +1.0
    advs = rloo_advantage([0.0, 1.0], normalize=False)
    assert len(advs) == 2
    assert abs(advs[0] - (-1.0)) < 1e-6
    assert abs(advs[1] - 1.0) < 1e-6


def test_rloo_advantage_group_of_one():
    assert rloo_advantage([0.7]) == [0.0]


def test_rloo_advantage_zero_variance():
    # All rewards equal → all advantages zero
    advs = rloo_advantage([0.5, 0.5, 0.5], normalize=False)
    assert all(abs(a) < 1e-8 for a in advs)


def test_rloo_advantage_three():
    rewards = [0.0, 0.5, 1.0]
    advs = rloo_advantage(rewards, normalize=False)
    # A[0] = 0.0 - (0.5+1.0)/2 = 0.0 - 0.75 = -0.75
    # A[1] = 0.5 - (0.0+1.0)/2 = 0.5 - 0.5  =  0.0
    # A[2] = 1.0 - (0.0+0.5)/2 = 1.0 - 0.25 =  0.75
    assert abs(advs[0] - (-0.75)) < 1e-6
    assert abs(advs[1] - 0.0) < 1e-6
    assert abs(advs[2] - 0.75) < 1e-6


def test_rloo_advantage_normalize():
    advs = rloo_advantage([0.0, 1.0])
    mean_a = sum(advs) / len(advs)
    assert abs(mean_a) < 1e-5
    std_a = (sum((a - mean_a) ** 2 for a in advs) / len(advs)) ** 0.5
    assert abs(std_a - 1.0) < 1e-5


def test_rloo_advantage_default_is_normalized():
    advs = rloo_advantage([0.0, 1.0])
    mean_a = sum(advs) / len(advs)
    assert abs(mean_a) < 1e-5


def test_rloo_algo_end_to_end():
    pytest.importorskip("torch")
    from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
    from hermes_agentic_rl.algos.rloo import RLOOAlgo, RLOOConfig
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

    b = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=2, seed=42))
    records = []
    for i in range(4):
        prompt = b.tokenizer.encode("test prompt")
        out = b.generate(prompt, max_new_tokens=3, temperature=1.0, seed=i)
        records.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=float(i) / 4,
                group_id="g0",
            )
        )

    algo = RLOOAlgo(RLOOConfig(clip_eps=0.2, kl_coef=0.0))
    loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.n_records == 4
    assert isinstance(stats.mean_advantage, float)
    if loss.requires_grad:
        loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in b.trainable_parameters()
    )
    assert has_grad, "no gradient for RLOO"


# ---------------------------------------------------------------------------
# Entropy Schedulers
# ---------------------------------------------------------------------------
from hermes_agentic_rl.algos.entropy_schedule import (
    CosineEntropySchedule,
    ExponentialEntropySchedule,
    LinearEntropySchedule,
    TargetEntropyPID,
    make_entropy_scheduler,
)


def test_linear_entropy_schedule():
    s = LinearEntropySchedule(start=0.1, end=0.0, total_steps=100)
    assert abs(s.get_coef(0) - 0.1) < 1e-9
    assert abs(s.get_coef(100) - 0.0) < 1e-9
    assert abs(s.get_coef(50) - 0.05) < 1e-9


def test_exp_entropy_schedule():
    s = ExponentialEntropySchedule(start=0.1, decay=0.9, floor=1e-5)
    assert abs(s.get_coef(0) - 0.1) < 1e-9
    assert s.get_coef(10) < s.get_coef(0)
    assert s.get_coef(1000) >= 1e-5  # floor respected


def test_cosine_entropy_schedule():
    s = CosineEntropySchedule(peak=0.1, end=0.0, total_steps=100, warmup_steps=10)
    # at step 0: warmup start (0)
    assert abs(s.get_coef(0)) < 1e-9
    # at step 10: peak
    assert abs(s.get_coef(10) - 0.1) < 1e-9
    # at step 100: should be ≈ end=0
    assert abs(s.get_coef(100) - 0.0) < 1e-9
    # monotone after warmup
    assert s.get_coef(20) >= s.get_coef(80)


def test_target_entropy_pid():
    pid = TargetEntropyPID(target_entropy=2.0, init_coef=0.01, lr=0.1)
    # entropy below target → coef should increase
    coef_after = pid.update(current_entropy=1.0)
    assert coef_after > 0.01
    # entropy above target → coef should decrease
    pid2 = TargetEntropyPID(target_entropy=2.0, init_coef=0.1, lr=0.1)
    coef_after2 = pid2.update(current_entropy=3.0)
    assert coef_after2 < 0.1


def test_make_entropy_scheduler_factory():
    for kind in ("linear", "exp", "cosine", "pid"):
        s = make_entropy_scheduler(kind)
        assert s is not None
    with pytest.raises(ValueError):
        make_entropy_scheduler("unknown_kind")


# ---------------------------------------------------------------------------
# PPO asymmetric clip
# ---------------------------------------------------------------------------


def test_ppo_config_has_clip_eps_high():
    from hermes_agentic_rl.algos.ppo import PPOConfig

    cfg = PPOConfig()
    assert hasattr(cfg, "clip_eps_high")
    assert cfg.clip_eps_high >= cfg.clip_eps


def test_ppo_asymmetric_clip_forward():
    pytest.importorskip("torch")
    from hermes_agentic_rl.algos.base import RolloutBatch, RolloutRecord
    from hermes_agentic_rl.algos.ppo import PPO, PPOConfig
    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend

    b = TinyCausalLMBackend(
        TinyBackendConfig(dim=16, n_heads=2, n_layers=2, seed=7, with_value_head=True)
    )
    records = []
    prompt = b.tokenizer.encode("ppo test")
    for i in range(3):
        out = b.generate(prompt, max_new_tokens=3, temperature=1.0, seed=i)
        records.append(
            RolloutRecord(
                prompt_ids=list(prompt),
                response_ids=list(out.response_ids),
                old_logprobs=list(out.logprobs),
                reward=float(i),
                group_id="g0",
            )
        )

    algo = PPO(PPOConfig(clip_eps=0.2, clip_eps_high=0.28, entropy_coef=0.0, kl_coef=0.0))
    _loss, stats = algo.compute_loss(b, None, RolloutBatch(records))
    assert stats.n_records == 3
    assert "value_loss" in stats.extra


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------
from hermes_agentic_rl.offline.per_buffer import (
    PrioritizedReplayBuffer,
    TrainSample,
)


def _make_sample(reward: float = 1.0) -> TrainSample:
    return TrainSample(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        reward=reward,
        advantage=reward,
    )


def test_per_buffer_add_and_sample():
    buf = PrioritizedReplayBuffer(maxlen=100, alpha=0.6, beta=0.4)
    for i in range(10):
        buf.add_sample(_make_sample(float(i)), priority=float(i + 1))
    assert len(buf) == 10
    indices, samples, weights = buf.sample_prioritized(batch_size=5)
    assert len(indices) == 5
    assert len(samples) == 5
    assert len(weights) == 5
    assert all(0.0 <= w <= 1.0 for w in weights)


def test_per_buffer_update_priorities():
    buf = PrioritizedReplayBuffer(alpha=0.6, beta=0.4)
    for i in range(5):
        buf.add_sample(_make_sample(1.0), priority=1.0)
    indices, _, _ = buf.sample_prioritized(batch_size=5)
    old_max = buf._max_priority
    buf.update_priorities(indices, td_errors=[10.0] * len(indices))
    assert buf._max_priority > old_max


def test_per_buffer_maxlen_eviction():
    buf = PrioritizedReplayBuffer(maxlen=3, alpha=0.6, beta=0.4)
    for i in range(5):
        buf.add_sample(_make_sample(float(i)))
    assert len(buf) == 3
    assert len(buf._priorities) == 3


def test_per_buffer_snapshot_stats():
    buf = PrioritizedReplayBuffer(alpha=0.6, beta=0.4)
    stats = buf.snapshot_stats()
    assert stats["size"] == 0
    buf.add_sample(_make_sample(1.0), priority=2.0)
    stats = buf.snapshot_stats()
    assert stats["size"] == 1
    assert stats["max_priority"] == 2.0


def test_per_buffer_anneal_beta():
    buf = PrioritizedReplayBuffer(beta=0.4)
    new_beta = buf.anneal_beta(step=500, total_steps=1000, beta_end=1.0)
    assert new_beta > 0.4
    assert new_beta <= 1.0


# ---------------------------------------------------------------------------
# MixedCurriculumEnv
# ---------------------------------------------------------------------------
import asyncio

from hermes_agentic_rl.envs.echo_task_env import EchoTaskEnv, build_default_echo_dataset


def _echo():
    return EchoTaskEnv(build_default_echo_dataset())


def test_mixed_curriculum_weight_normalization():
    from hermes_agentic_rl.envs.curriculum import MixedCurriculumEnv

    levels = [_echo(), _echo(), _echo()]
    env = MixedCurriculumEnv(levels, weights=[1.0, 2.0, 1.0])
    total = sum(env.weights)
    assert abs(total - 1.0) < 1e-9


def test_mixed_curriculum_unequal_weights_rejected():
    from hermes_agentic_rl.envs.curriculum import MixedCurriculumEnv

    with pytest.raises(ValueError):
        MixedCurriculumEnv([_echo()], weights=[0.5, 0.5])


def test_mixed_curriculum_observe_updates_weights():
    from hermes_agentic_rl.envs.curriculum import MixedCurriculumEnv

    levels = [_echo(), _echo()]
    env = MixedCurriculumEnv(levels, window=5, adapt_lr=0.2, min_weight=0.05)
    for _ in range(10):
        env.observe(1.0, level=0)
    # After many successes on level 0, weight distribution should shift
    assert sum(env.weights) == pytest.approx(1.0, abs=1e-6)


def test_mixed_curriculum_snapshot():
    from hermes_agentic_rl.envs.curriculum import MixedCurriculumEnv

    env = MixedCurriculumEnv([_echo(), _echo()])
    snap = env.snapshot()
    assert "weights" in snap
    assert "per_level" in snap
    assert len(snap["per_level"]) == 2


def test_mixed_curriculum_get_next_item():
    from hermes_agentic_rl.envs.curriculum import MixedCurriculumEnv

    env = MixedCurriculumEnv([_echo(), _echo()])

    async def _run():
        await env.setup()
        item = await env.get_next_item()
        return item

    item = asyncio.run(_run())
    assert "_curriculum_level" in item
    assert item["_curriculum_level"] in (0, 1)
