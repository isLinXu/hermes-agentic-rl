"""P0-3: multi-stream unified training (MixedCurriculumEnv).

Covers weighted mixture sampling, adaptive reweighting, the trainer's
``_observe_env_reward`` level-forwarding helper, end-to-end training that
surfaces per-stream metrics via ``env_snapshot``, and the CLI builder.
"""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.curriculum import MixedCurriculumEnv
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig
from hermes_agentic_rl.trainers.on_policy import _observe_env_reward


def _echo():
    return EchoTaskEnv(build_default_echo_dataset())


def test_weighted_sampling_favours_heavier_stream():
    env = MixedCurriculumEnv([_echo(), _echo()], weights=[9.0, 1.0], seed=0)
    assert env.weights == pytest.approx([0.9, 0.1], abs=1e-9)

    async def _draw(n):
        await env.setup()
        return [(await env.get_next_item())["_curriculum_level"] for _ in range(n)]

    levels = asyncio.run(_draw(300))
    n0 = levels.count(0)
    # ~90% should be stream 0; allow generous slack for RNG.
    assert n0 > 220


def test_observe_attributes_reward_per_stream():
    env = MixedCurriculumEnv([_echo(), _echo()], window=5, adapt_lr=0.3, seed=1)
    base = list(env.weights)
    # Stream 1 keeps failing => its weight should rise (difficulty-prioritised).
    for _ in range(20):
        env.observe(0.0, level=1)
        env.observe(1.0, level=0)
    snap = env.snapshot()
    assert sum(env.weights) == pytest.approx(1.0, abs=1e-6)
    assert env.weights[1] > base[1]
    assert snap["per_level"][1]["count"] == 20
    assert snap["per_level"][1]["recent_mean"] == pytest.approx(0.0)


def test_min_weight_floor_respected():
    env = MixedCurriculumEnv([_echo(), _echo()], window=3, adapt_lr=2.0, min_weight=0.1, seed=2)
    # Hammer stream 0 with successes so its weight is driven down hard.
    for _ in range(50):
        env.observe(1.0, level=0)
    assert min(env.weights) >= 0.1 - 1e-9
    assert sum(env.weights) == pytest.approx(1.0, abs=1e-6)


def test_observe_env_reward_helper_forwards_level():
    seen = []

    class Env:
        def observe(self, reward, level=None):
            seen.append((reward, level))

    _observe_env_reward(Env(), {"_curriculum_level": 1}, 0.5)
    assert seen == [(0.5, 1)]

    # Env whose observe only takes reward still works (TypeError fallback).
    seen2 = []

    class LegacyEnv:
        def observe(self, reward):
            seen2.append(reward)

    _observe_env_reward(LegacyEnv(), {"_curriculum_level": 1}, 0.7)
    assert seen2 == [0.7]

    # Env without observe is a no-op (no exception).
    _observe_env_reward(object(), {"_curriculum_level": 0}, 0.1)


def test_trainer_drives_multi_stream_and_reports_per_stream_metrics():
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    env = MixedCurriculumEnv([_echo(), _echo()], weights=[1.0, 1.0], window=2, seed=0)
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    records = []
    cfg = GRPOTrainerConfig(
        n_iters=3,
        group_size=3,
        prompts_per_iter=2,
        max_new_tokens=4,
        lr=5e-3,
        log_every=100,
        seed=1,
        metrics_sink=records.append,
    )
    trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
    stats = trainer.train()

    assert len(stats.iters) == 3
    last = records[-1]
    assert "env_snapshot" in last
    snap = last["env_snapshot"]
    assert len(snap["per_level"]) == 2
    # 3 iters * 2 prompts = 6 items sampled across the two streams.
    assert snap["total_items"] == 6
    # observe forwarded the level => per-stream counts add up to all rollouts.
    total_counts = sum(p["count"] for p in snap["per_level"])
    assert total_counts == 3 * 2 * 3  # iters * prompts * group_size

    # Per-stream batch metadata is flattened into the iter record. Each iter's
    # 2 prompts * 3 group_size = 6 records are split across the sampled streams;
    # the per-stream counts must sum to the batch size.
    iter_rec = records[0]
    stream_keys = [k for k in iter_rec if k.startswith("stream/") and k.endswith("/count")]
    assert stream_keys, "expected per-stream count keys in the metrics record"
    assert sum(iter_rec[k] for k in stream_keys) == 6.0
    assert iter_rec["n_streams"] >= 1
    # share fractions over the iter's streams sum to 1.0
    share_keys = [k for k in iter_rec if k.startswith("stream/") and k.endswith("/share")]
    assert sum(iter_rec[k] for k in share_keys) == pytest.approx(1.0, abs=1e-9)


def test_cli_build_multi_stream():
    from hermes_agentic_rl.cli.train_rl import _build_env_and_rewards

    cfg = {
        "environment": {
            "type": "multi_stream",
            "window": 10,
            "streams": [
                {"type": "echo", "weight": 3.0},
                {"type": "echo", "weight": 1.0},
            ],
        }
    }
    env, rm = _build_env_and_rewards(cfg)
    assert isinstance(env, MixedCurriculumEnv)
    assert env.weights == pytest.approx([0.75, 0.25], abs=1e-9)
    assert len(env.levels) == 2
    assert hasattr(rm, "evaluate")
