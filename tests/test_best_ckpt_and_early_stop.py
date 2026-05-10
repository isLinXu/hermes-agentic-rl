"""v0.7: best_checkpoint separate dir + early-stop semantics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


def _make_trainer(tmp_path, **overrides):
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    cfg = GRPOTrainerConfig(
        n_iters=overrides.pop("n_iters", 6),
        group_size=2,
        prompts_per_iter=1,
        lr=1e-3,
        max_new_tokens=4,
        temperature=1.0,
        log_every=100,
        seed=0,
        output_dir=tmp_path,
        checkpoint_every=2,
        **overrides,
    )
    return GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)


def test_save_best_checkpoint_creates_separate_dir(tmp_path):
    trainer = _make_trainer(tmp_path, save_best_checkpoint=True)
    trainer.train()

    regular = tmp_path / "checkpoints"
    best = tmp_path / "checkpoints_best"
    assert regular.is_dir()
    assert best.is_dir()
    # Best dir must contain exactly the iter that scored highest — and NEVER
    # get pruned (keep_last=0 for the best manager).
    best_iters = sorted(best.iterdir())
    assert len(best_iters) >= 1
    # Best reward must correspond to one of the persisted dirs.
    assert any(str(trainer._best_iter).zfill(5) in p.name for p in best_iters)


def test_save_best_checkpoint_disabled_by_default(tmp_path):
    trainer = _make_trainer(tmp_path)  # save_best_checkpoint defaults to False
    trainer.train()
    assert (tmp_path / "checkpoints").is_dir()
    assert not (tmp_path / "checkpoints_best").exists()


def test_early_stop_halts_when_no_improvement(tmp_path):
    # Use a tiny patience so test converges fast; min_delta large enough that
    # repeated iters on identical task won't register as "improvement".
    trainer = _make_trainer(
        tmp_path,
        n_iters=20,
        early_stop_patience=2,
        early_stop_min_delta=0.5,  # any realistic delta won't exceed this
    )
    stats = trainer.train()
    # We expect fewer than n_iters records because early-stop triggered.
    assert len(stats.iters) < 20
    assert trainer._early_stopped is True


def test_early_stop_respects_patience_threshold(tmp_path):
    # patience=0 must NOT early-stop regardless of rewards.
    trainer = _make_trainer(tmp_path, n_iters=4, early_stop_patience=0)
    stats = trainer.train()
    assert len(stats.iters) == 4
    assert trainer._early_stopped is False
