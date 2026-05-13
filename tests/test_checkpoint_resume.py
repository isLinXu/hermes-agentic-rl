"""Checkpoint + auto_resume end-to-end test for OnPolicyTrainer (GRPO path)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.trainers.checkpoint import CheckpointManager
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


def _make_trainer(out: Path, *, resume_from=None, auto_resume: bool = False, n_iters: int = 4):
    backend = TinyCausalLMBackend(
        TinyBackendConfig(dim=12, n_heads=2, n_layers=1, seed=0)
    )
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    cfg = GRPOTrainerConfig(
        n_iters=n_iters,
        group_size=2,
        prompts_per_iter=1,
        lr=1e-3,
        max_new_tokens=3,
        log_every=100,
        seed=0,
        checkpoint_every=2,
        keep_last_checkpoints=5,
        output_dir=out,
        resume_from=resume_from,
        auto_resume=auto_resume,
    )
    return GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)


def test_checkpoints_saved_on_interval(tmp_path: Path) -> None:
    out = tmp_path / "run1"
    trainer = _make_trainer(out, n_iters=5)
    trainer.train()

    mgr = CheckpointManager(out / "checkpoints")
    iters = mgr.list_checkpoints()
    assert iters, "no checkpoints were saved"
    # With checkpoint_every=2 + final flush, we expect checkpoints for
    # iters {2, 4, 4 (final dup)} — dedup is fine.
    assert any(i >= 2 for i in iters)


def test_auto_resume_continues_from_latest(tmp_path: Path) -> None:
    out = tmp_path / "run2"

    # Phase 1: run 4 iters, produce checkpoints.
    t1 = _make_trainer(out, n_iters=4)
    s1 = t1.train()
    assert len(s1.iters) == 4

    # Phase 2: run with auto_resume=True → should continue, not restart.
    t2 = _make_trainer(out, auto_resume=True, n_iters=6)
    s2 = t2.train()

    # Stats should contain iters from the first run (restored) + the new
    # iterations. Latest ckpt is iter=3 (final flush), so start_iter=4 → 2
    # new iters appended (iters 4, 5).
    iters_indices = [r["iter"] for r in s2.iters]
    # contains the tail of run-1 and fresh iters 4+5
    assert 4 in iters_indices
    assert 5 in iters_indices


def test_resume_from_specific_iter(tmp_path: Path) -> None:
    out = tmp_path / "run3"
    t1 = _make_trainer(out, n_iters=4)
    t1.train()
    mgr = CheckpointManager(out / "checkpoints")
    available = mgr.list_checkpoints()
    assert available
    target = available[0]  # earliest

    t2 = _make_trainer(out, resume_from=target, n_iters=target + 2)
    s2 = t2.train()
    # Training resumed from `target + 1`, so new records include iter=target+1.
    iters_indices = [r["iter"] for r in s2.iters]
    assert (target + 1) in iters_indices


def test_resume_from_missing_iter_raises(tmp_path: Path) -> None:
    out = tmp_path / "run4"
    t1 = _make_trainer(out, n_iters=2)
    t1.train()
    with pytest.raises(RuntimeError):
        _make_trainer(out, resume_from=999, n_iters=3)


def test_auto_resume_no_checkpoint_is_noop(tmp_path: Path) -> None:
    """Auto-resume without any prior checkpoints should NOT error."""
    out = tmp_path / "run5"
    trainer = _make_trainer(out, auto_resume=True, n_iters=2)
    stats = trainer.train()
    assert len(stats.iters) == 2
