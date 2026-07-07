"""Checkpoint save / resume operations extracted from ``on_policy.py``.

Keeps the ``OnPolicyTrainer`` thin by moving the ~260-line checkpoint
save / load / resume logic into stateless helper functions that operate
on the trainer's attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from hermes_agentic_rl.trainers._rollout_helpers import config_to_dict as _config_to_dict

if TYPE_CHECKING:
    from hermes_agentic_rl.trainers.on_policy import OnPolicyTrainer


def save_full_checkpoint(
    trainer: OnPolicyTrainer,
    it: int,
    manager: Any | None = None,
    *,
    background: bool = False,
) -> None:
    """Save a {model, optimizer, rng, stats} bundle via CheckpointManager."""
    target_mgr = manager or trainer._ckpt_manager
    if target_mgr is None or not hasattr(trainer.policy, "model"):
        return
    from hermes_agentic_rl.trainers.checkpoint import (
        CheckpointState,
        capture_rng_state,
        snapshot_state_to_cpu,
    )

    state = CheckpointState(
        iteration=it,
        model_state=trainer.policy.model.state_dict(),  # type: ignore[attr-defined]
        optimizer_state=trainer._optim.state_dict(),
        rng_state=capture_rng_state(),
        stats=list(trainer.stats.iters),
        config=_config_to_dict(trainer.cfg),
        best_reward=trainer._best_reward,
        best_iteration=trainer._best_iter,
        running_stats=(
            trainer._reward_rms.state_dict() if trainer._reward_rms is not None else None
        ),
        kl_ctrl_state=(
            trainer._kl_ctrl.state_dict() if trainer._kl_ctrl is not None else None
        ),
        ema_state=(
            {
                k: v.detach().cpu()
                for k, v in trainer._ema.shadow.model.state_dict().items()  # type: ignore[attr-defined]
            }
            if trainer._ema is not None and hasattr(trainer._ema.shadow, "model")
            else None
        ),
        prm_head_state=(
            trainer._prm_pipeline.prm.head.state_dict()
            if trainer._prm_pipeline is not None
            else None
        ),
        curriculum_state=(
            trainer._curriculum_scheduler.state_dict()
            if trainer._curriculum_scheduler is not None
            else None
        ),
    )
    use_async = (
        background
        and trainer.cfg.async_checkpoint
        and trainer._async_ckpt_saver is not None
        and manager is None
    )
    if use_async:
        saver = trainer._async_ckpt_saver
        assert saver is not None
        saver.submit(target_mgr, snapshot_state_to_cpu(state))
    else:
        target_mgr.save(state)


def maybe_resume(trainer: OnPolicyTrainer) -> None:
    """Restore from the latest checkpoint if auto-resume is enabled."""
    if trainer._ckpt_manager is None:
        return
    from hermes_agentic_rl.trainers.checkpoint import (
        CheckpointState,
        restore_rng_state,
    )

    target: CheckpointState | None = None
    resume_from = trainer.cfg.resume_from
    if resume_from is not None and resume_from != "latest":
        try:
            target = trainer._ckpt_manager.load(int(resume_from))
        except Exception:
            target = None
        if target is None:
            raise RuntimeError(
                f"resume_from={resume_from!r} requested but checkpoint not found"
            )
    elif resume_from == "latest" or trainer.cfg.auto_resume:
        target = trainer._ckpt_manager.load_latest()
        if target is None:
            return
    else:
        return

    if not hasattr(trainer.policy, "model"):
        return
    trainer.policy.model.load_state_dict(target.model_state)  # type: ignore[attr-defined]
    if target.optimizer_state is not None:
        try:
            trainer._optim.load_state_dict(target.optimizer_state)
        except Exception:
            pass
    if target.rng_state is not None:
        try:
            restore_rng_state(target.rng_state)
        except Exception:
            pass
    trainer.stats.iters = list(target.stats)
    trainer._best_reward = float(target.best_reward)
    trainer._best_iter = int(target.best_iteration)
    trainer._start_iter = int(target.iteration) + 1

    running_stats = getattr(target, "running_stats", None)
    if running_stats is not None and trainer._reward_rms is not None:
        try:
            trainer._reward_rms.load_state_dict(running_stats)
        except Exception:
            pass

    kl_ctrl_state = getattr(target, "kl_ctrl_state", None)
    if kl_ctrl_state is not None and trainer._kl_ctrl is not None:
        try:
            trainer._kl_ctrl.load_state_dict(kl_ctrl_state)
        except Exception:
            pass

    ema_state = getattr(target, "ema_state", None)
    if (
        ema_state is not None
        and trainer._ema is not None
        and hasattr(trainer._ema.shadow, "model")
    ):
        try:
            trainer._ema.shadow.model.load_state_dict(ema_state, strict=False)  # type: ignore[attr-defined]
        except Exception:
            pass

    prm_head_state = getattr(target, "prm_head_state", None)
    if prm_head_state is not None and trainer._prm_pipeline is not None:
        try:
            trainer._prm_pipeline.prm.head.load_state_dict(prm_head_state)
        except Exception:
            pass

    curriculum_state = getattr(target, "curriculum_state", None)
    if curriculum_state is not None and trainer._curriculum_scheduler is not None:
        try:
            trainer._curriculum_scheduler.load_state_dict(curriculum_state)
        except Exception:
            pass

    print(
        f"[train] resumed from iter={target.iteration} "
        f"(best_reward={trainer._best_reward:.4f} start_iter={trainer._start_iter})"
    )


def save_checkpoint(trainer: OnPolicyTrainer, it: int) -> None:
    """Save a lightweight policy-only checkpoint (legacy format)."""
    if trainer.cfg.output_dir is None:
        return
    out = Path(trainer.cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"policy_iter_{it:04d}.pt"
    if hasattr(trainer.policy, "model"):
        torch.save(trainer.policy.model.state_dict(), target)  # type: ignore[attr-defined]
