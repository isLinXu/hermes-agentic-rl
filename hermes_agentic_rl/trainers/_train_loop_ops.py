"""Per-iteration side-effect operations extracted from ``on_policy.py``.

The ``train()`` method had ~190 lines of inline side-effects per iteration
(EMA update, PRM co-training, dynamic reward balancer, curriculum scheduler,
memory shaper, reference policy re-clone, eval hook, best-reward tracking,
checkpointing, early-stop). This module packages each into a focused
function so ``train()`` stays readable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes_agentic_rl.trainers.on_policy import OnPolicyTrainer


def run_ema_update(trainer: OnPolicyTrainer, record: dict[str, Any]) -> None:
    """Update EMA shadow weights after the learner step."""
    if trainer._ema is not None:
        trainer._ema.update(trainer.policy)
        record["ema_rollout"] = 1.0
        record["ema_tau"] = trainer._ema.current_tau()


def run_prm_cotraining(
    trainer: OnPolicyTrainer, it: int, record: dict[str, Any]
) -> None:
    """PRM online co-training step using the latest rollout batch."""
    if trainer._prm_pipeline is not None:
        last_batch = getattr(trainer, "_last_train_batch", None)
        if last_batch is not None:
            tokenizer = getattr(trainer.policy, "tokenizer", None)
            if tokenizer is not None:
                try:
                    prm_metrics = trainer._prm_pipeline.step(it, last_batch, tokenizer=tokenizer)
                    if prm_metrics:
                        record.update(prm_metrics)
                except Exception as exc:
                    record["prm_error"] = str(exc)


def run_dynamic_balancer(trainer: OnPolicyTrainer, it: int, record: dict[str, Any]) -> None:
    """Dynamic reward balancer: update component weights based on per-batch variance."""
    if trainer._dynamic_balancer is not None:
        try:
            new_weights = trainer._dynamic_balancer.observe_batch(it, record)
            for comp in getattr(trainer.reward_manager, "components", []):
                name = getattr(comp, "name", None)
                if name is not None and name in new_weights:
                    comp.weight = new_weights[name]
            record.update(trainer._dynamic_balancer.snapshot())
        except Exception:
            pass


def run_curriculum_scheduler(trainer: OnPolicyTrainer, it: int, record: dict[str, Any]) -> None:
    """Curriculum scheduler: observe batch stats and maybe advance stage."""
    if trainer._curriculum_scheduler is not None:
        try:
            trainer._curriculum_scheduler.observe_batch_stats(it, record)
            if trainer._curriculum_scheduler.should_advance():
                old_name = trainer._curriculum_scheduler.get_current_stage().name
                trainer._curriculum_scheduler.advance()
                new_stage = trainer._curriculum_scheduler.get_current_stage()
                if new_stage is not None:
                    record["curriculum_advanced"] = 1.0
                    record["curriculum_from"] = old_name
                    record["curriculum_to"] = new_stage.name
                    if trainer._dynamic_balancer is not None:
                        stage_weights = trainer._curriculum_scheduler.get_stage_weights()
                        for wname, wval in stage_weights.items():
                            if wname in trainer._dynamic_balancer.base_weights:
                                trainer._dynamic_balancer.base_weights[wname] = wval
            record.update(trainer._curriculum_scheduler.snapshot())
        except Exception:
            pass


def run_memory_shaper_snapshot(trainer: OnPolicyTrainer, record: dict[str, Any]) -> None:
    """Memory shaper snapshot for logging."""
    if trainer._memory_shaper is not None:
        record.update(trainer._memory_shaper.snapshot())


def run_ref_policy_reclone(trainer: OnPolicyTrainer, it: int, record: dict[str, Any]) -> None:
    """Reference policy periodic re-clone (prevents KL drift)."""
    ref_every = int(getattr(trainer.cfg, "ref_update_every", 0))
    if ref_every > 0 and trainer.ref_policy is not None and it > 0 and it % ref_every == 0:
        if hasattr(trainer.policy, "clone_frozen"):
            trainer.ref_policy = trainer.policy.clone_frozen()  # type: ignore[attr-defined]
            record["ref_policy_updated"] = 1.0


def run_eval_hook(
    trainer: OnPolicyTrainer, it: int, record: dict[str, Any]
) -> None:
    """Eval hook: periodically evaluate with deterministic decoding."""
    eval_every = int(getattr(trainer.cfg, "eval_every", 0))
    if eval_every > 0 and it > 0 and it % eval_every == 0:
        eval_record = trainer._run_eval_hook(it)
        if eval_record:
            record.update(eval_record)


def update_best_tracking(
    trainer: OnPolicyTrainer, it: int, record: dict[str, Any]
) -> bool:
    """Best-reward tracking + best checkpoint. Returns True if improved."""
    mean_r = float(record.get("mean_reward", 0.0))
    improved = mean_r > (trainer._best_reward + trainer.cfg.early_stop_min_delta)
    if improved:
        trainer._best_reward = mean_r
        trainer._best_iter = it
        trainer._iters_since_best = 0
        if trainer._best_ckpt_manager is not None and hasattr(trainer.policy, "model"):
            from hermes_agentic_rl.trainers._checkpoint_ops import save_full_checkpoint

            save_full_checkpoint(trainer, it, manager=trainer._best_ckpt_manager)
    else:
        trainer._iters_since_best += 1
    return improved


def run_logging(trainer: OnPolicyTrainer, it: int, record: dict[str, Any]) -> None:
    """Log to logger, metrics sink, and profiler output."""
    if trainer.cfg.log_every and (it % trainer.cfg.log_every == 0):
        trainer.logger(record)
    if trainer.cfg.metrics_sink is not None:
        try:
            trainer.cfg.metrics_sink(record)
        except Exception:
            pass
    if trainer._profiler.enabled and trainer._profile_output_path is not None:
        from hermes_agentic_rl.trainers.profiling import append_jsonl

        profile_record = {
            "iter": it,
            "algo": trainer.algo_name,
            **{
                k: v
                for k, v in record.items()
                if isinstance(k, str) and k.startswith("time_")
            },
        }
        append_jsonl(trainer._profile_output_path, profile_record)


def run_periodic_checkpoints(trainer: OnPolicyTrainer, it: int) -> None:
    """Save lightweight + full checkpoints on their respective schedules."""
    if (
        trainer.cfg.save_every
        and trainer.cfg.output_dir is not None
        and trainer.cfg.save_every > 0
        and it > 0
        and it % trainer.cfg.save_every == 0
    ):
        from hermes_agentic_rl.trainers._checkpoint_ops import save_checkpoint

        save_checkpoint(trainer, it)
    if (
        trainer._ckpt_manager is not None
        and trainer.cfg.checkpoint_every > 0
        and it > 0
        and it % trainer.cfg.checkpoint_every == 0
    ):
        from hermes_agentic_rl.trainers._checkpoint_ops import save_full_checkpoint

        save_full_checkpoint(trainer, it, background=True)


def check_early_stop(trainer: OnPolicyTrainer, it: int) -> bool:
    """Return True if early-stop should fire."""
    if (
        trainer.cfg.early_stop_patience > 0
        and trainer._iters_since_best >= trainer.cfg.early_stop_patience
    ):
        print(
            f"[train] early stop at iter={it} "
            f"(best={trainer._best_reward:.4f} @ iter {trainer._best_iter}; "
            f"patience={trainer.cfg.early_stop_patience} exhausted)"
        )
        trainer._early_stopped = True
        return True
    return False


def run_final_checkpoint(trainer: OnPolicyTrainer, last_iter: int) -> None:
    """Emit a final checkpoint + flush async saver."""
    if trainer._ckpt_manager is not None and trainer.cfg.n_iters > int(
        getattr(trainer, "_start_iter", 0)
    ):
        from hermes_agentic_rl.trainers._checkpoint_ops import save_full_checkpoint

        save_full_checkpoint(trainer, last_iter)
    if trainer._async_ckpt_saver is not None:
        try:
            trainer._async_ckpt_saver.flush()
        finally:
            trainer._async_ckpt_saver.close()


def run_all_post_iter(
    trainer: OnPolicyTrainer, it: int, record: dict[str, Any]
) -> bool:
    """Execute all per-iteration side-effects. Returns True if early-stopped."""
    run_ema_update(trainer, record)
    run_prm_cotraining(trainer, it, record)
    run_dynamic_balancer(trainer, it, record)
    run_curriculum_scheduler(trainer, it, record)
    run_memory_shaper_snapshot(trainer, record)
    run_ref_policy_reclone(trainer, it, record)
    run_eval_hook(trainer, it, record)
    update_best_tracking(trainer, it, record)
    run_logging(trainer, it, record)
    run_periodic_checkpoints(trainer, it)
    return bool(check_early_stop(trainer, it))
