"""Training Orchestrator — thin coordination layer over OnPolicyTrainer.

Design Rationale
----------------
``OnPolicyTrainer.train()`` (~180 lines) interleaves three concerns:

1. **Execution**: rollout collection, gradient update, reward computation.
2. **Side effects**: EMA, PRM, dynamic balancer, curriculum, SFT, eval,
   checkpoint, early-stop, logging.
3. **Bookkeeping**: stats tracking, best-reward tracking, metrics sink.

The orchestrator extracts concern #2 (side-effect coordination) into a
separate class so the trainer's ``train()`` can focus on the core loop.
This is a **non-breaking** refactor: ``OnPolicyTrainer.train()`` still
works unchanged — the orchestrator is an *opt-in* alternative entry point.

Usage::

    trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
    orchestrator = TrainingOrchestrator(trainer)
    stats = orchestrator.run()

The orchestrator calls ``trainer._one_iter()`` (execution) and then runs
side effects itself, mirroring the exact logic that was in ``train()``.
Over time, ``train()`` can be simplified to delegate here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hermes_agentic_rl.trainers.on_policy import OnPolicyTrainer, TrainStats

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Coordinate side effects around ``OnPolicyTrainer._one_iter()``.

    This class wraps an existing ``OnPolicyTrainer`` (or any subclass like
    ``GRPOTrainer``) and provides an alternative ``run()`` method that
    performs the same training loop as ``trainer.train()`` but with cleaner
    separation of concerns.

    Attributes
    ----------
    trainer : OnPolicyTrainer
        The wrapped trainer instance.
    """

    def __init__(self, trainer: OnPolicyTrainer) -> None:
        self.trainer = trainer
        self._best_reward: float = float("-inf")
        self._best_iter: int = -1
        self._iters_since_best: int = 0
        self._early_stopped: bool = False

    def run(self) -> TrainStats:
        """Execute the full training loop with orchestrated side effects.

        Returns
        -------
        TrainStats
            Aggregated training statistics from the trainer.
        """
        t = self.trainer
        cfg = t.cfg

        # Bootstrap SFT (runs before the main loop, same as train()).
        self._run_bootstrap_sft()

        start = self._get_start_iter()
        last_iter = start

        for it in range(start, cfg.n_iters):
            last_iter = it
            self._sync_vllm_weights(it)
            stats = asyncio.run(t._one_iter(it))
            record = self._build_record(it, stats)
            self._run_side_effects(it, record)
            t.stats.add(record)

            if self._should_early_stop(it):
                break

        self._save_final_checkpoint(last_iter)
        return t.stats

    # ------------------------------------------------------------------
    # Side-effect runners (extracted from OnPolicyTrainer.train())
    # ------------------------------------------------------------------

    def _run_bootstrap_sft(self) -> None:
        """Run bootstrap SFT if configured."""
        t = self.trainer
        if t.cfg.bootstrap_sft_rounds > 0:
            bootstrap_record = t._maybe_run_bootstrap_sft()
            if bootstrap_record and t.cfg.log_every:
                t.logger(bootstrap_record)

    def _get_start_iter(self) -> int:
        """Get the starting iteration (for resume)."""
        t = self.trainer
        if t.cfg.auto_resume and t.cfg.resume_from is None:
            return getattr(t, "_resume_iter", 0)
        if isinstance(t.cfg.resume_from, int):
            return t.cfg.resume_from
        return 0

    def _sync_vllm_weights(self, it: int) -> None:
        """Sync weights to vLLM rollout backend before rollout."""
        t = self.trainer
        if t._vllm_rollout is not None and it > 0:
            sync_every = max(1, int(t.cfg.vllm_sync_every))
            if it % sync_every == 0:
                t._sync_weights_to_vllm(t.policy)

    def _build_record(self, it: int, stats: Any) -> dict[str, Any]:
        """Build the per-iteration record dict."""
        t = self.trainer
        record = {"iter": it, "algo": t.algo_name, **stats.as_dict()}

        # Lagrangian dual step
        if t.lagrangian is not None:
            t.lagrangian.dual_step()
            record["lagrangian"] = t.lagrangian.snapshot()

        # Interleaved SFT
        sft_metrics = t._maybe_run_interleaved_sft(it)
        if sft_metrics:
            record.update(sft_metrics)

        # Env snapshot
        env_snapshot = getattr(t.env, "snapshot", None)
        if callable(env_snapshot):
            try:
                record["env_snapshot"] = env_snapshot()
            except Exception:
                pass

        return record

    def _run_side_effects(self, it: int, record: dict[str, Any]) -> None:
        """Run all per-iteration side effects (EMA, PRM, curriculum, etc.)."""
        self._run_ema_update(record)
        self._run_prm_pipeline(it, record)
        self._run_dynamic_balancer(it, record)
        self._run_curriculum(it, record)
        self._run_memory_shaper(record)
        self._run_ref_update(it, record)
        self._run_eval_hook(it, record)
        self._track_best_reward(it, record)
        self._log_and_checkpoint(it, record)

    def _run_ema_update(self, record: dict[str, Any]) -> None:
        """Update EMA shadow weights."""
        t = self.trainer
        if t._ema is not None:
            t._ema.update(t.policy)
            record["ema_rollout"] = 1.0
            record["ema_tau"] = t._ema.current_tau()

    def _run_prm_pipeline(self, it: int, record: dict[str, Any]) -> None:
        """Run PRM online co-training."""
        t = self.trainer
        if t._prm_pipeline is not None:
            last_batch = getattr(t, "_last_train_batch", None)
            if last_batch is not None:
                tokenizer = getattr(t.policy, "tokenizer", None)
                if tokenizer is not None:
                    try:
                        prm_metrics = t._prm_pipeline.step(it, last_batch, tokenizer=tokenizer)
                        if prm_metrics:
                            record.update(prm_metrics)
                    except Exception as e:
                        record["prm_error"] = str(e)

    def _run_dynamic_balancer(self, it: int, record: dict[str, Any]) -> None:
        """Update dynamic reward component weights."""
        t = self.trainer
        if t._dynamic_balancer is not None:
            try:
                new_weights = t._dynamic_balancer.observe_batch(it, record)
                for comp in getattr(t.reward_manager, "components", []):
                    name = getattr(comp, "name", None)
                    if name is not None and name in new_weights:
                        comp.weight = new_weights[name]
                record.update(t._dynamic_balancer.snapshot())
            except Exception:
                pass

    def _run_curriculum(self, it: int, record: dict[str, Any]) -> None:
        """Observe batch stats and maybe advance curriculum stage."""
        t = self.trainer
        if t._curriculum_scheduler is not None:
            try:
                t._curriculum_scheduler.observe_batch_stats(it, record)
                if t._curriculum_scheduler.should_advance():
                    old_name = t._curriculum_scheduler.get_current_stage().name
                    t._curriculum_scheduler.advance()
                    new_stage = t._curriculum_scheduler.get_current_stage()
                    if new_stage is not None:
                        record["curriculum_advanced"] = 1.0
                        record["curriculum_from"] = old_name
                        record["curriculum_to"] = new_stage.name
                        if t._dynamic_balancer is not None:
                            stage_weights = t._curriculum_scheduler.get_stage_weights()
                            for wname, wval in stage_weights.items():
                                if wname in t._dynamic_balancer.base_weights:
                                    t._dynamic_balancer.base_weights[wname] = wval
                    record.update(t._curriculum_scheduler.snapshot())
            except Exception:
                pass

    def _run_memory_shaper(self, record: dict[str, Any]) -> None:
        """Snapshot memory shaper for logging."""
        t = self.trainer
        if t._memory_shaper is not None:
            record.update(t._memory_shaper.snapshot())

    def _run_ref_update(self, it: int, record: dict[str, Any]) -> None:
        """Periodically re-clone reference policy."""
        t = self.trainer
        ref_every = int(getattr(t.cfg, "ref_update_every", 0))
        if ref_every > 0 and t.ref_policy is not None and it > 0 and it % ref_every == 0:
            if hasattr(t.policy, "clone_frozen"):
                t.ref_policy = t.policy.clone_frozen()
                record["ref_policy_updated"] = 1.0

    def _run_eval_hook(self, it: int, record: dict[str, Any]) -> None:
        """Run periodic evaluation with deterministic decoding."""
        t = self.trainer
        eval_every = int(getattr(t.cfg, "eval_every", 0))
        if eval_every > 0 and it > 0 and it % eval_every == 0:
            eval_record = t._run_eval_hook(it)
            if eval_record:
                record.update(eval_record)

    def _track_best_reward(self, it: int, record: dict[str, Any]) -> None:
        """Track best reward and save best checkpoint."""
        t = self.trainer
        mean_r = float(record.get("mean_reward", 0.0))
        improved = mean_r > (self._best_reward + t.cfg.early_stop_min_delta)
        if improved:
            self._best_reward = mean_r
            self._best_iter = it
            self._iters_since_best = 0
            if t._best_ckpt_manager is not None and hasattr(t.policy, "model"):
                t._save_full_checkpoint(it, manager=t._best_ckpt_manager)
        else:
            self._iters_since_best += 1

    def _log_and_checkpoint(self, it: int, record: dict[str, Any]) -> None:
        """Log, checkpoint, and send to metrics sink."""
        t = self.trainer
        if t.cfg.log_every and (it % t.cfg.log_every == 0):
            t.logger(record)
        if t.cfg.metrics_sink is not None:
            try:
                t.cfg.metrics_sink(record)
            except Exception:
                pass
        if (
            t.cfg.save_every
            and t.cfg.output_dir is not None
            and t.cfg.save_every > 0
            and it > 0
            and it % t.cfg.save_every == 0
        ):
            t._save_checkpoint(it)
        if (
            t._ckpt_manager is not None
            and t.cfg.checkpoint_every > 0
            and it > 0
            and it % t.cfg.checkpoint_every == 0
        ):
            t._save_full_checkpoint(it, background=True)

    def _should_early_stop(self, it: int) -> bool:
        """Check early stopping condition."""
        t = self.trainer
        if t.cfg.early_stop_patience > 0 and self._iters_since_best >= t.cfg.early_stop_patience:
            print(
                f"[orchestrator] early stop at iter={it} "
                f"(best={self._best_reward:.4f} @ iter {self._best_iter}; "
                f"patience={t.cfg.early_stop_patience} exhausted)"
            )
            self._early_stopped = True
            return True
        return False

    def _save_final_checkpoint(self, last_iter: int) -> None:
        """Save final checkpoint and flush async saver."""
        t = self.trainer
        if t._ckpt_manager is not None and t.cfg.n_iters > 0:
            t._save_full_checkpoint(last_iter)
        if t._async_ckpt_saver is not None:
            try:
                t._async_ckpt_saver.flush()
            finally:
                t._async_ckpt_saver.close()
