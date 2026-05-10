"""Real GRPO Trainer — thin wrapper over OnPolicyTrainer.

GRPOTrainer(policy, env, reward_manager, cfg)
    cfg: GRPOTrainerConfig (compatible with v0.2 fields)

The actual training loop lives in ``on_policy.OnPolicyTrainer``. This file
only maps GRPO-specific config into the shared skeleton + instantiates the
GRPO algorithm.

Backward compatibility:
  - ``GRPOTrainerConfig`` keeps all v0.2 fields. New fields (loss_agg,
    max_len_for_dr_grpo) are opt-in with safe defaults.
  - ``GRPOTrainer`` retains the public surface: ``.train()``, ``.stats``,
    ``.policy``, ``.ref_policy``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.base_env import BaseEnv
from hermes_agentic_rl.trainers.on_policy import (
    AgentLoopFactory,
    OnPolicyTrainer,
    OnPolicyTrainerConfig,
    TrainStats,
)


@dataclass(slots=True)
class GRPOTrainerConfig:
    # --- shared ---
    n_iters: int = 20
    group_size: int = 4
    prompts_per_iter: int = 2
    lr: float = 1e-3
    max_new_tokens: int = 16
    temperature: float = 1.0
    use_reference: bool = False
    log_every: int = 1
    save_every: int = 0
    output_dir: Path | None = None
    seed: int | None = 0
    multi_turn: bool = False
    grad_clip: float = 1.0
    metrics_sink: Callable[[dict[str, Any]], None] | None = None
    # --- GRPO-specific ---
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    advantage_eps: float = 1e-6
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token"
    max_len_for_dr_grpo: int = 256
    # --- v0.5: per-token advantage (REINFORCE++) ---
    per_token_advantage: bool = False
    advantage_norm: Literal["group", "batch", "whiten"] = "group"
    # --- v0.5: interleaved SFT ---
    interleave_sft_every: int = 0  # 0 = disabled
    interleave_sft_samples: int = 32
    interleave_sft_lr: float = 1e-4
    # --- v0.5: batch generate ---
    batch_generate: bool = False
    # --- v0.6: checkpoint / resume ---
    checkpoint_every: int = 0
    keep_last_checkpoints: int = 3
    resume_from: int | str | None = None
    auto_resume: bool = False
    # --- v0.6: KL estimator ---
    kl_estimator: Literal["k1", "k2", "k3"] = "k1"
    # --- v0.7: best checkpoint + early stopping ---
    save_best_checkpoint: bool = False
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4


# v0.2 kept this as a bespoke TrainStats; re-export the shared one for BC.
GRPOTrainStats = TrainStats


class GRPOTrainer(OnPolicyTrainer):
    """GRPO flavor of the on-policy trainer."""

    algo_name = "grpo"

    def __init__(
        self,
        policy: LLMBackend,
        env: BaseEnv,
        reward_manager: RewardManager,
        cfg: GRPOTrainerConfig | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
        *,
        agent_loop_factory: AgentLoopFactory | None = None,
        rollout_pool: Any = None,
        lagrangian: Any = None,
    ) -> None:
        cfg = cfg or GRPOTrainerConfig()
        shared = OnPolicyTrainerConfig(
            n_iters=cfg.n_iters,
            group_size=cfg.group_size,
            prompts_per_iter=cfg.prompts_per_iter,
            lr=cfg.lr,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            grad_clip=cfg.grad_clip,
            use_reference=cfg.use_reference,
            multi_turn=cfg.multi_turn,
            log_every=cfg.log_every,
            save_every=cfg.save_every,
            output_dir=cfg.output_dir,
            seed=cfg.seed,
            metrics_sink=cfg.metrics_sink,
            checkpoint_every=cfg.checkpoint_every,
            keep_last_checkpoints=cfg.keep_last_checkpoints,
            resume_from=cfg.resume_from,
            auto_resume=cfg.auto_resume,
            save_best_checkpoint=cfg.save_best_checkpoint,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
        )
        algo = GRPO(
            GRPOConfig(
                clip_eps=cfg.clip_eps,
                kl_coef=cfg.kl_coef,
                entropy_coef=cfg.entropy_coef,
                advantage_eps=cfg.advantage_eps,
                loss_agg=cfg.loss_agg,
                max_len_for_dr_grpo=cfg.max_len_for_dr_grpo,
                per_token_advantage=cfg.per_token_advantage,
                advantage_norm=cfg.advantage_norm,
                kl_estimator=cfg.kl_estimator,
            )
        )
        super().__init__(
            policy=policy,
            env=env,
            reward_manager=reward_manager,
            algo=algo,
            cfg=shared,
            agent_loop_factory=agent_loop_factory,
            logger=logger,
            rollout_pool=rollout_pool,
            lagrangian=lagrangian,
        )
        self._grpo_cfg = cfg  # keep public access to config type v0.2 tests expect
