"""PPO Trainer — thin wrapper over OnPolicyTrainer.

Requires a backend with a value head (``policy.supports_value_head()``).
Most fields mirror ``GRPOTrainerConfig``; the PPO-specific ones cover GAE
and value-loss settings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hermes_agentic_rl.algos.ppo import PPO, PPOConfig
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
class PPOTrainerConfig:
    # shared
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
    # PPO-specific
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    vf_clip_eps: float = 0.2
    entropy_coef: float = 0.0
    kl_coef: float = 0.0
    gamma: float = 1.0
    lam: float = 0.95
    normalize_advantage: bool = True
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token"
    max_len_for_dr_grpo: int = 256
    # --- v0.6: checkpoint / resume ---
    checkpoint_every: int = 0
    keep_last_checkpoints: int = 3
    resume_from: int | str | None = None
    auto_resume: bool = False
    # --- v0.6: KL estimator ---
    kl_estimator: Literal["k1", "k2", "k3"] = "k1"
    # --- v0.7: advantage whitening (clip to ±advantage_clip after norm) ---
    whiten_advantage: bool = False
    advantage_clip: float = 3.0
    # --- v0.7: best checkpoint + early stopping ---
    save_best_checkpoint: bool = False
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4


PPOTrainStats = TrainStats


class PPOTrainer(OnPolicyTrainer):
    """PPO flavor — requires a value-head backend."""

    algo_name = "ppo"

    def __init__(
        self,
        policy: LLMBackend,
        env: BaseEnv,
        reward_manager: RewardManager,
        cfg: PPOTrainerConfig | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
        *,
        agent_loop_factory: AgentLoopFactory | None = None,
        rollout_pool: Any = None,
        lagrangian: Any = None,
    ) -> None:
        cfg = cfg or PPOTrainerConfig()
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
        algo = PPO(
            PPOConfig(
                clip_eps=cfg.clip_eps,
                vf_coef=cfg.vf_coef,
                vf_clip_eps=cfg.vf_clip_eps,
                entropy_coef=cfg.entropy_coef,
                kl_coef=cfg.kl_coef,
                gamma=cfg.gamma,
                lam=cfg.lam,
                normalize_advantage=cfg.normalize_advantage,
                loss_agg=cfg.loss_agg,
                max_len_for_dr_grpo=cfg.max_len_for_dr_grpo,
                kl_estimator=cfg.kl_estimator,
                whiten_advantage=cfg.whiten_advantage,
                advantage_clip=cfg.advantage_clip,
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
        self._ppo_cfg = cfg

    def _validate_backend(self, policy: LLMBackend) -> None:
        if not policy.supports_value_head():
            raise RuntimeError(
                "PPOTrainer requires a value-head backend; use "
                "TinyBackendConfig(with_value_head=True) or switch to GRPOTrainer"
            )
