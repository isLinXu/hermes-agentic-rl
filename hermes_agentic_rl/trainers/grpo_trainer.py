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

import logging
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
    TrainStats,
)
from hermes_agentic_rl.trainers.on_policy_config import (
    build_shared_on_policy_config,
)

logger = logging.getLogger(__name__)


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
    log_format: str = "text"
    save_every: int = 0
    output_dir: Path | None = None
    seed: int | None = 0
    multi_turn: bool = False
    multi_turn_credit: dict[str, Any] | None = None
    grad_clip: float = 1.0
    metrics_sink: Callable[[dict[str, Any]], None] | None = None
    profile: bool = False
    profile_output_path: Path | None = None
    update_epochs: int = 1
    minibatch_size: int = 0
    shuffle_minibatches: bool = True
    # --- GRPO-specific ---
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28
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
    interleave_sft_epochs: int = 1
    interleave_sft_batch_size: int = 8
    bootstrap_sft_rounds: int = 0
    bootstrap_sft_samples: int = 32
    bootstrap_sft_lr: float = 1e-4
    bootstrap_sft_epochs: int = 1
    # --- v0.5: batch generate ---
    batch_generate: bool = False
    # --- v0.6: checkpoint / resume ---
    checkpoint_every: int = 0
    keep_last_checkpoints: int = 3
    resume_from: int | str | None = None
    auto_resume: bool = False
    # --- v0.6: KL estimator ---
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"
    # --- v0.7: best checkpoint + early stopping ---
    save_best_checkpoint: bool = False
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4
    # --- v0.8/v0.9 shared trainer controls ---
    target_kl: float = 0.0
    adaptive_kl: bool = False
    adaptive_kl_horizon: float = 10000.0
    adaptive_kl_min: float = 1e-4
    adaptive_kl_max: float = 10.0
    normalize_reward: bool = False
    reward_norm_clip: float = 10.0
    amp_dtype: str = "fp32"
    grad_accum_steps: int = 1
    vllm_rollout_model: str | None = None
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = 4096
    vllm_gpu_memory_utilization: float = 0.90
    vllm_enable_prefix_caching: bool = True
    vllm_sync_every: int = 1
    distributed_strategy: str = "none"
    fsdp_cpu_offload: bool = False
    flash_attention: bool = False
    gradient_checkpointing: bool = False
    async_checkpoint: bool = False
    stability_preset: str = "none"
    token_budget: dict[str, Any] | None = None
    entropy_schedule: dict[str, Any] | None = None
    use_ema_rollout: bool = False
    ema_tau: float = 0.005
    ema_tau_start: float | None = None
    ema_tau_warmup_steps: int = 0
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    lr_warmup_start_lr: float = 0.0
    lr_total_steps: int = 0
    lr_end_lr: float = 0.0
    pipeline_rollouts: bool = False
    replay_buffer: dict[str, Any] | None = None
    replay_mix_ratio: float = 0.25
    # --- Curriculum scheduler (forwarded to OnPolicyTrainerConfig) ---
    # When set, OnPolicyTrainer creates an internal CurriculumScheduler
    # and runs observe/advance inside the training loop (on_policy.py:594-620, 1457-1479).
    curriculum: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Fatal cross-field validation — catches misconfigurations that
        would otherwise silently produce garbage gradients or crashes
        deep in the training loop.

        Non-fatal warnings are handled by ``validate_on_policy_config``;
        the checks here are *hard* constraints that make the config
        objectively invalid.
        """
        # GRPO requires group_size >= 2 for group normalization.
        # Warn (don't raise) so that single-sample unit tests and debug
        # configs still work — the trainer will produce a warning at runtime.
        if self.group_size < 2:
            import warnings

            warnings.warn(
                f"GRPOTrainerConfig.group_size={self.group_size} < 2: "
                "GRPO requires group_size >= 2 for group-normalized advantages. "
                "Advantages will be zero — use only for testing/debugging.",
                stacklevel=3,
            )

        # EMA rollout and vLLM rollout are mutually exclusive.
        if self.use_ema_rollout and self.vllm_rollout_model:
            raise ValueError(
                "GRPOTrainerConfig: use_ema_rollout=True is incompatible with "
                "vllm_rollout_model — EMA shadow cannot sync to a separate vLLM process"
            )

        # Learning rate must be positive.
        if self.lr <= 0:
            raise ValueError(
                f"GRPOTrainerConfig.lr={self.lr} must be positive"
            )

        # clip_eps must be in a sane range.
        if not (0 < self.clip_eps < 1.0):
            raise ValueError(
                f"GRPOTrainerConfig.clip_eps={self.clip_eps} must be in (0, 1)"
            )

        # If DR-GPPO loss aggregation is used, max_len_for_dr_grpo must be > 0.
        if self.loss_agg == "dr_grpo" and self.max_len_for_dr_grpo <= 0:
            raise ValueError(
                "GRPOTrainerConfig: loss_agg='dr_grpo' requires "
                "max_len_for_dr_grpo > 0"
            )

        # Gradient accumulation steps must be positive.
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"GRPOTrainerConfig.grad_accum_steps={self.grad_accum_steps} "
                "must be >= 1"
            )

        # Log non-fatal warnings from the shared validator.
        from hermes_agentic_rl.trainers.on_policy_config import (
            validate_on_policy_config,
        )
        shared = build_shared_on_policy_config(self)
        for warning in validate_on_policy_config(shared):
            logger.warning(warning)


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
        shared = build_shared_on_policy_config(cfg)
        algo = GRPO(
            GRPOConfig(
                clip_eps=cfg.clip_eps,
                clip_eps_high=cfg.clip_eps_high,
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
