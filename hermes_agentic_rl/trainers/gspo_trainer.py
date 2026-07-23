"""GSPO Trainer — thin wrapper over OnPolicyTrainer.

GSPO shares the same trainer-level controls as GRPO, but swaps the algorithm
object for sequence-level ratio optimization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hermes_agentic_rl.algos.gspo import GSPO, GSPOConfig
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
from hermes_agentic_rl.trainers.stability import StabilityPreset


@dataclass(slots=True)
class GSPOTrainerConfig:
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
    multi_turn_credit: dict[str, Any] | None = None
    grad_clip: float = 1.0
    metrics_sink: Callable[[dict[str, Any]], None] | None = None
    update_epochs: int = 1
    minibatch_size: int = 0
    shuffle_minibatches: bool = True
    # --- GSPO-specific ---
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    advantage_eps: float = 1e-6
    advantage_norm: Literal["group", "dapo", "batch", "whiten", "none"] = "group"
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"
    log_ratio_clip: float = 40.0
    # --- interleaved SFT ---
    interleave_sft_every: int = 0
    interleave_sft_samples: int = 32
    interleave_sft_lr: float = 1e-4
    interleave_sft_epochs: int = 1
    interleave_sft_batch_size: int = 8
    bootstrap_sft_rounds: int = 0
    bootstrap_sft_samples: int = 32
    bootstrap_sft_lr: float = 1e-4
    bootstrap_sft_epochs: int = 1
    # --- batch generation / checkpointing ---
    batch_generate: bool = False
    checkpoint_every: int = 0
    keep_last_checkpoints: int = 3
    resume_from: int | str | None = None
    auto_resume: bool = False
    save_best_checkpoint: bool = False
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4
    # --- shared trainer controls ---
    target_kl: float = 0.0
    adaptive_kl: bool = False
    adaptive_kl_horizon: float = 10000.0
    adaptive_kl_min: float = 1e-4
    adaptive_kl_max: float = 10.0
    normalize_reward: bool = False
    reward_norm_clip: float = 10.0
    stability_preset: StabilityPreset = "none"
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
    token_budget: dict[str, Any] | None = None
    entropy_schedule: dict[str, Any] | None = None
    use_ema_rollout: bool = False
    ema_tau: float = 0.005
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    lr_warmup_start_lr: float = 0.0
    lr_total_steps: int = 0
    lr_end_lr: float = 0.0


GSPOTrainStats = TrainStats


class GSPOTrainer(OnPolicyTrainer):
    """GSPO flavor of the on-policy trainer."""

    algo_name = "gspo"

    def __init__(
        self,
        policy: LLMBackend,
        env: BaseEnv,
        reward_manager: RewardManager,
        cfg: GSPOTrainerConfig | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
        *,
        agent_loop_factory: AgentLoopFactory | None = None,
        rollout_pool: Any = None,
        lagrangian: Any = None,
    ) -> None:
        cfg = cfg or GSPOTrainerConfig()
        shared = build_shared_on_policy_config(cfg)
        algo = GSPO(
            GSPOConfig(
                clip_eps=cfg.clip_eps,
                clip_eps_high=cfg.clip_eps_high,
                kl_coef=cfg.kl_coef,
                entropy_coef=cfg.entropy_coef,
                advantage_eps=cfg.advantage_eps,
                advantage_norm=cfg.advantage_norm,
                kl_estimator=cfg.kl_estimator,
                log_ratio_clip=cfg.log_ratio_clip,
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
        self._gspo_cfg = cfg

    def _validate_backend(self, policy: LLMBackend) -> None:
        """GSPO requires a trainable policy backend with score_batch() support."""
        cls_name = type(policy).__name__
        if not policy.is_trainable():
            raise RuntimeError(
                f"{cls_name} is not trainable. GSPOTrainer requires a backend "
                f"with trainable_parameters() returning non-empty list."
            )
