"""Hybrid (GRPO + OPD) Trainer — thin wrapper over OnPolicyTrainer.

OpenClaw-RL §3.3 hybrid objective ``L = w_rl·L_GRPO + w_opd·L_OPD``. Like the
GRPO / GSPO wrappers this only maps config into the shared on-policy skeleton
and instantiates :class:`HybridAlgo`.

The OPD branch needs per-token teacher log-probs. They are produced online by
``rewards.opd_teacher.TeacherLogprobFiller`` when ``opd_teacher_fill=True`` (the
trainer wires the filler from the shared config). Without the filler the OPD
branch is a no-op and the objective degrades to plain GRPO.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hermes_agentic_rl.algos.grpo import GRPOConfig
from hermes_agentic_rl.algos.hybrid import HybridAlgo, HybridConfig
from hermes_agentic_rl.algos.opd import OPDConfig
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


@dataclass(slots=True)
class HybridTrainerConfig:
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
    # --- hybrid weights ---
    w_rl: float = 1.0
    w_opd: float = 1.0
    w_rl_schedule: dict[str, Any] | None = None
    w_opd_schedule: dict[str, Any] | None = None
    # --- GRPO branch ---
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28
    kl_coef: float = 0.02
    entropy_coef: float = 0.0
    advantage_eps: float = 1e-6
    advantage_norm: Literal["group", "batch", "whiten"] = "group"
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token"
    max_len_for_dr_grpo: int = 256
    per_token_advantage: bool = False
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"
    # --- OPD branch ---
    opd_kl_coef: float = 0.02
    opd_clip_eps: float = 0.2
    opd_clip_eps_high: float = 0.28
    opd_adv_diff_clip: float = 1.0
    opd_skip_missing_hints: bool = False
    # --- OPD teacher-logprob closed loop (shared config consumes these) ---
    opd_teacher_fill: bool = True
    opd_hint_template: str = "\n\n[HINT_START]{hint}[HINT_END]\n"
    opd_teacher_max_hint_tokens: int = 128
    opd_capability_axis_weights: dict[str, float] | None = None
    # In-trainer hint extraction (recovers directive hints from next-state).
    opd_hint_extractor: dict[str, Any] | None = None
    # --- interleaved / bootstrap SFT ---
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
    stability_preset: str = "none"
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


HybridTrainStats = TrainStats


class HybridTrainer(OnPolicyTrainer):
    """Hybrid GRPO + OPD flavor of the on-policy trainer."""

    algo_name = "hybrid"

    def __init__(
        self,
        policy: LLMBackend,
        env: BaseEnv,
        reward_manager: RewardManager,
        cfg: HybridTrainerConfig | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
        *,
        agent_loop_factory: AgentLoopFactory | None = None,
        rollout_pool: Any = None,
        lagrangian: Any = None,
    ) -> None:
        cfg = cfg or HybridTrainerConfig()
        shared = build_shared_on_policy_config(cfg)
        algo = HybridAlgo(
            HybridConfig(
                w_rl=cfg.w_rl,
                w_opd=cfg.w_opd,
                grpo=GRPOConfig(
                    clip_eps=cfg.clip_eps,
                    clip_eps_high=cfg.clip_eps_high,
                    kl_coef=cfg.kl_coef,
                    entropy_coef=cfg.entropy_coef,
                    advantage_eps=cfg.advantage_eps,
                    advantage_norm=cfg.advantage_norm,
                    loss_agg=cfg.loss_agg,
                    max_len_for_dr_grpo=cfg.max_len_for_dr_grpo,
                    per_token_advantage=cfg.per_token_advantage,
                    kl_estimator=cfg.kl_estimator,
                ),
                opd=OPDConfig(
                    kl_coef=cfg.opd_kl_coef,
                    clip_eps=cfg.opd_clip_eps,
                    clip_eps_high=cfg.opd_clip_eps_high,
                    adv_diff_clip=cfg.opd_adv_diff_clip,
                    skip_missing_hints=cfg.opd_skip_missing_hints,
                    kl_estimator=cfg.kl_estimator,
                ),
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
        self._hybrid_cfg = cfg
