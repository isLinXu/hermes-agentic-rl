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

import torch

from hermes_agentic_rl.algos.base import RolloutBatch
from hermes_agentic_rl.algos.ppo import PPO, PPOConfig
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.base_env import BaseEnv
from hermes_agentic_rl.trainers.on_policy import (
    AgentLoopFactory,
    OnPolicyTrainer,
    TrainStats,
)
from hermes_agentic_rl.trainers.on_policy_config import build_shared_on_policy_config


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
    batch_generate: bool = False
    interleave_sft_every: int = 0
    interleave_sft_samples: int = 32
    interleave_sft_lr: float = 1e-4
    interleave_sft_epochs: int = 1
    interleave_sft_batch_size: int = 8
    bootstrap_sft_rounds: int = 0
    bootstrap_sft_samples: int = 32
    bootstrap_sft_lr: float = 1e-4
    bootstrap_sft_epochs: int = 1
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
    replay_buffer: dict[str, Any] | None = None
    replay_mix_ratio: float = 0.25


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
        shared = build_shared_on_policy_config(cfg)
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

    def _prepare_update_batch(self, batch: RolloutBatch) -> RolloutBatch:
        prepared: list[Any] = []
        for record in batch.records:
            copied = type(record)(
                prompt_ids=list(record.prompt_ids),
                response_ids=list(record.response_ids),
                old_logprobs=list(record.old_logprobs),
                reward=float(record.reward),
                group_id=str(record.group_id),
                metadata=dict(record.metadata),
            )
            if copied.response_ids:
                with torch.no_grad():
                    _logp, _ent, values = self.policy.score_with_value(
                        copied.prompt_ids,
                        copied.response_ids,
                    )
                n = min(len(copied.response_ids), int(values.numel()))
                if n > 0:
                    copied.metadata["_ppo_old_values"] = [
                        float(v) for v in values[-n:].detach().cpu().tolist()
                    ]
            prepared.append(copied)
        return RolloutBatch(records=prepared)
