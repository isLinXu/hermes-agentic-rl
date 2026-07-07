"""Shared on-policy trainer base class.

Both GRPO and PPO follow the same skeleton:

    for iter in range(n_iters):
        batch = collect_group_rollouts(policy, env, G=group_size)
        rewards = reward_manager.evaluate(batch)
        loss, stats = algo.compute_loss(policy, ref_policy, batch)
        optim.zero_grad(); loss.backward(); optim.step()
        log(stats); maybe_save_checkpoint()

The only things that vary are:
  - The `Algo` object (GRPO vs PPO, each with its own config)
  - Whether the backend needs a value head
  - Per-turn record generation (optional, enabled by `agent_loop_factory`
    that returns a MultiTurnAgentLoop)

This module extracts that skeleton so GRPOTrainer / PPOTrainer become thin
subclasses.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
import torch.nn.functional as F

from hermes_agentic_rl.agent_loop.base import BaseAgentLoop
from hermes_agentic_rl.agent_loop.policy_loop import PolicyAgentLoop
from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
)
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.backends.batch_generate import BatchRolloutGenerator
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.mdp.state_encoder import PromptStateEncoder
from hermes_agentic_rl.trainers._rollout_helpers import (
    batch_single_turn_trajectory as _batch_single_turn_trajectory,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    config_to_dict as _config_to_dict,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    extract_next_state as _extract_next_state,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    extract_rl as _extract_rl,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    grad_l2_norm as _grad_l2_norm,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    param_l2_norm as _param_l2_norm,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    reward_component_payload as _reward_component_payload,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    rollout_temperature_from_meta as _rollout_temperature_from_meta,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    summarize_batch_metadata as _summarize_batch_metadata,
)
from hermes_agentic_rl.trainers._rollout_helpers import (
    turn_group_id as _turn_group_id,
)
from hermes_agentic_rl.trainers.batch_stats import (
    _rl_dense_reward_metadata,
    _teacher_responses_from_env,
)
from hermes_agentic_rl.trainers.minibatch_builder import (
    aggregate_update_stats as _aggregate_update_stats_fn,
)
from hermes_agentic_rl.trainers.minibatch_builder import (
    build_update_batches as _build_update_batches_fn,
)
from hermes_agentic_rl.trainers.minibatch_builder import (
    split_minibatches as _split_minibatches_fn,
)
from hermes_agentic_rl.trainers.multi_turn_credit import assign_multi_turn_rewards
from hermes_agentic_rl.trainers.on_policy_config import OnPolicyTrainerConfig
from hermes_agentic_rl.trainers.profiling import (
    StepProfiler,
    append_jsonl,
    format_json_record,
)
from hermes_agentic_rl.trainers.train_stats import TrainStats as _TrainStats
from hermes_agentic_rl.utils.coerce import coerce_float

_module_logger = logging.getLogger(__name__)


class AgentLoopFactory(Protocol):
    """Creates a fresh BaseAgentLoop per rollout (for seed isolation)."""

    def __call__(self, *, backend: LLMBackend, seed: int | None) -> BaseAgentLoop: ...


DistributedStrategy = Literal["none", "ddp", "fsdp"]
DistributedPrecision = Literal["fp16", "bf16", "fp32", "auto"]


def _distributed_strategy(value: str) -> DistributedStrategy:
    normalized = value if value in {"none", "ddp", "fsdp"} else "none"
    return cast(DistributedStrategy, normalized)


def _distributed_precision(value: str) -> DistributedPrecision:
    normalized = value if value in {"fp16", "bf16", "fp32", "auto"} else "fp32"
    return cast(DistributedPrecision, normalized)


def default_policy_loop_factory(
    *,
    max_new_tokens: int,
    temperature: float,
) -> AgentLoopFactory:
    def _make(*, backend: LLMBackend, seed: int | None) -> BaseAgentLoop:
        return PolicyAgentLoop(
            backend=backend,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )

    return _make


def _observe_env_reward(env: Any, item: dict[str, Any] | None, reward: float) -> None:
    """Feed a rollout reward to a curriculum / multi-stream env, if it wants it.

    Duck-typed: a plain :class:`BaseEnv` has no ``observe`` and is skipped.
    Curriculum / multi-stream envs expose ``observe(reward, level=None)`` — we
    forward the stream/level tag that ``get_next_item`` stamped onto the item
    (``_curriculum_level``) so per-stream adaptive reweighting can attribute the
    reward to the right stream. Envs whose ``observe`` only takes ``reward``
    (e.g. ``LetterCountingEnv``) still work via the fallback.
    """
    observe = getattr(env, "observe", None)
    if not callable(observe):
        return
    level = item.get("_curriculum_level") if isinstance(item, dict) else None
    try:
        observe(float(reward), level=level)
    except TypeError:
        try:
            observe(float(reward))
        except Exception:
            pass
    except Exception:
        pass


def _scheduled_scalar(
    schedule: dict[str, Any] | None,
    *,
    default: float,
    iter_idx: int,
    total_steps: int,
) -> float:
    if not isinstance(schedule, dict) or not schedule:
        return float(default)

    mode = str(schedule.get("mode", schedule.get("kind", "linear"))).lower()
    start = float(schedule.get("start", schedule.get("value", default)))
    end = float(schedule.get("end", start))
    steps = max(1, int(schedule.get("total_steps", total_steps)))
    progress = max(0.0, min(1.0, float(iter_idx) / float(steps)))

    if mode == "constant":
        return float(schedule.get("value", start))
    if mode == "linear":
        return start + (end - start) * progress
    if mode == "cosine":
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError("hybrid weight schedule mode must be one of: constant, linear, cosine")


# TrainStats is defined in train_stats.py; re-export here for backward compat.
TrainStats = _TrainStats


class OnPolicyTrainer:
    """Generic rollout → loss → step loop, parameterized by a BaseAlgo.

    Subclasses only need to provide ``self.algo`` and (optionally) override
    ``_validate_backend`` / ``_build_loop``.

    Distributed rollouts (opt-in): pass a pre-started
    ``hermes_agentic_rl.distributed.MPRolloutPool`` via ``rollout_pool``. The
    trainer broadcasts weights every iter and submits ``prompts_per_iter *
    group_size`` rollout tasks in parallel. The learner-side loss computation
    is unchanged.
    """

    algo_name: str = "on_policy"
    _reward_shaping_fn: Callable[[list[RolloutRecord]], list[RolloutRecord]] | None = None

    def __init__(
        self,
        policy: LLMBackend,
        env: BaseEnv,
        reward_manager: RewardManager,
        algo: BaseAlgo,
        cfg: OnPolicyTrainerConfig | None = None,
        *,
        agent_loop_factory: AgentLoopFactory | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
        rollout_pool: Any = None,
        lagrangian: Any = None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.reward_manager = reward_manager
        self.algo = algo
        self.cfg = cfg or OnPolicyTrainerConfig()
        self.rollout_pool = rollout_pool
        self.lagrangian = lagrangian

        self._validate_backend(policy)

        # ── v0.9: FSDP / DDP wrapping (BEFORE optimizer creation) ──
        self._fsdp_enabled = False
        if self.cfg.distributed_strategy not in ("none", ""):
            from hermes_agentic_rl.trainers.distributed import (
                DistributedConfig,
                wrap_for_distributed,
            )

            dist_cfg = DistributedConfig(
                strategy=_distributed_strategy(self.cfg.distributed_strategy),
                fsdp_cpu_offload=self.cfg.fsdp_cpu_offload,
                mixed_precision=_distributed_precision(self.cfg.amp_dtype),
            )
            if hasattr(policy, "model"):
                wrapped, self._fsdp_enabled = wrap_for_distributed(
                    policy.model,
                    dist_cfg,
                )
                policy.model = wrapped  # type: ignore[attr-defined]

        params = list(policy.trainable_parameters())
        if not params:
            raise RuntimeError("policy has no trainable parameters")
        self._trainable_params = params
        self.optim = torch.optim.AdamW(params, lr=self.cfg.lr)

        # ── LR scheduler (step-based, grad-accum-aware) ──────────────────
        from hermes_agentic_rl.trainers.lr_schedule import make_lr_scheduler

        total_steps = int(self.cfg.lr_total_steps) or int(self.cfg.n_iters)
        # When grad_accum > 1, each "iteration" may produce multiple optimizer
        # steps. The scheduler counts *effective optimizer steps* so warm-up /
        # decay align with actual parameter updates, not just iterations.
        self._lr_sched = make_lr_scheduler(
            kind=self.cfg.lr_schedule,
            lr=self.cfg.lr,
            total_steps=total_steps,
            warmup_steps=int(self.cfg.lr_warmup_steps),
            warmup_start_lr=float(self.cfg.lr_warmup_start_lr),
            end_lr=float(self.cfg.lr_end_lr),
        )
        self._lr_optim_step_counter: int = 0  # tracks actual optimizer steps

        # ── v0.9: AMP context ──
        from hermes_agentic_rl.trainers.mixed_precision import AMPContext

        self._amp = AMPContext(
            dtype=self.cfg.amp_dtype,
            enabled=(self.cfg.amp_dtype not in ("fp32", "float32", "none", "")),
        )

        # ── v0.9: Gradient accumulation ──
        from hermes_agentic_rl.trainers.mixed_precision import GradientAccumulator

        self._grad_accum = GradientAccumulator(steps=self.cfg.grad_accum_steps)

        # ── EMA + vLLM conflict check (before any heavy init) ──
        if self.cfg.use_ema_rollout and self.cfg.vllm_rollout_model:
            from hermes_agentic_rl.runtime.errors import RuntimeConfigurationError

            raise RuntimeConfigurationError(
                "EMA rollout is incompatible with vLLM rollout. "
                "Set use_ema_rollout=False or remove vllm_rollout_model."
            )

        # ── v0.9: vLLM rollout backend (generation-only) ──
        self._vllm_rollout: Any = None
        if self.cfg.vllm_rollout_model:
            from hermes_agentic_rl.backends.vllm_backend import (
                VLLMRolloutBackend,
                VLLMRolloutConfig,
            )

            self._vllm_rollout = VLLMRolloutBackend(
                VLLMRolloutConfig(
                    model=self.cfg.vllm_rollout_model,
                    tensor_parallel_size=self.cfg.vllm_tensor_parallel_size,
                    max_model_len=self.cfg.vllm_max_model_len,
                    gpu_memory_utilization=self.cfg.vllm_gpu_memory_utilization,
                    enable_prefix_caching=self.cfg.vllm_enable_prefix_caching,
                )
            )
            # Initial sync: push learner weights to vLLM.
            if hasattr(policy, "model"):
                self._sync_weights_to_vllm(policy)

        self.ref_policy: LLMBackend | None = None
        if self.cfg.use_reference and hasattr(policy, "clone_frozen"):
            self.ref_policy = policy.clone_frozen()  # type: ignore[attr-defined]

        # ── EMA shadow for stable local rollouts ──
        self._ema: Any = None
        if self.cfg.use_ema_rollout:
            from hermes_agentic_rl.trainers.ema import EMAModel

            self._ema = EMAModel(
                policy,
                tau=self.cfg.ema_tau,
                tau_start=self.cfg.ema_tau_start,
                tau_warmup_steps=self.cfg.ema_tau_warmup_steps,
            )

        self.agent_loop_factory = agent_loop_factory or default_policy_loop_factory(
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )
        self._prompt_encoder = PromptStateEncoder(self.policy.tokenizer)
        self.logger = logger or self._default_logger
        self._profiler = StepProfiler(enabled=bool(getattr(self.cfg, "profile", False)))
        self._profile_output_path = (
            Path(self.cfg.profile_output_path) if self.cfg.profile_output_path is not None else None
        )
        self._gradient_checkpointing_supported = False
        self.stats = TrainStats()
        self._seed_counter = 0

        # v0.8: reward normalizer + adaptive KL controller (opt-in).
        from hermes_agentic_rl.trainers.kl_controller import (
            KLControllerType,
            build_kl_controller,
        )
        from hermes_agentic_rl.trainers.ppo_utils import RunningMeanStd

        self._reward_rms: RunningMeanStd | None = (
            RunningMeanStd() if self.cfg.normalize_reward else None
        )
        self._kl_ctrl: KLControllerType | None = None
        if self.cfg.adaptive_kl and self.cfg.target_kl > 0 and self.cfg.use_reference:
            algo_cfg = getattr(self.algo, "cfg", None)
            init_beta = float(getattr(algo_cfg, "kl_coef", 0.02)) or 0.02
            kl_kind = str(getattr(self.cfg, "adaptive_kl_type", "p") or "p")
            self._kl_ctrl = build_kl_controller(
                kind=kl_kind,  # type: ignore[arg-type]
                init_kl_coef=init_beta,
                target_kl=float(self.cfg.target_kl),
                horizon=float(self.cfg.adaptive_kl_horizon),
                min_coef=float(self.cfg.adaptive_kl_min),
                max_coef=float(self.cfg.adaptive_kl_max),
                Kp=float(getattr(self.cfg, "adaptive_kl_Kp", 0.1)),
                Ki=float(getattr(self.cfg, "adaptive_kl_Ki", 0.01)),
                Kd=float(getattr(self.cfg, "adaptive_kl_Kd", 0.005)),
                I_max=float(getattr(self.cfg, "adaptive_kl_I_max", 2.0)),
            )
        # Entropy-coefficient scheduler (opt-in via cfg.entropy_schedule).
        # The dict mirrors the stability-preset shape, e.g.
        #   {"mode": "linear", "start": 0.01, "end": 0.001}
        #   {"mode": "pid", "target_entropy": 1.5}
        # Previously this config field (and the values injected by the
        # "standard"/"aggressive" stability presets) was silently dropped.
        self._entropy_sched: Any = None
        if self.cfg.entropy_schedule:
            from hermes_agentic_rl.algos.entropy_schedule import (
                make_entropy_scheduler,
            )

            sched_kwargs = dict(self.cfg.entropy_schedule)
            kind = str(sched_kwargs.pop("mode", sched_kwargs.pop("kind", "linear")))
            # Default total_steps to the full run length for step schedules
            # so users do not have to restate it in YAML.
            if kind in ("linear", "cosine") and "total_steps" not in sched_kwargs:
                sched_kwargs["total_steps"] = max(1, int(self.cfg.n_iters))
            try:
                self._entropy_sched = make_entropy_scheduler(kind, **sched_kwargs)
            except (TypeError, ValueError) as exc:
                import warnings as _warnings

                _warnings.warn(
                    f"Ignoring invalid entropy_schedule {self.cfg.entropy_schedule!r}: {exc}",
                    stacklevel=2,
                )
                self._entropy_sched = None

        # Iteration to start from — updated by _maybe_resume().
        self._start_iter = 0
        self._best_reward = 0.0
        self._best_iter = 0
        # Expose optimizer via stable alias for checkpoint helpers.
        self._optim = self.optim

        # Initialize checkpoint manager lazily (only when output_dir + checkpoint_every).
        self._ckpt_manager = None
        self._best_ckpt_manager = None
        self._async_ckpt_saver = None
        if self.cfg.output_dir is not None and (
            self.cfg.checkpoint_every > 0
            or self.cfg.auto_resume
            or self.cfg.resume_from is not None
        ):
            from hermes_agentic_rl.trainers.checkpoint import CheckpointManager

            self._ckpt_manager = CheckpointManager(
                Path(self.cfg.output_dir) / "checkpoints",
                keep_last=self.cfg.keep_last_checkpoints,
            )
            if self.cfg.async_checkpoint:
                from hermes_agentic_rl.trainers.checkpoint import AsyncCheckpointSaver

                self._async_ckpt_saver = AsyncCheckpointSaver()
            if self.cfg.save_best_checkpoint:
                # keep_last=0 ⇒ never prune the best bundle
                self._best_ckpt_manager = CheckpointManager(
                    Path(self.cfg.output_dir) / "checkpoints_best",
                    keep_last=0,
                )

        # Early-stop bookkeeping.
        self._iters_since_best = 0
        self._early_stopped = False
        self._batch_rollout_generator: BatchRolloutGenerator | None = None
        if (
            self.cfg.batch_generate
            and agent_loop_factory is None
            and not self.cfg.multi_turn
            and hasattr(self.policy, "model")
        ):
            self._batch_rollout_generator = BatchRolloutGenerator(
                self.policy,
                batch_size=max(1, self.cfg.group_size),
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
            )

        # Pipelined (double-buffered) rollout bookkeeping.
        #   _pending_expected: count of in-flight rollout tasks dispatched for
        #     the next iteration (None = nothing in flight).
        #   _update_version: monotonically increasing count of completed
        #     gradient updates — the learner's "policy version".
        #   _pending_dispatch_version: the policy version captured when the
        #     in-flight rollouts were dispatched; their staleness on drain is
        #     ``_update_version - _pending_dispatch_version``.
        #   _rollout_staleness: staleness of the rollouts consumed this iter,
        #     surfaced as the ``rollout_staleness`` stat.
        self._pending_expected: int | None = None
        self._update_version: int = 0
        self._pending_dispatch_version: int | None = None
        self._rollout_staleness: int = 0

        # ── OPD teacher-logprob filler (closes the OPD loop) ──────────────
        self._teacher_filler: Any = None
        if getattr(self.cfg, "opd_teacher_fill", False):
            from hermes_agentic_rl.rewards.opd_teacher import (
                TeacherFillConfig,
                TeacherLogprobFiller,
            )

            self._teacher_filler = TeacherLogprobFiller(
                self.policy,
                TeacherFillConfig(
                    enabled=True,
                    hint_template=str(
                        getattr(
                            self.cfg,
                            "opd_hint_template",
                            "\n\n[HINT_START]{hint}[HINT_END]\n",
                        )
                    ),
                    max_hint_tokens=int(getattr(self.cfg, "opd_teacher_max_hint_tokens", 128)),
                    capability_axis_weights=getattr(self.cfg, "opd_capability_axis_weights", None),
                ),
            )

        # ── OPD in-trainer hint extractor (recovers directive hints) ──────
        # Runs BEFORE the teacher fill: any record that has a next-state signal
        # but no ``opd_hint`` gets one from a configurable judge, so OPD fires
        # even when the env / reward did not pre-populate a hint. Opt-in via the
        # ``opd.hint_extractor`` YAML block; ``None`` keeps legacy behaviour.
        self._hint_extractor: Any = None
        # Persistent stats so metrics are always emitted (0 when extractor is off).
        self._last_hint_extract_stats: dict[str, float] | None = None
        hint_cfg = getattr(self.cfg, "opd_hint_extractor", None)
        if hint_cfg:
            from hermes_agentic_rl.rewards.opd_hint_extractor import (
                build_opd_hint_extractor,
            )

            self._hint_extractor = build_opd_hint_extractor(hint_cfg)

        # ── Replay buffer (off-policy mixing with TIS correction) ───────────
        self._replay_buffer: Any = None
        replay_cfg = getattr(self.cfg, "replay_buffer", None)
        if replay_cfg:
            from hermes_agentic_rl.trainers.replay_buffer import (
                build_replay_buffer_from_config,
            )

            self._replay_buffer = build_replay_buffer_from_config(replay_cfg)

        # ── PRM online co-training pipeline ────────────────────────────────
        # Enabled via cfg.prm_pipeline = {train_every: N, ...} in YAML.
        # The pipeline generates step-level pseudo-labels from each iter's
        # rollout batch and trains the PRM every train_every iters in place.
        self._prm_pipeline: Any = None
        prm_cfg_dict = getattr(self.cfg, "prm_pipeline", None)
        if isinstance(prm_cfg_dict, dict) and prm_cfg_dict:
            from hermes_agentic_rl.rewards.prm import ProcessRewardModel
            from hermes_agentic_rl.trainers.prm_pipeline import (
                build_prm_pipeline_from_config,
            )

            # Build PRM model from the same backbone as the policy (shared trunk).
            if hasattr(policy, "model"):
                try:
                    prm_model = ProcessRewardModel(
                        policy,
                        freeze_base=bool(prm_cfg_dict.get("prm_freeze_base", True)),
                    )
                    prm_model_path = prm_cfg_dict.get("prm_model_path")
                    if prm_model_path is not None:
                        import torch as _torch

                        state = _torch.load(str(prm_model_path), map_location="cpu")
                        prm_model.head.load_state_dict(state)
                    self._prm_pipeline = build_prm_pipeline_from_config(prm_cfg_dict, prm_model)
                except Exception as _prm_e:
                    import warnings

                    warnings.warn(
                        f"PRM pipeline init failed, continuing without PRM: {_prm_e}",
                        stacklevel=2,
                    )

        # ── Memory-aware reward shaper (cross-session improvement bonus) ───
        # Differentiator: rewards improvement over the agent's OWN history,
        # not just the current batch baseline. Opt-in via cfg.memory_reward.
        self._memory_shaper: Any = None
        memory_cfg = getattr(self.cfg, "memory_reward", None)
        if isinstance(memory_cfg, dict) and memory_cfg:
            from hermes_agentic_rl.rewards.memory_reward_shaper import (
                MemoryAwareRewardShaper,
                MemoryRewardConfig,
            )

            _mem_cfg = MemoryRewardConfig(
                alpha=float(memory_cfg.get("alpha", 0.9)),
                bonus_coef=float(memory_cfg.get("bonus_coef", 0.1)),
                clip_max=float(memory_cfg.get("clip_max", 0.5)),
                sigma_floor=float(memory_cfg.get("sigma_floor", 0.1)),
                min_obs=int(memory_cfg.get("min_obs", 3)),
                axis_bonus_coef={
                    str(k): float(v) for k, v in (memory_cfg.get("axis_bonus_coef") or {}).items()
                },
                task_key_mode=str(memory_cfg.get("task_key_mode", "task_id")),
                apply_inplace=bool(memory_cfg.get("apply_inplace", True)),
            )
            self._memory_shaper = MemoryAwareRewardShaper(_mem_cfg)

        # ── Dynamic reward balancer (adaptive component weight scheduling) ──
        # Differentiator: automatically adjusts component weights based on
        # their informative variance. OpenClaw-RL uses static weights.
        self._dynamic_balancer: Any = None
        dynamic_cfg = getattr(self.cfg, "dynamic_reward_balancer", None)
        if isinstance(dynamic_cfg, dict) and dynamic_cfg:
            base_weights = dynamic_cfg.get("base_weights")
            if isinstance(base_weights, dict) and base_weights:
                from hermes_agentic_rl.rewards.dynamic_reward_balancer import (
                    build_dynamic_balancer_from_config,
                )

                self._dynamic_balancer = build_dynamic_balancer_from_config(
                    dynamic_cfg, {str(k): float(v) for k, v in base_weights.items()}
                )

        # ── Curriculum scheduler (progressive stage advancement) ──
        # Integrates with DynamicRewardBalancer to adjust component weights
        # based on curriculum stage progression. This ensures the model
        # learns tool call structure before content before summary quality.
        self._curriculum_scheduler: Any = None
        curriculum_cfg = getattr(self.cfg, "curriculum", None)
        if isinstance(curriculum_cfg, dict) and curriculum_cfg:
            from hermes_agentic_rl.curriculum import (
                CurriculumScheduler,
                CurriculumSchedulerConfig,
                create_default_curriculum,
            )

            stages = create_default_curriculum()
            # Allow custom stages via config
            custom_stages = curriculum_cfg.get("stages")
            if isinstance(custom_stages, list) and custom_stages:
                from hermes_agentic_rl.curriculum import CurriculumStage

                stages = [CurriculumStage(**s) if isinstance(s, dict) else s for s in custom_stages]
            sched_cfg = CurriculumSchedulerConfig(
                auto_advance=bool(curriculum_cfg.get("auto_advance", True)),
                min_iters_per_stage=int(curriculum_cfg.get("min_iters_per_stage", 10)),
                allow_regression=bool(curriculum_cfg.get("allow_regression", False)),
                regression_factor=float(curriculum_cfg.get("regression_factor", 0.5)),
            )
            self._curriculum_scheduler = CurriculumScheduler(stages=stages, cfg=sched_cfg)

        # ── Staleness-adaptive TIS controller ──────────────────────────────
        # When pipeline_rollouts or replay_buffer is enabled, the staleness
        # of consumed rollouts varies per iter. This controller dynamically
        # adjusts the TIS rho_clip: high trust (high clip) for fresh rollouts,
        # conservative (low clip) for stale ones. Overrides the algo's fixed
        # tis_rho_clip when active.
        self._staleness_tis: Any = None
        staleness_tis_cfg = getattr(self.cfg, "staleness_adaptive_tis", None)
        if isinstance(staleness_tis_cfg, dict) and staleness_tis_cfg:
            from hermes_agentic_rl.algos.common.staleness_adaptive_tis import (
                StalenessAdaptiveTIS,
                StalenessSchedule,
            )

            schedule = StalenessSchedule(
                max_rho_clip=float(staleness_tis_cfg.get("max_rho_clip", 2.0)),
                min_rho_clip=float(staleness_tis_cfg.get("min_rho_clip", 1.0)),
                max_staleness=int(staleness_tis_cfg.get("max_staleness", 10)),
                interpolation=str(staleness_tis_cfg.get("interpolation", "linear")),
                rho_floor=float(staleness_tis_cfg.get("rho_floor", 0.0)),
            )
            self._staleness_tis = StalenessAdaptiveTIS(
                schedule=schedule,
                window_size=int(staleness_tis_cfg.get("window_size", 10)),
                enabled=bool(staleness_tis_cfg.get("enabled", True)),
            )

        # ── LoRA hot-reload manager ────────────────────────────────────────
        # When lora_hot_reload config is set and a vLLM rollout backend is
        # available, inject LoRA adapters into the policy model and create a
        # LoRAHotReloadManager that merges LoRA deltas → shadow weights → vLLM
        # sync after each optimizer step.
        self._lora_hot_reload: Any = None
        lora_cfg = getattr(self.cfg, "lora_hot_reload", None)
        if isinstance(lora_cfg, dict) and lora_cfg and self._vllm_rollout is not None:
            from hermes_agentic_rl.peft.lora import (
                LoRAConfig,
                inject_lora,
            )
            from hermes_agentic_rl.peft.lora_hot_reload import (
                LoRAHotReloadManager,
            )

            lora_config = LoRAConfig(
                r=int(lora_cfg.get("rank", 8)),
                alpha=float(lora_cfg.get("alpha", 16.0)),
                dropout=float(lora_cfg.get("dropout", 0.0)),
                target_patterns=tuple(lora_cfg.get("target_patterns", ("qkv", "proj"))),
                freeze_base=True,
            )
            if hasattr(policy, "model"):
                adapter = inject_lora(policy.model, lora_config)  # type: ignore[attr-defined]
                self._lora_hot_reload = LoRAHotReloadManager(
                    adapter=adapter,
                    base_model=policy.model,  # type: ignore[attr-defined]
                    vllm_backend=self._vllm_rollout,
                    sync_every=int(lora_cfg.get("sync_every", 1)),
                    shadow_device=str(lora_cfg.get("shadow_device", "cpu")),
                )
                _module_logger.info(
                    "LoRA hot-reload enabled: rank=%d, alpha=%.1f, sync_every=%d",
                    lora_config.r,
                    lora_config.alpha,
                    int(lora_cfg.get("sync_every", 1)),
                )

        self._maybe_resume()

    # ------------------------------------------------------------------
    # Optimizer step helper (eliminates duplicate unscale/clip/step/LR logic)
    # ------------------------------------------------------------------

    def _optimizer_step(self, *, requires_grad: bool) -> float:
        """Execute one optimizer step: unscale → clip → step → zero_grad → LR.

        Returns the gradient norm (0.0 if no gradients were applied).
        """
        grad_norm = 0.0
        if requires_grad:
            self._amp.unscale_(self.optim)
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                grad_norm_raw = torch.nn.utils.clip_grad_norm_(
                    self._trainable_params,
                    max_norm=self.cfg.grad_clip,
                )
                grad_norm = float(grad_norm_raw.detach().item())
            else:
                grad_norm = _grad_l2_norm(self._trainable_params)
            self._amp.step(self.optim)
            self.optim.zero_grad()
            self._amp.update()
            self._grad_accum.finish_step()
            self._lr_optim_step_counter += 1
            new_lr = self._lr_sched.get_lr(self._lr_optim_step_counter)
            for pg in self.optim.param_groups:
                pg["lr"] = new_lr
        else:
            self.optim.zero_grad()
            self._grad_accum.finish_step()
        return grad_norm

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------

    def _validate_backend(self, policy: LLMBackend) -> None:
        """Override to require a value head, etc."""

    def _rollout_backend(self) -> LLMBackend:
        """Return the backend used for rollout generation.

        When EMA rollout is enabled, returns the EMA shadow; otherwise returns
        the learner policy itself.
        """
        if self._ema is not None:
            return self._ema.as_rollout_backend()
        return self.policy

    def _sync_weights_to_vllm(self, policy: LLMBackend) -> None:
        """Push learner state_dict to vLLM rollout engine.

        Handles FSDP case: if the model is FSDP-wrapped, gathers shards
        first with ``gather_fsdp_state_dict``.
        """
        if self._vllm_rollout is None:
            return
        if not hasattr(policy, "model"):
            return
        if self._fsdp_enabled:
            from hermes_agentic_rl.trainers.distributed import (
                gather_fsdp_state_dict,
            )

            state = gather_fsdp_state_dict(policy.model)  # type: ignore[attr-defined]
        else:
            state = {
                k: v.detach().cpu()
                for k, v in policy.model.state_dict().items()  # type: ignore[attr-defined]
            }
        self._vllm_rollout.sync_weights_from(state)

    def _prepare_update_batch(self, batch: RolloutBatch) -> RolloutBatch:
        """Hook for subclasses to freeze rollout-time signals before SGD epochs."""
        return batch

    def _set_gradient_checkpointing(self, enabled: bool) -> bool:
        if not bool(getattr(self.cfg, "gradient_checkpointing", False)):
            return False
        toggle = getattr(self.policy, "set_gradient_checkpointing", None)
        if callable(toggle):
            try:
                applied = bool(toggle(bool(enabled)))
                self._gradient_checkpointing_supported = (
                    self._gradient_checkpointing_supported or applied
                )
                return applied
            except Exception:
                return False
        return False

    def _apply_hybrid_weight_schedules(self, iter_idx: int) -> dict[str, float]:
        algo_cfg = getattr(self.algo, "cfg", None)
        if algo_cfg is None or not hasattr(algo_cfg, "w_rl") or not hasattr(algo_cfg, "w_opd"):
            return {}

        metrics: dict[str, float] = {}
        total_steps = max(1, int(self.cfg.n_iters) - 1)
        schedule_specs = {
            "w_rl": getattr(self.cfg, "w_rl_schedule", None),
            "w_opd": getattr(self.cfg, "w_opd_schedule", None),
        }
        for attr, schedule in schedule_specs.items():
            if not isinstance(schedule, dict) or not schedule:
                continue
            current = float(getattr(algo_cfg, attr))
            value = _scheduled_scalar(
                schedule,
                default=current,
                iter_idx=iter_idx,
                total_steps=total_steps,
            )
            setattr(algo_cfg, attr, value)
            metrics[attr] = float(value)
            metrics[f"{attr}_schedule_active"] = 1.0
        return metrics

    def _maybe_normalize_rewards(self, records: list[RolloutRecord]) -> list[RolloutRecord]:
        """Apply running-reward normalization if enabled.

        Records are mutated in-place: ``reward`` is replaced by the
        whitened value (clipped to ``±reward_norm_clip``), and the raw
        scalar survives under ``metadata['raw_reward']``. Logging still
        uses the raw reward via ``mean_reward`` because the algo reads
        from each record after this step.
        """
        if self._reward_rms is None or not records:
            return records
        raws = [float(r.reward) for r in records]
        self._reward_rms.update(raws)
        clip = float(self.cfg.reward_norm_clip)
        for rec, raw in zip(records, raws, strict=True):
            # Preserve raw for logging/diagnosis.
            if "raw_reward" not in rec.metadata:
                rec.metadata["raw_reward"] = raw
            norm = self._reward_rms.normalize(raw)
            if clip > 0:
                norm = max(-clip, min(clip, norm))
            rec.reward = float(norm)
            rec.metadata["normalized_reward"] = float(norm)
        return records

    def _preserve_group_boundaries(self) -> bool:
        return self.algo_name == "grpo"

    # ------------------------------------------------------------------
    # rollout → records
    # ------------------------------------------------------------------

    def _next_seed(self) -> int | None:
        if self.cfg.seed is None:
            return None
        self._seed_counter += 1
        return self.cfg.seed + self._seed_counter

    async def _collect_group(
        self,
        item: dict[str, Any],
        *,
        temperature: float | None = None,
        group_size: int | None = None,
    ) -> list[RolloutRecord]:
        if self._batch_rollout_generator is not None:
            return await self._collect_group_batched(
                item,
                temperature=temperature,
                group_size=group_size,
            )

        instruction = self.env.format_prompt(item)
        records: list[RolloutRecord] = []
        group_id = str(item.get("task_id", "group"))
        for _g in range(group_size if group_size is not None else self.cfg.group_size):
            loop = self.agent_loop_factory(backend=self.policy, seed=self._next_seed())
            trajectory: Trajectory = await RolloutManager(loop).collect(item, instruction)
            summary = await self.reward_manager.evaluate(item, trajectory, tool_context=None)
            # curriculum / multi-stream feedback — duck-typed; forwards the
            # stream/level tag so per-stream reweighting can attribute reward.
            _observe_env_reward(self.env, item, float(summary.final_score))
            if self.lagrangian is not None:
                try:
                    self.lagrangian.measure(item, trajectory)
                except Exception:
                    pass
            rl_meta = _extract_rl(trajectory)
            if rl_meta is None:
                raise RuntimeError("Agent loop must emit trajectory.metadata['runtime']['rl']")
            dense_meta = _rl_dense_reward_metadata(rl_meta)
            rollout_temperature = _rollout_temperature_from_meta(
                rl_meta,
                fallback=temperature if temperature is not None else self.cfg.temperature,
            )

            base_meta: dict[str, Any] = {
                "final_output": trajectory.final_output,
                "reward_components": [
                    _reward_component_payload(component) for component in summary.components
                ],
                "finished_naturally": bool(trajectory.finished_naturally),
                "turns_used": trajectory.turns_used,
                "tool_calls_count": sum(len(step.tool_calls) for step in trajectory.steps),
                "tool_results_count": sum(len(step.tool_results) for step in trajectory.steps),
                "final_output_chars": len(trajectory.final_output or ""),
                "reward_summary_metadata": dict(summary.metadata),
                "rollout_temperature": rollout_temperature,
                **dense_meta,
            }

            # Propagate the OPD directive hint (written by NextStatePRM) so the
            # teacher-logprob filler / OPD branch can consume it. Without this
            # the hint stays buried in runtime metadata and OPD never fires.
            opd_hint = rl_meta.get("opd_hint")
            if isinstance(opd_hint, str) and opd_hint.strip():
                base_meta["opd_hint"] = opd_hint

            # Propagate the raw next-state signal so the in-trainer
            # OPDHintExtractor can recover a hint when none was pre-populated.
            next_state = _extract_next_state(trajectory)
            if isinstance(next_state, str) and next_state.strip():
                base_meta["next_state"] = next_state

            # Multi-stream tag: stamp the sampled stream/level so
            # _summarize_batch_metadata can break reward/count down per stream.
            stream_level = item.get("_curriculum_level")
            if stream_level is not None:
                try:
                    base_meta["stream_level"] = int(stream_level)
                except (TypeError, ValueError):
                    pass

            if self.cfg.multi_turn and rl_meta.get("turns"):
                teacher_responses = _teacher_responses_from_env(
                    self.env,
                    item,
                    n_turns=len(rl_meta["turns"]),
                )
                turn_rewards = assign_multi_turn_rewards(
                    trajectory,
                    final_reward=float(summary.final_score),
                    n_turns=len(rl_meta["turns"]),
                    cfg=self.cfg.multi_turn_credit,
                    teacher_responses=teacher_responses,
                )
                # Emit one RolloutRecord per turn, each scored under its true
                # rollout context. Reward assignment is configurable:
                # legacy shared final reward, terminal-only, discounted, or
                # hybrid with local tool/feedback shaping.
                for t_idx, turn in enumerate(rl_meta["turns"]):
                    turn_group_id = _turn_group_id(group_id, t_idx)
                    credit_meta = (
                        dict(turn_rewards[t_idx])
                        if t_idx < len(turn_rewards)
                        else {
                            "reward": float(summary.final_score),
                            "final_component": float(summary.final_score),
                            "local_component": 0.0,
                            "weighted_final_component": float(summary.final_score),
                            "weighted_local_component": 0.0,
                            "mode": "shared",
                        }
                    )
                    records.append(
                        RolloutRecord(
                            prompt_ids=list(turn["prompt_prefix_ids"]),
                            response_ids=list(turn["response_ids"]),
                            old_logprobs=list(turn["old_logprobs"]),
                            reward=coerce_float(
                                credit_meta.get("reward"),
                                default=float(summary.final_score),
                            ),
                            group_id=turn_group_id,
                            metadata={
                                **base_meta,
                                "prompt_group_id": group_id,
                                "turn_group_id": turn_group_id,
                                "turn_index": t_idx,
                                "prompt_tokens": len(turn["prompt_prefix_ids"]),
                                "response_tokens": len(turn["response_ids"]),
                                "rollout_final_reward": float(summary.final_score),
                                "turn_credit": credit_meta,
                            },
                        )
                    )
            else:
                records.append(
                    RolloutRecord(
                        prompt_ids=list(rl_meta["prompt_ids"]),
                        response_ids=list(rl_meta["response_ids"]),
                        old_logprobs=list(rl_meta["old_logprobs"]),
                        reward=float(summary.final_score),
                        group_id=group_id,
                        metadata={
                            **base_meta,
                            "prompt_tokens": len(rl_meta["prompt_ids"]),
                            "response_tokens": len(rl_meta["response_ids"]),
                        },
                    )
                )
        return records

    async def _collect_group_batched(
        self,
        item: dict[str, Any],
        *,
        temperature: float | None = None,
        group_size: int | None = None,
    ) -> list[RolloutRecord]:
        instruction = self.env.format_prompt(item)
        encoder = PromptStateEncoder(self.policy.tokenizer)
        prompt_ids = list(encoder.encode({"instruction": instruction}).prompt_ids)
        if self._batch_rollout_generator is None:
            raise RuntimeError("batched rollout collection requires a batch rollout generator")
        _gs = group_size if group_size is not None else self.cfg.group_size
        _temp = temperature if temperature is not None else self.cfg.temperature
        outputs = self._batch_rollout_generator.generate(
            [prompt_ids for _ in range(_gs)],
            seed=self._next_seed(),
        )

        group_id = str(item.get("task_id", "group"))
        records: list[RolloutRecord] = []
        for gen in outputs:
            response_text = self.policy.tokenizer.decode(gen.response_ids)
            trajectory = _batch_single_turn_trajectory(
                item=item,
                instruction=instruction,
                response_text=response_text,
                prompt_ids=prompt_ids,
                response_ids=list(gen.response_ids),
                old_logprobs=list(gen.logprobs),
                temperature=_temp,
                finished=gen.finished,
            )
            summary = await self.reward_manager.evaluate(item, trajectory, tool_context=None)
            _observe_env_reward(self.env, item, float(summary.final_score))
            if self.lagrangian is not None:
                try:
                    self.lagrangian.measure(item, trajectory)
                except Exception:
                    pass
            rl_meta = _extract_rl(trajectory) or {}
            dense_meta = _rl_dense_reward_metadata(rl_meta)
            opd_hint = rl_meta.get("opd_hint")
            opd_meta: dict[str, Any] = (
                {"opd_hint": opd_hint} if isinstance(opd_hint, str) and opd_hint.strip() else {}
            )
            next_state = _extract_next_state(trajectory)
            if isinstance(next_state, str) and next_state.strip():
                opd_meta["next_state"] = next_state
            stream_level = item.get("_curriculum_level")
            if stream_level is not None:
                try:
                    opd_meta["stream_level"] = int(stream_level)
                except (TypeError, ValueError):
                    pass
            records.append(
                RolloutRecord(
                    prompt_ids=list(prompt_ids),
                    response_ids=list(gen.response_ids),
                    old_logprobs=list(gen.logprobs),
                    reward=float(summary.final_score),
                    group_id=group_id,
                    metadata={
                        **opd_meta,
                        "final_output": trajectory.final_output,
                        "reward_components": [
                            _reward_component_payload(component) for component in summary.components
                        ],
                        "finished_naturally": bool(trajectory.finished_naturally),
                        "turns_used": trajectory.turns_used,
                        "tool_calls_count": sum(len(step.tool_calls) for step in trajectory.steps),
                        "tool_results_count": sum(
                            len(step.tool_results) for step in trajectory.steps
                        ),
                        "final_output_chars": len(trajectory.final_output or ""),
                        "prompt_tokens": len(prompt_ids),
                        "response_tokens": len(gen.response_ids),
                        "rollout_temperature": float(_temp),
                        "reward_summary_metadata": dict(summary.metadata),
                        **dense_meta,
                    },
                )
            )
        return records

    async def _dispatch_distributed(self) -> int:
        """Broadcast current learner weights + submit one iter of rollout tasks.

        Returns the number of dispatched tasks (the ``expected`` count for the
        matching :meth:`_drain_distributed`). The weight ``state_dict`` is
        snapshotted synchronously here, so the learner may safely continue
        mutating its parameters (e.g. an overlapping update step) while the
        pool's workers roll out against the broadcast snapshot. This snapshot
        boundary is what makes the pipelined (double-buffered) schedule safe.
        """
        from hermes_agentic_rl.distributed.mp_pool import RolloutTask

        # 1) broadcast current learner weights (synchronous snapshot)
        state = {k: v.detach().cpu() for k, v in self.policy.model.state_dict().items()}  # type: ignore[attr-defined]
        self.rollout_pool.broadcast_weights(state)

        # 2) build tasks
        tasks: list[RolloutTask] = []
        seq = 0
        for _ in range(self.cfg.prompts_per_iter):
            item = await self.env.get_next_item()
            instruction = self.env.format_prompt(item)
            for _g in range(self.cfg.group_size):
                tasks.append(
                    RolloutTask(
                        task_id=str(item.get("task_id", "group")),
                        item=dict(item),
                        instruction=instruction,
                        seed=self._next_seed(),
                        task_seq=seq,
                    )
                )
                seq += 1

        # 3) submit (non-blocking: workers process asynchronously)
        self.rollout_pool.submit_tasks(tasks)
        return len(tasks)

    def _drain_distributed(self, expected: int) -> list[RolloutRecord]:
        """Block until ``expected`` rollout results return, then flatten."""
        results = self.rollout_pool.drain(expected=expected)

        records: list[RolloutRecord] = []
        for r in results:
            # Distributed results don't echo the item, so per-stream level
            # attribution is unavailable here; observe globally (level=None).
            _observe_env_reward(self.env, None, float(r["final_score"]))
            for rec_dict in r["records"]:
                records.append(
                    RolloutRecord(
                        prompt_ids=list(rec_dict["prompt_ids"]),
                        response_ids=list(rec_dict["response_ids"]),
                        old_logprobs=list(rec_dict["old_logprobs"]),
                        reward=float(rec_dict["reward"]),
                        group_id=str(rec_dict["group_id"]),
                        metadata=dict(rec_dict.get("metadata", {})),
                    )
                )
        return records

    async def _collect_distributed(self) -> list[RolloutRecord]:
        """Synchronous (BSP) fan-out: dispatch one iter, wait, flatten."""
        expected = await self._dispatch_distributed()
        return self._drain_distributed(expected)

    # ------------------------------------------------------------------
    # train loop
    # ------------------------------------------------------------------

    def _pipeline_enabled(self) -> bool:
        return bool(getattr(self.cfg, "pipeline_rollouts", False)) and self.rollout_pool is not None

    async def _collect_for_iter(self, iter_idx: int) -> list[RolloutRecord]:
        """Collect this iteration's rollouts.

        Three modes:
          * **pipelined** (``pipeline_rollouts`` + a rollout pool): drain the
            rollouts dispatched during the *previous* iteration, then
            immediately dispatch the *next* iteration's rollouts so they
            overlap with this iteration's gradient update. Tolerates 1 step of
            policy staleness — the broadcast snapshot is taken before the
            update, so workers roll out against ``W_{t-1}`` while the learner
            advances to ``W_t``. PPO/GRPO ratio clipping absorbs the lag.
          * **distributed BSP** (pool, no pipeline): dispatch + wait inline.
          * **local**: in-process group collection.
        """
        if iter_idx == 0:
            await self.env.setup()

        if self._pipeline_enabled():
            if self._pending_expected is None:
                self._pending_expected = await self._dispatch_distributed()
                self._pending_dispatch_version = self._update_version
            # The rollouts about to be drained were dispatched at this version;
            # their staleness is how many updates have landed since.
            dispatch_version = self._pending_dispatch_version or 0
            records = self._drain_distributed(self._pending_expected)
            self._rollout_staleness = self._update_version - dispatch_version
            # Prefetch the next iter's rollouts BEFORE the update runs, so the
            # broadcast captures pre-update weights (1-step staleness) and the
            # rollout overlaps with this iter's gradient step.
            if iter_idx + 1 < self.cfg.n_iters:
                self._pending_expected = await self._dispatch_distributed()
                self._pending_dispatch_version = self._update_version
            else:
                self._pending_expected = None
                self._pending_dispatch_version = None
            return records

        # Synchronous paths consume freshly-generated rollouts: zero staleness.
        self._rollout_staleness = 0
        if self.rollout_pool is not None:
            return await self._collect_distributed()

        batch_records: list[RolloutRecord] = []
        for _ in range(self.cfg.prompts_per_iter):
            item = await self.env.get_next_item()
            batch_records.extend(await self._collect_group(item))
        return batch_records

    async def _one_iter(self, iter_idx: int) -> AlgoUpdateStats:
        self._profiler.reset()
        if self.lagrangian is not None:
            self.lagrangian.begin_iter()

        with self._profiler.measure("iter.total"):
            self._set_gradient_checkpointing(False)
            with self._profiler.measure("rollout.collect"):
                batch_records = await self._collect_for_iter(iter_idx)
            with self._profiler.measure("update.total"):
                gc_applied = self._set_gradient_checkpointing(True)
                try:
                    stats = self._update_on_records(batch_records, iter_idx)
                finally:
                    self._set_gradient_checkpointing(False)
        stats.extra.update(self._profiler.as_metrics())
        if bool(getattr(self.cfg, "gradient_checkpointing", False)):
            stats.extra["gradient_checkpointing"] = 1.0 if gc_applied else 0.0
            stats.extra["gradient_checkpointing_supported"] = (
                1.0 if self._gradient_checkpointing_supported else 0.0
            )
        return stats

    def _update_on_records(
        self, batch_records: list[RolloutRecord], iter_idx: int
    ) -> AlgoUpdateStats:
        # v0.8: running-reward normalization BEFORE prepare so the reward
        # used for advantage computation is whitened, while `raw_reward`
        # survives in metadata for logging.
        with self._profiler.measure("reward.normalize"):
            batch_records = self._maybe_normalize_rewards(batch_records)

        # ── Replay buffer mixing (off-policy with TIS correction) ──────────
        # Mix in stale-but-recent records from the replay buffer. Current
        # records are pushed into the buffer for future iters but are NOT
        # reused this iter (avoids double-counting). TIS/V-trace in the
        # algo corrects for staleness automatically.
        with self._profiler.measure("replay.mix"):
            if self._replay_buffer is not None:
                batch_records = self._replay_buffer.mix_with_current(
                    current=batch_records,
                    mix_ratio=float(getattr(self.cfg, "replay_mix_ratio", 0.25)),
                    policy_version=self._update_version,
                )

        # Cache the current-iter batch for PRM pipeline co-training.
        # We snapshot before potential in-place mutations below.
        if self._prm_pipeline is not None:
            self._last_train_batch = RolloutBatch(records=list(batch_records))

        # ── Memory-aware reward shaping (cross-session improvement bonus) ──
        # Runs AFTER normalization but BEFORE advantage computation so the
        # bonus is included in the advantage signal.
        with self._profiler.measure("reward.memory_shape"):
            if self._memory_shaper is not None:
                batch_records = self._memory_shaper.shape_records(
                    batch_records,
                    tokenizer=getattr(self.policy, "tokenizer", None),
                )
        # Stamp iteration index into metadata so curriculum-aware shaping
        # functions (see rewards/shaping.py :: curriculum_shaping) can scale
        # their effect with training progress.
        for rec in batch_records:
            meta = getattr(rec, "metadata", None)
            if isinstance(meta, dict):
                meta["_trainer_iter"] = int(iter_idx)

        # Apply reward shaping (if configured via subclass constructor).
        with self._profiler.measure("reward.shape"):
            if self._reward_shaping_fn is not None:
                batch_records = list(self._reward_shaping_fn(batch_records))

        # OPD in-trainer hint extraction: recover directive hints from the
        # next-state signal for records that don't already carry one. Must run
        # BEFORE the teacher fill so newly-stamped hints get teacher logprobs.
        # OpenClaw-RL §3.2 produces hints from the PRM judge; this is the
        # in-trainer equivalent that keeps OPD from silently degrading to GRPO.
        with self._profiler.measure("opd.hint_extract"):
            if self._hint_extractor is not None:
                hint_stats = self._hint_extractor.extract(batch_records)
                self._last_hint_extract_stats = hint_stats.as_dict()
            else:
                # Always emit zero-value OPD hint metrics so dashboards don't lose
                # the series when the extractor is disabled or not yet configured.
                self._last_hint_extract_stats = {
                    "opd_hint_n_records": float(len(batch_records)),
                    "opd_hint_n_with_next_state": 0.0,
                    "opd_hint_n_judged": 0.0,
                    "opd_hint_n_added": 0.0,
                    "opd_hint_n_rejected_quality": 0.0,
                    "opd_hint_n_errors": 0.0,
                    "opd_hint_effective_rate": 0.0,
                }

        # OPD teacher-logprob fill: re-score hinted records under a
        # hint-enhanced context so the OPD / Hybrid branch has a real teacher
        # distribution (OpenClaw-RL §3.2). No-op when no hints are present.
        self._last_teacher_fill_stats: dict[str, float] | None = None
        with self._profiler.measure("opd.teacher_fill"):
            if self._teacher_filler is not None:
                fill_stats = self._teacher_filler.fill(batch_records)
                self._last_teacher_fill_stats = fill_stats.as_dict()

        with self._profiler.measure("update.prepare_batch"):
            batch = self._prepare_update_batch(RolloutBatch(records=batch_records))

        # v0.8: adaptive KL — sync β into algo.cfg BEFORE computing loss for
        # this iter. The previous iter's KL drove the update.
        if self._kl_ctrl is not None:
            algo_cfg = getattr(self.algo, "cfg", None)
            if algo_cfg is not None and hasattr(algo_cfg, "kl_coef"):
                algo_cfg.kl_coef = float(self._kl_ctrl.value)

        # Entropy coefficient schedule — sync entropy_coef into algo.cfg
        # BEFORE computing loss. For step schedules (linear/exp/cosine) the
        # coefficient is a function of the iteration. For the PID controller
        # the coefficient is whatever the previous iter's entropy drove it to
        # (updated at the end of this method).
        entropy_coef_applied: float | None = None
        if self._entropy_sched is not None:
            algo_cfg = getattr(self.algo, "cfg", None)
            if algo_cfg is not None and hasattr(algo_cfg, "entropy_coef"):
                from hermes_agentic_rl.algos.entropy_schedule import (
                    TargetEntropyPID,
                )

                if isinstance(self._entropy_sched, TargetEntropyPID):
                    coef = float(self._entropy_sched.coef)
                else:
                    coef = float(self._entropy_sched.step(iter_idx))
                algo_cfg.entropy_coef = coef
                entropy_coef_applied = coef

        schedule_metrics = self._apply_hybrid_weight_schedules(iter_idx)

        with self._profiler.measure("update.build_batches"):
            update_batches = self._build_update_batches(batch, iter_idx=iter_idx)
        per_step_stats: list[AlgoUpdateStats] = []
        early_stopped = False
        last_approx_kl = 0.0

        target_kl = float(getattr(self.cfg, "target_kl", 0.0) or 0.0)

        self.optim.zero_grad()
        for mb_idx, mini_batch in enumerate(update_batches):
            # v0.9: autocast the forward pass
            with self._amp.autocast_ctx():
                with self._profiler.measure("algo.compute_loss"):
                    loss, stats = self.algo.compute_loss(self.policy, self.ref_policy, mini_batch)
                if self.lagrangian is not None:
                    loss = self.lagrangian.penalty_term(loss)

            grad_norm = 0.0
            grad_accum_denom = float(self.cfg.grad_accum_steps)
            if loss.requires_grad:
                # v0.9: AMP scale + divide by grad_accum_steps
                self._amp.scale(loss / grad_accum_denom).backward()

            # v0.9: only step when grad_accum counter fires. Unscale and clip
            # BEFORE optimizer.step(); doing it after step silently made
            # grad_clip a no-op on normal minibatches.
            did_step = self._grad_accum.advance()
            if did_step:
                with self._profiler.measure("optimizer.step"):
                    grad_norm = self._optimizer_step(requires_grad=loss.requires_grad)

            stats.extra["grad_norm"] = grad_norm
            stats.extra["param_norm"] = _param_l2_norm(self._trainable_params)
            stats.extra["lr"] = float(self.optim.param_groups[0].get("lr", 0.0))
            stats.extra["optimizer_step_applied"] = 1.0 if did_step else 0.0
            stats.extra["amp_scale"] = self._amp.get_scale()
            stats.extra["grad_accum_step"] = float(mb_idx + 1)
            stats.extra["grad_clip_triggered"] = (
                1.0
                if (
                    self.cfg.grad_clip and self.cfg.grad_clip > 0 and grad_norm > self.cfg.grad_clip
                )
                else 0.0
            )
            per_step_stats.append(stats)

            # v0.8: ratio-based early stop.
            ak = float(stats.extra.get("approx_kl", 0.0) or 0.0)
            last_approx_kl = ak
            if target_kl > 0 and ak > 1.5 * target_kl:
                early_stopped = True
                break

        # v0.9: drain remaining grad_accum steps if any.
        if self._grad_accum.has_pending():
            # Force a final step with whatever's in the buffer.
            with self._profiler.measure("optimizer.step"):
                self._optimizer_step(requires_grad=True)

        # v0.8: feed the last-seen approx_kl into the adaptive controller.
        if self._kl_ctrl is not None and per_step_stats:
            # Use the mean approx_kl across all executed minibatches.
            kl_vals = [float(s.extra.get("approx_kl", 0.0) or 0.0) for s in per_step_stats]
            mean_kl = sum(kl_vals) / max(1, len(kl_vals))
            new_beta = self._kl_ctrl.update(mean_kl, n_steps=len(kl_vals))
            for s in per_step_stats:
                s.extra["adaptive_kl_coef"] = float(new_beta)

        with self._profiler.measure("update.aggregate_stats"):
            agg = self._aggregate_update_stats(
                batch=batch,
                step_stats=per_step_stats,
                n_update_batches=len(update_batches),
            )
        # Entropy schedule bookkeeping: record the coef used this iter and
        # (for PID) feed the measured entropy back to drive the next iter.
        if self._entropy_sched is not None:
            from hermes_agentic_rl.algos.entropy_schedule import TargetEntropyPID

            if entropy_coef_applied is not None:
                agg.extra["entropy_coef"] = float(entropy_coef_applied)
            if isinstance(self._entropy_sched, TargetEntropyPID):
                next_coef = float(self._entropy_sched.update(float(agg.entropy)))
                agg.extra["entropy_coef_next"] = next_coef
        if early_stopped:
            agg.extra["early_stopped_by_kl"] = 1.0
            agg.extra["last_minibatch_approx_kl"] = last_approx_kl
        if schedule_metrics:
            agg.extra.update(schedule_metrics)
        if self._reward_rms is not None:
            agg.extra["reward_norm_mean"] = float(self._reward_rms.mean)
            agg.extra["reward_norm_std"] = float(self._reward_rms.std)
            # Restore mean_reward to RAW scale for logging (the normalized
            # scalar that drove the gradient is in `mean_advantage`).
            raws = [float(r.metadata.get("raw_reward", r.reward)) for r in batch.records]
            if raws:
                agg.mean_reward = sum(raws) / len(raws)
        if self._last_teacher_fill_stats:
            agg.extra.update(self._last_teacher_fill_stats)
        if getattr(self, "_last_hint_extract_stats", None):
            agg.extra.update(self._last_hint_extract_stats)
        # P0-2 observability: how stale were the rollouts that drove this update
        # (0 for synchronous/BSP, ~1 in steady-state pipelined mode), and the
        # learner's policy version (number of completed updates so far).
        agg.extra["rollout_staleness"] = float(self._rollout_staleness)
        agg.extra["policy_version"] = float(self._update_version)

        # Staleness-adaptive TIS: observe this iter's staleness and push the
        # adjusted rho_clip into the algo config for the NEXT iter's loss
        # computation. The algo's fixed tis_rho_clip is overridden when active.
        if self._staleness_tis is not None:
            self._staleness_tis.observe_staleness(self._rollout_staleness)
            tis_cfg = self._staleness_tis.get_config()
            algo_cfg = getattr(self.algo, "cfg", None)
            if algo_cfg is not None and hasattr(algo_cfg, "tis_rho_clip"):
                if tis_cfg.enabled:
                    algo_cfg.tis_rho_clip = float(tis_cfg.rho_clip)
            agg.extra.update(self._staleness_tis.stats())

        self._update_version += 1
        # Replay buffer stats (when enabled).
        if self._replay_buffer is not None:
            agg.extra.update(self._replay_buffer.stats.as_dict())
            n_replayed = sum(
                1
                for rec in batch.records
                if isinstance(getattr(rec, "metadata", None), dict)
                and rec.metadata.get("_replay_sampled")
            )
            agg.extra["replay_n_in_batch"] = float(n_replayed)
        return agg

    def train(self) -> TrainStats:
        start = int(getattr(self, "_start_iter", 0))
        if start == 0:
            bootstrap_record = self._maybe_run_bootstrap_sft()
            if bootstrap_record:
                self.stats.add(bootstrap_record)
                if self.cfg.log_every:
                    self.logger(bootstrap_record)
                if self.cfg.metrics_sink is not None:
                    try:
                        self.cfg.metrics_sink(bootstrap_record)
                    except Exception:
                        pass
        last_iter = start
        for it in range(start, self.cfg.n_iters):
            last_iter = it
            # v0.9: sync weights to vLLM before rollout.
            if self._vllm_rollout is not None and it > 0:
                if self._lora_hot_reload is not None:
                    # LoRA hot-reload: merge deltas and push to vLLM.
                    self._lora_hot_reload.sync_to_vllm()
                else:
                    sync_every = max(1, int(self.cfg.vllm_sync_every))
                    if it % sync_every == 0:
                        self._sync_weights_to_vllm(self.policy)
            stats = asyncio.run(self._one_iter(it))
            if self.lagrangian is not None:
                self.lagrangian.dual_step()
            record = {"iter": it, "algo": self.algo_name, **stats.as_dict()}
            sft_metrics = self._maybe_run_interleaved_sft(it)
            if sft_metrics:
                record.update(sft_metrics)
            if self.lagrangian is not None:
                record["lagrangian"] = self.lagrangian.snapshot()
            env_snapshot = getattr(self.env, "snapshot", None)
            if callable(env_snapshot):
                try:
                    record["env_snapshot"] = env_snapshot()
                except Exception:
                    pass
            self.stats.add(record)

            # EMA shadow update (after the learner step).
            if self._ema is not None:
                self._ema.update(self.policy)
                record["ema_rollout"] = 1.0
                record["ema_tau"] = self._ema.current_tau()

            # PRM online co-training (after policy update, using latest batch).
            if self._prm_pipeline is not None:
                _last_batch = getattr(self, "_last_train_batch", None)
                if _last_batch is not None:
                    tokenizer = getattr(self.policy, "tokenizer", None)
                    if tokenizer is not None:
                        try:
                            prm_metrics = self._prm_pipeline.step(
                                it, _last_batch, tokenizer=tokenizer
                            )
                            if prm_metrics:
                                record.update(prm_metrics)
                        except Exception as _prm_e:
                            record["prm_error"] = str(_prm_e)

            # Dynamic reward balancer: update component weights based on
            # per-batch variance (differentiator over OpenClaw-RL static weights).
            if self._dynamic_balancer is not None:
                try:
                    new_weights = self._dynamic_balancer.observe_batch(it, record)
                    # Apply updated weights to reward_manager components.
                    for comp in getattr(self.reward_manager, "components", []):
                        name = getattr(comp, "name", None)
                        if name is not None and name in new_weights:
                            comp.weight = new_weights[name]
                    record.update(self._dynamic_balancer.snapshot())
                except Exception:
                    pass

            # Curriculum scheduler: observe batch stats and maybe advance stage.
            # This integrates with DynamicRewardBalancer by adjusting the
            # base_weights based on curriculum stage progression.
            if self._curriculum_scheduler is not None:
                try:
                    self._curriculum_scheduler.observe_batch_stats(it, record)
                    if self._curriculum_scheduler.should_advance():
                        old_name = self._curriculum_scheduler.get_current_stage().name
                        self._curriculum_scheduler.advance()
                        new_stage = self._curriculum_scheduler.get_current_stage()
                        if new_stage is not None:
                            record["curriculum_advanced"] = 1.0
                            record["curriculum_from"] = old_name
                            record["curriculum_to"] = new_stage.name
                            # Update dynamic balancer base weights if present
                            if self._dynamic_balancer is not None:
                                stage_weights = self._curriculum_scheduler.get_stage_weights()
                                for wname, wval in stage_weights.items():
                                    if wname in self._dynamic_balancer.base_weights:
                                        self._dynamic_balancer.base_weights[wname] = wval
                    record.update(self._curriculum_scheduler.snapshot())
                except Exception:
                    pass

            # Memory shaper snapshot for logging.
            if self._memory_shaper is not None:
                record.update(self._memory_shaper.snapshot())

            # Reference policy periodic re-clone (prevents KL drift).
            ref_every = int(getattr(self.cfg, "ref_update_every", 0))
            if ref_every > 0 and self.ref_policy is not None and it > 0 and it % ref_every == 0:
                if hasattr(self.policy, "clone_frozen"):
                    self.ref_policy = self.policy.clone_frozen()
                    record["ref_policy_updated"] = 1.0

            # Eval hook: periodically evaluate with deterministic decoding.
            eval_every = int(getattr(self.cfg, "eval_every", 0))
            if eval_every > 0 and it > 0 and it % eval_every == 0:
                eval_record = self._run_eval_hook(it)
                if eval_record:
                    record.update(eval_record)

            # Best-reward tracking + best checkpoint + early-stop counter.
            mean_r = float(record.get("mean_reward", 0.0))
            improved = mean_r > (self._best_reward + self.cfg.early_stop_min_delta)
            if improved:
                self._best_reward = mean_r
                self._best_iter = it
                self._iters_since_best = 0
                if self._best_ckpt_manager is not None and hasattr(self.policy, "model"):
                    self._save_full_checkpoint(it, manager=self._best_ckpt_manager)
            else:
                self._iters_since_best += 1

            if self.cfg.log_every and (it % self.cfg.log_every == 0):
                self.logger(record)
            if self.cfg.metrics_sink is not None:
                try:
                    self.cfg.metrics_sink(record)
                except Exception:
                    # metrics must never break training
                    pass
            if self._profiler.enabled and self._profile_output_path is not None:
                profile_record = {
                    "iter": it,
                    "algo": self.algo_name,
                    **{
                        k: v
                        for k, v in record.items()
                        if isinstance(k, str) and k.startswith("time_")
                    },
                }
                append_jsonl(self._profile_output_path, profile_record)
            if (
                self.cfg.save_every
                and self.cfg.output_dir is not None
                and self.cfg.save_every > 0
                and it > 0
                and it % self.cfg.save_every == 0
            ):
                self._save_checkpoint(it)
            if (
                self._ckpt_manager is not None
                and self.cfg.checkpoint_every > 0
                and it > 0
                and it % self.cfg.checkpoint_every == 0
            ):
                self._save_full_checkpoint(it, background=True)

            # Early-stop check (after all per-iter side effects).
            if (
                self.cfg.early_stop_patience > 0
                and self._iters_since_best >= self.cfg.early_stop_patience
            ):
                print(
                    f"[train] early stop at iter={it} "
                    f"(best={self._best_reward:.4f} @ iter {self._best_iter}; "
                    f"patience={self.cfg.early_stop_patience} exhausted)"
                )
                self._early_stopped = True
                break

        # Always emit a final checkpoint if ckpt manager is active.
        if self._ckpt_manager is not None and self.cfg.n_iters > start:
            self._save_full_checkpoint(last_iter)
        if self._async_ckpt_saver is not None:
            try:
                self._async_ckpt_saver.flush()
            finally:
                self._async_ckpt_saver.close()
        return self.stats

    # ------------------------------------------------------------------
    # checkpoint helpers
    # ------------------------------------------------------------------

    def _save_full_checkpoint(
        self,
        it: int,
        manager: Any | None = None,
        *,
        background: bool = False,
    ) -> None:
        """Save a {model, optimizer, rng, stats} bundle via CheckpointManager.

        If ``manager`` is None, uses the default checkpoint manager. Passing an
        alternate manager (e.g. ``self._best_ckpt_manager``) writes to a
        separate directory with its own retention policy.

        When ``background=True`` and ``async_checkpoint`` is enabled, the CPU
        snapshot is taken synchronously but disk I/O runs on a worker thread.
        Best and final checkpoints always call with ``background=False``.
        """
        target_mgr = manager or self._ckpt_manager
        if target_mgr is None or not hasattr(self.policy, "model"):
            return
        from hermes_agentic_rl.trainers.checkpoint import (
            CheckpointState,
            capture_rng_state,
            snapshot_state_to_cpu,
        )

        state = CheckpointState(
            iteration=it,
            model_state=self.policy.model.state_dict(),  # type: ignore[attr-defined]
            optimizer_state=self._optim.state_dict(),
            rng_state=capture_rng_state(),
            stats=list(self.stats.iters),
            config=_config_to_dict(self.cfg),
            best_reward=self._best_reward,
            best_iteration=self._best_iter,
            # Reward normalizer state (RunningMeanStd).
            running_stats=(self._reward_rms.state_dict() if self._reward_rms is not None else None),
            # Adaptive KL controller state (P or PID).
            kl_ctrl_state=(self._kl_ctrl.state_dict() if self._kl_ctrl is not None else None),
            # EMA shadow weights so rollout behavior is deterministic on resume.
            ema_state=(
                {
                    k: v.detach().cpu()
                    for k, v in self._ema.shadow.model.state_dict().items()  # type: ignore[attr-defined]
                }
                if self._ema is not None and hasattr(self._ema.shadow, "model")
                else None
            ),
            # PRM head weights for warm-start on next run.
            prm_head_state=(
                self._prm_pipeline.prm.head.state_dict() if self._prm_pipeline is not None else None
            ),
            # Curriculum scheduler state for resuming stage progression.
            curriculum_state=(
                self._curriculum_scheduler.state_dict()
                if self._curriculum_scheduler is not None
                else None
            ),
        )
        use_async = (
            background
            and self.cfg.async_checkpoint
            and self._async_ckpt_saver is not None
            and manager is None
        )
        if use_async:
            saver = self._async_ckpt_saver
            assert saver is not None
            saver.submit(target_mgr, snapshot_state_to_cpu(state))
        else:
            target_mgr.save(state)

    def _maybe_resume(self) -> None:
        if self._ckpt_manager is None:
            return
        from hermes_agentic_rl.trainers.checkpoint import (
            CheckpointState,
            restore_rng_state,
        )

        target: CheckpointState | None = None
        resume_from = self.cfg.resume_from
        if resume_from is not None and resume_from != "latest":
            try:
                target = self._ckpt_manager.load(int(resume_from))
            except Exception:
                target = None
            if target is None:
                raise RuntimeError(
                    f"resume_from={resume_from!r} requested but checkpoint not found"
                )
        elif resume_from == "latest" or self.cfg.auto_resume:
            target = self._ckpt_manager.load_latest()
            if target is None:
                return  # nothing to resume from; fresh start
        else:
            return

        if not hasattr(self.policy, "model"):
            return
        self.policy.model.load_state_dict(target.model_state)  # type: ignore[attr-defined]
        if target.optimizer_state is not None:
            try:
                self._optim.load_state_dict(target.optimizer_state)
            except Exception:
                pass  # optimizer mismatch shouldn't break resume
        if target.rng_state is not None:
            try:
                restore_rng_state(target.rng_state)
            except Exception:
                pass
        self.stats.iters = list(target.stats)
        self._best_reward = float(target.best_reward)
        self._best_iter = int(target.best_iteration)
        # Resume from the NEXT iteration — we already finished `iteration`.
        self._start_iter = int(target.iteration) + 1

        # Restore reward normalizer state.
        running_stats = getattr(target, "running_stats", None)
        if running_stats is not None and self._reward_rms is not None:
            try:
                self._reward_rms.load_state_dict(running_stats)
            except Exception:
                pass

        # Restore adaptive KL controller state.
        kl_ctrl_state = getattr(target, "kl_ctrl_state", None)
        if kl_ctrl_state is not None and self._kl_ctrl is not None:
            try:
                self._kl_ctrl.load_state_dict(kl_ctrl_state)
            except Exception:
                pass

        # Restore EMA shadow weights.
        ema_state = getattr(target, "ema_state", None)
        if ema_state is not None and self._ema is not None and hasattr(self._ema.shadow, "model"):
            try:
                self._ema.shadow.model.load_state_dict(ema_state, strict=False)  # type: ignore[attr-defined]
            except Exception:
                pass  # shape mismatch → shadow will re-track from scratch

        # Restore PRM head weights for warm-start.
        prm_head_state = getattr(target, "prm_head_state", None)
        if prm_head_state is not None and self._prm_pipeline is not None:
            try:
                self._prm_pipeline.prm.head.load_state_dict(prm_head_state)
            except Exception:
                pass

        # Restore curriculum scheduler state.
        curriculum_state = getattr(target, "curriculum_state", None)
        if curriculum_state is not None and self._curriculum_scheduler is not None:
            try:
                self._curriculum_scheduler.load_state_dict(curriculum_state)
            except Exception:
                pass

        print(
            f"[train] resumed from iter={target.iteration} "
            f"(best_reward={self._best_reward:.4f} start_iter={self._start_iter})"
        )

    def _save_checkpoint(self, it: int) -> None:
        if self.cfg.output_dir is None:
            return
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"policy_iter_{it:04d}.pt"
        if hasattr(self.policy, "model"):
            torch.save(self.policy.model.state_dict(), target)  # type: ignore[attr-defined]

    def _default_logger(self, rec: dict[str, Any]) -> None:
        if str(getattr(self.cfg, "log_format", "text")).lower() == "json":
            print(format_json_record(rec))
        else:
            print(self._format_log(rec))

    def _format_log(self, rec: dict[str, Any]) -> str:
        keys = [
            "iter",
            "algo",
            "mean_reward",
            "loss",
            "policy_loss",
            "value_loss",
            "turn_credit_reward_mean",
            "turn_credit_local_component_mean",
            "turn_credit_final_component_mean",
            "sft_loss",
            "mean_advantage",
            "kl",
            "clip_frac",
            "n_updated",
        ]
        # Keys that are internal / noisy and should not appear in the log.
        _suppressed = {
            "n_tokens",
            "ratio_mean",
            "optimizer_step_applied",
            "amp_scale",
            "grad_accum_step",
        }
        parts = []
        seen_keys: set[str] = set()
        for k in keys:
            v = rec.get(k)
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif v is not None:
                parts.append(f"{k}={v}")
            seen_keys.add(k)
        # Append any extra numeric keys not in the primary list and not suppressed.
        for k, v in rec.items():
            if k in seen_keys or k in _suppressed:
                continue
            if isinstance(v, int | float):
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        return "[train] " + " ".join(parts)

    def _build_update_batches(self, batch: RolloutBatch, *, iter_idx: int) -> list[RolloutBatch]:
        # Delegates to minibatch_builder for testability and code-size reduction.
        return _build_update_batches_fn(
            batch,
            update_epochs=max(1, int(self.cfg.update_epochs)),
            minibatch_size=int(self.cfg.minibatch_size),
            shuffle_minibatches=bool(self.cfg.shuffle_minibatches),
            preserve_group_boundaries=self._preserve_group_boundaries(),
            seed=self.cfg.seed,
            iter_idx=iter_idx,
        )

    def _split_minibatches(
        self,
        batch: RolloutBatch,
        *,
        minibatch_size: int,
        iter_idx: int,
        epoch_idx: int,
    ) -> list[RolloutBatch]:
        from hermes_agentic_rl.trainers.minibatch_builder import _make_rng

        return _split_minibatches_fn(
            batch,
            minibatch_size=minibatch_size,
            shuffle=bool(self.cfg.shuffle_minibatches),
            preserve_group_boundaries=self._preserve_group_boundaries(),
            rng=_make_rng(seed=self.cfg.seed, iter_idx=iter_idx, epoch_idx=epoch_idx),
        )

    def _minibatch_rng(self, *, iter_idx: int, epoch_idx: int) -> random.Random:
        from hermes_agentic_rl.trainers.minibatch_builder import _make_rng

        return _make_rng(seed=self.cfg.seed, iter_idx=iter_idx, epoch_idx=epoch_idx)

    def _aggregate_update_stats(
        self,
        *,
        batch: RolloutBatch,
        step_stats: list[AlgoUpdateStats],
        n_update_batches: int,
    ) -> AlgoUpdateStats:
        return _aggregate_update_stats_fn(
            batch=batch,
            step_stats=step_stats,
            n_update_batches=n_update_batches,
            update_epochs=max(1, int(self.cfg.update_epochs)),
            minibatch_size=int(self.cfg.minibatch_size),
            summarize_batch_meta_fn=_summarize_batch_metadata,
        )

    def _run_eval_hook(self, iter_idx: int) -> dict[str, Any] | None:
        """Run a quick evaluation roll-out with deterministic decoding.

        Uses the same env but sets temperature=0 (greedy) and collects
        ``eval_prompts`` items. Returns a dict of ``eval_*`` metrics or None
        if the env has no items.
        """
        n_eval = max(1, int(getattr(self.cfg, "eval_prompts", 4)))
        eval_temp = float(getattr(self.cfg, "eval_temperature", 0.0))

        eval_records: list[RolloutRecord] = []
        for _ in range(n_eval):
            try:
                item = asyncio.run(self.env.get_next_item())
            except StopIteration:
                break
            try:
                records = asyncio.run(
                    self._collect_group(
                        item,
                        temperature=eval_temp,
                        group_size=1,
                    )
                )
                eval_records.extend(records)
            except Exception:
                continue

        if not eval_records:
            return None

        rewards = [float(rec.reward) for rec in eval_records]
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        return {
            "eval_mean_reward": mean_r,
            "eval_n_records": float(len(eval_records)),
            "eval_temperature": eval_temp,
        }

    def _maybe_run_interleaved_sft(self, iter_idx: int) -> dict[str, Any]:
        every = max(0, int(self.cfg.interleave_sft_every))
        if every <= 0 or iter_idx <= 0 or iter_idx % every != 0:
            return {}
        samples = asyncio.run(
            self._collect_supervised_samples(int(self.cfg.interleave_sft_samples))
        )
        if not samples:
            raise RuntimeError(
                "interleave_sft is enabled, but the active environment produced no "
                "supervised samples. Implement build_supervised_samples(item) on the env "
                "or disable interleave_sft_every."
            )
        return self._run_supervised_updates(
            samples,
            lr=float(self.cfg.interleave_sft_lr),
            epochs=max(1, int(self.cfg.interleave_sft_epochs)),
        )

    def _maybe_run_bootstrap_sft(self) -> dict[str, Any]:
        rounds = max(0, int(self.cfg.bootstrap_sft_rounds))
        if rounds <= 0:
            return {}
        asyncio.run(self.env.setup())

        losses: list[float] = []
        total_samples = 0
        total_steps = 0
        for _ in range(rounds):
            samples = asyncio.run(
                self._collect_supervised_samples(int(self.cfg.bootstrap_sft_samples))
            )
            if not samples:
                raise RuntimeError(
                    "bootstrap_sft is enabled, but the active environment produced no "
                    "supervised samples. Implement build_supervised_samples(item) on the env "
                    "or disable bootstrap_sft_rounds."
                )
            metrics = self._run_supervised_updates(
                samples,
                lr=float(self.cfg.bootstrap_sft_lr),
                epochs=max(1, int(self.cfg.bootstrap_sft_epochs)),
            )
            if "sft_loss" in metrics:
                losses.append(float(metrics["sft_loss"]))
            total_samples += int(metrics.get("n_sft_samples", 0))
            total_steps += int(metrics.get("n_sft_steps", 0))

        return {
            "iter": -1,
            "algo": "sft_bootstrap",
            "mean_reward": 0.0,
            "loss": (sum(losses) / len(losses)) if losses else 0.0,
            "sft_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "n_sft_samples": total_samples,
            "n_sft_steps": total_steps,
            "bootstrap_sft_rounds": rounds,
        }

    async def _collect_supervised_samples(self, n_items: int) -> list[SupervisedSample]:
        out: list[SupervisedSample] = []
        for _ in range(max(1, n_items)):
            item = await self.env.get_next_item()
            out.extend(self.env.build_supervised_samples(item))
        return [
            sample
            for sample in out
            if str(sample.instruction).strip() and str(sample.response).strip()
        ]

    def _run_supervised_updates(
        self,
        samples: list[SupervisedSample],
        *,
        lr: float,
        epochs: int,
    ) -> dict[str, Any]:
        batch_size = max(1, min(int(self.cfg.interleave_sft_batch_size), len(samples)))
        prev_lrs = [float(group["lr"]) for group in self.optim.param_groups]
        for group in self.optim.param_groups:
            group["lr"] = float(lr)

        losses: list[float] = []
        n_steps = 0
        try:
            for epoch_idx in range(epochs):
                ordered = list(samples)
                rng_epoch = self._minibatch_rng(
                    iter_idx=len(self.stats.iters) + 1,
                    epoch_idx=epoch_idx + 1,
                )
                rng_epoch.shuffle(ordered)
                for start in range(0, len(ordered), batch_size):
                    batch = ordered[start : start + batch_size]
                    if not batch:
                        continue
                    inp, labels, loss_mask = self._collate_supervised_batch(batch)
                    self.optim.zero_grad()
                    logits = self._forward_model_logits(inp)
                    ce = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        reduction="none",
                    ).reshape(labels.shape)
                    loss = (ce * loss_mask.float()).sum() / loss_mask.sum().clamp(min=1)
                    loss.backward()
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._trainable_params,
                            max_norm=self.cfg.grad_clip,
                        )
                    self.optim.step()
                    losses.append(float(loss.detach().item()))
                    n_steps += 1
        finally:
            for group, lr in zip(self.optim.param_groups, prev_lrs, strict=False):
                group["lr"] = lr

        return {
            "sft_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "n_sft_samples": len(samples),
            "n_sft_steps": n_steps,
        }

    def _collate_supervised_batch(
        self, batch: list[SupervisedSample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows: list[tuple[list[int], int]] = []
        max_len = self._policy_max_sequence_length()
        for sample in batch:
            obs = self._prompt_encoder.encode({"instruction": sample.instruction})
            prompt_ids = list(obs.prompt_ids)
            if sample.prompt_suffix:
                prompt_ids.extend(self.policy.tokenizer.encode(sample.prompt_suffix))
            response_ids = self.policy.tokenizer.encode(sample.response, add_eos=True)
            full = prompt_ids + response_ids
            if max_len is not None and len(full) > max_len:
                drop = len(full) - max_len
                full = full[drop:]
                prompt_ids = prompt_ids[drop:] if drop < len(prompt_ids) else []
            rows.append((full, len(prompt_ids)))

        max_len = max(len(full_ids) for full_ids, _prompt_len in rows)
        device = self._trainable_params[0].device
        pad_id = int(getattr(self.policy.tokenizer, "pad_id", 0))
        inp = torch.full((len(rows), max_len - 1), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(rows), max_len - 1), pad_id, dtype=torch.long, device=device)
        loss_mask = torch.zeros((len(rows), max_len - 1), dtype=torch.bool, device=device)

        for row_idx, (full_ids, prompt_len) in enumerate(rows):
            input_ids = full_ids[:-1]
            target_ids = full_ids[1:]
            n = len(input_ids)
            if n <= 0:
                continue
            inp[row_idx, :n] = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels[row_idx, :n] = torch.tensor(target_ids, dtype=torch.long, device=device)
            start = max(0, prompt_len - 1)
            loss_mask[row_idx, start:n] = True
        return inp, labels, loss_mask

    def _policy_max_sequence_length(self) -> int | None:
        cfg = getattr(self.policy, "cfg", None)
        for source in (cfg, getattr(self.policy, "model", None)):
            if source is None:
                continue
            max_len = getattr(source, "max_len", None)
            if isinstance(max_len, int) and max_len > 0:
                return max_len
        return None

    def _forward_model_logits(self, inp: torch.Tensor) -> torch.Tensor:
        out = self.policy.model(inp)  # type: ignore[attr-defined]
        return out.logits if hasattr(out, "logits") else out


# NOTE: Module-level helpers moved to _rollout_helpers.py and imported above.
