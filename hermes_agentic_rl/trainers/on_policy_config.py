from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OnPolicyTrainerConfig:
    n_iters: int = 20
    group_size: int = 4
    prompts_per_iter: int = 2
    lr: float = 1e-3
    max_new_tokens: int = 16
    temperature: float = 1.0
    grad_clip: float = 1.0
    use_reference: bool = False
    # How often to re-clone the reference policy from the current policy.
    # 0 = never (classic frozen-ref), N = every N iters. Prevents KL drift
    # in long training runs. When adaptive_kl is also enabled, the ref-update
    # interval acts as a ceiling on how far the KL anchor can drift.
    ref_update_every: int = 0
    multi_turn: bool = False
    multi_turn_credit: dict[str, Any] | None = None
    log_every: int = 1
    log_format: str = "text"
    save_every: int = 0
    output_dir: Path | None = None
    seed: int | None = 0
    metrics_sink: Callable[[dict[str, Any]], None] | None = None
    profile: bool = False
    profile_output_path: Path | None = None
    batch_generate: bool = False
    update_epochs: int = 1
    minibatch_size: int = 0
    shuffle_minibatches: bool = True
    interleave_sft_every: int = 0
    interleave_sft_samples: int = 32
    interleave_sft_lr: float = 1e-4
    interleave_sft_epochs: int = 1
    interleave_sft_batch_size: int = 8
    bootstrap_sft_rounds: int = 0
    bootstrap_sft_samples: int = 32
    bootstrap_sft_lr: float = 1e-4
    bootstrap_sft_epochs: int = 1
    checkpoint_every: int = 0
    keep_last_checkpoints: int = 3
    resume_from: int | str | None = None
    auto_resume: bool = False
    save_best_checkpoint: bool = False
    # When True, periodic checkpoints are written from a background
    # thread so the training loop is not blocked on disk I/O. The
    # tensor snapshot is still produced synchronously (so subsequent
    # iterations may safely mutate the live model), but the multi-second
    # serialisation runs in parallel with the next iteration. Best-bundle
    # and final checkpoints always remain synchronous.
    async_checkpoint: bool = False
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4
    target_kl: float = 0.0
    adaptive_kl: bool = False
    adaptive_kl_horizon: float = 10000.0
    adaptive_kl_min: float = 1e-4
    adaptive_kl_max: float = 10.0
    # KL controller type: "p" = InstructGPT proportional (default, backward-
    # compatible); "pid" = PID controller (DeepSeek/DAPO style, eliminates
    # steady-state error and reacts to oscillations).
    adaptive_kl_type: str = "p"
    # PID gains — only used when adaptive_kl_type = "pid".
    adaptive_kl_Kp: float = 0.1
    adaptive_kl_Ki: float = 0.01
    adaptive_kl_Kd: float = 0.005
    adaptive_kl_I_max: float = 2.0
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
    token_budget: dict[str, Any] | None = None
    entropy_schedule: dict[str, Any] | None = None
    use_ema_rollout: bool = False
    ema_tau: float = 0.005
    # Optional EMA tau warm-up: anneal tau from ``ema_tau_start`` down to
    # ``ema_tau`` over ``ema_tau_warmup_steps`` updates so the shadow tracks
    # the fast-moving learner early in training. ema_tau_start=None disables.
    ema_tau_start: float | None = None
    ema_tau_warmup_steps: int = 0
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    lr_warmup_start_lr: float = 0.0
    lr_total_steps: int = 0
    lr_end_lr: float = 0.0
    # --- OPD teacher-logprob closed loop (OpenClaw-RL §3.2) ---
    # When True, after rewards are computed each iter, re-score hinted records
    # under a hint-enhanced context (self-distillation) to fill
    # ``metadata["teacher_logprobs"]`` so the OPD / Hybrid branch fires.
    opd_teacher_fill: bool = False
    opd_hint_template: str = "\n\n[HINT_START]{hint}[HINT_END]\n"
    opd_teacher_max_hint_tokens: int = 128
    # Optional {axis_name: weight} map for capability-axis-aware OPD weighting.
    opd_capability_axis_weights: dict[str, float] | None = None
    # --- OPD in-trainer hint extraction (closes the "OPD silently dies" gap) ---
    opd_hint_extractor: dict[str, Any] | None = None
    # --- P0-2: pipelined (double-buffered) rollout/update ---
    # When True AND a rollout_pool is attached, dispatch iter N+1's rollouts
    # before iter N's gradient update so they overlap (tolerates 1-step policy
    # staleness; PPO/GRPO ratio clipping absorbs the lag).
    pipeline_rollouts: bool = False
    # --- Replay buffer (off-policy mixing with TIS correction) ---
    # When enabled, a fraction of recent-but-not-current rollouts are mixed into
    # each update. TIS/V-trace corrects for staleness. See replay_buffer.py.
    replay_buffer: dict[str, Any] | None = None
    replay_mix_ratio: float = 0.25
    # --- Hybrid objective weight schedules ---
    # Optional schedules for HybridAlgo weights. Dict shape:
    # {mode: linear|cosine|constant, start: float, end: float, total_steps: int}
    w_rl_schedule: dict[str, Any] | None = None
    w_opd_schedule: dict[str, Any] | None = None
    # --- Eval hook (periodic evaluation during training) ---
    # When > 0, run evaluation every N iters using the same env but with
    # deterministic decoding (temperature=0). Results are logged as
    # ``eval_*`` keys in the iteration record.
    eval_every: int = 0
    eval_prompts: int = 4
    eval_temperature: float = 0.0
    # --- PRM online co-training pipeline ---
    # When set, a ProcessRewardModel is trained online every ``prm_train_every``
    # iters using ORM pseudo-labels derived from the current rollout batch.
    # Dict keys match PRMPipelineConfig fields plus ``prm_model_path`` (optional
    # path to a pre-trained PRM head for warm start).
    prm_pipeline: dict[str, Any] | None = None
    # --- Memory-aware reward shaping (cross-session improvement bonus) ---
    # Differentiator over OpenClaw-RL: rewards improvement over the agent's OWN
    # per-task history, not just the current batch baseline. Keys: alpha,
    # bonus_coef, clip_max, sigma_floor, min_obs, axis_bonus_coef, task_key_mode.
    memory_reward: dict[str, Any] | None = None
    # --- Dynamic reward balancer (adaptive component weight scheduling) ---
    # Differentiator over OpenClaw-RL: automatically adjusts reward component
    # weights based on their informative variance. Required keys: base_weights
    # (dict of {component_name: base_weight}). Optional: alpha, variance_floor,
    # warmup_iters, max_weight_change_ratio, preserve_scale, component_warmup.
    dynamic_reward_balancer: dict[str, Any] | None = None
    # --- Curriculum scheduler (progressive stage advancement) ---
    # When set, a CurriculumScheduler is created that tracks per-stage metrics
    # and advances through curriculum stages when mastery conditions are met.
    # Dict keys: auto_advance (bool), min_iters_per_stage (int),
    # allow_regression (bool), regression_factor (float),
    # stages (optional list of CurriculumStage dicts).
    curriculum: dict[str, Any] | None = None
    # --- Staleness-adaptive TIS (dynamic rho_clip based on observed staleness) ---
    # When set, a StalenessAdaptiveTIS controller is created that observes
    # the pipeline/replay staleness each iter and dynamically adjusts the
    # TIS rho_clip. Dict keys: max_rho_clip, min_rho_clip, max_staleness,
    # interpolation ("linear"|"exp"), rho_floor, window_size, enabled.
    # Overrides the algo's fixed tis_rho_clip when active.
    staleness_adaptive_tis: dict[str, Any] | None = None
    # --- LoRA hot-reload (merge LoRA deltas → vLLM sync without full retrain) ---
    # When set, a LoRAHotReloadManager is created that merges LoRA adapter
    # deltas into shadow base weights and pushes to vLLM after each optimizer
    # step. Dict keys: rank, alpha, target_patterns, sync_every, shadow_device.
    # Requires vllm_rollout_model to be set.
    lora_hot_reload: dict[str, Any] | None = None


def build_shared_on_policy_config(source: Any) -> OnPolicyTrainerConfig:
    """Build an ``OnPolicyTrainerConfig`` from any dataclass / object that has
    overlapping field names.

    This replaces the verbose per-trainer field-by-field mapping in
    ``GRPOTrainer.__init__`` and ``PPOTrainer.__init__`` (60+ lines each).
    Only fields whose name matches an ``OnPolicyTrainerConfig`` field are
    copied; the algorithm-specific fields on ``source`` are ignored.

    The contract is:
      * If ``source`` exposes an attribute with the same name as a field on
        ``OnPolicyTrainerConfig``, the value is forwarded.
      * Missing attributes fall back to the field default — adding a new
        shared field on ``OnPolicyTrainerConfig`` only requires touching one
        file (this module).

    The function does not mutate ``source``.
    """
    shared_fields = {f.name for f in fields(OnPolicyTrainerConfig)}
    kwargs: dict[str, Any] = {}
    for name in shared_fields:
        if hasattr(source, name):
            kwargs[name] = getattr(source, name)
    return OnPolicyTrainerConfig(**kwargs)


def validate_on_policy_config(cfg: OnPolicyTrainerConfig) -> list[str]:
    """Return non-fatal cross-field configuration warnings."""

    warnings: list[str] = []
    if cfg.adaptive_kl and not cfg.use_reference:
        warnings.append("adaptive_kl requires use_reference=True to affect the KL term")
    if cfg.adaptive_kl and cfg.target_kl <= 0:
        warnings.append("adaptive_kl requires target_kl > 0")
    if cfg.vllm_rollout_model and cfg.distributed_strategy == "fsdp":
        warnings.append("vLLM rollout + FSDP may require explicit weight-sync validation")
    if cfg.grad_accum_steps > 1 and cfg.minibatch_size > 0:
        warnings.append("grad_accum_steps + minibatch_size changes effective batch size")
    if cfg.use_ema_rollout and cfg.vllm_rollout_model:
        warnings.append("EMA rollout cannot be combined with vLLM rollout")
    if cfg.lr_warmup_steps > 0 and cfg.lr_schedule == "constant":
        warnings.append("lr_warmup_steps is ignored when lr_schedule='constant'")
    if cfg.log_format not in {"text", "json"}:
        warnings.append("log_format should be either 'text' or 'json'")
    if cfg.profile_output_path is not None and not cfg.profile:
        warnings.append("profile_output_path is ignored when profile=False")
    for schedule_name, schedule in {
        "w_rl_schedule": cfg.w_rl_schedule,
        "w_opd_schedule": cfg.w_opd_schedule,
    }.items():
        if isinstance(schedule, dict) and schedule:
            mode = str(schedule.get("mode", schedule.get("kind", "linear"))).lower()
            if mode not in {"constant", "linear", "cosine"}:
                warnings.append(f"{schedule_name}.mode should be one of: constant, linear, cosine")
    if cfg.replay_buffer and not cfg.use_reference:
        warnings.append(
            "replay_buffer works best with use_reference=True so KL to the "
            "reference policy constrains off-policy drift"
        )
    if cfg.adaptive_kl_type == "pid":
        if cfg.adaptive_kl_Kp <= 0:
            warnings.append(
                "adaptive_kl_type='pid' but adaptive_kl_Kp <= 0; "
                "the proportional gain should be positive (default 0.1)"
            )
        if cfg.adaptive_kl_I_max <= 0:
            warnings.append(
                "adaptive_kl_I_max <= 0; PID integral anti-windup will be "
                "ineffective — set a positive value (default 2.0)"
            )
    if cfg.lora_hot_reload and not cfg.vllm_rollout_model:
        warnings.append(
            "lora_hot_reload requires vllm_rollout_model to be set — "
            "LoRA deltas are merged and synced to vLLM"
        )
    if cfg.staleness_adaptive_tis and not (
        cfg.pipeline_rollouts or cfg.replay_buffer
    ):
        warnings.append(
            "staleness_adaptive_tis has no effect without pipeline_rollouts "
            "or replay_buffer — staleness is always 0 in synchronous mode"
        )
    return warnings
