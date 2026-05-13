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
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
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
from hermes_agentic_rl.trainers.multi_turn_credit import assign_multi_turn_rewards


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
    multi_turn: bool = False            # if True, emit per-turn RolloutRecords
    multi_turn_credit: dict[str, Any] | None = None
    log_every: int = 1
    save_every: int = 0
    output_dir: Path | None = None
    seed: int | None = 0
    metrics_sink: Callable[[dict[str, Any]], None] | None = None  # optional live sink
    batch_generate: bool = False
    update_epochs: int = 1
    minibatch_size: int = 0  # 0 = full batch
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
    # --- checkpoint / resume (v0.6) ---
    # 0 = never save a resumable checkpoint (legacy .pt-only save_every still
    # controls the flat state_dict dump). When > 0, CheckpointManager saves a
    # full {model, optimizer, rng, stats} bundle under `<output_dir>/checkpoints/`.
    checkpoint_every: int = 0
    keep_last_checkpoints: int = 3
    # If set to a positive int, try to resume from `<output_dir>/checkpoints/iter_{N}`.
    # "latest" = resume from the newest checkpoint found (None = no resume).
    resume_from: int | str | None = None
    # If True, resume from the latest checkpoint automatically (no-op if none exists).
    auto_resume: bool = False
    # --- v0.7: best-checkpoint + early stopping ---
    # When True AND checkpoint_every > 0: additionally save the iter with the
    # highest mean_reward under `<output_dir>/checkpoints_best/iter_{N}/`.
    # Independent of keep_last_checkpoints — the best is NEVER pruned.
    save_best_checkpoint: bool = False
    # Early stop: halt training when no new best mean_reward appears for
    # `early_stop_patience` consecutive iterations. 0 disables early-stop.
    # A tiny positive delta is required to count as improvement.
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4
    # --- v0.8: trust-region + reward stabilization (all opt-in) ---
    # Per-minibatch ratio-based early stop. When the policy drifts too far
    # inside one update-epoch, abort remaining epochs for this iter.
    # `target_kl`: threshold on minibatch approx_kl (K2). 0 disables.
    target_kl: float = 0.0
    # Adaptive KL controller (InstructGPT A.2). Scales the algo's kl_coef
    # between iters so KL stays near `target_kl`. Requires `target_kl > 0`
    # AND `use_reference = True` (else there is no KL term to scale).
    adaptive_kl: bool = False
    adaptive_kl_horizon: float = 10000.0
    adaptive_kl_min: float = 1e-4
    adaptive_kl_max: float = 10.0
    # Running reward normalization: whitens scalar rewards with running
    # mean/std so the advantage scale is stable across iters. The raw
    # reward is preserved in `mean_reward` for logging; records get
    # `_normalized_reward` in metadata.
    normalize_reward: bool = False
    reward_norm_clip: float = 10.0
    # --- v0.9: scale-up (Qwen-7B+ support) ---
    # Mixed-precision training. "fp16", "bf16", "fp32", or "auto" (picks best).
    amp_dtype: str = "fp32"
    # Gradient accumulation steps. Loss is divided by this, grads are summed.
    # optimizer.step() fires every `grad_accum_steps` micro-batches.
    grad_accum_steps: int = 1
    # vLLM rollout backend (generation-only). When set, the trainer creates a
    # VLLMRolloutBackend and syncs weights every iter.
    # String: model name/path for vLLM. None = no vLLM, use policy backend.
    vllm_rollout_model: str | None = None
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = 4096
    vllm_gpu_memory_utilization: float = 0.90
    vllm_enable_prefix_caching: bool = True
    # Sync weights to vLLM every N iterations (1 = every iter, 0 = never).
    vllm_sync_every: int = 1
    # FSDP / DDP distributed strategy. "none" / "ddp" / "fsdp".
    distributed_strategy: str = "none"
    fsdp_cpu_offload: bool = False
    # FlashAttention. When True, HF backends use attn_implementation="flash_attention_2".
    # Tiny backend uses torch.nn.functional.scaled_dot_product_attention.
    flash_attention: bool = False


@dataclass(slots=True)
class TrainStats:
    iters: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def add(self, record: dict[str, Any]) -> None:
        self.iters.append(record)

    def best_reward(self) -> float:
        return max((r["mean_reward"] for r in self.iters), default=0.0)

    def last_reward(self) -> float:
        return self.iters[-1]["mean_reward"] if self.iters else 0.0

    def mean_reward_delta(self) -> float:
        if len(self.iters) < 2:
            return 0.0
        return self.iters[-1]["mean_reward"] - self.iters[0]["mean_reward"]


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
                    policy.model, dist_cfg,
                )
                policy.model = wrapped  # type: ignore[attr-defined]

        params = list(policy.trainable_parameters())
        if not params:
            raise RuntimeError("policy has no trainable parameters")
        self._trainable_params = params
        self.optim = torch.optim.AdamW(params, lr=self.cfg.lr)

        # ── v0.9: AMP context ──
        from hermes_agentic_rl.trainers.mixed_precision import AMPContext

        self._amp = AMPContext(
            dtype=self.cfg.amp_dtype,
            enabled=(self.cfg.amp_dtype not in ("fp32", "float32", "none", "")),
        )

        # ── v0.9: Gradient accumulation ──
        from hermes_agentic_rl.trainers.mixed_precision import GradientAccumulator

        self._grad_accum = GradientAccumulator(steps=self.cfg.grad_accum_steps)

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

        self.agent_loop_factory = agent_loop_factory or default_policy_loop_factory(
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )
        self._prompt_encoder = PromptStateEncoder(self.policy.tokenizer)
        self.logger = logger or (lambda rec: print(self._format_log(rec)))
        self.stats = TrainStats()
        self._seed_counter = 0

        # v0.8: reward normalizer + adaptive KL controller (opt-in).
        from hermes_agentic_rl.trainers.ppo_utils import (
            AdaptiveKLController,
            RunningMeanStd,
        )
        self._reward_rms: RunningMeanStd | None = (
            RunningMeanStd() if self.cfg.normalize_reward else None
        )
        self._kl_ctrl: AdaptiveKLController | None = None
        if self.cfg.adaptive_kl and self.cfg.target_kl > 0 and self.cfg.use_reference:
            algo_cfg = getattr(self.algo, "cfg", None)
            init_beta = float(getattr(algo_cfg, "kl_coef", 0.02)) or 0.02
            self._kl_ctrl = AdaptiveKLController(
                init_kl_coef=init_beta,
                target_kl=float(self.cfg.target_kl),
                horizon=float(self.cfg.adaptive_kl_horizon),
                min_coef=float(self.cfg.adaptive_kl_min),
                max_coef=float(self.cfg.adaptive_kl_max),
            )
        # Iteration to start from — updated by _maybe_resume().
        self._start_iter = 0
        self._best_reward = 0.0
        self._best_iter = 0
        # Expose optimizer via stable alias for checkpoint helpers.
        self._optim = self.optim

        # Initialize checkpoint manager lazily (only when output_dir + checkpoint_every).
        self._ckpt_manager = None
        self._best_ckpt_manager = None
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

        self._maybe_resume()

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------

    def _validate_backend(self, policy: LLMBackend) -> None:
        """Override to require a value head, etc."""

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

    def _maybe_normalize_rewards(
        self, records: list[RolloutRecord]
    ) -> list[RolloutRecord]:
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

    async def _collect_group(self, item: dict[str, Any]) -> list[RolloutRecord]:
        if self._batch_rollout_generator is not None:
            return await self._collect_group_batched(item)

        instruction = self.env.format_prompt(item)
        records: list[RolloutRecord] = []
        group_id = str(item.get("task_id", "group"))
        for _g in range(self.cfg.group_size):
            loop = self.agent_loop_factory(backend=self.policy, seed=self._next_seed())
            trajectory: Trajectory = await RolloutManager(loop).collect(item, instruction)
            summary = await self.reward_manager.evaluate(item, trajectory, tool_context=None)
            # curriculum feedback — duck-typed; safe if env doesn't support it
            observe = getattr(self.env, "observe", None)
            if callable(observe):
                try:
                    observe(float(summary.final_score))
                except Exception:
                    pass
            if self.lagrangian is not None:
                try:
                    self.lagrangian.measure(item, trajectory)
                except Exception:
                    pass
            rl_meta = _extract_rl(trajectory)
            if rl_meta is None:
                raise RuntimeError(
                    "Agent loop must emit trajectory.metadata['runtime']['rl']"
                )
            rollout_temperature = _rollout_temperature_from_meta(
                rl_meta,
                fallback=self.cfg.temperature,
            )

            base_meta = {
                "final_output": trajectory.final_output,
                "reward_components": [
                    _reward_component_payload(component)
                    for component in summary.components
                ],
                "finished_naturally": bool(trajectory.finished_naturally),
                "turns_used": trajectory.turns_used,
                "tool_calls_count": sum(len(step.tool_calls) for step in trajectory.steps),
                "tool_results_count": sum(len(step.tool_results) for step in trajectory.steps),
                "final_output_chars": len(trajectory.final_output or ""),
                "reward_summary_metadata": dict(summary.metadata),
                "rollout_temperature": rollout_temperature,
            }

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
                            reward=float(credit_meta.get("reward", summary.final_score)),
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

    async def _collect_group_batched(self, item: dict[str, Any]) -> list[RolloutRecord]:
        instruction = self.env.format_prompt(item)
        encoder = PromptStateEncoder(self.policy.tokenizer)
        prompt_ids = list(encoder.encode({"instruction": instruction}).prompt_ids)
        if self._batch_rollout_generator is None:
            raise RuntimeError("batched rollout collection requires a batch rollout generator")
        outputs = self._batch_rollout_generator.generate(
            [prompt_ids for _ in range(self.cfg.group_size)],
            seed=self._next_seed(),
        )

        observe = getattr(self.env, "observe", None)
        group_id = str(item.get("task_id", "group"))
        records: list[RolloutRecord] = []
        for gen in outputs:
            response_text = self.policy.tokenizer.decode(gen.response_ids)
            trajectory = Trajectory(
                task_id=item["task_id"],
                prompt=instruction,
                steps=[],
                final_output=response_text,
                finished_naturally=gen.finished,
                turns_used=1,
                metadata={
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response_text},
                    ],
                    "runtime": {
                        "runtime": "policy_agent_loop",
                        "prompt": instruction,
                    "rl": {
                        "prompt_ids": list(prompt_ids),
                        "response_ids": list(gen.response_ids),
                        "old_logprobs": list(gen.logprobs),
                        "temperature": self.cfg.temperature,
                        },
                    },
                },
            )
            summary = await self.reward_manager.evaluate(item, trajectory, tool_context=None)
            if callable(observe):
                try:
                    observe(float(summary.final_score))
                except Exception:
                    pass
            if self.lagrangian is not None:
                try:
                    self.lagrangian.measure(item, trajectory)
                except Exception:
                    pass
            records.append(
                RolloutRecord(
                    prompt_ids=list(prompt_ids),
                    response_ids=list(gen.response_ids),
                    old_logprobs=list(gen.logprobs),
                    reward=float(summary.final_score),
                    group_id=group_id,
                    metadata={
                        "final_output": trajectory.final_output,
                        "reward_components": [
                            _reward_component_payload(component)
                            for component in summary.components
                        ],
                        "finished_naturally": bool(trajectory.finished_naturally),
                        "turns_used": trajectory.turns_used,
                        "tool_calls_count": 0,
                        "tool_results_count": 0,
                    "final_output_chars": len(trajectory.final_output or ""),
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(gen.response_ids),
                    "rollout_temperature": float(self.cfg.temperature),
                    "reward_summary_metadata": dict(summary.metadata),
                },
            )
            )
        return records

    async def _collect_distributed(self) -> list[RolloutRecord]:
        """Fan out `prompts_per_iter * group_size` rollouts to the pool."""
        from hermes_agentic_rl.distributed.mp_pool import RolloutTask

        # 1) broadcast current learner weights
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

        # 3) submit and drain
        self.rollout_pool.submit_tasks(tasks)
        results = self.rollout_pool.drain(expected=len(tasks))

        # 4) flatten to RolloutRecord, feed curriculum observer
        observe = getattr(self.env, "observe", None)
        records: list[RolloutRecord] = []
        for r in results:
            if callable(observe):
                try:
                    observe(float(r["final_score"]))
                except Exception:
                    pass
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

    # ------------------------------------------------------------------
    # train loop
    # ------------------------------------------------------------------

    async def _one_iter(self, iter_idx: int) -> AlgoUpdateStats:
        if iter_idx == 0:
            await self.env.setup()

        if self.lagrangian is not None:
            self.lagrangian.begin_iter()

        if self.rollout_pool is not None:
            batch_records = await self._collect_distributed()
        else:
            batch_records = []
            for _ in range(self.cfg.prompts_per_iter):
                item = await self.env.get_next_item()
                batch_records.extend(await self._collect_group(item))

        # v0.8: running-reward normalization BEFORE prepare so the reward
        # used for advantage computation is whitened, while `raw_reward`
        # survives in metadata for logging.
        batch_records = self._maybe_normalize_rewards(batch_records)
        batch = self._prepare_update_batch(RolloutBatch(records=batch_records))

        # v0.8: adaptive KL — sync β into algo.cfg BEFORE computing loss for
        # this iter. The previous iter's KL drove the update.
        if self._kl_ctrl is not None:
            algo_cfg = getattr(self.algo, "cfg", None)
            if algo_cfg is not None and hasattr(algo_cfg, "kl_coef"):
                algo_cfg.kl_coef = float(self._kl_ctrl.value)

        update_batches = self._build_update_batches(batch, iter_idx=iter_idx)
        per_step_stats: list[AlgoUpdateStats] = []
        early_stopped = False
        last_approx_kl = 0.0

        target_kl = float(getattr(self.cfg, "target_kl", 0.0) or 0.0)

        self.optim.zero_grad()
        for mb_idx, mini_batch in enumerate(update_batches):
            # v0.9: autocast the forward pass
            with self._amp.autocast_ctx():
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
            if did_step and loss.requires_grad:
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
            elif did_step:
                self.optim.zero_grad()
                self._grad_accum.finish_step()

            stats.extra["grad_norm"] = grad_norm
            stats.extra["param_norm"] = _param_l2_norm(self._trainable_params)
            stats.extra["lr"] = float(self.optim.param_groups[0].get("lr", 0.0))
            stats.extra["optimizer_step_applied"] = 1.0 if did_step else 0.0
            stats.extra["amp_scale"] = self._amp.get_scale()
            stats.extra["grad_accum_step"] = float(mb_idx + 1)
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
            self._amp.unscale_(self.optim)
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_params, max_norm=self.cfg.grad_clip,
                )
            self._amp.step(self.optim)
            self.optim.zero_grad()
            self._amp.update()
            self._grad_accum.finish_step()

        # v0.8: feed the last-seen approx_kl into the adaptive controller.
        if self._kl_ctrl is not None and per_step_stats:
            # Use the mean approx_kl across all executed minibatches.
            kl_vals = [float(s.extra.get("approx_kl", 0.0) or 0.0) for s in per_step_stats]
            mean_kl = sum(kl_vals) / max(1, len(kl_vals))
            new_beta = self._kl_ctrl.update(mean_kl, n_steps=len(kl_vals))
            for s in per_step_stats:
                s.extra["adaptive_kl_coef"] = float(new_beta)

        agg = self._aggregate_update_stats(
            batch=batch,
            step_stats=per_step_stats,
            n_update_batches=len(update_batches),
        )
        if early_stopped:
            agg.extra["early_stopped_by_kl"] = 1.0
            agg.extra["last_minibatch_approx_kl"] = last_approx_kl
        if self._reward_rms is not None:
            agg.extra["reward_norm_mean"] = float(self._reward_rms.mean)
            agg.extra["reward_norm_std"] = float(self._reward_rms.std)
            # Restore mean_reward to RAW scale for logging (the normalized
            # scalar that drove the gradient is in `mean_advantage`).
            raws = [
                float(r.metadata.get("raw_reward", r.reward))
                for r in batch.records
            ]
            if raws:
                agg.mean_reward = sum(raws) / len(raws)
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
                self._save_full_checkpoint(it)

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
        return self.stats

    # ------------------------------------------------------------------
    # checkpoint helpers
    # ------------------------------------------------------------------

    def _save_full_checkpoint(self, it: int, manager: Any | None = None) -> None:
        """Save a {model, optimizer, rng, stats} bundle via CheckpointManager.

        If ``manager`` is None, uses the default checkpoint manager. Passing an
        alternate manager (e.g. ``self._best_ckpt_manager``) writes to a
        separate directory with its own retention policy.
        """
        target_mgr = manager or self._ckpt_manager
        if target_mgr is None or not hasattr(self.policy, "model"):
            return
        from hermes_agentic_rl.trainers.checkpoint import (
            CheckpointState,
            capture_rng_state,
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
        )
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
        parts = []
        for k in keys:
            v = rec.get(k)
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif v is not None:
                parts.append(f"{k}={v}")
        return "[train] " + " ".join(parts)

    def _build_update_batches(self, batch: RolloutBatch, *, iter_idx: int) -> list[RolloutBatch]:
        update_epochs = max(1, int(self.cfg.update_epochs))
        minibatch_size = int(self.cfg.minibatch_size)
        if minibatch_size == 0:
            minibatch_size = len(batch.records)
        minibatch_size = max(1, minibatch_size)

        batches: list[RolloutBatch] = []
        for epoch_idx in range(update_epochs):
            if (
                minibatch_size >= len(batch.records)
                or len(batch.records) <= 1
            ):
                batches.append(RolloutBatch(records=list(batch.records)))
                continue
            epoch_batches = self._split_minibatches(
                batch,
                minibatch_size=minibatch_size,
                iter_idx=iter_idx,
                epoch_idx=epoch_idx,
            )
            if not epoch_batches:
                batches.append(RolloutBatch(records=list(batch.records)))
            else:
                batches.extend(epoch_batches)
        return batches

    def _split_minibatches(
        self,
        batch: RolloutBatch,
        *,
        minibatch_size: int,
        iter_idx: int,
        epoch_idx: int,
    ) -> list[RolloutBatch]:
        rng = self._minibatch_rng(iter_idx=iter_idx, epoch_idx=epoch_idx)
        if self._preserve_group_boundaries():
            groups = [list(group) for group in batch.by_group().values()]
            if self.cfg.shuffle_minibatches:
                rng.shuffle(groups)
            out: list[RolloutBatch] = []
            current: list[Any] = []
            current_size = 0
            for group in groups:
                group_size = len(group)
                if current and current_size + group_size > minibatch_size:
                    out.append(RolloutBatch(records=list(current)))
                    current = []
                    current_size = 0
                current.extend(group)
                current_size += group_size
                if current_size >= minibatch_size:
                    out.append(RolloutBatch(records=list(current)))
                    current = []
                    current_size = 0
            if current:
                out.append(RolloutBatch(records=list(current)))
            return out

        records = list(batch.records)
        if self.cfg.shuffle_minibatches:
            rng.shuffle(records)
        return [
            RolloutBatch(records=records[start : start + minibatch_size])
            for start in range(0, len(records), minibatch_size)
        ]

    def _minibatch_rng(self, *, iter_idx: int, epoch_idx: int) -> random.Random:
        seed = self.cfg.seed
        if seed is None:
            return random.Random()
        return random.Random(int(seed) + (iter_idx * 1009) + (epoch_idx * 9173))

    def _aggregate_update_stats(
        self,
        *,
        batch: RolloutBatch,
        step_stats: list[AlgoUpdateStats],
        n_update_batches: int,
    ) -> AlgoUpdateStats:
        total_records = len(batch.records)
        if not step_stats:
            return AlgoUpdateStats(
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                entropy=0.0,
                mean_reward=0.0,
                mean_advantage=0.0,
                clip_frac=0.0,
                n_records=total_records,
                extra={
                    "n_updated": 0,
                    "n_optimizer_steps": 0,
                    "update_epochs": max(1, int(self.cfg.update_epochs)),
                    "n_minibatches": n_update_batches,
                },
            )

        def _weight(stat: AlgoUpdateStats) -> int:
            return max(1, int(stat.n_records))

        total_weight = sum(_weight(stat) for stat in step_stats)

        def _weighted(attr: str) -> float:
            return sum(float(getattr(stat, attr)) * _weight(stat) for stat in step_stats) / max(
                1, total_weight
            )

        extras: dict[str, Any] = {}
        first_extra = step_stats[0].extra
        if "algo" in first_extra:
            extras["algo"] = first_extra["algo"]
        extras["n_updated"] = sum(int(stat.extra.get("n_updated", 0)) for stat in step_stats)
        extras["n_optimizer_steps"] = len(step_stats)
        extras["update_epochs"] = max(1, int(self.cfg.update_epochs))
        extras["n_minibatches"] = n_update_batches
        extras["minibatch_size"] = (
            len(batch.records) if int(self.cfg.minibatch_size) <= 0 else int(self.cfg.minibatch_size)
        )

        numeric_means: dict[str, list[tuple[float, int]]] = {}
        for stat in step_stats:
            for key, value in stat.extra.items():
                if key in {"n_updated", "algo"}:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    numeric_means.setdefault(key, []).append((float(value), _weight(stat)))
        for key, values in numeric_means.items():
            denom = sum(weight for _, weight in values)
            extras[key] = sum(value * weight for value, weight in values) / max(1, denom)
        extras.update(_summarize_batch_metadata(batch))

        return AlgoUpdateStats(
            loss=_weighted("loss"),
            policy_loss=_weighted("policy_loss"),
            kl=_weighted("kl"),
            entropy=_weighted("entropy"),
            mean_reward=_weighted("mean_reward"),
            mean_advantage=_weighted("mean_advantage"),
            clip_frac=_weighted("clip_frac"),
            n_records=total_records,
            extra=extras,
        )

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


def _config_to_dict(cfg: Any) -> dict[str, Any]:
    """Shallow dataclass-to-dict for checkpoint config snapshot."""
    out: dict[str, Any] = {}
    for name in getattr(cfg, "__slots__", []) or []:
        try:
            v = getattr(cfg, name)
        except AttributeError:
            continue
        if isinstance(v, Path):
            out[name] = str(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[name] = v
        else:
            out[name] = repr(v)
    return out


def _turn_group_id(prompt_group_id: str, turn_index: int) -> str:
    return f"{prompt_group_id}::turn:{turn_index}"


def _teacher_responses_from_env(
    env: BaseEnv,
    item: dict[str, Any],
    *,
    n_turns: int,
) -> list[str | None] | None:
    samples = env.build_supervised_samples(item)
    if not samples:
        return None

    out: list[str | None] = [None] * max(1, n_turns)
    explicit = False
    for sample in samples:
        turn_index = sample.metadata.get("turn_index")
        if isinstance(turn_index, int) and 0 <= turn_index < len(out):
            out[turn_index] = str(sample.response)
            explicit = True

    if not explicit:
        if len(samples) == len(out):
            for idx, sample in enumerate(samples):
                out[idx] = str(sample.response)
        elif len(out) == 1 and samples:
            out[0] = str(samples[0].response)

    if all(response is None or not str(response).strip() for response in out):
        return None
    return out


def _series_stats(
    values: list[float],
    *,
    include_mean: bool = True,
) -> dict[str, float]:
    if not values:
        return {}
    summary: dict[str, float] = {
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }
    if include_mean:
        summary["mean"] = sum(values) / len(values)
    return summary


def _sanitize_metric_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return cleaned or "metric"


def _reward_component_payload(component: Any) -> dict[str, Any]:
    payload = {
        "name": getattr(component, "name", "reward"),
        "score": getattr(component, "score", 0.0),
        "weight": getattr(component, "weight", 1.0),
    }
    metadata = getattr(component, "metadata", None)
    if isinstance(metadata, dict):
        numeric_metadata = {
            _sanitize_metric_name(str(key)): float(value)
            for key, value in metadata.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric_metadata:
            payload["metadata"] = numeric_metadata
    return payload


def _grad_l2_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        total += float((grad.detach().float() ** 2).sum().item())
    return total ** 0.5


def _param_l2_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for param in params:
        total += float((param.detach().float() ** 2).sum().item())
    return total ** 0.5


def _summarize_batch_metadata(batch: RolloutBatch) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    rewards = [float(rec.reward) for rec in batch.records]
    reward_stats = _series_stats(rewards, include_mean=False)
    summary.update({f"reward_{key}": value for key, value in reward_stats.items()})

    groups = batch.by_group()
    if groups:
        summary["n_groups"] = len(groups)
        group_size_stats = _series_stats([float(len(rows)) for rows in groups.values()])
        if group_size_stats:
            summary["records_per_group"] = group_size_stats
        group_reward_std = [
            _series_stats([float(rec.reward) for rec in rows], include_mean=False).get("std", 0.0)
            for rows in groups.values()
        ]
        group_reward_std_stats = _series_stats(group_reward_std)
        if group_reward_std_stats:
            summary["group_reward_std"] = group_reward_std_stats

    prompt_tokens = [float(len(rec.prompt_ids)) for rec in batch.records]
    prompt_stats = _series_stats(prompt_tokens)
    if prompt_stats:
        summary["prompt_tokens"] = prompt_stats

    response_tokens = [float(len(rec.response_ids)) for rec in batch.records]
    response_stats = _series_stats(response_tokens)
    if response_stats:
        summary["response_tokens"] = response_stats

    generation_metrics = {
        "final_output_chars": [
            float(rec.metadata["final_output_chars"])
            for rec in batch.records
            if isinstance(rec.metadata.get("final_output_chars"), (int, float))
        ],
        "turns_used": [
            float(rec.metadata["turns_used"])
            for rec in batch.records
            if isinstance(rec.metadata.get("turns_used"), (int, float))
        ],
        "tool_calls_count": [
            float(rec.metadata["tool_calls_count"])
            for rec in batch.records
            if isinstance(rec.metadata.get("tool_calls_count"), (int, float))
        ],
        "tool_results_count": [
            float(rec.metadata["tool_results_count"])
            for rec in batch.records
            if isinstance(rec.metadata.get("tool_results_count"), (int, float))
        ],
        "rollout_temperature": [
            float(rec.metadata["rollout_temperature"])
            for rec in batch.records
            if isinstance(rec.metadata.get("rollout_temperature"), (int, float))
            and not isinstance(rec.metadata.get("rollout_temperature"), bool)
        ],
    }
    for key, values in generation_metrics.items():
        stats = _series_stats(values)
        if stats:
            summary[key] = stats

    finished_naturally = [
        1.0 if bool(rec.metadata["finished_naturally"]) else 0.0
        for rec in batch.records
        if "finished_naturally" in rec.metadata
    ]
    if finished_naturally:
        summary["finished_naturally_rate"] = sum(finished_naturally) / len(finished_naturally)

    reward_component_scores: dict[str, list[float]] = {}
    reward_component_weights: dict[str, list[float]] = {}
    reward_component_metadata: dict[str, dict[str, list[float]]] = {}
    reward_summary_numeric: dict[str, list[float]] = {}
    for rec in batch.records:
        components = rec.metadata.get("reward_components")
        if isinstance(components, list):
            for component in components:
                if not isinstance(component, dict):
                    continue
                name = _sanitize_metric_name(str(component.get("name", "reward")))
                score = component.get("score")
                if isinstance(score, (int, float)):
                    reward_component_scores.setdefault(name, []).append(float(score))
                weight = component.get("weight")
                if isinstance(weight, (int, float)):
                    reward_component_weights.setdefault(name, []).append(float(weight))
                metadata = component.get("metadata")
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)):
                            safe_key = _sanitize_metric_name(str(key))
                            reward_component_metadata.setdefault(name, {}).setdefault(
                                safe_key,
                                [],
                            ).append(float(value))
        reward_meta = rec.metadata.get("reward_summary_metadata")
        if isinstance(reward_meta, dict):
            for key, value in reward_meta.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    reward_summary_numeric.setdefault(_sanitize_metric_name(str(key)), []).append(
                        float(value)
                    )

    if reward_component_scores:
        summary["reward_components"] = {}
        for name, values in sorted(reward_component_scores.items()):
            component_summary = _series_stats(values)
            weight_values = reward_component_weights.get(name, [])
            if weight_values:
                component_summary["weight_mean"] = sum(weight_values) / len(weight_values)
            summary["reward_components"][name] = component_summary

    if reward_component_metadata:
        summary["reward_component_metadata"] = {
            component_name: {
                key: _series_stats(values)
                for key, values in sorted(metadata.items())
                if values
            }
            for component_name, metadata in sorted(reward_component_metadata.items())
        }

    if reward_summary_numeric:
        summary["reward_summary"] = {
            key: _series_stats(values)
            for key, values in sorted(reward_summary_numeric.items())
            if values
        }

    turn_credit_rows: list[dict[str, Any]] = []
    for rec in batch.records:
        turn_credit = rec.metadata.get("turn_credit")
        if isinstance(turn_credit, dict):
            turn_credit_rows.append(turn_credit)
    if turn_credit_rows:
        numeric_keys = [
            "reward",
            "final_component",
            "local_component",
            "judge_component",
            "teacher_component",
            "weighted_final_component",
            "weighted_local_component",
        ]
        summary["n_turn_records"] = len(turn_credit_rows)
        summary["turn_credit"] = {}
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in turn_credit_rows
                if isinstance(row.get(key), (int, float))
            ]
            stats = _series_stats(values)
            if stats:
                summary["turn_credit"][key] = stats
                # Also expose as flat `turn_credit_<key>_<stat>` keys so
                # downstream tests / log formatters / CSV writers don't
                # need to walk the nested dict.
                for stat_name, stat_val in stats.items():
                    summary[f"turn_credit_{key}_{stat_name}"] = stat_val

        turn_indices = [
            float(rec.metadata["turn_index"])
            for rec in batch.records
            if isinstance(rec.metadata.get("turn_index"), int)
        ]
        turn_index_stats = _series_stats(turn_indices)
        if turn_index_stats:
            summary["turn_index"] = turn_index_stats

        rollout_final_rewards = [
            float(rec.metadata["rollout_final_reward"])
            for rec in batch.records
            if isinstance(rec.metadata.get("rollout_final_reward"), (int, float))
        ]
        rollout_reward_stats = _series_stats(rollout_final_rewards)
        if rollout_reward_stats:
            summary["rollout_final_reward"] = rollout_reward_stats
    return summary


def _extract_rl(trajectory: Trajectory) -> dict[str, Any] | None:
    runtime_block = trajectory.metadata.get("runtime")
    if isinstance(runtime_block, dict):
        rl = runtime_block.get("rl")
        if isinstance(rl, dict):
            return rl
    rl = trajectory.metadata.get("rl")
    if isinstance(rl, dict):
        return rl
    return None


def _rollout_temperature_from_meta(
    rl_meta: dict[str, Any],
    *,
    fallback: float,
) -> float:
    raw = rl_meta.get("temperature", fallback)
    if isinstance(raw, bool):
        return float(fallback)
    if isinstance(raw, (int, float)):
        return float(raw)
    return float(fallback)
