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
from hermes_agentic_rl.core.types import RolloutStep, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.mdp.state_encoder import PromptStateEncoder
from hermes_agentic_rl.trainers.batch_stats import _rl_dense_reward_metadata
from hermes_agentic_rl.trainers.multi_turn_credit import assign_multi_turn_rewards
from hermes_agentic_rl.trainers.on_policy_config import OnPolicyTrainerConfig


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

    # -- v0.10 convenience accessors -------------------------------------

    def reward_curve(self) -> list[float]:
        """Return the per-iteration mean_reward series."""
        return [float(r.get("mean_reward", 0.0)) for r in self.iters]

    def mean_kl(self) -> float:
        """Average KL divergence across all iterations."""
        kls = [float(r.get("kl", 0.0)) for r in self.iters]
        return sum(kls) / len(kls) if kls else 0.0

    def loss_curve(self) -> list[float]:
        """Return the per-iteration loss series."""
        return [float(r.get("loss", 0.0)) for r in self.iters]

    def get_column(self, key: str) -> list[Any]:
        """Extract a single column from all iteration records."""
        return [r.get(key) for r in self.iters]

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with key training statistics."""
        return {
            "n_iters": len(self.iters),
            "best_reward": self.best_reward(),
            "last_reward": self.last_reward(),
            "mean_kl": self.mean_kl(),
        }

    def to_dataframe(self):
        """Convert to a pandas DataFrame (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(self.iters)


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
            kind = str(
                sched_kwargs.pop("mode", sched_kwargs.pop("kind", "linear"))
            )
            # Default total_steps to the full run length for step schedules
            # so users do not have to restate it in YAML.
            if (
                kind in ("linear", "cosine")
                and "total_steps" not in sched_kwargs
            ):
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
                    max_hint_tokens=int(
                        getattr(self.cfg, "opd_teacher_max_hint_tokens", 128)
                    ),
                    capability_axis_weights=getattr(
                        self.cfg, "opd_capability_axis_weights", None
                    ),
                ),
            )

        # ── OPD in-trainer hint extractor (recovers directive hints) ──────
        # Runs BEFORE the teacher fill: any record that has a next-state signal
        # but no ``opd_hint`` gets one from a configurable judge, so OPD fires
        # even when the env / reward did not pre-populate a hint. Opt-in via the
        # ``opd.hint_extractor`` YAML block; ``None`` keeps legacy behaviour.
        self._hint_extractor: Any = None
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

        self._maybe_resume()

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
                raise RuntimeError(
                    "Agent loop must emit trajectory.metadata['runtime']['rl']"
                )
            dense_meta = _rl_dense_reward_metadata(rl_meta)
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
                temperature=self.cfg.temperature,
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
            opd_meta = (
                {"opd_hint": opd_hint}
                if isinstance(opd_hint, str) and opd_hint.strip()
                else {}
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
                            _reward_component_payload(component)
                            for component in summary.components
                        ],
                        "finished_naturally": bool(trajectory.finished_naturally),
                        "turns_used": trajectory.turns_used,
                        "tool_calls_count": sum(len(step.tool_calls) for step in trajectory.steps),
                        "tool_results_count": sum(len(step.tool_results) for step in trajectory.steps),
                    "final_output_chars": len(trajectory.final_output or ""),
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(gen.response_ids),
                    "rollout_temperature": float(self.cfg.temperature),
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
        return (
            bool(getattr(self.cfg, "pipeline_rollouts", False))
            and self.rollout_pool is not None
        )

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
        if self.lagrangian is not None:
            self.lagrangian.begin_iter()

        batch_records = await self._collect_for_iter(iter_idx)
        return self._update_on_records(batch_records, iter_idx)

    def _update_on_records(
        self, batch_records: list[RolloutRecord], iter_idx: int
    ) -> AlgoUpdateStats:
        # v0.8: running-reward normalization BEFORE prepare so the reward
        # used for advantage computation is whitened, while `raw_reward`
        # survives in metadata for logging.
        batch_records = self._maybe_normalize_rewards(batch_records)

        # ── Replay buffer mixing (off-policy with TIS correction) ──────────
        # Mix in stale-but-recent records from the replay buffer. Current
        # records are pushed into the buffer for future iters but are NOT
        # reused this iter (avoids double-counting). TIS/V-trace in the
        # algo corrects for staleness automatically.
        if self._replay_buffer is not None:
            batch_records = self._replay_buffer.mix_with_current(
                current=batch_records,
                mix_ratio=float(getattr(self.cfg, "replay_mix_ratio", 0.25)),
                policy_version=self._update_version,
            )

        # Stamp iteration index into metadata so curriculum-aware shaping
        # functions (see rewards/shaping.py :: curriculum_shaping) can scale
        # their effect with training progress.
        for rec in batch_records:
            meta = getattr(rec, "metadata", None)
            if isinstance(meta, dict):
                meta["_trainer_iter"] = int(iter_idx)

        # Apply reward shaping (if configured via subclass constructor).
        if getattr(self, "_reward_shaping_fn", None) is not None:
            batch_records = list(self._reward_shaping_fn(batch_records))

        # OPD in-trainer hint extraction: recover directive hints from the
        # next-state signal for records that don't already carry one. Must run
        # BEFORE the teacher fill so newly-stamped hints get teacher logprobs.
        # OpenClaw-RL §3.2 produces hints from the PRM judge; this is the
        # in-trainer equivalent that keeps OPD from silently degrading to GRPO.
        self._last_hint_extract_stats: dict[str, float] | None = None
        if self._hint_extractor is not None:
            hint_stats = self._hint_extractor.extract(batch_records)
            self._last_hint_extract_stats = hint_stats.as_dict()

        # OPD teacher-logprob fill: re-score hinted records under a
        # hint-enhanced context so the OPD / Hybrid branch has a real teacher
        # distribution (OpenClaw-RL §3.2). No-op when no hints are present.
        self._last_teacher_fill_stats: dict[str, float] | None = None
        if self._teacher_filler is not None:
            fill_stats = self._teacher_filler.fill(batch_records)
            self._last_teacher_fill_stats = fill_stats.as_dict()

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
                # Grad-accum-aware LR scheduling: advance per *optimizer step*
                # (not per iteration) so warm-up/decay align with actual
                # parameter updates.
                self._lr_optim_step_counter += 1
                new_lr = self._lr_sched.get_lr(self._lr_optim_step_counter)
                for pg in self.optim.param_groups:
                    pg["lr"] = new_lr
            elif did_step:
                self.optim.zero_grad()
                self._grad_accum.finish_step()

            stats.extra["grad_norm"] = grad_norm
            stats.extra["param_norm"] = _param_l2_norm(self._trainable_params)
            stats.extra["lr"] = float(self.optim.param_groups[0].get("lr", 0.0))
            stats.extra["optimizer_step_applied"] = 1.0 if did_step else 0.0
            stats.extra["amp_scale"] = self._amp.get_scale()
            stats.extra["grad_accum_step"] = float(mb_idx + 1)
            stats.extra["grad_clip_triggered"] = (
                1.0
                if (self.cfg.grad_clip and self.cfg.grad_clip > 0 and grad_norm > self.cfg.grad_clip)
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
            self._amp.unscale_(self.optim)
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_params, max_norm=self.cfg.grad_clip,
                )
            self._amp.step(self.optim)
            self.optim.zero_grad()
            self._amp.update()
            self._grad_accum.finish_step()
            self._lr_optim_step_counter += 1
            new_lr = self._lr_sched.get_lr(self._lr_optim_step_counter)
            for pg in self.optim.param_groups:
                pg["lr"] = new_lr

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
        if self._last_teacher_fill_stats:
            agg.extra.update(self._last_teacher_fill_stats)
        if getattr(self, "_last_hint_extract_stats", None):
            agg.extra.update(self._last_hint_extract_stats)
        # P0-2 observability: how stale were the rollouts that drove this update
        # (0 for synchronous/BSP, ~1 in steady-state pipelined mode), and the
        # learner's policy version (number of completed updates so far).
        agg.extra["rollout_staleness"] = float(self._rollout_staleness)
        agg.extra["policy_version"] = float(self._update_version)
        self._update_version += 1
        # Replay buffer stats (when enabled).
        if self._replay_buffer is not None:
            agg.extra.update(self._replay_buffer.stats.as_dict())
            n_replayed = sum(
                1 for rec in batch.records
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

            # Reference policy periodic re-clone (prevents KL drift).
            ref_every = int(getattr(self.cfg, "ref_update_every", 0))
            if (
                ref_every > 0
                and self.ref_policy is not None
                and it > 0
                and it % ref_every == 0
            ):
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
        )
        use_async = (
            background
            and self.cfg.async_checkpoint
            and self._async_ckpt_saver is not None
            and manager is None
        )
        if use_async:
            self._async_ckpt_saver.submit(target_mgr, snapshot_state_to_cpu(state))
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
        # Keys that are internal / noisy and should not appear in the log.
        _suppressed = {
            "n_tokens", "ratio_mean", "optimizer_step_applied",
            "amp_scale", "grad_accum_step",
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
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
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

    def _run_eval_hook(self, iter_idx: int) -> dict[str, Any] | None:
        """Run a quick evaluation roll-out with deterministic decoding.

        Uses the same env but sets temperature=0 (greedy) and collects
        ``eval_prompts`` items. Returns a dict of ``eval_*`` metrics or None
        if the env has no items.
        """
        n_eval = max(1, int(getattr(self.cfg, "eval_prompts", 4)))
        eval_temp = float(getattr(self.cfg, "eval_temperature", 0.0))

        try:
            eval_records = asyncio.run(
                self._collect_group(
                    n_items=n_eval,
                    temperature=eval_temp,
                    group_size=1,  # greedy → 1 sample per prompt
                )
            )
        except Exception:
            return None

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

    # Per-stream breakdown (multi-stream unified training). Records sampled from
    # a MixedCurriculumEnv carry a ``stream_level`` tag; aggregate reward / count
    # / share per stream so the metrics sink sees where the optimizer budget went
    # and how each stream is performing — complementing the env-level snapshot.
    stream_rewards: dict[int, list[float]] = {}
    for rec in batch.records:
        lvl = rec.metadata.get("stream_level")
        if isinstance(lvl, int) and not isinstance(lvl, bool):
            stream_rewards.setdefault(lvl, []).append(float(rec.reward))
    if stream_rewards:
        summary["n_streams"] = len(stream_rewards)
        total_stream_recs = sum(len(v) for v in stream_rewards.values())
        for lvl, rws in sorted(stream_rewards.items()):
            summary[f"stream/{lvl}/count"] = float(len(rws))
            summary[f"stream/{lvl}/share"] = float(len(rws)) / total_stream_recs
            summary[f"stream/{lvl}/mean_reward"] = sum(rws) / len(rws)
            std = _series_stats(rws, include_mean=False).get("std")
            if std is not None:
                summary[f"stream/{lvl}/reward_std"] = std

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


def _batch_single_turn_trajectory(
    *,
    item: dict[str, Any],
    instruction: str,
    response_text: str,
    prompt_ids: list[int],
    response_ids: list[int],
    old_logprobs: list[float],
    temperature: float,
    finished: bool,
) -> Trajectory:
    """Build a single-turn trajectory for batched rollout collection.

    Batched generation bypasses RolloutManager, so we must populate at least
    one :class:`RolloutStep` so trajectory-level rewards (tool-call fidelity,
    turn discount, conditional gating on ``traj.steps``) behave like the
    agent-loop path.
    """
    return Trajectory(
        task_id=item["task_id"],
        prompt=instruction,
        steps=[
            RolloutStep(
                turn_index=0,
                assistant_message=response_text,
            )
        ],
        final_output=response_text,
        finished_naturally=finished,
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
                    "response_ids": list(response_ids),
                    "old_logprobs": list(old_logprobs),
                    "temperature": temperature,
                },
            },
        },
    )


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


def _extract_next_state(trajectory: Trajectory) -> str | None:
    """Pull the next-state signal from a trajectory's runtime metadata.

    Mirrors ``NextStatePRMComponent`` which reads
    ``trajectory.metadata["runtime"]["next_state"]``. Falls back to a top-level
    ``next_state`` key. Returns None when no textual signal is present.
    """
    runtime_block = trajectory.metadata.get("runtime")
    if isinstance(runtime_block, dict):
        val = runtime_block.get("next_state")
        if isinstance(val, str) and val.strip():
            return val
    val = trajectory.metadata.get("next_state")
    if isinstance(val, str) and val.strip():
        return val
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
