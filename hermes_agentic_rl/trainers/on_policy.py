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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from hermes_agentic_rl.agent_loop.base import BaseAgentLoop
from hermes_agentic_rl.agent_loop.policy_loop import PolicyAgentLoop
from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
)
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv


class AgentLoopFactory(Protocol):
    """Creates a fresh BaseAgentLoop per rollout (for seed isolation)."""

    def __call__(self, *, backend: LLMBackend, seed: int | None) -> BaseAgentLoop: ...


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
    log_every: int = 1
    save_every: int = 0
    output_dir: Path | None = None
    seed: int | None = 0
    metrics_sink: Callable[[dict[str, Any]], None] | None = None  # optional live sink
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


@dataclass(slots=True)
class TrainStats:
    iters: list[dict[str, Any]] = field(default_factory=list)

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

        params = list(policy.trainable_parameters())
        if not params:
            raise RuntimeError("policy has no trainable parameters")
        self.optim = torch.optim.AdamW(params, lr=self.cfg.lr)

        self.ref_policy: LLMBackend | None = None
        if self.cfg.use_reference and hasattr(policy, "clone_frozen"):
            self.ref_policy = policy.clone_frozen()  # type: ignore[attr-defined]

        self.agent_loop_factory = agent_loop_factory or default_policy_loop_factory(
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )
        self.logger = logger or (lambda rec: print(self._format_log(rec)))
        self.stats = TrainStats()
        self._seed_counter = 0
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

        self._maybe_resume()

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------

    def _validate_backend(self, policy: LLMBackend) -> None:
        """Override to require a value head, etc."""

    # ------------------------------------------------------------------
    # rollout → records
    # ------------------------------------------------------------------

    def _next_seed(self) -> int | None:
        if self.cfg.seed is None:
            return None
        self._seed_counter += 1
        return self.cfg.seed + self._seed_counter

    async def _collect_group(self, item: dict[str, Any]) -> list[RolloutRecord]:
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

            base_meta = {
                "final_output": trajectory.final_output,
                "reward_components": [
                    {"name": c.name, "score": c.score} for c in summary.components
                ],
                "turns_used": trajectory.turns_used,
            }

            if self.cfg.multi_turn and rl_meta.get("turns"):
                # Emit one RolloutRecord per turn, each scored under its true
                # rollout context. Reward is shared across turns of the same
                # rollout — group-normalization still works because same
                # group_id means same prompt (different rollouts).
                for t_idx, turn in enumerate(rl_meta["turns"]):
                    records.append(
                        RolloutRecord(
                            prompt_ids=list(turn["prompt_prefix_ids"]),
                            response_ids=list(turn["response_ids"]),
                            old_logprobs=list(turn["old_logprobs"]),
                            reward=float(summary.final_score),
                            group_id=group_id,
                            metadata={**base_meta, "turn_index": t_idx},
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
                        metadata=base_meta,
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
        batch = RolloutBatch(records=batch_records)

        self.optim.zero_grad()
        loss, stats = self.algo.compute_loss(self.policy, self.ref_policy, batch)
        if self.lagrangian is not None:
            loss = self.lagrangian.penalty_term(loss)
        if loss.requires_grad:
            loss.backward()
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.trainable_parameters()),
                    max_norm=self.cfg.grad_clip,
                )
            self.optim.step()

        return stats

    def train(self) -> TrainStats:
        start = int(getattr(self, "_start_iter", 0))
        last_iter = start
        for it in range(start, self.cfg.n_iters):
            last_iter = it
            stats = asyncio.run(self._one_iter(it))
            if self.lagrangian is not None:
                self.lagrangian.dual_step()
            record = {"iter": it, "algo": self.algo_name, **stats.as_dict()}
            if self.lagrangian is not None:
                record["lagrangian"] = self.lagrangian.snapshot()
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
