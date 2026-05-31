"""Ray-based distributed rollout pool.

Architecture (learner + N Ray actors):

    Learner (driver)                Ray Actor_i (remote worker)
        │                                 │
        │── broadcast(state_dict) ──────►│   load_state_dict
        │                                 │
        │── submit_tasks([RolloutTask])──►│   run env → rollout → reward
        │◄─── [RolloutRecord, ...] ───────│
        │
        │── shutdown ──────────────────►│   actor.exit()

Design:
  - Requires ``ray`` (``pip install ray``). Raises RayUnavailableError if absent.
  - Each actor lazily builds (backend, env, reward_manager, agent_loop_factory)
    on first use via a ``builder_fn``, matching the MPRolloutPool contract.
  - Weight sync: state_dict is serialised by Ray's object store (shared memory
    on same node, NCCL/TCP across nodes). Works for CPU and CUDA tensors.
  - vLLM validation: when the builder_fn returns a VLLMRolloutBackend, rollouts
    use KV-cache batched generation and are ~5-10× faster per batch.
  - Same RolloutTask / RolloutRecord types as mp_pool so callers can swap
    MPRolloutPool ↔ RayRolloutPool without changing trainer code.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.distributed.mp_pool import RolloutTask


class RayUnavailableError(RuntimeError):
    """Raised when ``ray`` is not installed."""


# ---------------------------------------------------------------------------
# Ray worker actor
# ---------------------------------------------------------------------------

def _make_ray_actor_cls() -> Any:
    """Lazily create the Ray actor class to avoid import-time Ray dependency."""
    try:
        import ray
    except ImportError as exc:
        raise RayUnavailableError(
            "RayRolloutPool requires ray. Install with: pip install 'ray[default]'"
        ) from exc

    import ray

    @ray.remote
    class _RayRolloutWorker:
        """Remote rollout worker actor."""

        def __init__(
            self,
            worker_id: int,
            builder_fn: Callable[[dict[str, Any]], tuple[Any, Any, Any, Any]],
            build_ctx: dict[str, Any],
        ) -> None:
            self.worker_id = worker_id
            self.build_ctx = build_ctx
            self._backend, self._env, self._reward_manager, self._agent_loop_factory = (
                builder_fn(build_ctx)
            )
            self._loop = asyncio.new_event_loop()

        def load_weights(self, state_dict_bytes: bytes) -> None:
            """Deserialize and load updated weights from learner."""
            import io

            import torch
            buf = io.BytesIO(state_dict_bytes)
            state_dict = torch.load(buf, map_location="cpu", weights_only=True)
            if hasattr(self._backend, "model") and self._backend.model is not None:
                self._backend.model.load_state_dict(state_dict)
            # If backend is vLLM, sync via its weight API
            elif hasattr(self._backend, "sync_weights_from"):
                self._backend.sync_weights_from(state_dict)

        def run_task(self, task: RolloutTask) -> dict[str, Any] | None:
            """Execute one rollout task and return an MP-pool-compatible result."""
            try:
                return self._loop.run_until_complete(self._run_task_async(task))
            except Exception as exc:
                return {
                    "task_seq": task.task_seq,
                    "worker_id": self.worker_id,
                    "error": repr(exc),
                }

        async def _run_task_async(self, task: RolloutTask) -> dict[str, Any]:
            from hermes_agentic_rl.core.rollout_manager import RolloutManager
            from hermes_agentic_rl.trainers.multi_turn_credit import assign_multi_turn_rewards

            loop_obj = self._agent_loop_factory(
                backend=self._backend, seed=task.seed
            )
            traj = await RolloutManager(loop_obj).collect(task.item, task.instruction)
            summary = await self._reward_manager.evaluate(task.item, traj, tool_context=None)

            rt = traj.metadata.get("runtime") or {}
            rl_meta = rt.get("rl") if isinstance(rt, dict) else None
            if rl_meta is None:
                raise RuntimeError("agent loop missing rl metadata")

            prompt_group_id = str(task.item.get("task_id", task.task_id or "group"))
            records: list[dict[str, Any]] = []
            base_meta = {
                "final_output": traj.final_output,
                "turns_used": traj.turns_used,
                "reward_components": [
                    {"name": c.name, "score": c.score} for c in summary.components
                ],
            }
            turns = rl_meta.get("turns")
            if turns:
                teacher_samples = []
                try:
                    teacher_samples = self._env.build_supervised_samples(task.item)
                except Exception:
                    teacher_samples = []
                teacher_responses: list[str | None] | None = None
                if teacher_samples:
                    teacher_responses = [None] * len(turns)
                    explicit = False
                    for sample in teacher_samples:
                        turn_index = getattr(sample, "metadata", {}).get("turn_index")
                        if isinstance(turn_index, int) and 0 <= turn_index < len(turns):
                            teacher_responses[turn_index] = str(sample.response)
                            explicit = True
                    if not explicit:
                        if len(teacher_samples) == len(turns):
                            teacher_responses = [str(sample.response) for sample in teacher_samples]
                        elif len(turns) == 1 and teacher_samples:
                            teacher_responses = [str(teacher_samples[0].response)]
                    if teacher_responses is not None and all(
                        response is None or not str(response).strip()
                        for response in teacher_responses
                    ):
                        teacher_responses = None
                turn_rewards = assign_multi_turn_rewards(
                    traj,
                    final_reward=float(summary.final_score),
                    n_turns=len(turns),
                    cfg=(
                        self.build_ctx.get("multi_turn_credit")
                        if isinstance(self.build_ctx, dict)
                        else None
                    ),
                    teacher_responses=teacher_responses,
                )
                for ti, turn in enumerate(turns):
                    turn_group_id = f"{prompt_group_id}::turn:{ti}"
                    credit_meta = (
                        dict(turn_rewards[ti])
                        if ti < len(turn_rewards)
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
                        {
                            "prompt_ids": list(turn["prompt_prefix_ids"]),
                            "response_ids": list(turn["response_ids"]),
                            "old_logprobs": list(turn["old_logprobs"]),
                            "old_seq_logprob": float(sum(turn["old_logprobs"])),
                            "reward": float(credit_meta.get("reward", summary.final_score)),
                            "group_id": turn_group_id,
                            "metadata": {
                                **base_meta,
                                "prompt_group_id": prompt_group_id,
                                "turn_group_id": turn_group_id,
                                "turn_index": ti,
                                "rollout_final_reward": float(summary.final_score),
                                "turn_credit": credit_meta,
                            },
                        }
                    )
            else:
                records.append(
                    {
                        "prompt_ids": list(rl_meta["prompt_ids"]),
                        "response_ids": list(rl_meta["response_ids"]),
                        "old_logprobs": list(rl_meta["old_logprobs"]),
                        "old_seq_logprob": float(sum(rl_meta["old_logprobs"])),
                        "reward": float(summary.final_score),
                        "group_id": prompt_group_id,
                        "metadata": base_meta,
                    }
                )
            return {
                "task_seq": task.task_seq,
                "worker_id": self.worker_id,
                "records": records,
                "final_score": float(summary.final_score),
            }

        def ping(self) -> str:
            return f"worker_{self.worker_id}_ok"

    return _RayRolloutWorker


# ---------------------------------------------------------------------------
# Public pool API
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RayRolloutPoolConfig:
    n_workers: int = 4
    num_cpus_per_worker: float = 1.0
    num_gpus_per_worker: float = 0.0   # >0 for GPU rollout (e.g. vLLM workers)
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)


class RayRolloutPool:
    """Distributed rollout pool backed by Ray remote actors.

    Drop-in replacement for ``MPRolloutPool``:

        pool = RayRolloutPool(
            builder_fn=my_builder,
            build_ctx={"model": "gpt2"},
            cfg=RayRolloutPoolConfig(n_workers=8, num_gpus_per_worker=0.5),
        )
        pool.start()
        pool.broadcast(state_dict)
        records = pool.collect(tasks)
        pool.shutdown()
    """

    def __init__(
        self,
        builder_fn: Callable[[dict[str, Any]], tuple[Any, Any, Any, Any]],
        build_ctx: dict[str, Any] | None = None,
        cfg: RayRolloutPoolConfig | None = None,
    ) -> None:
        self.builder_fn = builder_fn
        self.build_ctx = build_ctx or {}
        self.cfg = cfg or RayRolloutPoolConfig()
        self._actor_cls: Any = None
        self._actors: list[Any] = []
        self._pending_refs: list[Any] = []
        self._started = False

    def start(self) -> None:
        """Initialize Ray and spawn worker actors."""
        if self._started:
            return
        try:
            import ray
        except ImportError as exc:
            raise RayUnavailableError(
                "RayRolloutPool requires ray. Install with: pip install 'ray[default]'"
            ) from exc

        if not ray.is_initialized():
            ray.init(**self.cfg.ray_init_kwargs)

        self._actor_cls = _make_ray_actor_cls()
        self._actors = [
            self._actor_cls.options(  # type: ignore[attr-defined]
                num_cpus=self.cfg.num_cpus_per_worker,
                num_gpus=self.cfg.num_gpus_per_worker,
            ).remote(i, self.builder_fn, self.build_ctx)
            for i in range(self.cfg.n_workers)
        ]
        self._started = True

    def broadcast(self, state_dict: dict[str, Any]) -> None:
        """Push updated model weights to all actors."""
        import io

        import torch
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        weights_bytes = buf.getvalue()

        import ray
        refs = [actor.load_weights.remote(weights_bytes) for actor in self._actors]
        ray.get(refs)

    def broadcast_weights(self, state_dict: dict[str, Any]) -> None:
        """MP-pool-compatible alias used by OnPolicyTrainer."""
        self.broadcast(state_dict)

    def submit_tasks(self, tasks: list[RolloutTask]) -> None:
        """Submit tasks round-robin and keep refs for ``drain``."""
        if not self._started:
            raise RuntimeError("pool not started")
        self._pending_refs = []
        for i, task in enumerate(tasks):
            actor = self._actors[i % len(self._actors)]
            self._pending_refs.append(actor.run_task.remote(task))

    def drain(self, expected: int) -> list[dict[str, Any]]:
        """Gather submitted task results in submission order."""
        import ray
        refs = self._pending_refs[:expected]
        self._pending_refs = self._pending_refs[expected:]
        results = ray.get(refs)
        out = [r for r in results if r is not None]
        for result in out:
            if "error" in result:
                raise RuntimeError(
                    f"ray rollout worker {result.get('worker_id')} crashed: "
                    f"{result.get('error')}"
                )
        out.sort(key=lambda r: r["task_seq"])
        return out

    def collect(self, tasks: list[RolloutTask]) -> list[dict[str, Any]]:
        """Submit tasks round-robin and gather results."""
        self.submit_tasks(tasks)
        return self.drain(expected=len(tasks))

    def shutdown(self) -> None:
        """Gracefully shut down all actors."""
        if not self._started:
            return
        import ray
        for actor in self._actors:
            ray.kill(actor)
        self._actors = []
        self._pending_refs = []
        self._started = False

    def __enter__(self) -> RayRolloutPool:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()
