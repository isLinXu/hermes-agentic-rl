"""Multiprocessing-based rollout worker pool.

Architecture (learner ↔ N workers):

    Learner                    Worker_i (child process)
      │                           │
      │── broadcast(state_dict) ─►│   load into local policy
      │                           │
      │── submit(RolloutTask)  ──►│   run agent_loop → rollout → reward
      │                           │     → RolloutRecord
      │◄── results  ─────────────│
      │                           │
      │── shutdown ─────────────►│   clean exit

Design constraints:
  - Pure stdlib (no Ray / no torch.distributed / no NCCL required)
  - Workers reconstruct the backend + env + reward_manager + agent_loop
    locally via `builder_fn(build_ctx) -> (backend, env, reward_manager,
    agent_loop_factory)`; we cannot pickle these objects directly because
    they carry tokenizers / torch modules / closures.
  - Weight sync: state_dict is torch.saved to a bytes buffer and put on a
    queue. Each worker calls backend.model.load_state_dict() at task start.
  - One task = one rollout. Coarser batching (M tasks per send) is a free
    optimization the caller can do by submitting M tasks before draining.

Correctness notes:
  - old_logprobs are captured on the worker under the **broadcast** weights,
    so ratio = exp(new_logp - old_logp) computed on the learner with the
    same weights is valid. The Trainer must update weights BEFORE calling
    collect() on the next iter; the pool's broadcast() does this.
  - Seeds: each task carries its own seed, so rollouts are reproducible
    regardless of worker assignment.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.utils.coerce import coerce_float

BuilderFn = Callable[[dict[str, Any]], tuple[Any, Any, Any, Any]]
"""Signature: builder_fn(build_ctx) -> (backend, env, reward_manager, agent_loop_factory)."""


@dataclass(slots=True)
class RolloutTask:
    task_id: str
    item: dict[str, Any]
    instruction: str
    seed: int | None = None
    task_seq: int = 0


@dataclass(slots=True)
class MPRolloutPoolConfig:
    n_workers: int = 2
    ctx_method: str = "spawn"  # "fork" is faster but unsafe with torch on macOS
    task_timeout: float = 120.0
    worker_startup_timeout: float = 120.0
    build_ctx: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Worker main
# ---------------------------------------------------------------------------


def _worker_main(
    worker_id: int,
    builder_fn: BuilderFn,
    build_ctx: dict[str, Any],
    task_q: mp.Queue[Any],
    result_q: mp.Queue[Any],
    weight_q: mp.Queue[Any],
) -> None:
    """Run forever: pull weights → pull tasks → emit results."""
    import asyncio

    import torch  # local import so parent survives if torch missing

    from hermes_agentic_rl.core.rollout_manager import RolloutManager

    try:
        backend, _env_unused, reward_manager, agent_loop_factory = builder_fn(build_ctx)
    except Exception as exc:  # pragma: no cover — defensive
        result_q.put(("builder_error", worker_id, repr(exc), traceback.format_exc()))
        return

    result_q.put(("ready", worker_id))
    current_weights_version = -1

    def _sync_weights() -> None:
        nonlocal current_weights_version
        # drain weight_q, keep only the latest
        latest = None
        while True:
            try:
                latest = weight_q.get_nowait()
            except Exception:
                break
        if latest is None:
            return
        version, blob = latest
        if version <= current_weights_version:
            return
        if hasattr(backend, "model") and blob is not None:
            buf = io.BytesIO(blob)
            state = torch.load(buf, map_location="cpu")
            backend.model.load_state_dict(state)
            backend.model.eval()
        current_weights_version = version

    while True:
        try:
            msg = task_q.get()
        except Exception:
            break
        if msg is None:
            break  # shutdown sentinel
        kind, payload = msg
        if kind == "task":
            task: RolloutTask = payload
            _sync_weights()
            try:
                loop = agent_loop_factory(backend=backend, seed=task.seed)

                async def _run(
                    task_item=task.item,
                    task_instruction=task.instruction,
                    task_loop=loop,
                    _task_task_id=task.task_id,
                    _task_task_seq=task.task_seq,
                ):
                    traj = await RolloutManager(task_loop).collect(task_item, task_instruction)
                    summary = await reward_manager.evaluate(task_item, traj, tool_context=None)
                    from hermes_agentic_rl.trainers.multi_turn_credit import (
                        assign_multi_turn_rewards,
                    )

                    runtime_block = traj.metadata.get("runtime") or {}
                    rl_meta = (
                        runtime_block.get("rl") if isinstance(runtime_block, dict) else None
                    ) or traj.metadata.get("rl")
                    if rl_meta is None:
                        raise RuntimeError("agent loop missing rl metadata")
                    prompt_group_id = str(task_item.get("task_id", _task_task_id or "group"))
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
                            teacher_samples = _env_unused.build_supervised_samples(task_item)
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
                                    teacher_responses = [
                                        str(sample.response) for sample in teacher_samples
                                    ]
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
                                build_ctx.get("multi_turn_credit")
                                if isinstance(build_ctx, dict)
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
                                    "reward": coerce_float(
                                        credit_meta.get("reward"),
                                        default=float(summary.final_score),
                                    ),
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
                                "reward": float(summary.final_score),
                                "group_id": prompt_group_id,
                                "metadata": base_meta,
                            }
                        )
                    return {
                        "task_seq": _task_task_seq,
                        "worker_id": worker_id,
                        "records": records,
                        "final_score": float(summary.final_score),
                    }

                out = asyncio.run(_run())
                result_q.put(("ok", out))
            except Exception as exc:  # pragma: no cover — defensive
                result_q.put(
                    (
                        "task_error",
                        {
                            "task_seq": task.task_seq,
                            "worker_id": worker_id,
                            "error": repr(exc),
                            "tb": traceback.format_exc(),
                        },
                    )
                )
        elif kind == "shutdown":
            break


# ---------------------------------------------------------------------------
# Learner-side pool
# ---------------------------------------------------------------------------


class MPRolloutPool:
    """Public API for a learner process."""

    def __init__(self, cfg: MPRolloutPoolConfig, builder_fn: BuilderFn) -> None:
        self.cfg = cfg
        self.builder_fn = builder_fn
        self._ctx: Any = mp.get_context(cfg.ctx_method)
        self._task_qs: list[mp.Queue[Any]] = []
        self._weight_qs: list[mp.Queue[Any]] = []
        self._result_q: mp.Queue[Any] = self._ctx.Queue()
        self._procs: list[mp.Process] = []
        self._version = 0
        self._started = False

    # --- lifecycle ---

    def start(self) -> None:
        if self._started:
            return
        for wid in range(self.cfg.n_workers):
            tq: mp.Queue[Any] = self._ctx.Queue()
            wq: mp.Queue[Any] = self._ctx.Queue()
            p = self._ctx.Process(
                target=_worker_main,
                args=(wid, self.builder_fn, self.cfg.build_ctx, tq, self._result_q, wq),
                daemon=True,
            )
            p.start()
            self._task_qs.append(tq)
            self._weight_qs.append(wq)
            self._procs.append(p)
        self._wait_for_workers_ready()
        self._started = True

    def _wait_for_workers_ready(self) -> None:
        import time

        ready = 0
        deadline = time.monotonic() + float(self.cfg.worker_startup_timeout)
        while ready < self.cfg.n_workers:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"only {ready}/{self.cfg.n_workers} rollout workers became ready "
                    f"within {self.cfg.worker_startup_timeout}s"
                )
            try:
                kind, payload = self._result_q.get(timeout=1.0)
            except Exception:
                continue
            if kind == "ready":
                ready += 1
            elif kind == "builder_error":
                raise RuntimeError(f"worker builder failed: {payload}")

    def shutdown(self) -> None:
        if not self._started:
            return
        for tq in self._task_qs:
            try:
                tq.put(("shutdown", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()
        self._task_qs.clear()
        self._weight_qs.clear()
        self._procs.clear()
        self._started = False

    def __enter__(self) -> MPRolloutPool:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.shutdown()

    # --- broadcast weights ---

    def broadcast_weights(self, state_dict: dict[str, Any]) -> None:
        """Push the latest policy state_dict to every worker."""
        import torch

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        blob = buf.getvalue()
        self._version += 1
        for wq in self._weight_qs:
            wq.put((self._version, blob))

    # --- submit / drain ---

    def submit_tasks(self, tasks: list[RolloutTask]) -> None:
        """Distribute tasks round-robin across workers."""
        if not self._started:
            raise RuntimeError("pool not started")
        for i, t in enumerate(tasks):
            self._task_qs[i % len(self._task_qs)].put(("task", t))

    def drain(self, expected: int) -> list[dict[str, Any]]:
        """Collect `expected` results. Raises on worker error."""
        out: list[dict[str, Any]] = []
        while len(out) < expected:
            kind, payload = self._result_q.get(timeout=self.cfg.task_timeout)
            if kind == "ok":
                out.append(payload)
            elif kind == "task_error":
                raise RuntimeError(
                    f"rollout worker {payload.get('worker_id')} crashed: "
                    f"{payload.get('error')}\n{payload.get('tb')}"
                )
            elif kind == "builder_error":
                raise RuntimeError(f"worker builder failed: {payload}")
        # preserve submission order (by task_seq)
        out.sort(key=lambda r: r["task_seq"])
        return out
