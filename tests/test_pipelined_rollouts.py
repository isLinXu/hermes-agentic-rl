"""P0-2 step 1: pipelined (double-buffered) rollout/update.

Verifies the scheduling contract without spawning real worker processes: a
lightweight in-process fake pool records the order of weight broadcasts vs
gradient updates and generates valid rollout records so the update path runs.

Key property: with ``pipeline_rollouts=True`` the dispatch (weight broadcast)
for iteration N+1 happens BEFORE the gradient update of iteration N, so the
next iter's rollout overlaps the current update (1-step staleness). With it
off, each broadcast immediately precedes its own update (BSP).
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


class FakeRolloutPool:
    """In-process stand-in for MPRolloutPool that logs broadcast ordering."""

    def __init__(self, backend: TinyCausalLMBackend, events: list[str]) -> None:
        self.backend = backend
        self.events = events
        self._tasks: list = []
        self._broadcasts = 0
        self._rng = random.Random(0)

    def broadcast_weights(self, state_dict) -> None:  # type: ignore[no-untyped-def]
        self.events.append(f"broadcast_{self._broadcasts}")
        self._broadcasts += 1

    def submit_tasks(self, tasks) -> None:  # type: ignore[no-untyped-def]
        self._tasks = list(tasks)

    def drain(self, expected: int):  # type: ignore[no-untyped-def]
        results = []
        for task in self._tasks:
            prompt_ids = self.backend.tokenizer.encode(task.instruction)
            out = self.backend.generate(prompt_ids, max_new_tokens=4, seed=task.seed)
            rec = {
                "prompt_ids": list(prompt_ids),
                "response_ids": list(out.response_ids),
                "old_logprobs": list(out.logprobs),
                # variance within the same group_id => non-zero GRPO advantage
                "reward": self._rng.random(),
                "group_id": str(task.task_id),
                "metadata": {},
            }
            results.append({"final_score": rec["reward"], "records": [rec]})
        self._tasks = []
        return results


def _run(pipeline: bool, events: list[str]):
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    pool = FakeRolloutPool(backend, events)
    cfg = GRPOTrainerConfig(
        n_iters=3,
        group_size=4,
        prompts_per_iter=1,
        max_new_tokens=4,
        lr=5e-3,
        log_every=100,
        seed=1,
        pipeline_rollouts=pipeline,
    )
    trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg, rollout_pool=pool)

    orig_update = trainer._update_on_records

    def _logged_update(records, iter_idx):  # type: ignore[no-untyped-def]
        events.append(f"update_{iter_idx}")
        return orig_update(records, iter_idx)

    trainer._update_on_records = _logged_update  # type: ignore[assignment]
    stats = trainer.train()
    assert len(stats.iters) == 3
    return stats


def test_pipelined_dispatches_next_before_current_update():
    events: list[str] = []
    _run(pipeline=True, events=events)
    # One broadcast per iter (prefetched), three updates.
    broadcasts = [e for e in events if e.startswith("broadcast")]
    updates = [e for e in events if e.startswith("update")]
    assert len(broadcasts) == 3
    assert updates == ["update_0", "update_1", "update_2"]
    # The dispatch for iter 1 (2nd broadcast) must precede update_0 (overlap).
    assert events.index(broadcasts[1]) < events.index("update_0")
    # The dispatch for iter 2 (3rd broadcast) must precede update_1.
    assert events.index(broadcasts[2]) < events.index("update_1")


def test_bsp_dispatches_after_previous_update():
    events: list[str] = []
    _run(pipeline=False, events=events)
    broadcasts = [e for e in events if e.startswith("broadcast")]
    assert len(broadcasts) == 3
    # BSP: each broadcast immediately precedes its OWN update, so the 2nd
    # broadcast comes AFTER update_0 (no overlap).
    assert events.index(broadcasts[1]) > events.index("update_0")


def test_pipelined_reports_one_step_staleness():
    events: list[str] = []
    stats = _run(pipeline=True, events=events)
    staleness = [it["rollout_staleness"] for it in stats.iters]
    versions = [it["policy_version"] for it in stats.iters]
    # iter0 consumes freshly-dispatched rollouts (no update has landed yet),
    # then steady-state is exactly 1 step stale.
    assert staleness == [0.0, 1.0, 1.0]
    assert versions == [0.0, 1.0, 2.0]


def test_bsp_reports_zero_staleness():
    events: list[str] = []
    stats = _run(pipeline=False, events=events)
    assert [it["rollout_staleness"] for it in stats.iters] == [0.0, 0.0, 0.0]


def test_pipeline_disabled_without_pool():
    # pipeline_rollouts=True but no pool => falls back to local collection.
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    cfg = GRPOTrainerConfig(
        n_iters=2,
        group_size=3,
        prompts_per_iter=1,
        max_new_tokens=4,
        lr=5e-3,
        log_every=100,
        seed=1,
        pipeline_rollouts=True,
    )
    trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
    assert trainer._pipeline_enabled() is False
    stats = trainer.train()
    assert len(stats.iters) == 2
