"""Tests for the v0.9.3 framework extensions.

* RewardComposer (per-component normalization + conditional gating
  + turn discount).
* FaultTolerantRolloutPool (retry, timeout extension, restart).

These tests run without GPU and without spawning real subprocesses; the
fault-tolerant pool is exercised through a stub ``inner`` object that
mimics ``MPRolloutPool`` only where the wrapper actually touches it.
"""

from __future__ import annotations

import asyncio
import queue as queue_mod
from dataclasses import dataclass
from typing import Any

import pytest

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward
from hermes_agentic_rl.rewards.composer import (
    RewardComposer,
    RewardComposerConfig,
)

# ---------------------------------------------------------------------------
# Test rewards: tiny deterministic stubs
# ---------------------------------------------------------------------------


class _ConstReward(BaseReward):
    """Returns a fixed score regardless of input. Used for stat seeding."""

    def __init__(self, name: str, score: float, weight: float = 1.0) -> None:
        self.name = name
        self._score = score
        self.weight = weight

    async def evaluate(self, item, trajectory, tool_context):
        return RewardResult(
            name=self.name,
            score=self._score,
            reason="const",
            weight=self.weight,
        )


class _SequenceReward(BaseReward):
    """Returns scores from a list, advancing one entry per call."""

    def __init__(self, name: str, scores: list[float], weight: float = 1.0) -> None:
        self.name = name
        self._scores = list(scores)
        self.weight = weight
        self._idx = 0

    async def evaluate(self, item, trajectory, tool_context):
        s = self._scores[self._idx % len(self._scores)]
        self._idx += 1
        return RewardResult(name=self.name, score=s, reason="seq", weight=self.weight)


def _traj(turns: int = 1) -> Trajectory:
    return Trajectory(
        task_id="t",
        prompt="p",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=turns,
    )


# ---------------------------------------------------------------------------
# RewardComposer
# ---------------------------------------------------------------------------


def test_composer_default_behaves_like_weighted_sum():
    composer = RewardComposer(
        components=[_ConstReward("a", 1.0, weight=1.0), _ConstReward("b", 0.0, weight=1.0)],
    )
    summary = asyncio.run(composer.evaluate({}, _traj(), tool_context=None))
    assert abs(summary.final_score - 0.5) < 1e-9
    assert summary.metadata["composer"] == "reward_composer"
    assert summary.metadata["skipped_components"] == []


def test_composer_normalize_whitens_after_two_samples():
    seq = _SequenceReward("x", [1.0, 3.0, 5.0], weight=1.0)
    composer = RewardComposer(
        components=[seq],
        config={"normalize": {"x": True}},
    )
    s1 = asyncio.run(composer.evaluate({}, _traj(), tool_context=None))
    _s2 = asyncio.run(composer.evaluate({}, _traj(), tool_context=None))
    s3 = asyncio.run(composer.evaluate({}, _traj(), tool_context=None))

    # First sample: count=1 → whiten returns raw value (no stats yet).
    assert abs(s1.final_score - 1.0) < 1e-9
    # By the third sample the running mean is 3.0, std ≈ 2.0; whitened ≈ 1.0.
    # We just assert it's positive (above mean) and finite.
    assert s3.final_score > 0
    assert s3.components[0].metadata["normalized_score"] == s3.final_score
    assert s3.components[0].metadata["raw_score"] == 5.0


def test_composer_condition_skips_component():
    triggered: list[str] = []

    class _Fail(BaseReward):
        name = "should_skip"

        def __init__(self, weight: float = 1.0) -> None:
            self.weight = weight

        async def evaluate(self, item, trajectory, tool_context):
            triggered.append(self.name)
            return RewardResult(
                name=self.name, score=99.0, reason="should not run", weight=self.weight
            )

    composer = RewardComposer(
        components=[_ConstReward("base", 0.5), _Fail()],
        config={"conditions": {"should_skip": lambda item, traj: traj.turns_used > 5}},
    )
    summary = asyncio.run(composer.evaluate({}, _traj(turns=1), tool_context=None))
    assert "should_skip" in summary.metadata["skipped_components"]
    assert triggered == []
    assert abs(summary.final_score - 0.5) < 1e-9


def test_composer_turn_discount_applied():
    composer = RewardComposer(
        components=[_ConstReward("d", 1.0, weight=1.0)],
        config={"turn_discount": {"d": 0.5}},
    )
    summary = asyncio.run(composer.evaluate({}, _traj(turns=3), tool_context=None))
    # 0.5 ** 3 = 0.125
    assert abs(summary.final_score - 0.125) < 1e-9
    assert summary.components[0].metadata["turn_discount_gamma"] == 0.5


def test_composer_all_skipped_returns_zero():
    composer = RewardComposer(
        components=[_ConstReward("a", 1.0)],
        config={"conditions": {"a": lambda *_: False}},
    )
    summary = asyncio.run(composer.evaluate({}, _traj(), tool_context=None))
    assert summary.final_score == 0.0
    assert "a" in summary.metadata["skipped_components"]


def test_composer_condition_exception_falls_back_to_active(caplog):
    def boom(_item, _traj):
        raise RuntimeError("buggy condition")

    import logging
    import warnings

    composer = RewardComposer(
        components=[_ConstReward("a", 0.7)],
        config={"conditions": {"a": boom}},
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING):
            summary = asyncio.run(composer.evaluate({}, _traj(), tool_context=None))
    assert abs(summary.final_score - 0.7) < 1e-9
    assert any("condition raised" in str(rec.message) for rec in w)
    assert any("condition raised" in rec.message for rec in caplog.records)


def test_composer_reset_stats_clears_running_state():
    seq = _SequenceReward("y", [1.0, 2.0, 3.0])
    composer = RewardComposer(
        components=[seq],
        config={"normalize": {"y": True}},
    )
    asyncio.run(composer.evaluate({}, _traj(), tool_context=None))
    asyncio.run(composer.evaluate({}, _traj(), tool_context=None))
    snap1 = composer.snapshot_stats()
    assert snap1["y"]["count"] == 2.0
    composer.reset_stats()
    snap2 = composer.snapshot_stats()
    assert snap2["y"]["count"] == 0.0


def test_composer_dataclass_config_round_trip():
    cfg = RewardComposerConfig(normalize={"a": True}, turn_discount={"a": 0.9})
    composer = RewardComposer(components=[_ConstReward("a", 1.0)], config=cfg)
    assert composer.cfg.normalize == {"a": True}
    assert composer.cfg.turn_discount == {"a": 0.9}


def test_composer_unknown_aggregator_raises():
    with pytest.raises(ValueError):
        RewardComposer(
            components=[_ConstReward("a", 1.0)],
            config={"aggregator": "median"},
        )


# ---------------------------------------------------------------------------
# FaultTolerantRolloutPool — tested with a stub inner pool
# ---------------------------------------------------------------------------


@dataclass
class _FakeProc:
    alive: bool = True

    def is_alive(self) -> bool:
        return self.alive


class _StubInnerPool:
    """Minimal MPRolloutPool look-alike for unit-testing the fault wrapper.

    The wrapper accesses these attributes/methods on its inner pool:
    ``broadcast_weights``, ``submit_tasks``, ``start``, ``shutdown``,
    ``_result_q``, ``_task_qs``, ``_procs``, ``_weight_qs``.
    """

    def __init__(self, n_workers: int = 2) -> None:
        self._result_q: queue_mod.Queue[Any] = queue_mod.Queue()
        self._task_qs: list[queue_mod.Queue[Any]] = [queue_mod.Queue() for _ in range(n_workers)]
        self._weight_qs: list[queue_mod.Queue[Any]] = [queue_mod.Queue() for _ in range(n_workers)]
        self._procs: list[_FakeProc] = [_FakeProc() for _ in range(n_workers)]
        self.broadcasts: list[Any] = []
        self.start_calls = 0
        self.shutdown_calls = 0

    def broadcast_weights(self, state_dict):
        self.broadcasts.append(state_dict)

    def submit_tasks(self, tasks):
        for i, t in enumerate(tasks):
            self._task_qs[i % len(self._task_qs)].put(("task", t))

    def start(self):
        self.start_calls += 1
        for p in self._procs:
            p.alive = True

    def shutdown(self):
        self.shutdown_calls += 1
        for p in self._procs:
            p.alive = False


@dataclass
class _Task:
    task_seq: int
    item: dict
    instruction: str = ""
    seed: int | None = None


def _ok(seq: int) -> tuple[str, dict[str, Any]]:
    return ("ok", {"task_seq": seq, "worker_id": 0, "records": [], "final_score": 0.0})


def _err(seq: int, msg: str = "boom") -> tuple[str, dict[str, Any]]:
    return ("task_error", {"task_seq": seq, "worker_id": 0, "error": msg, "tb": ""})


def test_fault_tolerant_drain_happy_path():
    from hermes_agentic_rl.distributed.fault_tolerant_pool import (
        FaultTolerantPoolConfig,
        FaultTolerantRolloutPool,
    )

    inner = _StubInnerPool()
    pool = FaultTolerantRolloutPool(
        inner,
        FaultTolerantPoolConfig(max_retries=2, poll_interval=0.05),
    )
    pool.submit_tasks([_Task(task_seq=0, item={}), _Task(task_seq=1, item={})])
    inner._result_q.put(_ok(0))
    inner._result_q.put(_ok(1))
    out = pool.drain(expected=2)
    assert [r["task_seq"] for r in out] == [0, 1]
    m = pool.metrics()
    assert m["succeeded"] == 2 and m["retried"] == 0 and m["failed"] == 0


def test_fault_tolerant_retries_then_succeeds():
    from hermes_agentic_rl.distributed.fault_tolerant_pool import (
        FaultTolerantPoolConfig,
        FaultTolerantRolloutPool,
    )

    inner = _StubInnerPool()
    pool = FaultTolerantRolloutPool(
        inner,
        FaultTolerantPoolConfig(max_retries=3, poll_interval=0.05),
    )
    pool.submit_tasks([_Task(task_seq=42, item={})])

    # First attempt fails, then succeeds.
    inner._result_q.put(_err(42))
    inner._result_q.put(_ok(42))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pool.drain(expected=1)
    assert out == [{"task_seq": 42, "worker_id": 0, "records": [], "final_score": 0.0}]
    m = pool.metrics()
    assert m["succeeded"] == 1 and m["retried"] == 1


def test_fault_tolerant_exhausts_retries():
    from hermes_agentic_rl.distributed.fault_tolerant_pool import (
        FaultTolerantPoolConfig,
        FaultTolerantRolloutPool,
        RolloutPoolFailure,
    )

    inner = _StubInnerPool()
    pool = FaultTolerantRolloutPool(
        inner,
        FaultTolerantPoolConfig(max_retries=2, poll_interval=0.05),
    )
    pool.submit_tasks([_Task(task_seq=99, item={})])
    inner._result_q.put(_err(99, "bang"))
    inner._result_q.put(_err(99, "bang2"))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RolloutPoolFailure, match="bang2"):
            pool.drain(expected=1)
    assert pool.metrics()["failed"] == 1


def test_fault_tolerant_timeout_extends_when_workers_alive():
    from hermes_agentic_rl.distributed.fault_tolerant_pool import (
        FaultTolerantPoolConfig,
        FaultTolerantRolloutPool,
    )

    inner = _StubInnerPool()
    pool = FaultTolerantRolloutPool(
        inner,
        FaultTolerantPoolConfig(
            max_retries=2,
            poll_interval=0.02,
            worker_timeout=0.05,
        ),
    )
    pool.submit_tasks([_Task(task_seq=7, item={})])

    # The first drain attempt times out (queue is empty). After the timeout
    # the workers are still alive, so the wrapper extends the deadline.
    # We immediately enqueue a result to unblock the second poll.
    import threading
    import time

    def _enqueue_after_delay():
        time.sleep(0.07)
        inner._result_q.put(_ok(7))

    threading.Thread(target=_enqueue_after_delay, daemon=True).start()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pool.drain(expected=1)
    assert out[0]["task_seq"] == 7
    assert pool.metrics()["timeouts"] >= 1
