"""Tests for A3: judge caching + process-reward aggregation (o + mean step PRM).

Covers:
  * JudgeCache hit/miss/eviction bookkeeping and content addressing.
  * cached_judge wrapper avoids re-querying identical (response, next_state).
  * ProcessRewardAggregator implements final = o + (1/m)·Σ rᵢ.
  * Falls back to the single next_state when no per-step list is present.
  * Outcome resolution via metadata key and callable.
"""

from __future__ import annotations

import asyncio

from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.judge_cache import JudgeCache, cached_judge
from hermes_agentic_rl.rewards.next_state_prm import NextStateJudge, NextStatePRM, PRMVote
from hermes_agentic_rl.rewards.process_reward import (
    ProcessRewardAggregator,
    ProcessRewardConfig,
)

# ---------------------------------------------------------------------------
# JudgeCache
# ---------------------------------------------------------------------------


def test_judge_cache_hit_miss_and_content_addressing():
    cache = JudgeCache(max_entries=16)
    k1 = cache.make_key("j", "resp", "ns")
    k2 = cache.make_key("j", "resp", "ns")
    k3 = cache.make_key("j", "resp", "different")
    assert k1 == k2 and k1 != k3

    found, _ = cache.get(k1)
    assert found is False
    cache.put(k1, "value")
    found, val = cache.get(k1)
    assert found is True and val == "value"
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_judge_cache_lru_eviction():
    cache = JudgeCache(max_entries=2)
    for i in range(3):
        cache.put(cache.make_key("j", str(i), ""), i)
    assert cache.stats.evictions == 1


def test_cached_judge_avoids_recompute():
    calls = {"n": 0}

    def judge(_resp: str, _ns: str) -> str:
        calls["n"] += 1
        return "hint"

    cache = JudgeCache()
    wrapped = cached_judge(judge, cache, judge_id="t")

    async def run() -> None:
        a = await wrapped("r", "n")
        b = await wrapped("r", "n")  # identical → cached
        c = await wrapped("r", "other")  # different → recompute
        assert a == b == c == "hint"

    asyncio.run(run())
    assert calls["n"] == 2  # only 2 distinct pairs queried
    assert cache.stats.hits == 1


def test_cached_judge_supports_async():
    async def judge(_resp: str, _ns: str) -> str:
        return "async-hint"

    cache = JudgeCache()
    wrapped = cached_judge(judge, cache, judge_id="t")
    assert asyncio.run(wrapped("r", "n")) == "async-hint"
    assert asyncio.run(wrapped("r", "n")) == "async-hint"
    assert cache.stats.hits == 1


# ---------------------------------------------------------------------------
# ProcessRewardAggregator
# ---------------------------------------------------------------------------


def _vote_judge(verdict_map: dict[str, int]) -> NextStateJudge:
    async def fn(_response: str, next_state: str) -> PRMVote:
        return PRMVote(vote=verdict_map.get(next_state, 0), hint=None, raw=next_state)

    return NextStateJudge(fn)


def _traj(metadata: dict) -> Trajectory:
    return Trajectory(
        task_id="t",
        prompt="p",
        steps=[],
        final_output="ans",
        finished_naturally=True,
        turns_used=1,
        metadata=metadata,
    )


def test_process_reward_aggregates_outcome_plus_mean_steps():
    # 3 steps: GOOD(+1), GOOD(+1), BAD(-1) → mean = (1+1-1)/3 = 0.333
    judge = _vote_judge({"s_good1": 1, "s_good2": 1, "s_bad": -1})
    prm = NextStatePRM(judge)
    agg = ProcessRewardAggregator(prm, ProcessRewardConfig(outcome_default=0.5))

    traj = _traj(
        {
            "outcome_reward": 0.5,
            "runtime": {"step_next_states": ["s_good1", "s_good2", "s_bad"]},
        }
    )
    res = asyncio.run(agg.evaluate({}, traj, None))
    # final = 1.0*0.5 (outcome) + 1.0*0.333 (process mean)
    assert abs(res.score - (0.5 + (1 + 1 - 1) / 3)) < 1e-6


def test_process_reward_falls_back_to_single_next_state():
    judge = _vote_judge({"only": 1})
    prm = NextStatePRM(judge)
    agg = ProcessRewardAggregator(prm, ProcessRewardConfig(outcome_default=0.0))
    traj = _traj({"runtime": {"next_state": "only"}})
    res = asyncio.run(agg.evaluate({}, traj, None))
    assert abs(res.score - 1.0) < 1e-6  # 0 outcome + 1.0 single-step process


def test_process_reward_outcome_callable_takes_precedence():
    judge = _vote_judge({})  # all neutral → process mean 0
    prm = NextStatePRM(judge)
    agg = ProcessRewardAggregator(
        prm,
        ProcessRewardConfig(),
        outcome_fn=lambda _t: 0.9,
    )
    traj = _traj({"outcome_reward": 0.1, "runtime": {"next_state": "x"}})
    res = asyncio.run(agg.evaluate({}, traj, None))
    assert abs(res.score - 0.9) < 1e-6  # callable wins over metadata key


def test_process_reward_clip():
    judge = _vote_judge({"s": 1})
    prm = NextStatePRM(judge)
    agg = ProcessRewardAggregator(prm, ProcessRewardConfig(outcome_default=5.0, clip=1.0))
    traj = _traj({"runtime": {"next_state": "s"}})
    res = asyncio.run(agg.evaluate({}, traj, None))
    assert res.score == 1.0  # clipped from 5+1=6
