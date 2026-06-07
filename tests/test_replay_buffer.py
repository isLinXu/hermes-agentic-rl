"""Tests for ReplayBuffer — ring buffer with recency-weighted sampling."""

from __future__ import annotations

import pytest

from hermes_agentic_rl.trainers.replay_buffer import (
    ReplayBuffer,
    ReplayBufferStats,
    build_replay_buffer_from_config,
)


class _FakeRecord:
    """Minimal record with metadata dict (duck-types as RolloutRecord)."""

    __slots__ = ("reward", "metadata")

    def __init__(self, reward: float = 0.0, metadata: dict | None = None) -> None:
        self.reward = reward
        self.metadata = metadata if metadata is not None else {}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestReplayBufferConstruction:
    def test_default_construction(self):
        buf = ReplayBuffer()
        assert buf.capacity == 2048
        assert buf.size == 0
        assert buf.recency_alpha == 0.0

    def test_custom_construction(self):
        buf = ReplayBuffer(capacity=512, recency_alpha=1.5, seed=42)
        assert buf.capacity == 512
        assert buf.recency_alpha == 1.5

    def test_minimum_capacity(self):
        buf = ReplayBuffer(capacity=0)
        assert buf.capacity == 1


# ---------------------------------------------------------------------------
# Push / overflow
# ---------------------------------------------------------------------------


class TestReplayBufferPush:
    def test_push_increments_size(self):
        buf = ReplayBuffer(capacity=10)
        buf.push([_FakeRecord(1.0)], policy_version=0)
        assert buf.size == 1

    def test_push_multiple_records(self):
        buf = ReplayBuffer(capacity=10)
        buf.push([_FakeRecord(i) for i in range(5)], policy_version=0)
        assert buf.size == 5

    def test_ring_overwrite(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push([_FakeRecord(i)], policy_version=i)
        assert buf.size == 3
        assert buf.stats.overflow_count == 2

    def test_empty_push_noop(self):
        buf = ReplayBuffer(capacity=10)
        buf.push([], policy_version=0)
        assert buf.size == 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestReplayBufferSampling:
    def test_uniform_sampling(self):
        buf = ReplayBuffer(capacity=100, recency_alpha=0.0, seed=42)
        records = [_FakeRecord(i) for i in range(20)]
        buf.push(records, policy_version=0)
        sampled = buf.sample(5)
        assert len(sampled) == 5
        assert all(isinstance(r, _FakeRecord) for r in sampled)

    def test_recency_sampling(self):
        buf = ReplayBuffer(capacity=100, recency_alpha=2.0, seed=42)
        records = [_FakeRecord(i) for i in range(20)]
        buf.push(records, policy_version=0)
        sampled = buf.sample(5)
        assert len(sampled) == 5

    def test_sample_more_than_available(self):
        buf = ReplayBuffer(capacity=100)
        buf.push([_FakeRecord(i) for i in range(3)], policy_version=0)
        sampled = buf.sample(10)
        assert len(sampled) == 3

    def test_sample_empty_buffer(self):
        buf = ReplayBuffer(capacity=100)
        assert buf.sample(5) == []


# ---------------------------------------------------------------------------
# mix_with_current
# ---------------------------------------------------------------------------


class TestReplayBufferMixWithCurrent:
    def test_no_mix_when_zero_ratio(self):
        buf = ReplayBuffer(capacity=100)
        current = [_FakeRecord(i) for i in range(10)]
        result = buf.mix_with_current(current, mix_ratio=0.0, policy_version=0)
        assert len(result) == 10

    def test_mix_adds_replay_records(self):
        buf = ReplayBuffer(capacity=100, seed=42)
        # Pre-fill buffer with some records
        buf.push([_FakeRecord(i) for i in range(20)], policy_version=0)
        current = [_FakeRecord(i) for i in range(10)]
        result = buf.mix_with_current(current, mix_ratio=0.5, policy_version=1)
        # Should have 10 current + ~5 replay
        assert len(result) >= 10
        # Replay records should be tagged
        replay_tagged = [r for r in result if r.metadata.get("_replay_sampled")]
        assert len(replay_tagged) > 0

    def test_current_records_are_pushed_into_buffer(self):
        buf = ReplayBuffer(capacity=100)
        current = [_FakeRecord(i) for i in range(10)]
        buf.mix_with_current(current, mix_ratio=0.0, policy_version=0)
        assert buf.size == 10

    def test_replay_records_have_staleness(self):
        buf = ReplayBuffer(capacity=100, seed=42)
        buf.push([_FakeRecord(i) for i in range(10)], policy_version=0)
        current = [_FakeRecord(i) for i in range(5)]
        result = buf.mix_with_current(current, mix_ratio=1.0, policy_version=5)
        for rec in result:
            if rec.metadata.get("_replay_sampled"):
                assert rec.metadata.get("_replay_staleness", 0) >= 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestBuildFromConfig:
    def test_none_config_returns_none(self):
        assert build_replay_buffer_from_config(None) is None

    def test_disabled_returns_none(self):
        assert build_replay_buffer_from_config({"enabled": False}) is None

    def test_enabled_creates_buffer(self):
        buf = build_replay_buffer_from_config({
            "enabled": True,
            "capacity": 512,
            "recency_alpha": 1.0,
            "seed": 42,
        })
        assert isinstance(buf, ReplayBuffer)
        assert buf.capacity == 512
        assert buf.recency_alpha == 1.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestReplayBufferStats:
    def test_stats_as_dict(self):
        stats = ReplayBufferStats(push_count=10, sample_count=5, current_capacity=8)
        d = stats.as_dict()
        assert d["replay_push_count"] == 10.0
        assert d["replay_sample_count"] == 5.0
        assert d["replay_capacity_used"] == 8.0

    def test_clear_resets_size(self):
        buf = ReplayBuffer(capacity=10)
        buf.push([_FakeRecord(i) for i in range(5)], policy_version=0)
        assert buf.size == 5
        buf.clear()
        assert buf.size == 0

    def test_snapshot(self):
        buf = ReplayBuffer(capacity=10, recency_alpha=0.5, seed=42)
        buf.push([_FakeRecord(i) for i in range(3)], policy_version=0)
        snap = buf.snapshot()
        assert snap["capacity"] == 10
        assert snap["recency_alpha"] == 0.5
        assert snap["size"] == 3
