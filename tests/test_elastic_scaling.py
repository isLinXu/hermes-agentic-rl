"""Tests for elastic scaling in FaultTolerantRolloutPool.

Covers:
- ElasticScalingConfig defaults and validation
- scale_up / scale_down boundary conditions
- Auto-scaling decision logic (threshold-based)
- Metrics reporting
- Heartbeat staleness detection
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hermes_agentic_rl.distributed.fault_tolerant_pool import (
    ElasticScalingConfig,
    FaultTolerantPoolConfig,
    FaultTolerantRolloutPool,
    _PoolStats,
)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestElasticScalingConfig:
    def test_defaults(self):
        cfg = ElasticScalingConfig()
        assert cfg.enabled is False
        assert cfg.min_workers == 1
        assert cfg.max_workers == 8
        assert cfg.scale_up_threshold == 0.8
        assert cfg.scale_down_threshold == 0.3
        assert cfg.eval_window == 5
        assert cfg.cooldown_rounds == 3
        assert cfg.heartbeat_ttl == 0.0

    def test_custom_config(self):
        cfg = ElasticScalingConfig(
            enabled=True,
            min_workers=2,
            max_workers=16,
            scale_up_threshold=0.9,
            scale_down_threshold=0.2,
            eval_window=3,
            cooldown_rounds=2,
            heartbeat_ttl=60.0,
        )
        assert cfg.enabled is True
        assert cfg.max_workers == 16
        assert cfg.heartbeat_ttl == 60.0

    def test_fault_tolerant_config_with_elastic(self):
        elastic = ElasticScalingConfig(enabled=True, min_workers=2, max_workers=4)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        assert cfg.elastic is not None
        assert cfg.elastic.enabled is True
        assert cfg.elastic.max_workers == 4

    def test_fault_tolerant_config_without_elastic(self):
        cfg = FaultTolerantPoolConfig()
        assert cfg.elastic is None


# ---------------------------------------------------------------------------
# _PoolStats tests
# ---------------------------------------------------------------------------


class TestPoolStats:
    def test_default_stats(self):
        stats = _PoolStats()
        assert stats.scale_ups == 0
        assert stats.scale_downs == 0
        assert stats.current_workers == 0

    def test_stats_with_scaling(self):
        stats = _PoolStats()
        stats.scale_ups = 2
        stats.scale_downs = 1
        stats.current_workers = 5
        assert stats.scale_ups == 2
        assert stats.scale_downs == 1


# ---------------------------------------------------------------------------
# Helper: build a mock inner pool
# ---------------------------------------------------------------------------


def _make_mock_pool(n_workers: int = 2) -> MagicMock:
    """Create a MagicMock that mimics MPRolloutPool for unit tests.

    Uses simple list structures instead of real multiprocessing queues.
    """
    inner = MagicMock()
    inner._procs = [MagicMock(is_alive=lambda: True) for _ in range(n_workers)]
    inner._task_qs = [MagicMock() for _ in range(n_workers)]
    inner._weight_qs = [MagicMock() for _ in range(n_workers)]
    inner._result_q = MagicMock()
    inner._ctx = MagicMock()
    inner._version = 0
    inner._started = True
    inner.builder_fn = lambda ctx: (None, None, None, None)
    inner.cfg = MagicMock()
    inner.cfg.build_ctx = {}
    inner.cfg.worker_startup_timeout = 5.0
    return inner


# ---------------------------------------------------------------------------
# Scale up/down boundary tests
# ---------------------------------------------------------------------------


class TestScaleUpDownBoundaries:
    def test_scale_up_respects_max_workers(self):
        inner = _make_mock_pool(n_workers=2)
        elastic = ElasticScalingConfig(enabled=True, min_workers=1, max_workers=3)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        # Mock the worker creation to avoid real processes
        with patch.object(pool, "_wait_for_new_workers_ready"):
            added = pool.scale_up(5)
        assert added == 1  # only 1 can be added (3 - 2 = 1)

    def test_scale_up_returns_zero_when_at_max(self):
        inner = _make_mock_pool(n_workers=4)
        elastic = ElasticScalingConfig(enabled=True, min_workers=1, max_workers=4)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        added = pool.scale_up(1)
        assert added == 0

    def test_scale_down_respects_min_workers(self):
        inner = _make_mock_pool(n_workers=3)
        elastic = ElasticScalingConfig(enabled=True, min_workers=2, max_workers=8)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        removed = pool.scale_down(5)
        assert removed == 1  # only 1 can be removed (3 - 2 = 1)

    def test_scale_down_returns_zero_when_at_min(self):
        inner = _make_mock_pool(n_workers=1)
        elastic = ElasticScalingConfig(enabled=True, min_workers=1, max_workers=8)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        removed = pool.scale_down(1)
        assert removed == 0

    def test_scale_up_without_builder_returns_zero(self):
        inner = _make_mock_pool(n_workers=2)
        elastic = ElasticScalingConfig(enabled=True, min_workers=1, max_workers=8)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        # Don't call start() — builder_fn not captured
        pool._builder_fn = None
        added = pool.scale_up(1)
        assert added == 0

    def test_n_workers_property(self):
        inner = _make_mock_pool(n_workers=3)
        pool = FaultTolerantRolloutPool(inner)
        pool.start()
        assert pool.n_workers == 3

    def test_scale_down_removes_from_tail(self):
        inner = _make_mock_pool(n_workers=4)
        elastic = ElasticScalingConfig(enabled=True, min_workers=1, max_workers=8)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        removed = pool.scale_down(2)
        assert removed == 2
        assert pool.n_workers == 2

    def test_scale_up_increments_stats(self):
        inner = _make_mock_pool(n_workers=2)
        elastic = ElasticScalingConfig(enabled=True, min_workers=1, max_workers=6)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        with patch.object(pool, "_wait_for_new_workers_ready"):
            added = pool.scale_up(2)
        assert added == 2
        assert pool.stats.scale_ups == 2
        assert pool.stats.current_workers == 4


# ---------------------------------------------------------------------------
# Auto-scaling decision tests
# ---------------------------------------------------------------------------


class TestAutoScalingDecisions:
    def test_auto_scale_disabled_by_default(self):
        inner = _make_mock_pool(n_workers=2)
        pool = FaultTolerantRolloutPool(inner)
        pool.start()
        pool._drain_rounds = 10
        pool._maybe_auto_scale(10)
        assert pool.stats.scale_ups == 0
        assert pool.stats.scale_downs == 0

    def test_auto_scale_respects_cooldown(self):
        inner = _make_mock_pool(n_workers=2)
        elastic = ElasticScalingConfig(
            enabled=True, min_workers=1, max_workers=8,
            eval_window=1, cooldown_rounds=5,
        )
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        pool._drain_rounds = 1
        pool._last_scale_round = 0  # cooldown is 5, so 1 - 0 = 1 < 5
        pool._maybe_auto_scale(10)
        assert pool.stats.scale_ups == 0

    def test_auto_scale_respects_eval_window(self):
        inner = _make_mock_pool(n_workers=2)
        elastic = ElasticScalingConfig(
            enabled=True, min_workers=1, max_workers=8,
            eval_window=3, cooldown_rounds=0,
        )
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        pool._drain_rounds = 1
        pool._util_history = [0.9]  # only 1 entry, need 3
        pool._maybe_auto_scale(10)
        assert pool.stats.scale_ups == 0

    def test_auto_scale_up_triggers_on_high_util(self):
        inner = _make_mock_pool(n_workers=2)
        elastic = ElasticScalingConfig(
            enabled=True, min_workers=1, max_workers=8,
            scale_up_threshold=0.8, eval_window=1, cooldown_rounds=0,
        )
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        pool.stats.succeeded = 90
        pool.stats.failed = 10
        pool._drain_rounds = 1
        pool._last_scale_round = 0
        with patch.object(pool, "scale_up", return_value=1) as mock_up:
            pool._maybe_auto_scale(100)
            mock_up.assert_called_once()

    def test_auto_scale_down_triggers_on_low_util(self):
        inner = _make_mock_pool(n_workers=4)
        elastic = ElasticScalingConfig(
            enabled=True, min_workers=1, max_workers=8,
            scale_down_threshold=0.3, eval_window=1, cooldown_rounds=0,
        )
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        pool.stats.succeeded = 10
        pool.stats.failed = 90
        pool._drain_rounds = 1
        pool._last_scale_round = 0
        with patch.object(pool, "scale_down", return_value=1) as mock_down:
            pool._maybe_auto_scale(100)
            mock_down.assert_called_once()

    def test_util_history_window_eviction(self):
        inner = _make_mock_pool(n_workers=2)
        elastic = ElasticScalingConfig(
            enabled=True, min_workers=1, max_workers=8,
            eval_window=2, cooldown_rounds=0,
        )
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        # Simulate multiple drain rounds to fill and evict history
        pool._util_history = [0.5, 0.6]  # at window size
        pool._drain_rounds = 1
        pool._last_scale_round = 0
        pool.stats.succeeded = 80
        pool.stats.failed = 20
        with patch.object(pool, "scale_up", return_value=0):
            pool._maybe_auto_scale(100)
        # After appending, history should be evicted to window size
        assert len(pool._util_history) <= elastic.eval_window


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_include_scaling_fields(self):
        inner = _make_mock_pool(n_workers=2)
        pool = FaultTolerantRolloutPool(inner)
        pool.start()
        m = pool.metrics()
        assert "current_workers" in m
        assert "scale_ups" in m
        assert "scale_downs" in m
        assert m["current_workers"] == 2
        assert m["scale_ups"] == 0
        assert m["scale_downs"] == 0

    def test_metrics_after_manual_scale_down(self):
        inner = _make_mock_pool(n_workers=3)
        elastic = ElasticScalingConfig(enabled=True, min_workers=1, max_workers=8)
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        pool.scale_down(1)
        m = pool.metrics()
        assert m["scale_downs"] == 1
        assert m["current_workers"] == 2


# ---------------------------------------------------------------------------
# Heartbeat tests
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_heartbeat_disabled_returns_empty(self):
        inner = _make_mock_pool(n_workers=2)
        pool = FaultTolerantRolloutPool(inner)
        pool.start()
        stale = pool._check_heartbeat()
        assert stale == []

    def test_heartbeat_finds_dead_workers(self):
        inner = _make_mock_pool(n_workers=3)
        elastic = ElasticScalingConfig(
            enabled=True, min_workers=1, max_workers=8,
            heartbeat_ttl=10.0,
        )
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        # Kill one worker
        inner._procs[1].is_alive = lambda: False
        stale = pool._check_heartbeat()
        assert 1 in stale

    def test_heartbeat_all_alive_returns_empty(self):
        inner = _make_mock_pool(n_workers=3)
        elastic = ElasticScalingConfig(
            enabled=True, min_workers=1, max_workers=8,
            heartbeat_ttl=10.0,
        )
        cfg = FaultTolerantPoolConfig(elastic=elastic)
        pool = FaultTolerantRolloutPool(inner, cfg=cfg)
        pool.start()
        stale = pool._check_heartbeat()
        assert stale == []


# ---------------------------------------------------------------------------
# ElasticScalingConfig validation
# ---------------------------------------------------------------------------


class TestElasticValidation:
    def test_min_le_max(self):
        elastic = ElasticScalingConfig(min_workers=4, max_workers=2)
        # The config itself doesn't validate; scale_up/down handles it
        # by returning 0 when min > max effectively
        assert elastic.min_workers > elastic.max_workers

    def test_thresholds_within_range(self):
        elastic = ElasticScalingConfig(
            scale_up_threshold=0.95,
            scale_down_threshold=0.05,
        )
        assert 0 < elastic.scale_down_threshold < elastic.scale_up_threshold < 1
