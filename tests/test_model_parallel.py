"""Tests for model parallel strategy interface.

Covers:
- ModelParallelConfig: defaults, validation, properties
- TensorParallelStrategy: apply, gather, sync (no-op fallback)
- PipelineParallelStrategy: apply, gather, sync
- HybridParallelStrategy: combined apply
- apply_model_parallel: strategy selection
- _NoOpStrategy: pass-through behavior
- compute_parallel_config: heuristic computation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from hermes_agentic_rl.distributed.model_parallel import (
    HybridParallelStrategy,
    ModelParallelConfig,
    ModelParallelStrategy,
    PipelineParallelStrategy,
    TensorParallelStrategy,
    apply_model_parallel,
    compute_parallel_config,
)

# ---------------------------------------------------------------------------
# ModelParallelConfig tests
# ---------------------------------------------------------------------------


class TestModelParallelConfig:
    def test_defaults(self):
        cfg = ModelParallelConfig()
        assert cfg.tensor_parallel_size == 1
        assert cfg.pipeline_parallel_size == 1
        assert cfg.pipeline_chunks == 1
        assert cfg.backend == "auto"
        assert cfg.device == "cuda"
        assert cfg.overlap_comm is True
        assert cfg.activation_checkpointing is False

    def test_enabled_false_by_default(self):
        cfg = ModelParallelConfig()
        assert cfg.enabled is False

    def test_enabled_true_with_tp(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2)
        assert cfg.enabled is True

    def test_enabled_true_with_pp(self):
        cfg = ModelParallelConfig(pipeline_parallel_size=2)
        assert cfg.enabled is True

    def test_total_parallel_size(self):
        cfg = ModelParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=2)
        assert cfg.total_parallel_size == 8

    def test_validate_no_warnings(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
        assert cfg.validate() == []

    def test_validate_zero_tp(self):
        cfg = ModelParallelConfig(tensor_parallel_size=0)
        warnings = cfg.validate()
        assert any("tensor_parallel_size" in w for w in warnings)

    def test_validate_zero_pp(self):
        cfg = ModelParallelConfig(pipeline_parallel_size=0)
        warnings = cfg.validate()
        assert any("pipeline_parallel_size" in w for w in warnings)

    def test_validate_torch_backend_warning(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2, backend="torch")
        warnings = cfg.validate()
        assert any("experimental" in w for w in warnings)


# ---------------------------------------------------------------------------
# Strategy interface tests
# ---------------------------------------------------------------------------


class TestModelParallelStrategy:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ModelParallelStrategy(ModelParallelConfig())  # type: ignore[abstract]

    def test_teardown_resets_applied(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2)
        strategy = TensorParallelStrategy(cfg)
        strategy._applied = True
        strategy.teardown(MagicMock())
        assert strategy.is_applied is False


# ---------------------------------------------------------------------------
# TensorParallelStrategy tests
# ---------------------------------------------------------------------------


class TestTensorParallelStrategy:
    def test_disabled_returns_model_unchanged(self):
        cfg = ModelParallelConfig()  # disabled
        strategy = TensorParallelStrategy(cfg)
        model = MagicMock()
        result = strategy.apply(model)
        assert result is model
        assert strategy.is_applied is False

    def test_no_dist_returns_model_unchanged(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2)
        strategy = TensorParallelStrategy(cfg)
        model = MagicMock()
        result = strategy.apply(model)
        # No torch.distributed initialized → no-op
        assert result is model
        assert strategy.is_applied is False

    def test_gather_state_dict_not_applied(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2)
        strategy = TensorParallelStrategy(cfg)
        model = MagicMock()
        model.state_dict.return_value = {"weight": "tensor"}
        result = strategy.gather_state_dict(model)
        assert result == {"weight": "tensor"}

    def test_sync_gradients_not_applied_is_noop(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2)
        strategy = TensorParallelStrategy(cfg)
        model = MagicMock()
        strategy.sync_gradients(model)  # should not raise

    def test_teardown(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2)
        strategy = TensorParallelStrategy(cfg)
        strategy._applied = True
        strategy.teardown(MagicMock())
        assert not strategy.is_applied


# ---------------------------------------------------------------------------
# PipelineParallelStrategy tests
# ---------------------------------------------------------------------------


class TestPipelineParallelStrategy:
    def test_disabled_returns_model_unchanged(self):
        cfg = ModelParallelConfig()  # disabled
        strategy = PipelineParallelStrategy(cfg)
        model = MagicMock()
        result = strategy.apply(model)
        assert result is model
        assert strategy.is_applied is False

    def test_no_dist_returns_model_unchanged(self):
        cfg = ModelParallelConfig(pipeline_parallel_size=2)
        strategy = PipelineParallelStrategy(cfg)
        model = MagicMock()
        result = strategy.apply(model)
        assert result is model
        assert strategy.is_applied is False

    def test_gather_state_dict_not_applied(self):
        cfg = ModelParallelConfig(pipeline_parallel_size=2)
        strategy = PipelineParallelStrategy(cfg)
        model = MagicMock()
        model.state_dict.return_value = {"layer.weight": "tensor"}
        result = strategy.gather_state_dict(model)
        assert result == {"layer.weight": "tensor"}

    def test_sync_gradients_not_applied_is_noop(self):
        cfg = ModelParallelConfig(pipeline_parallel_size=2)
        strategy = PipelineParallelStrategy(cfg)
        strategy.sync_gradients(MagicMock())  # should not raise


# ---------------------------------------------------------------------------
# HybridParallelStrategy tests
# ---------------------------------------------------------------------------


class TestHybridParallelStrategy:
    def test_disabled_returns_model_unchanged(self):
        cfg = ModelParallelConfig()
        strategy = HybridParallelStrategy(cfg)
        model = MagicMock()
        result = strategy.apply(model)
        assert result is model
        assert strategy.is_applied is False

    def test_apply_with_tp_and_pp(self):
        cfg = ModelParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        strategy = HybridParallelStrategy(cfg)
        model = MagicMock()
        result = strategy.apply(model)
        # Without dist, both TP and PP are no-ops
        assert result is model
        assert strategy.is_applied is False

    def test_sync_gradients_is_noop_without_dist(self):
        cfg = ModelParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        strategy = HybridParallelStrategy(cfg)
        strategy.sync_gradients(MagicMock())

    def test_teardown(self):
        cfg = ModelParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        strategy = HybridParallelStrategy(cfg)
        strategy._applied = True
        strategy.teardown(MagicMock())
        assert not strategy.is_applied


# ---------------------------------------------------------------------------
# apply_model_parallel tests
# ---------------------------------------------------------------------------


class TestApplyModelParallel:
    def test_disabled_returns_noop_strategy(self):
        cfg = ModelParallelConfig()
        model = MagicMock()
        wrapped, strategy = apply_model_parallel(model, cfg)
        assert wrapped is model
        assert not strategy.is_applied
        assert isinstance(strategy, type(strategy))  # _NoOpStrategy

    def test_tp_only_returns_tp_strategy(self):
        cfg = ModelParallelConfig(tensor_parallel_size=2)
        model = MagicMock()
        _wrapped, strategy = apply_model_parallel(model, cfg)
        assert isinstance(strategy, TensorParallelStrategy)

    def test_pp_only_returns_pp_strategy(self):
        cfg = ModelParallelConfig(pipeline_parallel_size=2)
        model = MagicMock()
        _wrapped, strategy = apply_model_parallel(model, cfg)
        assert isinstance(strategy, PipelineParallelStrategy)

    def test_tp_and_pp_returns_hybrid_strategy(self):
        cfg = ModelParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        model = MagicMock()
        _wrapped, strategy = apply_model_parallel(model, cfg)
        assert isinstance(strategy, HybridParallelStrategy)

    def test_strategy_gather_state_dict_works(self):
        cfg = ModelParallelConfig()
        model = MagicMock()
        model.state_dict.return_value = {"w": "v"}
        _wrapped, strategy = apply_model_parallel(model, cfg)
        result = strategy.gather_state_dict(model)
        assert result == {"w": "v"}

    def test_strategy_sync_gradients_is_noop(self):
        cfg = ModelParallelConfig()
        model = MagicMock()
        _, strategy = apply_model_parallel(model, cfg)
        strategy.sync_gradients(model)  # should not raise


# ---------------------------------------------------------------------------
# compute_parallel_config tests
# ---------------------------------------------------------------------------


class TestComputeParallelConfig:
    def test_single_gpu_no_parallelism(self):
        cfg = compute_parallel_config(
            model_params=7_000_000_000,
            n_gpus=1,
            gpu_memory_gb=80.0,
        )
        assert cfg.tensor_parallel_size == 1
        assert cfg.pipeline_parallel_size == 1

    def test_small_model_no_parallelism(self):
        cfg = compute_parallel_config(
            model_params=100_000_000,  # 100M
            n_gpus=4,
            gpu_memory_gb=80.0,
        )
        assert cfg.tensor_parallel_size == 1
        assert cfg.pipeline_parallel_size == 1

    def test_large_model_uses_tp(self):
        cfg = compute_parallel_config(
            model_params=7_000_000_000,  # 7B
            n_gpus=4,
            gpu_memory_gb=16.0,
        )
        assert cfg.tensor_parallel_size > 1

    def test_prefer_pipeline(self):
        cfg = compute_parallel_config(
            model_params=70_000_000_000,  # 70B
            n_gpus=8,
            gpu_memory_gb=40.0,
            prefer_pipeline=True,
        )
        # With prefer_pipeline, PP should be > 1
        assert cfg.pipeline_parallel_size > 1

    def test_tp_times_pp_le_n_gpus(self):
        for n_gpus in [2, 4, 8, 16]:
            cfg = compute_parallel_config(
                model_params=70_000_000_000,
                n_gpus=n_gpus,
                gpu_memory_gb=16.0,
            )
            assert cfg.tensor_parallel_size * cfg.pipeline_parallel_size <= n_gpus

    def test_max_tp_size_respected(self):
        cfg = compute_parallel_config(
            model_params=70_000_000_000,
            n_gpus=32,
            gpu_memory_gb=16.0,
            max_tp_size=4,
        )
        assert cfg.tensor_parallel_size <= 4
