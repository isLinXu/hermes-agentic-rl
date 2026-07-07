"""Tests for Optuna hyperparameter search integration."""

from __future__ import annotations

import pytest

from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainerConfig

# Skip all tests if Optuna is not installed.
optuna = pytest.importorskip("optuna")

from hermes_agentic_rl.tuning.hparam_search import (
    HyperparameterSearch,
    SearchDimension,
    SearchSpace,
    build_search_from_config,
)


class TestSearchDimension:
    """Test SearchDimension construction and validation."""

    def test_float_dimension(self):
        d = SearchDimension(name="lr", type="float", low=1e-5, high=1e-2, log=True)
        assert d.name == "lr"
        assert d.type == "float"
        assert d.log is True

    def test_int_dimension(self):
        d = SearchDimension(name="group_size", type="int", low=4, high=16)
        assert d.type == "int"
        assert d.low == 4
        assert d.high == 16

    def test_categorical_dimension(self):
        d = SearchDimension(
            name="advantage_norm",
            type="categorical",
            choices=["batch", "group"],
        )
        assert d.choices == ["batch", "group"]

    def test_float_requires_low_high(self):
        with pytest.raises(ValueError, match="requires low and high"):
            SearchDimension(name="lr", type="float", low=None, high=None)

    def test_categorical_requires_choices(self):
        with pytest.raises(ValueError, match="requires non-empty choices"):
            SearchDimension(name="x", type="categorical", choices=[])


class TestSearchSpace:
    """Test SearchSpace construction and sampling."""

    def test_from_config(self):
        config = {
            "dimensions": [
                {"name": "lr", "type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                {"name": "group_size", "type": "int", "low": 4, "high": 8},
                {"name": "advantage_norm", "type": "categorical", "choices": ["batch", "group"]},
            ],
        }
        space = SearchSpace.from_config(config)
        assert len(space.dimensions) == 3
        assert space.dimensions[0].name == "lr"
        assert space.dimensions[1].name == "group_size"
        assert space.dimensions[2].name == "advantage_norm"

    def test_sample_all(self):
        space = SearchSpace(
            dimensions=[
                SearchDimension(name="clip_eps", type="float", low=0.1, high=0.3),
                SearchDimension(name="group_size", type="int", low=4, high=8),
            ],
        )
        # Use a mock trial.
        class MockTrial:
            def suggest_float(self, name, low, high, log=False, step=None):
                return (low + high) / 2

            def suggest_int(self, name, low, high, log=False, step=None):
                return (low + high) // 2

            def suggest_categorical(self, name, choices):
                return choices[0]

        sampled = space.sample_all(MockTrial())
        assert "clip_eps" in sampled
        assert "group_size" in sampled
        assert sampled["clip_eps"] == pytest.approx(0.2)
        assert sampled["group_size"] == 6


class TestHyperparameterSearch:
    """Test the HyperparameterSearch class."""

    def test_init_requires_optuna(self):
        """If Optuna is available (we skipped otherwise), init should work."""
        space = SearchSpace(
            dimensions=[
                SearchDimension(name="lr", type="float", low=1e-5, high=1e-3, log=True),
            ],
        )
        base_cfg = GRPOTrainerConfig(n_iters=1, group_size=2, prompts_per_iter=1)
        search = HyperparameterSearch(
            space=space,
            base_config=base_cfg,
            n_trials=2,
            trial_iters=1,
        )
        assert search.n_trials == 2
        assert search.metric_key == "reward_mean"

    def test_run_search(self):
        """Run a tiny search and verify results."""
        space = SearchSpace(
            dimensions=[
                SearchDimension(name="lr", type="float", low=1e-4, high=1e-3, log=True),
                SearchDimension(name="clip_eps", type="float", low=0.1, high=0.3),
            ],
        )
        base_cfg = GRPOTrainerConfig(
            n_iters=1, group_size=2, prompts_per_iter=1, max_new_tokens=4,
        )
        search = HyperparameterSearch(
            space=space,
            base_config=base_cfg,
            n_trials=3,
            trial_iters=1,
            metric_key="reward_mean",
        )
        best_params, best_value = search.run()
        assert "lr" in best_params
        assert "clip_eps" in best_params
        assert isinstance(best_value, float)

    def test_summary_before_run(self):
        """summary() should return not_run status before run()."""
        space = SearchSpace(
            dimensions=[
                SearchDimension(name="lr", type="float", low=1e-4, high=1e-3),
            ],
        )
        base_cfg = GRPOTrainerConfig(n_iters=1, group_size=2, prompts_per_iter=1)
        search = HyperparameterSearch(
            space=space, base_config=base_cfg, n_trials=1,
        )
        result = search.summary()
        assert result["status"] == "not_run"

    def test_summary_after_run(self):
        """summary() should return results after run()."""
        space = SearchSpace(
            dimensions=[
                SearchDimension(name="lr", type="float", low=1e-4, high=1e-3),
            ],
        )
        base_cfg = GRPOTrainerConfig(
            n_iters=1, group_size=2, prompts_per_iter=1, max_new_tokens=4,
        )
        search = HyperparameterSearch(
            space=space, base_config=base_cfg, n_trials=2, trial_iters=1,
        )
        search.run()
        result = search.summary()
        assert result["n_trials"] == 2
        assert "best_value" in result
        assert "best_params" in result


class TestFactoryFunction:
    """Test build_search_from_config."""

    def test_returns_none_without_config(self):
        base_cfg = GRPOTrainerConfig(n_iters=1)
        assert build_search_from_config({}, base_cfg) is None

    def test_returns_none_without_dimensions(self):
        base_cfg = GRPOTrainerConfig(n_iters=1)
        result = build_search_from_config(
            {"hparam_search": {"n_trials": 10}},
            base_cfg,
        )
        assert result is None  # no dimensions

    def test_builds_search_with_dimensions(self):
        base_cfg = GRPOTrainerConfig(n_iters=1)
        config = {
            "hparam_search": {
                "n_trials": 5,
                "metric_key": "reward_mean",
                "direction": "maximize",
                "dimensions": [
                    {"name": "lr", "type": "float", "low": 1e-5, "high": 1e-3, "log": True},
                    {"name": "group_size", "type": "int", "low": 4, "high": 8},
                ],
            },
        }
        search = build_search_from_config(config, base_cfg)
        assert search is not None
        assert search.n_trials == 5
        assert len(search.space.dimensions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
