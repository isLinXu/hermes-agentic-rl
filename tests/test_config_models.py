"""Tests for Pydantic-backed config validation."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from hermes_agentic_rl.config import ConfigValidationError, validate_config
from hermes_agentic_rl.config_models import HermesRLConfigModel, validate_config_model


def test_pydantic_accepts_minimal_config() -> None:
    model = validate_config_model(
        {
            "runtime": {"integration": "fake", "max_agent_turns": 5},
            "environment": {"type": "echo"},
            "reward": {"aggregator": "weighted_sum"},
        }
    )
    assert model.runtime.integration == "fake"
    assert model.runtime.max_agent_turns == 5


def test_pydantic_accepts_hermes_integration() -> None:
    model = validate_config_model(
        {
            "runtime": {"integration": "hermes", "max_agent_turns": 10},
        }
    )
    assert model.runtime.integration == "hermes"


def test_pydantic_rejects_invalid_integration() -> None:
    with pytest.raises(pydantic.ValidationError):
        validate_config_model({"runtime": {"integration": "ghost", "max_agent_turns": 1}})


def test_pydantic_rejects_nonpositive_max_turns() -> None:
    with pytest.raises(pydantic.ValidationError):
        validate_config_model({"runtime": {"integration": "fake", "max_agent_turns": 0}})


def test_validate_config_uses_pydantic_when_available() -> None:
    with pytest.raises(ConfigValidationError, match="integration"):
        validate_config({"runtime": {"integration": "ghost", "max_agent_turns": 1}})


def test_validate_config_dict_fallback_without_pydantic(monkeypatch) -> None:
    import hermes_agentic_rl.config as config_mod

    monkeypatch.setattr(config_mod, "_PYDANTIC_AVAILABLE", False)
    with pytest.raises(ConfigValidationError):
        config_mod.validate_config(
            {"runtime": {"integration": "ghost", "max_agent_turns": 1}},
            use_pydantic=False,
        )


def test_hermes_rl_config_allows_extra_sections() -> None:
    model = HermesRLConfigModel.model_validate(
        {
            "runtime": {"integration": "fake"},
            "train_rl": {"n_iters": 10, "group_size": 4},
            "backend": {"name": "tiny"},
        }
    )
    assert model.train_rl["n_iters"] == 10
    assert model.backend["name"] == "tiny"
