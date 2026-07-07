"""Tests for client-server and quantization YAML config integration."""

from __future__ import annotations

import warnings

import pytest

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv, build_sim_tool_dataset
from hermes_agentic_rl.yaml_config import (
    build_client_server,
    build_trainer_config,
)


def _make_backend() -> TinyCausalLMBackend:
    return TinyCausalLMBackend(TinyBackendConfig(dim=16, n_layers=1, max_len=64))


class TestClientServerYAML:
    """Verify client-server config parsing and builder."""

    def test_section_skipped_in_trainer_config(self):
        """The ``client_server`` key should not be forwarded to GRPOTrainerConfig."""
        cfg = build_trainer_config({
            "n_iters": 5,
            "client_server": {"enabled": True, "sync_mode": "full"},
        })
        assert cfg.n_iters == 5

    def test_build_returns_none_when_disabled(self):
        """When client_server.enabled is False or missing, return None."""
        backend = _make_backend()
        assert build_client_server({}, backend) is None
        assert build_client_server({"client_server": {"enabled": False}}, backend) is None

    def test_build_returns_pair_when_enabled(self):
        """When client_server.enabled is True and backend has .model, return a pair."""
        backend = _make_backend()
        result = build_client_server(
            {"client_server": {"enabled": True, "sync_mode": "full"}},
            backend,
        )
        assert result is not None
        assert hasattr(result, "server")
        assert hasattr(result, "client")
        assert result.server.model is backend.model

    def test_build_returns_none_without_model(self):
        """When backend has no .model attribute, return None with a warning."""
        class NoModelBackend:
            pass
        result = build_client_server(
            {"client_server": {"enabled": True}},
            NoModelBackend(),
        )
        assert result is None

    def test_sync_mode_delta(self):
        """Delta sync mode should be accepted."""
        backend = _make_backend()
        result = build_client_server(
            {"client_server": {"enabled": True, "sync_mode": "delta"}},
            backend,
        )
        assert result is not None

    def test_sync_interval_forwarded(self):
        """sync_interval should be forwarded to the client."""
        backend = _make_backend()
        result = build_client_server(
            {"client_server": {
                "enabled": True,
                "sync_interval": 5.0,
            }},
            backend,
        )
        assert result is not None
        assert result.client._sync_interval == 5.0


class TestStalenessAndLoRAYAML:
    """Verify staleness-adaptive TIS and LoRA hot-reload are parsed from YAML."""

    def test_staleness_adaptive_tis_forwarded(self):
        """staleness_adaptive_tis should be forwarded to the trainer config."""
        cfg = build_trainer_config({
            "staleness_adaptive_tis": {
                "max_rho_clip": 2.0,
                "min_rho_clip": 1.0,
                "max_staleness": 5,
            },
        })
        assert cfg.staleness_adaptive_tis is not None
        assert cfg.staleness_adaptive_tis["max_rho_clip"] == 2.0

    def test_lora_hot_reload_forwarded(self):
        """lora_hot_reload should be forwarded to the trainer config."""
        cfg = build_trainer_config({
            "lora_hot_reload": {"rank": 4, "alpha": 8.0},
        })
        assert cfg.lora_hot_reload is not None
        assert cfg.lora_hot_reload["rank"] == 4

    def test_pipeline_rollouts_forwarded(self):
        """pipeline_rollouts should be forwarded."""
        cfg = build_trainer_config({"pipeline_rollouts": True})
        assert cfg.pipeline_rollouts is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
