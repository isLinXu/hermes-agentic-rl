"""Integration tests for staleness-adaptive TIS and LoRA hot-reload wiring.

These tests verify that the new config fields on OnPolicyTrainerConfig /
GRPOTrainerConfig correctly flow into the trainer's internal controllers,
and that the controllers interact with the training loop as expected.
"""

from __future__ import annotations

import warnings

import pytest

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.envs.sim_tool_env import SimToolEnv
from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig


def _make_tiny_backend() -> TinyCausalLMBackend:
    return TinyCausalLMBackend(TinyBackendConfig(dim=16, n_layers=1, max_len=64))


def _make_env() -> SimToolEnv:
    from hermes_agentic_rl.envs.sim_tool_env import build_sim_tool_dataset

    return SimToolEnv(build_sim_tool_dataset(n=4, seed=0))


class TestStalenessAdaptiveTISConfig:
    """Verify the staleness_adaptive_tis config flows through to the trainer."""

    def test_config_field_exists_on_grpo(self):
        """GRPOTrainerConfig should accept the new field."""
        cfg = GRPOTrainerConfig(
            staleness_adaptive_tis={
                "max_rho_clip": 2.0,
                "min_rho_clip": 1.0,
                "max_staleness": 5,
                "interpolation": "linear",
            },
        )
        assert cfg.staleness_adaptive_tis is not None
        assert cfg.staleness_adaptive_tis["max_rho_clip"] == 2.0

    def test_config_field_exists_on_shared(self):
        """OnPolicyTrainerConfig should have the field."""
        from hermes_agentic_rl.trainers.on_policy_config import (
            OnPolicyTrainerConfig,
        )

        cfg = OnPolicyTrainerConfig(
            staleness_adaptive_tis={"max_rho_clip": 1.5, "min_rho_clip": 0.5},
        )
        assert cfg.staleness_adaptive_tis is not None

    def test_warning_without_pipeline_or_replay(self):
        """Config validation should warn when staleness_adaptive_tis is set
        without pipeline_rollouts or replay_buffer."""
        from hermes_agentic_rl.trainers.on_policy_config import (
            OnPolicyTrainerConfig,
            validate_on_policy_config,
        )

        cfg = OnPolicyTrainerConfig(staleness_adaptive_tis={"max_rho_clip": 2.0})
        warns = validate_on_policy_config(cfg)
        assert any("staleness_adaptive_tis" in w for w in warns)


class TestLoRAHotReloadConfig:
    """Verify the lora_hot_reload config flows through to the trainer."""

    def test_config_field_exists_on_grpo(self):
        """GRPOTrainerConfig should accept the new field."""
        cfg = GRPOTrainerConfig(
            lora_hot_reload={"rank": 4, "alpha": 8.0},
        )
        assert cfg.lora_hot_reload is not None
        assert cfg.lora_hot_reload["rank"] == 4

    def test_warning_without_vllm(self):
        """Config validation should warn when lora_hot_reload is set
        without vllm_rollout_model."""
        from hermes_agentic_rl.trainers.on_policy_config import (
            OnPolicyTrainerConfig,
            validate_on_policy_config,
        )

        cfg = OnPolicyTrainerConfig(lora_hot_reload={"rank": 4})
        warns = validate_on_policy_config(cfg)
        assert any("lora_hot_reload" in w for w in warns)


class TestStalenessAdaptiveTISWiring:
    """Verify the StalenessAdaptiveTIS controller is created in the trainer."""

    def test_controller_created_when_configured(self):
        """When staleness_adaptive_tis is in the config, the trainer should
        have a _staleness_tis controller."""
        backend = _make_tiny_backend()
        env = _make_env()
        from hermes_agentic_rl.core.reward_manager import RewardManager

        rm = RewardManager(rewards=[])
        cfg = GRPOTrainerConfig(
            n_iters=1,
            group_size=2,
            prompts_per_iter=1,
            staleness_adaptive_tis={
                "max_rho_clip": 2.0,
                "min_rho_clip": 1.0,
                "max_staleness": 10,
                "enabled": True,
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        assert trainer._staleness_tis is not None
        assert trainer._staleness_tis.enabled is True
        assert trainer._staleness_tis.schedule.max_rho_clip == 2.0

    def test_controller_none_when_not_configured(self):
        """Without the config, _staleness_tis should be None."""
        backend = _make_tiny_backend()
        env = _make_env()
        from hermes_agentic_rl.core.reward_manager import RewardManager

        rm = RewardManager(rewards=[])
        cfg = GRPOTrainerConfig(n_iters=1, group_size=2, prompts_per_iter=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        assert trainer._staleness_tis is None

    def test_controller_observes_staleness_during_training(self):
        """After a training iteration, the controller should have observed
        at least one staleness value (0 for synchronous)."""
        backend = _make_tiny_backend()
        env = _make_env()
        from hermes_agentic_rl.core.reward_manager import RewardManager

        rm = RewardManager(rewards=[])
        cfg = GRPOTrainerConfig(
            n_iters=2,
            group_size=2,
            prompts_per_iter=1,
            max_new_tokens=4,
            staleness_adaptive_tis={
                "max_rho_clip": 2.0,
                "min_rho_clip": 1.0,
                "max_staleness": 10,
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        trainer.train()
        assert trainer._staleness_tis.step_count >= 2
        assert trainer._staleness_tis.mean_staleness == 0.0  # synchronous


class TestLoRAHotReloadWiring:
    """Verify LoRAHotReloadManager is NOT created without vLLM backend
    (it requires vLLM to function)."""

    def test_manager_none_without_vllm(self):
        """Without vllm_rollout_model, _lora_hot_reload should be None
        even if the config is set."""
        backend = _make_tiny_backend()
        env = _make_env()
        from hermes_agentic_rl.core.reward_manager import RewardManager

        rm = RewardManager(rewards=[])
        cfg = GRPOTrainerConfig(
            n_iters=1,
            group_size=2,
            prompts_per_iter=1,
            lora_hot_reload={"rank": 4, "alpha": 8.0},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = GRPOTrainer(policy=backend, env=env, reward_manager=rm, cfg=cfg)
        assert trainer._lora_hot_reload is None  # no vLLM → no manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
