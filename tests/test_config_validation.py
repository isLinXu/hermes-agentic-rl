"""Tests for config_validation.py — Pydantic + fallback validation."""

from __future__ import annotations

import pytest

from hermes_agentic_rl.config_validation import (
    is_pydantic_available,
    validate_config,
    validate_config_or_warn,
)

# ---------------------------------------------------------------------------
# Basic valid configs
# ---------------------------------------------------------------------------

class TestValidConfig:
    """Tests that valid configs pass validation."""

    def test_minimal_config(self):
        cfg = {"backend": "tiny", "n_iters": 5, "group_size": 4}
        result = validate_config(cfg)
        assert result is not None

    def test_full_config(self):
        cfg = {
            "backend": "hf",
            "model_name": "Qwen/Qwen2.5-1.5B",
            "n_iters": 100,
            "group_size": 8,
            "prompts_per_iter": 4,
            "lr": 1e-4,
            "max_new_tokens": 256,
            "temperature": 0.7,
            "update_epochs": 4,
            "grad_accum_steps": 2,
            "use_reference": True,
            "target_kl": 0.05,
            "adaptive_kl": True,
            "vllm_rollout_model": "Qwen/Qwen2.5-7B",
        }
        result = validate_config(cfg)
        assert result is not None

    def test_default_values_applied(self):
        if not is_pydantic_available():
            pytest.skip("Requires pydantic")
        from hermes_agentic_rl.config_validation import HermesConfig

        cfg = HermesConfig()  # type: ignore[abstract]
        assert cfg.backend == "tiny"
        assert cfg.n_iters == 20
        assert cfg.group_size == 4
        assert cfg.lr == 1e-3

    def test_extra_keys_allowed(self):
        """Unknown keys should not cause validation failure."""
        cfg = {"backend": "tiny", "n_iters": 5, "custom_field": 42}
        result = validate_config(cfg)
        assert result is not None


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------

class TestBackendValidation:
    def test_invalid_backend_raises(self):
        cfg = {"backend": "nonexistent", "n_iters": 5}
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_all_valid_backends(self):
        for backend in ("tiny", "hf", "vllm", "mock"):
            cfg = {"backend": backend, "n_iters": 1}
            result = validate_config(cfg)
            assert result is not None


# ---------------------------------------------------------------------------
# Numeric constraint validation
# ---------------------------------------------------------------------------

class TestNumericConstraints:
    def test_n_iters_must_be_positive(self):
        cfg = {"backend": "tiny", "n_iters": 0}
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_group_size_must_be_positive(self):
        cfg = {"backend": "tiny", "group_size": 0}
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_lr_must_be_positive(self):
        cfg = {"backend": "tiny", "lr": 0.0}
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_lr_negative_raises(self):
        cfg = {"backend": "tiny", "lr": -1e-3}
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_temperature_non_negative(self):
        cfg = {"backend": "tiny", "temperature": -0.1}
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_update_epochs_positive(self):
        cfg = {"backend": "tiny", "update_epochs": 0}
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# Composite block validation
# ---------------------------------------------------------------------------

class TestCompositeBlocks:
    def test_rewards_block_valid(self):
        cfg = {
            "backend": "tiny",
            "rewards": {
                "components": [
                    {"type": "ToolcallReward", "weight": 1.0},
                    {"type": "OutcomeReward", "weight": 2.0},
                ],
                "composer": {
                    "normalize": {"toolcall_reward": True},
                    "aggregator": "weighted_sum",
                    "parallel": True,
                },
            },
        }
        result = validate_config(cfg)
        assert result is not None

    def test_invalid_aggregator_raises(self):
        if not is_pydantic_available():
            pytest.skip("Requires pydantic")
        cfg = {
            "backend": "tiny",
            "rewards": {
                "components": [],
                "composer": {"aggregator": "bogus"},
            },
        }
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_ruler_block_valid(self):
        cfg = {
            "backend": "tiny",
            "ruler": {
                "rules": [
                    {"name": "exact", "template": "exact_match", "weight": 1.0},
                ],
                "default_weight": 2.0,
            },
        }
        result = validate_config(cfg)
        assert result is not None

    def test_curriculum_block_valid(self):
        cfg = {
            "backend": "tiny",
            "curriculum": {
                "auto_advance": True,
                "min_iters_per_stage": 10,
                "stages": [
                    {"name": "easy", "weight": 1.0, "mastery_threshold": 0.8},
                ],
            },
        }
        result = validate_config(cfg)
        assert result is not None


# ---------------------------------------------------------------------------
# Staleness TIS
# ---------------------------------------------------------------------------

class TestStalenessTIS:
    def test_valid_staleness_tis(self):
        cfg = {
            "backend": "tiny",
            "staleness_adaptive_tis": {
                "max_rho_clip": 1.5,
                "min_rho_clip": 0.5,
                "max_staleness": 5,
                "interpolation": "linear",
            },
        }
        result = validate_config(cfg)
        assert result is not None

    def test_invalid_interpolation_raises(self):
        if not is_pydantic_available():
            pytest.skip("Requires pydantic")
        cfg = {
            "backend": "tiny",
            "staleness_adaptive_tis": {"interpolation": "bogus"},
        }
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_min_gt_max_rho_clip_raises(self):
        if not is_pydantic_available():
            pytest.skip("Requires pydantic")
        cfg = {
            "backend": "tiny",
            "staleness_adaptive_tis": {
                "min_rho_clip": 2.0,
                "max_rho_clip": 1.0,
            },
        }
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# LoRA hot-reload
# ---------------------------------------------------------------------------

class TestLoRAHotReload:
    def test_lora_without_vllm_raises(self):
        cfg = {
            "backend": "tiny",
            "lora_hot_reload": {"rank": 8, "alpha": 16.0},
        }
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_lora_with_vllm_passes(self):
        cfg = {
            "backend": "vllm",
            "vllm_rollout_model": "Qwen/Qwen2.5-7B",
            "lora_hot_reload": {"rank": 8, "alpha": 16.0},
        }
        result = validate_config(cfg)
        assert result is not None

    def test_lora_rank_must_be_positive(self):
        if not is_pydantic_available():
            pytest.skip("Requires pydantic")
        cfg = {
            "backend": "vllm",
            "vllm_rollout_model": "model",
            "lora_hot_reload": {"rank": 0, "alpha": 16.0},
        }
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

class TestQuantization:
    def test_valid_quantization(self):
        cfg = {
            "backend": "vllm",
            "quantization": {
                "model_path": "/models/Qwen2.5-7B-GPTQ",
                "format": "gptq",
            },
        }
        result = validate_config(cfg)
        assert result is not None

    def test_invalid_format_raises(self):
        if not is_pydantic_available():
            pytest.skip("Requires pydantic")
        cfg = {
            "backend": "vllm",
            "quantization": {"model_path": "x", "format": "bogus"},
        }
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)

    def test_gpu_mem_utilization_range(self):
        if not is_pydantic_available():
            pytest.skip("Requires pydantic")
        cfg = {
            "backend": "vllm",
            "quantization": {"model_path": "x", "gpu_memory_utilization": 1.5},
        }
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# Client-Server
# ---------------------------------------------------------------------------

class TestClientServer:
    def test_valid_client_server(self):
        cfg = {
            "backend": "tiny",
            "client_server": {
                "enabled": True,
                "server_host": "0.0.0.0",
                "server_port": 8080,
            },
        }
        result = validate_config(cfg)
        assert result is not None

    def test_invalid_port_raises(self):
        if not is_pydantic_available():
            pytest.skip("Requires pydantic")
        cfg = {
            "backend": "tiny",
            "client_server": {"server_port": 99999},
        }
        with pytest.raises((ValueError, Exception)):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# Fallback / non-fatal path
# ---------------------------------------------------------------------------

class TestValidateOrWarn:
    def test_valid_config_returns_dict(self):
        cfg = {"backend": "tiny", "n_iters": 5}
        result = validate_config_or_warn(cfg)
        assert isinstance(result, dict)
        assert result["backend"] == "tiny"

    def test_invalid_config_returns_original(self):
        """On failure, validate_config_or_warn returns original dict."""
        cfg = {"backend": "nonexistent"}
        result = validate_config_or_warn(cfg)
        assert isinstance(result, dict)
        assert result["backend"] == "nonexistent"

    def test_pydantic_available_flag(self):
        # Should not raise regardless of pydantic availability.
        flag = is_pydantic_available()
        assert isinstance(flag, bool)
