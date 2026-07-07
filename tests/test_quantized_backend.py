"""Tests for the quantized rollout backend.

Tests focus on:
  1. Quantization format detection (no external dependencies needed).
  2. Config construction and validation.
  3. Factory function behavior.
  4. GGUF weight-sync rejection.
"""

from __future__ import annotations

import pytest

from hermes_agentic_rl.backends.quantized import (
    QuantFormat,
    QuantizedRolloutConfig,
    build_quantized_backend_from_config,
    detect_quant_format,
)


class TestDetectQuantFormat:
    """Test the quantization format detection logic."""

    def test_gguf_extension(self):
        assert detect_quant_format("/models/qwen.gguf") == QuantFormat.GGUF
        assert detect_quant_format("model.Q4_K_M.gguf") == QuantFormat.GGUF

    def test_gptq_pattern(self):
        assert detect_quant_format("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4") == QuantFormat.GPTQ
        assert detect_quant_format("Qwen/Qwen2.5-7B-gptq") == QuantFormat.GPTQ

    def test_awq_pattern(self):
        assert detect_quant_format("Qwen/Qwen2.5-7B-Instruct-AWQ") == QuantFormat.AWQ

    def test_none_for_plain_model(self):
        assert detect_quant_format("Qwen/Qwen2.5-7B-Instruct") == QuantFormat.NONE
        assert detect_quant_format("/models/base") == QuantFormat.NONE


class TestQuantizedRolloutConfig:
    """Test the config dataclass."""

    def test_auto_detect_from_path(self):
        cfg = QuantizedRolloutConfig(model_path="/models/test.gguf")
        assert cfg.quant_format == QuantFormat.GGUF

    def test_explicit_format_string(self):
        cfg = QuantizedRolloutConfig(
            model_path="/models/base",
            quant_format="gptq",
        )
        assert cfg.quant_format == QuantFormat.GPTQ

    def test_explicit_format_enum(self):
        cfg = QuantizedRolloutConfig(
            model_path="/models/base",
            quant_format=QuantFormat.AWQ,
        )
        assert cfg.quant_format == QuantFormat.AWQ

    def test_defaults(self):
        cfg = QuantizedRolloutConfig(model_path="/models/test.gguf")
        assert cfg.tensor_parallel_size == 1
        assert cfg.max_model_len == 4096
        assert cfg.gpu_memory_utilization == 0.90
        assert cfg.n_gpu_layers == -1


class TestFactoryFunction:
    """Test build_quantized_backend_from_config."""

    def test_returns_none_without_config(self):
        assert build_quantized_backend_from_config({}) is None

    def test_returns_none_without_model_path(self):
        assert build_quantized_backend_from_config({"quantization": {}}) is None

    def test_returns_none_with_empty_model_path(self):
        assert build_quantized_backend_from_config(
            {"quantization": {"model_path": ""}}
        ) is None

    def test_config_parsed_correctly(self):
        """The factory should parse the config dict correctly.
        Note: we don't build the actual backend (no vllm/llama-cpp installed),
        just verify the config parsing path doesn't error before the import."""
        config = {
            "quantization": {
                "model_path": "/models/test.gguf",
                "format": "gguf",
                "max_model_len": 2048,
                "n_gpu_layers": 0,
            }
        }
        # We expect a BackendUnavailableError since llama-cpp is not installed,
        # but the config parsing itself should work.
        from hermes_agentic_rl.backends.base import BackendUnavailableError
        with pytest.raises(BackendUnavailableError):
            build_quantized_backend_from_config(config)


class TestGGUFWeightSyncRejection:
    """Verify that GGUF backend correctly rejects weight sync."""

    def test_gguf_raises_on_sync(self):
        """GGUF backend should raise NotImplementedError on sync_weights_from."""
        # We can't construct the full backend without llama-cpp,
        # but we can test the logic by creating a subclass that skips __init__.
        from hermes_agentic_rl.backends.quantized import QuantizedRolloutBackend

        class TestGGUFBackend(QuantizedRolloutBackend):
            """Test subclass that skips engine initialization."""
            def __init__(self):  # type: ignore[no-untyped-def]
                self._supports_weight_sync = False
                self._format = QuantFormat.GGUF

        backend = TestGGUFBackend()
        with pytest.raises(NotImplementedError, match=r"GGUF.*frozen"):
            backend.sync_weights_from({})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
