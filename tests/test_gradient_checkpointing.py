from __future__ import annotations

import pytest

pytest.importorskip("torch")

from hermes_agentic_rl.backends.hf import HFCausalLMBackend


class _FakeConfig:
    use_cache = True


class _FakeHFModel:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.enabled = False
        self.input_require_grads_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.enabled = True

    def gradient_checkpointing_disable(self) -> None:
        self.enabled = False

    def enable_input_require_grads(self) -> None:
        self.input_require_grads_enabled = True


def test_hf_gradient_checkpointing_toggles_use_cache() -> None:
    backend = HFCausalLMBackend.__new__(HFCausalLMBackend)
    backend.model = _FakeHFModel()
    backend._gradient_checkpointing_enabled = False
    backend._gradient_checkpointing_prev_use_cache = None

    assert backend.set_gradient_checkpointing(True) is True
    assert backend.model.enabled is True
    assert backend.model.config.use_cache is False
    assert backend.model.input_require_grads_enabled is True

    assert backend.set_gradient_checkpointing(False) is True
    assert backend.model.enabled is False
    assert backend.model.config.use_cache is True
    assert backend._gradient_checkpointing_prev_use_cache is None
