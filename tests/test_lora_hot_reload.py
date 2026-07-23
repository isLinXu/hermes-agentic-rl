"""LoRA hot-reload manager tests: shadow merge, vLLM sync, save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.peft import LoRAConfig, LoRAHotReloadManager, inject_lora


class _MockVLLMBackend:
    """Minimal mock capturing sync_weights_from calls."""

    def __init__(self) -> None:
        self.synced: list[dict[str, Any]] = []
        self.sync_count = 0

    def sync_weights_from(self, state_dict: dict[str, Any]) -> None:
        self.synced.append({k: v.clone() for k, v in state_dict.items()})
        self.sync_count += 1


def _make_model_and_adapter():
    backend = TinyCausalLMBackend(TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2))
    adapter = inject_lora(
        backend.model,
        LoRAConfig(r=4, alpha=8, target_patterns=("qkv", "proj", "head")),
    )
    # Give lora_B some non-zero values so the delta is meaningful.
    for m in adapter.modules.values():
        m.lora_B.data.uniform_(-0.1, 0.1)
    return backend, adapter


def test_hot_reload_initializes_shadow_lazily():
    backend, adapter = _make_model_and_adapter()
    mock = _MockVLLMBackend()
    manager = LoRAHotReloadManager(adapter, backend.model, mock)
    assert not manager._shadow_initialized
    manager.sync_to_vllm()
    assert manager._shadow_initialized
    assert mock.sync_count == 1


def test_hot_reload_sync_pushes_merged_weights():
    backend, adapter = _make_model_and_adapter()
    mock = _MockVLLMBackend()
    manager = LoRAHotReloadManager(adapter, backend.model, mock)

    manager.sync_to_vllm()
    assert mock.sync_count == 1
    assert len(mock.synced[0]) > 0
    # All pushed tensors should be CPU.
    for v in mock.synced[0].values():
        assert v.device.type == "cpu"


def test_hot_reload_sync_every_throttle():
    backend, adapter = _make_model_and_adapter()
    mock = _MockVLLMBackend()
    manager = LoRAHotReloadManager(adapter, backend.model, mock, sync_every=3)

    for i in range(5):
        manager.sync_to_vllm()
    assert mock.sync_count == 1  # only at counter 3

    manager.sync_to_vllm()
    assert mock.sync_count == 2  # at counter 6


def test_hot_reload_no_vllm_returns_false():
    backend, adapter = _make_model_and_adapter()
    manager = LoRAHotReloadManager(adapter, backend.model, vllm_backend=None)
    assert manager.sync_to_vllm() is False


def test_hot_reload_merged_weights_differ_from_base():
    """When LoRA B is non-zero, merged weights should differ from base."""
    backend, adapter = _make_model_and_adapter()
    mock = _MockVLLMBackend()
    manager = LoRAHotReloadManager(adapter, backend.model, mock)
    manager.sync_to_vllm()

    base_state = backend.model.state_dict()
    merged = mock.synced[0]

    # At least one tensor should differ because LoRA B is non-zero.
    diffs = []
    for key in base_state:
        if key in merged:
            diffs.append(not torch.allclose(base_state[key].cpu(), merged[key], atol=1e-7))
    assert any(diffs), "Merged weights should differ from base when LoRA delta is non-zero"


def test_hot_reload_save_load_lora(tmp_path: Path):
    backend, adapter = _make_model_and_adapter()
    manager = LoRAHotReloadManager(adapter, backend.model)
    path = tmp_path / "adapter.lora"
    manager.save_lora(path)
    assert path.exists()

    # Zero out lora params, then reload.
    for m in adapter.modules.values():
        m.lora_A.data.zero_()
        m.lora_B.data.zero_()
    manager.load_lora(path)

    # Verify restored.
    for m in adapter.modules.values():
        assert not torch.allclose(m.lora_B.data, torch.zeros_like(m.lora_B.data))


def test_hot_reload_save_full(tmp_path: Path):
    backend, adapter = _make_model_and_adapter()
    manager = LoRAHotReloadManager(adapter, backend.model)
    path = tmp_path / "merged_full.pt"
    manager.save_full(path)
    assert path.exists()
    # Should be loadable as a state dict.
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict)
    assert len(loaded) > 0


def test_hot_reload_lora_param_ratio():
    backend, adapter = _make_model_and_adapter()
    manager = LoRAHotReloadManager(adapter, backend.model)
    ratio = manager.lora_param_ratio()
    assert 0.0 < ratio < 1.0
    assert manager.num_lora_params > 0


def test_hot_reload_sync_updates_when_lora_changes():
    """Second sync after updating LoRA weights should produce different merged output."""
    backend, adapter = _make_model_and_adapter()
    mock = _MockVLLMBackend()
    manager = LoRAHotReloadManager(adapter, backend.model, mock)

    manager.sync_to_vllm()
    first_merge = {k: v.clone() for k, v in mock.synced[0].items()}

    # Modify LoRA weights.
    for m in adapter.modules.values():
        m.lora_B.data.uniform_(-0.2, 0.2)

    manager.sync_to_vllm()
    second_merge = mock.synced[1]

    # At least one tensor should be different.
    diffs = []
    for key in first_merge:
        if key in second_merge:
            diffs.append(not torch.allclose(first_merge[key], second_merge[key], atol=1e-7))
    assert any(diffs), "Second sync should reflect updated LoRA weights"
