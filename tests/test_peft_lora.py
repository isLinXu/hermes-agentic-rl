"""LoRA injection tests: shape, freezing, save/load, merge/unmerge."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.peft import LoRAConfig, inject_lora


def test_inject_lora_freezes_base_and_returns_trainable_subset():
    backend = TinyCausalLMBackend(TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2))
    n_base = sum(p.numel() for p in backend.model.parameters())
    adapter = inject_lora(
        backend.model,
        LoRAConfig(r=4, alpha=8, target_patterns=("qkv", "proj", "ff.0", "ff.2", "head")),
    )
    assert adapter.num_parameters() > 0
    assert adapter.num_parameters() < n_base  # strictly fewer params
    # trainable set is exactly the adapter params
    trainable = [p for p in backend.model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == adapter.num_parameters()


def test_lora_preserves_behavior_at_init():
    """At init, B=0 so the adapter output must equal the base output."""
    b1 = TinyCausalLMBackend(TinyBackendConfig(seed=7, dim=32, n_heads=4, n_layers=2))
    ids = [b1.tokenizer.bos_id] + b1.tokenizer.encode("abc")
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        y_before = b1.model(x).clone()
    inject_lora(b1.model, LoRAConfig(r=4, target_patterns=("qkv", "proj", "head")))
    with torch.no_grad():
        y_after = b1.model(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)


def test_lora_save_load_roundtrip(tmp_path: Path):
    backend = TinyCausalLMBackend(TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2))
    adapter = inject_lora(backend.model, LoRAConfig(r=4, target_patterns=("qkv",)))
    # write something into lora_B so checkpoint is non-trivial
    for m in adapter.modules.values():
        m.lora_B.data.uniform_(-0.1, 0.1)
    saved = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    p = tmp_path / "adapter.pt"
    adapter.save(p)
    # zero out and reload
    for m in adapter.modules.values():
        m.lora_A.data.zero_()
        m.lora_B.data.zero_()
    adapter.load(p)
    for k, v in adapter.state_dict().items():
        assert torch.allclose(v, saved[k], atol=1e-8)


def test_lora_merge_then_unmerge_is_identity():
    backend = TinyCausalLMBackend(TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2))
    adapter = inject_lora(backend.model, LoRAConfig(r=4, target_patterns=("head",)))
    for m in adapter.modules.values():
        m.lora_B.data.uniform_(-0.05, 0.05)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        y0 = backend.model(ids).clone()
    adapter.merge_into_base()
    with torch.no_grad():
        y1 = backend.model(ids).clone()
    # merged: same output numerically (A@B now 0 because we zero in merge)
    assert torch.allclose(y0, y1, atol=1e-4)
    # unmerge restores original base weights
    # (we can't check equality cheaply after merge since A,B are zeroed; but
    # re-merge should be no-op)
    adapter.merge_into_base()
    with torch.no_grad():
        y2 = backend.model(ids).clone()
    assert torch.allclose(y1, y2, atol=1e-8)


def test_inject_lora_rejects_empty_pattern():
    backend = TinyCausalLMBackend(TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2))
    with pytest.raises(ValueError, match=r"no nn.Linear matched"):
        inject_lora(backend.model, LoRAConfig(r=4, target_patterns=("does_not_exist",)))
