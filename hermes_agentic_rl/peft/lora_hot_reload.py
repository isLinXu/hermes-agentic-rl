"""LoRA hot-reload manager — bridges LoRA training with vLLM weight sync.

This module provides ``LoRAHotReloadManager``, which:
  1. Wraps a ``LoRAAdapter`` and the learner's base model.
  2. After each optimizer step, merges the latest LoRA delta into a *shadow*
     copy of the base weights and pushes the merged result to vLLM via
     ``sync_weights_from``. The original base weights remain untouched so
     gradients continue to flow only through A/B.
  3. Supports periodic save / load of LoRA checkpoints for fault recovery.

Design rationale:
  - ART (OpenPipe) achieves zero-downtime LoRA reload by saving a LoRA file
    and asking vLLM to load it. Our vLLM backend uses ``load_weights`` /
    ``collective_rpc("update_weights")`` which expects full model weights,
    not LoRA deltas. The manager therefore *merges* before syncing, giving
    the same end result without requiring vLLM-side LoRA support.
  - The merge happens on a CPU shadow tensor so the learner's GPU memory
    is unaffected. The shadow is allocated once and reused.

Usage::

    from hermes_agentic_rl.peft.lora_hot_reload import LoRAHotReloadManager

    manager = LoRAHotReloadManager(adapter, base_model, vllm_backend)
    # After each optimizer step:
    manager.sync_to_vllm()
    # Save / load:
    manager.save_lora("checkpoints/iter_10.lora")
    manager.load_lora("checkpoints/iter_10.lora")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hermes_agentic_rl.peft.lora import LoRAAdapter

logger = logging.getLogger(__name__)


class LoRAHotReloadManager:
    """Manages LoRA delta → merged weights → vLLM sync pipeline.

    Parameters
    ----------
    adapter : LoRAAdapter
        The LoRA adapter wrapping the learner's model.
    base_model : nn.Module
        The underlying model (same object that ``adapter`` was injected into).
        Must expose ``state_dict()``.
    vllm_backend : Any, optional
        A ``VLLMRolloutBackend`` (or any object with ``sync_weights_from``).
        If ``None``, the manager only handles save/load but does not sync.
    sync_every : int
        Sync to vLLM every N calls to ``sync_to_vllm``. Default: 1 (every call).
    device : str
        Where to keep the shadow merged weights. ``"cpu"`` avoids GPU memory
        pressure; ``"cuda"`` is faster but doubles model memory.
    """

    def __init__(
        self,
        adapter: LoRAAdapter,
        base_model: nn.Module,
        vllm_backend: Any | None = None,
        *,
        sync_every: int = 1,
        shadow_device: str = "cpu",
    ) -> None:
        self.adapter = adapter
        self.base_model = base_model
        self.vllm_backend = vllm_backend
        self.sync_every = max(1, sync_every)
        self._sync_counter = 0
        self._shadow_device = shadow_device

        # Build a shadow copy of base weights (frozen, CPU by default).
        # This is where we apply the LoRA merge before sending to vLLM.
        self._shadow_state: dict[str, torch.Tensor] | None = None
        self._shadow_initialized = False

    def _ensure_shadow(self) -> None:
        """Lazily allocate the shadow state dict on first sync."""
        if self._shadow_initialized:
            return
        self._shadow_state = {}
        for name, param in self.base_model.state_dict().items():
            self._shadow_state[name] = param.detach().clone().to(self._shadow_device)
        self._shadow_initialized = True
        logger.info(
            "LoRAHotReloadManager: shadow state initialized (%d tensors on %s)",
            len(self._shadow_state),
            self._shadow_device,
        )

    def _compute_merged_state(self) -> dict[str, torch.Tensor]:
        """Merge current LoRA deltas into the shadow and return it.

        For each ``LoRALinear`` module in the adapter:
          shadow[name + ".base.weight"] += (A @ B)^T * scaling

        All other parameters are copied as-is from the shadow.
        """
        self._ensure_shadow()
        assert self._shadow_state is not None

        # Start from the current shadow (which holds the base weights).
        merged: dict[str, torch.Tensor] = {
            name: tensor.clone() for name, tensor in self._shadow_state.items()
        }

        # Apply LoRA deltas.
        for mod_name, lora_linear in self.adapter.modules.items():
            if lora_linear._merged:
                continue
            # Base weight key in the model's state_dict.
            base_key = f"{mod_name}.base.weight"
            if base_key not in merged:
                logger.warning("LoRAHotReloadManager: key %s not in shadow, skipping", base_key)
                continue

            # delta_W = (A @ B)^T * scaling  → shape [out_features, in_features]
            a = lora_linear.lora_A.detach().to(merged[base_key].dtype)
            b = lora_linear.lora_B.detach().to(merged[base_key].dtype)
            delta_w = (a @ b).t() * lora_linear.scaling

            merged[base_key] = merged[base_key] + delta_w.to(merged[base_key].device)

        return merged

    def sync_to_vllm(self) -> bool:
        """Merge current LoRA deltas and push to vLLM.

        Returns ``True`` if a sync actually happened, ``False`` if skipped
        by ``sync_every`` throttling.
        """
        if self.vllm_backend is None:
            return False

        self._sync_counter += 1
        if self._sync_counter % self.sync_every != 0:
            return False

        merged = self._compute_merged_state()
        # vLLM expects CPU tensors.
        cpu_state = {k: v.detach().cpu() for k, v in merged.items()}
        self.vllm_backend.sync_weights_from(cpu_state)
        logger.debug(
            "LoRAHotReloadManager: synced merged weights to vLLM (sync #%d)",
            self._sync_counter,
        )
        return True

    def save_lora(self, path: str | Path) -> None:
        """Save only the LoRA adapter weights (small file)."""
        self.adapter.save(path)

    def load_lora(self, path: str | Path) -> None:
        """Load LoRA adapter weights from file."""
        self.adapter.load(path)

    def save_full(self, path: str | Path) -> None:
        """Save merged full model weights (base + LoRA delta).

        Useful for deployment where vLLM loads a single checkpoint.
        """
        merged = self._compute_merged_state()
        torch.save(merged, Path(path))

    @property
    def num_lora_params(self) -> int:
        """Total number of trainable LoRA parameters."""
        return self.adapter.num_parameters()

    def lora_param_ratio(self) -> float:
        """Fraction of total model parameters that are LoRA-trainable."""
        total = sum(
            p.numel() for p in self.base_model.parameters()
        )
        if total == 0:
            return 0.0
        return self.num_lora_params / total
