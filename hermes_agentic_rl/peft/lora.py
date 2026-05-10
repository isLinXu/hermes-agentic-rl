"""LoRA (Low-Rank Adaptation) injection.

Wraps an existing ``nn.Linear`` with a parallel low-rank path::

    y = x W_base + x A B * (alpha / r)

where ``A ∈ R[in, r]`` and ``B ∈ R[r, out]`` are the only trainable tensors;
``W_base`` is frozen. At deploy time ``merge_into_base()`` folds ``A @ B``
back into ``W_base`` so inference has zero runtime overhead.

Implementation notes:
  - ``A`` is initialized with Kaiming-uniform; ``B`` is initialized to zero —
    at init the adapter is a no-op, which is what LoRA requires.
  - Optional dropout on the adapter path.
  - Target selection via substring match on the parameter's dotted name
    (e.g. ``"blocks.0.attn.qkv"``).
  - Multi-module injection: ``inject_lora`` walks the module tree, replaces
    matching ``nn.Linear`` with ``LoRALinear``, and freezes everything else.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn


@dataclass(slots=True)
class LoRAConfig:
    r: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_patterns: tuple[str, ...] = ("qkv", "proj")
    # Whether to also zero-initialize the base bias (no — untouched by default).
    freeze_base: bool = True


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that adds a LoRA branch."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # base (frozen)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        # lora A/B
        self.lora_A = nn.Parameter(torch.empty(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self._merged:
            return y
        delta = self.dropout(x) @ self.lora_A @ self.lora_B
        return y + delta * self.scaling

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Fold A @ B into base.weight and drop the LoRA branch.

        Idempotent: subsequent calls are no-ops.
        """
        if self._merged:
            return
        delta_w = (self.lora_A @ self.lora_B).t() * self.scaling  # [out, in]
        self.base.weight.add_(delta_w.to(self.base.weight.dtype))
        self.lora_A.zero_()
        self.lora_B.zero_()
        self._merged = True

    def unmerge(self) -> None:
        """Revert a merge so further LoRA training can resume."""
        if not self._merged:
            return
        delta_w = (self.lora_A @ self.lora_B).t() * self.scaling
        with torch.no_grad():
            self.base.weight.sub_(delta_w.to(self.base.weight.dtype))
        self._merged = False


@dataclass(slots=True)
class LoRAAdapter:
    """Opaque handle returned by `inject_lora` for save/load/merge."""

    modules: dict[str, LoRALinear] = field(default_factory=dict)
    cfg: LoRAConfig = field(default_factory=LoRAConfig)

    def parameters(self) -> Iterable[nn.Parameter]:
        for m in self.modules.values():
            yield m.lora_A
            yield m.lora_B

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, m in self.modules.items():
            out[f"{name}.lora_A"] = m.lora_A.detach().clone()
            out[f"{name}.lora_B"] = m.lora_B.detach().clone()
        return out

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, m in self.modules.items():
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state:
                m.lora_A.data.copy_(state[a_key])
            if b_key in state:
                m.lora_B.data.copy_(state[b_key])

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": {
                    "r": self.cfg.r,
                    "alpha": self.cfg.alpha,
                    "dropout": self.cfg.dropout,
                    "target_patterns": list(self.cfg.target_patterns),
                },
                "state": self.state_dict(),
                "module_names": list(self.modules.keys()),
            },
            p,
        )

    def load(self, path: str | Path) -> None:
        data = torch.load(Path(path), map_location="cpu")  # noqa: torch-load-unsafe
        self.load_state_dict(data["state"])

    def merge_into_base(self) -> None:
        for m in self.modules.values():
            m.merge_into_base()

    def unmerge(self) -> None:
        for m in self.modules.values():
            m.unmerge()


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(p in name for p in patterns)


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> LoRAAdapter:
    """Replace matching nn.Linear leaves with LoRALinear, freeze the rest."""
    adapter = LoRAAdapter(cfg=cfg)
    # collect (parent, attr_name, child) tuples first to avoid mutation-while-iterating
    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for mod_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full = f"{mod_name}.{child_name}" if mod_name else child_name
                if _matches(full, cfg.target_patterns):
                    replacements.append((module, child_name, full, child))

    if not replacements:
        raise ValueError(
            f"inject_lora: no nn.Linear matched target_patterns={cfg.target_patterns}. "
            "Inspect model.named_modules() to choose patterns."
        )

    for parent, attr, full_name, lin in replacements:
        wrapped = LoRALinear(lin, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
        setattr(parent, attr, wrapped)
        adapter.modules[full_name] = wrapped

    if cfg.freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in adapter.parameters():
            p.requires_grad_(True)

    return adapter
