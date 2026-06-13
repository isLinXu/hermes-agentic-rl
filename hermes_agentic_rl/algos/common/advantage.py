from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def group_normalize_advantage_tensor(
    rewards: torch.Tensor,  # [G] float
    eps: float = 1e-6,
) -> torch.Tensor:
    """Tensor-native group advantage normalization — avoids Python list loops.

    Equivalent to :func:`group_normalize_advantage` but operates directly on
    a [G] tensor. Returns a [G] tensor of advantages; caller is responsible
    for preserving group membership.

    When G == 1 returns a zero tensor (same semantics as the list version).
    When std < eps (all rewards identical) returns all zeros.
    """
    G = rewards.shape[0]
    if G == 0:
        return rewards.new_zeros(0)
    if G == 1:
        return rewards.new_zeros(1)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if float(std.item()) < eps:
        return rewards.new_zeros(G)
    return ((rewards - mean) / (std + eps)).detach()


def group_normalize_advantage(
    rewards: Sequence[float],
    eps: float = 1e-6,
) -> list[float]:
    """GRPO-style group normalization: (r - mean) / (std + eps).

    When a group has size 1 or zero variance, falls back to (r - mean) to keep
    the signal but avoid divide-by-zero. Returns advantages with the same order
    as `rewards`.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    if n == 1:
        return [0.0]
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        # All rewards identical → advantage = 0 (no learning signal, but that's
        # fine: GRPO just won't update on this group this step).
        return [0.0 for _ in rewards]
    return [(r - mean) / (std + eps) for r in rewards]


def dapo_group_advantage(
    rewards: Sequence[float],
    eps: float = 1e-6,
) -> list[float] | None:
    """DAPO-style group advantage: z-score when informative, else discard group.

    Returns ``None`` when all rewards in the group are identical (zero variance),
    signalling that the group should be filtered from the update.
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return None
    return [(r - mean) / (std + eps) for r in rewards]
