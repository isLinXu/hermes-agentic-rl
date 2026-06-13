"""Shared batch-preparation utilities for on-policy algorithms.

Several algorithms (GRPO, RLOO, GSPO, PPO) repeat nearly identical logic for:

1. Stacking old log-probabilities into a padded [B, T_max] tensor.
2. Building the advantage tensor [B, T_max] from per-record scalars/lists.
3. Computing the KL-to-reference penalty (batched).
4. Computing the entropy bonus (negative log-prob proxy).

Extracting these eliminates ~80 lines of duplication per algorithm and
ensures consistency (e.g., KL estimator, entropy formula) across the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from hermes_agentic_rl.algos.base import RolloutRecord
from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched
from hermes_agentic_rl.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# 1) Stack old log-probabilities
# ---------------------------------------------------------------------------

def stack_old_logprobs(
    records_with_adv: list[tuple[RolloutRecord, list[float]]],
    B: int,
    T_max: int,
    dtype: torch.dtype,
    device: torch.device,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Stack per-record ``old_logprobs`` into a right-aligned [B, T_max] tensor.

    Mirrors the backend convention: ``score_batch`` places each record's
    response in positions ``[0, R_i)`` where ``R_i = mask[i].sum()``.
    """
    import torch

    old_logp = torch.zeros(B, T_max, dtype=dtype, device=device)
    for i, (rec, _adv) in enumerate(records_with_adv):
        olp = rec.old_logprobs
        R_i = min(len(olp), int(mask[i].sum().item()))
        if R_i > 0:
            old_logp[i, :R_i] = torch.tensor(olp[-R_i:], dtype=dtype, device=device)
    return old_logp


# ---------------------------------------------------------------------------
# 2) Build advantage tensor
# ---------------------------------------------------------------------------

def build_advantage_tensor(
    records_with_adv: list[tuple[RolloutRecord, list[float]]],
    B: int,
    T_max: int,
    dtype: torch.dtype,
    device: torch.device,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Build a [B, T_max] advantage tensor from per-record scalars/lists.

    Supports both scalar advantages (broadcast across tokens) and per-token
    advantages (variable-length lists aligned to response length).
    """
    import torch

    has_per_token = any(len(a) > 1 for _, a in records_with_adv)
    if has_per_token:
        adv_tensor = torch.zeros(B, T_max, dtype=dtype, device=device)
        for i, (_rec, adv_list) in enumerate(records_with_adv):
            R_i = int(mask[i].sum().item())
            if R_i == 0 or not adv_list:
                continue
            vals = adv_list[-R_i:] if len(adv_list) >= R_i else adv_list
            adv_tensor[i, :len(vals)] = torch.tensor(vals, dtype=dtype, device=device)
    else:
        scalars = torch.tensor(
            [float(a[0]) if a else 0.0 for _, a in records_with_adv],
            dtype=dtype,
            device=device,
        )  # [B]
        adv_tensor = scalars.unsqueeze(-1) * mask.to(dtype)  # [B, T_max]
    return adv_tensor


# ---------------------------------------------------------------------------
# 3) Compute KL-to-reference penalty (batched)
# ---------------------------------------------------------------------------

@dataclass
class KLPenaltyResult:
    """Result of computing the KL-to-reference penalty."""

    kl_scalar: torch.Tensor  # scalar KL value (detached)
    kl_val: float  # Python float for stats


def compute_kl_penalty(
    new_logp: torch.Tensor,
    mask: torch.Tensor,
    prompt_ids_list: list[list[int]],
    response_ids_list: list[list[int]],
    score_temperature: float,
    ref_policy: LLMBackend | None,
    kl_coef: float,
    kl_estimator: str = "k3",
) -> KLPenaltyResult | None:
    """Compute the KL-to-reference penalty if applicable.

    Returns ``None`` when there is no reference policy or ``kl_coef`` is zero.
    """
    import torch

    if ref_policy is None or kl_coef <= 0:
        return None

    with torch.no_grad():
        ref_logp, _ref_mask = ref_policy.score_batch(
            prompt_ids_list,
            response_ids_list,
            temperature=score_temperature,
        )
    kl_scalar = kl_from_logprobs_batched(
        new_logp, ref_logp, mask, estimator=kl_estimator
    )
    return KLPenaltyResult(
        kl_scalar=kl_scalar,
        kl_val=float(kl_scalar.detach().item()),
    )


# ---------------------------------------------------------------------------
# 4) Compute entropy bonus
# ---------------------------------------------------------------------------

def compute_entropy_bonus(
    new_logp: torch.Tensor,
    mask: torch.Tensor,
    entropy_coef: float,
) -> tuple[float, torch.Tensor | None]:
    """Compute the entropy bonus (negative log-prob proxy).

    Returns ``(ent_val, ent_scalar)`` where ``ent_val`` is a Python float
    for stats and ``ent_scalar`` is the tensor to subtract from the total loss.
    When ``entropy_coef <= 0``, returns ``(0.0, None)``.
    """
    if entropy_coef <= 0:
        return 0.0, None

    dtype = new_logp.dtype
    mf = mask.to(dtype)
    ent_per_row = -(new_logp * mf).sum(dim=-1) / mf.sum(dim=-1).clamp(min=1)
    ent_scalar = ent_per_row.mean()
    ent_val = float(ent_scalar.detach().item())
    return ent_val, ent_scalar


# ---------------------------------------------------------------------------
# 5) Mean reward / advantage helpers
# ---------------------------------------------------------------------------

def mean_reward_from_records(records: list[RolloutRecord]) -> float:
    """Compute mean reward from a list of RolloutRecords."""
    if not records:
        return 0.0
    return sum(r.reward for r in records) / len(records)


def mean_advantage_from_tensors(
    adv_tensor: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute mean advantage from tensors, respecting the mask."""
    dtype = adv_tensor.dtype
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return 0.0
    return float((adv_tensor * mask.to(dtype)).sum().item()) / n_valid
