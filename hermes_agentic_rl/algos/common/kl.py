"""KL divergence estimators between two token-level log-prob streams.

Given per-token ``logπ_new`` and ``logπ_ref`` (both length [T]), let
``r = logπ_new - logπ_ref``. The three classical unbiased estimators
(Schulman 2020 — http://joschu.net/blog/kl-approx.html) are:

- **K1**: ``mean(r)`` — unbiased, high variance, CAN BE NEGATIVE
  (despite KL being non-negative). This is what GRPO / PPO codes
  have historically used. Cheap and differentiable.

- **K2**: ``mean(0.5 * r ** 2)`` — biased but always non-negative and
  low-variance. Not the estimator of KL(π_new ‖ π_ref); it estimates
  half the chi-squared divergence. Differentiable.

- **K3**: ``mean(exp(-r) - 1 + r)`` — unbiased, always non-negative,
  low variance, differentiable. Currently the recommended default for
  PPO/GRPO-style KL penalties (used by TRL, DeepSeek, verl).

All three return a scalar tensor with grad if the inputs have grad.
"""

from __future__ import annotations

from typing import Literal

import torch

KLEstimator = Literal["k1", "k2", "k3"]


def kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    estimator: KLEstimator = "k3",
) -> torch.Tensor:
    """Return a scalar KL estimate per-token-averaged.

    Args:
        new_logprobs: [T] logπ_new(a_t | s_t), differentiable.
        ref_logprobs: [T] logπ_ref(a_t | s_t), typically detached.
        estimator: "k1" | "k2" | "k3". Default "k3".

    Returns:
        Scalar tensor. For k1 may be negative; for k2/k3 always >= 0.
    """
    if new_logprobs.numel() == 0:
        return new_logprobs.new_zeros(())
    r = new_logprobs - ref_logprobs.to(dtype=new_logprobs.dtype, device=new_logprobs.device)
    if estimator == "k1":
        return r.mean()
    if estimator == "k2":
        return (0.5 * r.pow(2)).mean()
    if estimator == "k3":
        # exp(-r) - 1 + r = KL estimator (always >= 0)
        # Clamp r to avoid exp overflow when policy drifts too far from ref.
        r_clamped = r.clamp(min=-20.0, max=20.0)
        return (torch.exp(-r_clamped) - 1.0 + r_clamped).mean()
    raise ValueError(f"unknown KL estimator: {estimator}")


def kl_from_logprobs_batched(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    estimator: KLEstimator = "k3",
) -> torch.Tensor:
    """Masked batched KL divergence — scalar.

    Computes per-row KL (averaged over valid tokens), then averages over
    rows.  This is the canonical implementation shared by GRPO, PPO, and
    RLOO; previously each duplicated the same 8-line block.

    Args:
        new_logprobs: [B, T] current policy log-probs.
        ref_logprobs: [B, T] reference policy log-probs (detached expected).
        mask: [B, T] bool — True at valid (non-padding) positions.
        estimator: "k1" | "k2" | "k3".

    Returns:
        Scalar KL estimate (grad flows through new_logprobs).
    """
    if new_logprobs.numel() == 0:
        return new_logprobs.new_zeros(())
    dtype = new_logprobs.dtype
    mf = mask.to(dtype=dtype, device=new_logprobs.device)
    tokens_per_row = mf.sum(dim=-1).clamp(min=1)
    common_T = min(new_logprobs.shape[1], ref_logprobs.shape[1])
    r = (
        new_logprobs[:, :common_T]
        - ref_logprobs[:, :common_T].to(dtype=dtype, device=new_logprobs.device)
    ) * mf[:, :common_T]
    if estimator == "k1":
        kl_per_tok = r
    elif estimator == "k2":
        kl_per_tok = 0.5 * r.pow(2)
    elif estimator == "k3":
        r_c = r.clamp(min=-20.0, max=20.0)
        kl_per_tok = torch.exp(-r_c) - 1.0 + r_c
    else:
        raise ValueError(f"unknown KL estimator: {estimator}")
    kl_per_row = (kl_per_tok * mf[:, :common_T]).sum(dim=-1) / tokens_per_row
    return kl_per_row.mean()
