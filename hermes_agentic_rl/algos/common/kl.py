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
