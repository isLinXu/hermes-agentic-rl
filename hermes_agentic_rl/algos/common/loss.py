from __future__ import annotations

from typing import Literal

import torch


def clipped_surrogate_loss_batched(
    new_logprobs: torch.Tensor,  # [B, T]
    old_logprobs: torch.Tensor,  # [B, T]
    advantage: torch.Tensor,  # [B, T] or [B, 1] or [B]
    mask: torch.Tensor,  # [B, T] bool
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token",
    max_len_for_dr_grpo: int = 256,
    kl_estimator: Literal["k1", "k2", "k3"] = "k3",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Batched PPO-clipped surrogate with mask (v0.8).

    Args:
        new_logprobs, old_logprobs: [B, T] per-token logπ. Padding positions
            in ``new_logprobs`` are assumed to already be zero (backends
            using the new ``score_batch`` enforce this).
        advantage: broadcastable to [B, T]. Scalar GRPO advantage should
            be passed as [B, 1]; per-token GAE advantage as [B, T].
        mask: [B, T] bool — True at valid response tokens.
        clip_eps: lower clipping epsilon (ratio lower bound ``1 - clip_eps``).
        clip_eps_high: optional asymmetric upper clip (defaults to ``clip_eps``).
        loss_agg: aggregation across tokens and rollouts:
            - ``mean_token``: for each rollout, mean over its valid tokens;
              then mean over rollouts (GRPO/DeepSeek standard).
            - ``sum_token``: per-rollout token sum, then mean over rollouts.
            - ``dr_grpo``: per-rollout token sum ÷ ``max_len_for_dr_grpo``,
              then mean over rollouts (Liu 2024, removes length bias).
        kl_estimator: which KL estimator to use for ``approx_kl`` reporting:
            - ``k1``: first-order approximation ``logr`` (biased).
            - ``k2``: second-order ``0.5 * r²`` (symmetric, low variance).
            - ``k3``: Schulman ``exp(-r) - 1 + r`` (always non-negative,
              low variance, consistent with verl/TRL/DeepSeek default).
            Must match the ``kl_estimator`` configured on the calling Algo so
            that the reported ``approx_kl`` is comparable to the KL penalty
            term and can be used reliably for early-stopping thresholds.

    Returns:
        (loss [scalar], stats{clip_frac, ratio_mean, approx_kl})

    ``approx_kl`` is now computed using ``kl_estimator`` (default ``k3``),
    matching the standard used in the GRPO/RLOO/PPO penalty terms.
    """
    B = new_logprobs.shape[0]
    if B == 0 or new_logprobs.numel() == 0:
        zero = new_logprobs.new_zeros(())
        return zero, {
            "clip_frac": 0.0,
            "ratio_mean": 1.0,
            "approx_kl": 0.0,
            "n_tokens": 0,
        }

    adv = advantage.to(dtype=new_logprobs.dtype, device=new_logprobs.device).detach()
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)  # [B] → [B, 1]
    # Broadcast-safe
    mf = mask.to(dtype=new_logprobs.dtype)
    tokens_per_row = mf.sum(dim=-1).clamp(min=1)

    eps_high = clip_eps if clip_eps_high is None else clip_eps_high
    # Mixed-precision: cast old_logprobs to new_logprobs' dtype so the loss
    # stays in the lower precision (bfloat16 when AMP is active).
    old_lp = old_logprobs.to(dtype=new_logprobs.dtype)
    ratio = torch.exp(new_logprobs - old_lp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + eps_high) * adv
    loss_per_tok = -torch.minimum(surr1, surr2) * mf

    if loss_agg == "mean_token":
        per_row = loss_per_tok.sum(dim=-1) / tokens_per_row
        loss = per_row.mean()
    elif loss_agg == "sum_token":
        per_row = loss_per_tok.sum(dim=-1)
        loss = per_row.mean()
    elif loss_agg == "dr_grpo":
        per_row = loss_per_tok.sum(dim=-1) / float(max(1, max_len_for_dr_grpo))
        loss = per_row.mean()
    else:
        raise ValueError(f"unknown loss_agg: {loss_agg}")

    with torch.no_grad():
        # clip_frac: fraction of valid tokens where |ratio - 1| > eps.
        clip_threshold = max(clip_eps, eps_high)
        clipped_mask = (torch.abs(ratio - 1.0) > clip_threshold) & mask
        n_tok = mask.sum().clamp(min=1)
        clip_frac = clipped_mask.to(ratio.dtype).sum() / n_tok
        ratio_mean = (ratio * mf).sum() / n_tok
        # approx_kl uses the caller-selected estimator so it is consistent
        # with the KL penalty term and reliable for early-stopping thresholds.
        r_kl = (new_logprobs - old_logprobs) * mf
        if kl_estimator == "k1":
            per_row_kl = r_kl.sum(dim=-1) / tokens_per_row
        elif kl_estimator == "k2":
            per_row_kl = (0.5 * r_kl.pow(2)).sum(dim=-1) / tokens_per_row
        else:  # k3 — Schulman, always non-negative, verl/TRL/DeepSeek standard
            r_kl_c = r_kl.clamp(min=-20.0, max=20.0)
            per_row_kl = (torch.exp(-r_kl_c) - 1.0 + r_kl_c).sum(dim=-1) / tokens_per_row
        approx_kl = per_row_kl.mean()

    return loss, {
        "clip_frac": float(clip_frac.item()),
        "ratio_mean": float(ratio_mean.item()),
        "approx_kl": float(approx_kl.item()),
        "n_tokens": int(mask.sum().item()),
    }


def clipped_value_loss_batched(
    values_new: torch.Tensor,  # [B, T]
    values_old: torch.Tensor,  # [B, T]
    returns: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T] bool
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Batched clipped value loss (PPO)."""
    if values_new.numel() == 0:
        zero = values_new.new_zeros(())
        return zero, {"value_clip_frac": 0.0, "value_mean": 0.0}
    returns = returns.to(dtype=values_new.dtype, device=values_new.device).detach()
    values_old = values_old.to(dtype=values_new.dtype, device=values_new.device).detach()
    mf = mask.to(dtype=values_new.dtype)
    tokens_per_row = mf.sum(dim=-1).clamp(min=1)

    v_clipped = values_old + torch.clamp(values_new - values_old, -clip_eps, clip_eps)
    loss_unclipped = (values_new - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    per_tok = 0.5 * torch.maximum(loss_unclipped, loss_clipped) * mf
    per_row = per_tok.sum(dim=-1) / tokens_per_row
    loss = per_row.mean()

    with torch.no_grad():
        n_tok = mask.sum().clamp(min=1)
        clip_frac = (((values_new - values_old).abs() > clip_eps) & mask).to(
            values_new.dtype
        ).sum() / n_tok
        v_mean = (values_new * mf).sum() / n_tok
    return loss, {
        "value_clip_frac": float(clip_frac.item()),
        "value_mean": float(v_mean.item()),
    }


def clipped_surrogate_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: float | torch.Tensor,
    clip_eps: float = 0.2,
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token",
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO-style clipped surrogate for a single rollout.

    Args:
        new_logprobs: [T] differentiable per-token logπ_new
        old_logprobs: [T] fixed per-token logπ_old (detached)
        advantage: scalar advantage for the whole rollout (GRPO original
            formulation) OR a [T] tensor for token-level advantages (PPO-GAE
            / token-level GRPO).
        clip_eps: clipping epsilon.
        loss_agg: how to reduce token-level losses into a scalar.
            - "mean_token" (default): arithmetic mean over response tokens.
              Long responses contribute proportionally less per token.
            - "sum_token": sum (no length normalization). Long responses
              dominate. Not recommended outside debugging.
            - "dr_grpo": Dr.GRPO (Liu et al. 2024) — divide by a fixed
              constant ``MAX_LEN`` rather than the rollout's own length, so
              length bias is removed across groups. Here we use the
              rollout's own length but return the token-sum and leave the
              cross-rollout averaging to the caller. In the single-rollout
              API this reduces to "sum_token"; call sites that care about
              Dr.GRPO semantics should then divide by max_len externally.

    Returns:
        (loss_tensor (scalar to be MINIMIZED), stats dict)
    """
    if new_logprobs.numel() == 0:
        zero = new_logprobs.new_zeros(())
        return zero, {"clip_frac": 0.0, "ratio_mean": 1.0}

    ratio = torch.exp(new_logprobs - old_logprobs)
    adv: float | torch.Tensor
    if isinstance(advantage, torch.Tensor):
        adv = advantage.to(dtype=new_logprobs.dtype, device=new_logprobs.device).detach()
        if adv.numel() != new_logprobs.numel():
            # Broadcast scalar-ish advantage or truncate to shared length.
            n = min(adv.numel(), new_logprobs.numel())
            adv = adv[-n:]
            ratio = ratio[-n:]
    else:
        adv = float(advantage)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss_per_tok = -torch.minimum(surr1, surr2)

    if loss_agg == "mean_token":
        loss = loss_per_tok.mean()
    elif loss_agg in ("sum_token", "dr_grpo"):
        loss = loss_per_tok.sum()
    else:
        raise ValueError(f"unknown loss_agg: {loss_agg}")

    with torch.no_grad():
        clipped = (torch.abs(ratio - 1.0) > clip_eps).float().mean().item()
        ratio_mean = ratio.mean().item()
        # k3 KL estimator — consistent with batched variant and global default.
        log_r = (new_logprobs - old_logprobs).clamp(min=-20.0, max=20.0)
        approx_kl = float((torch.exp(-log_r) - 1.0 + log_r).mean().item())

    return loss, {
        "clip_frac": float(clipped),
        "ratio_mean": float(ratio_mean),
        "approx_kl": approx_kl,
        "n_tokens": int(new_logprobs.numel()),
    }


def clipped_value_loss(
    values_new: torch.Tensor,
    values_old: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO value-function loss with optional clipping (Schulman et al. 2017).

    loss = 0.5 * max( (V_new - R)^2, (clip(V_new, V_old ± ε) - R)^2 )
    """
    if values_new.numel() == 0:
        zero = values_new.new_zeros(())
        return zero, {"value_clip_frac": 0.0, "value_mean": 0.0}
    returns = returns.to(dtype=values_new.dtype, device=values_new.device).detach()
    values_old = values_old.to(dtype=values_new.dtype, device=values_new.device).detach()
    v_clipped = values_old + torch.clamp(values_new - values_old, -clip_eps, clip_eps)
    loss_unclipped = (values_new - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    loss = 0.5 * torch.maximum(loss_unclipped, loss_clipped).mean()
    with torch.no_grad():
        clipped_frac = ((values_new - values_old).abs() > clip_eps).float().mean().item()
        v_mean = values_new.mean().item()
    return loss, {"value_clip_frac": float(clipped_frac), "value_mean": float(v_mean)}
