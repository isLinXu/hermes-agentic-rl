from __future__ import annotations

from typing import Literal

import torch


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

    return loss, {"clip_frac": float(clipped), "ratio_mean": float(ratio_mean)}


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

