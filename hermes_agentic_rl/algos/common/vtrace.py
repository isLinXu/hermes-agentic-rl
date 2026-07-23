"""Off-policy corrections for asynchronous RL — V-trace + TIS.

Why this exists
---------------
Asynchronous (decoupled) RL — the OpenClaw-RL architecture, where Policy
Serving / Env / Judge / Training run as independent components with no
coordination barrier — necessarily consumes *stale* rollouts: a record was
sampled under policy version ``v_behavior`` but the learner has since advanced
to ``v_learner > v_behavior``. The importance ratio

    ρ_t = π_θ(a_t | s_t) / π_β(a_t | s_t)

is no longer ≈ 1, so a naive on-policy update is biased. Two standard fixes:

* **TIS (Truncated Importance Sampling)** — multiply each per-token advantage
  by a clipped IS weight ``min(ρ_t, c̄)``. Cheap, drop-in, and exactly what a
  PPO/GRPO clipped surrogate already approximates within its trust region; TIS
  extends correction *outside* the clip range for moderately stale data.

* **V-trace** (Espeholt 2018, IMPALA) — a recursive value-target correction
  using truncated IS weights ``ρ̄`` (for the TD target) and ``c̄`` (for the
  trace cutting). Used for PPO's value head under staleness.

Both are pure tensor functions here so they are trivially testable and can be
wired into GRPO (TIS on advantage) or PPO (V-trace on value targets) behind an
opt-in flag. Staleness=0 (synchronous/BSP) must reduce to the on-policy update.

Conventions match ``algos/common/loss.py``: tensors are ``[B, T]`` with a bool
``mask`` (True at valid response tokens), log-probs are per-token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TISConfig:
    """Config for truncated importance sampling.

    rho_clip: upper bound c̄ on the per-token IS weight ``min(ρ, c̄)``. 1.0
        recovers the most conservative (fully clipped) correction; larger
        values trust stale data more. OpenClaw-RL / IMPALA default ≈ 1.0–2.0.
    enabled: master switch (False → identity weight 1.0 everywhere).
    rho_floor: optional lower clamp on ρ (guards against vanishing weights from
        very negative log-ratios). 0.0 disables the floor.
    """

    rho_clip: float = 1.0
    enabled: bool = True
    rho_floor: float = 0.0


def importance_weights(
    new_logprobs: torch.Tensor,  # [B, T] log π_θ
    behavior_logprobs: torch.Tensor,  # [B, T] log π_β (rollout / old)
    mask: torch.Tensor,  # [B, T] bool
    *,
    rho_clip: float = 1.0,
    rho_floor: float = 0.0,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Per-token truncated IS weight ``clip(exp(logπ_θ − logπ_β), floor, c̄)``.

    Returns a detached ``[B, T]`` weight tensor (the correction multiplies a
    detached advantage; gradients flow through the policy term, not the weight).
    Masked positions are set to 0.
    """
    log_ratio = (new_logprobs - behavior_logprobs).clamp(min=-log_ratio_clip, max=log_ratio_clip)
    rho = torch.exp(log_ratio)
    if rho_floor > 0:
        rho = rho.clamp(min=rho_floor)
    rho = rho.clamp(max=rho_clip)
    return (rho * mask.to(rho.dtype)).detach()


def tis_corrected_advantage(
    advantage: torch.Tensor,  # [B, T] or [B, 1] or [B]
    new_logprobs: torch.Tensor,  # [B, T]
    behavior_logprobs: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T] bool
    cfg: TISConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply TIS to a (broadcastable) advantage tensor.

    Returns ``(corrected_advantage [B, T], stats)``. When ``cfg.enabled`` is
    False the advantage is returned broadcast to ``[B, T]`` unchanged, so the
    synchronous path is a strict no-op.
    """
    cfg = cfg or TISConfig()
    mf = mask.to(new_logprobs.dtype)
    adv = advantage.to(dtype=new_logprobs.dtype, device=new_logprobs.device)
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)
    adv = adv * mf  # broadcast scalar/[B,1] advantage to [B,T] over valid tokens

    if not cfg.enabled:
        return adv, {"tis_weight_mean": 1.0, "tis_clip_frac": 0.0}

    w = importance_weights(
        new_logprobs,
        behavior_logprobs,
        mask,
        rho_clip=cfg.rho_clip,
        rho_floor=cfg.rho_floor,
    )
    corrected = adv * w

    n_tok = float(mf.sum().item()) or 1.0
    w_mean = float((w * mf).sum().item()) / n_tok
    # fraction of (valid) tokens whose raw ρ exceeded the clip
    raw = torch.exp((new_logprobs - behavior_logprobs).clamp(-20, 20))
    clip_frac = float(((raw > cfg.rho_clip).to(mf.dtype) * mf).sum().item()) / n_tok
    return corrected, {"tis_weight_mean": w_mean, "tis_clip_frac": clip_frac}


@dataclass(slots=True)
class VTraceConfig:
    """Config for V-trace value-target correction.

    rho_bar: c̄_ρ — clip on the IS weight used in the TD target.
    c_bar:   c̄_c — clip on the IS weight used to cut the trace.
    gamma:   discount.
    enabled: master switch.
    """

    rho_bar: float = 1.0
    c_bar: float = 1.0
    gamma: float = 1.0
    enabled: bool = True


def vtrace_returns(
    rewards: torch.Tensor,  # [B, T] per-step reward (last token = terminal)
    values: torch.Tensor,  # [B, T] V(s_t) from the current critic
    bootstrap_value: torch.Tensor,  # [B] V(s_T) bootstrap
    new_logprobs: torch.Tensor,  # [B, T]
    behavior_logprobs: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T] bool
    cfg: VTraceConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute V-trace value targets ``vs`` and advantages.

    Returns ``(vs [B, T], advantages [B, T], stats)``. The advantage is the
    V-trace policy-gradient advantage ``ρ_t · (r_t + γ·vs_{t+1} − V(s_t))``.

    When ``cfg.enabled`` is False this reduces to the uncorrected n-step return
    (all IS weights = 1), i.e. the synchronous path.
    """
    cfg = cfg or VTraceConfig()
    dtype = values.dtype
    device = values.device
    B, T = values.shape
    mf = mask.to(dtype)

    if cfg.enabled:
        log_ratio = (new_logprobs - behavior_logprobs).clamp(-20, 20)
        rho = torch.exp(log_ratio)
        rho_t = rho.clamp(max=cfg.rho_bar)
        c_t = rho.clamp(max=cfg.c_bar)
    else:
        rho_t = torch.ones(B, T, dtype=dtype, device=device)
        c_t = torch.ones(B, T, dtype=dtype, device=device)
    rho_t = rho_t * mf
    c_t = c_t * mf

    # Next-state values: shift values left, append bootstrap at the end.
    next_values = torch.zeros(B, T, dtype=dtype, device=device)
    if T > 1:
        next_values[:, : T - 1] = values[:, 1:]
    next_values[:, T - 1] = bootstrap_value.to(dtype=dtype, device=device)

    # TD error δ_t = ρ_t (r_t + γ V(s_{t+1}) − V(s_t))
    deltas = rho_t * (rewards + cfg.gamma * next_values - values)

    # Recursive V-trace: vs_t = V(s_t) + δ_t + γ c_t (vs_{t+1} − V(s_{t+1}))
    vs_minus_v = torch.zeros(B, T, dtype=dtype, device=device)
    acc = torch.zeros(B, dtype=dtype, device=device)
    for t in range(T - 1, -1, -1):
        acc = deltas[:, t] + cfg.gamma * c_t[:, t] * acc
        vs_minus_v[:, t] = acc
    vs = vs_minus_v + values

    # Advantage uses vs_{t+1} (bootstrap for the last step).
    vs_next = torch.zeros(B, T, dtype=dtype, device=device)
    if T > 1:
        vs_next[:, : T - 1] = vs[:, 1:]
    vs_next[:, T - 1] = bootstrap_value.to(dtype=dtype, device=device)
    advantages = rho_t * (rewards + cfg.gamma * vs_next - values)

    n_tok = float(mf.sum().item()) or 1.0
    stats = {
        "vtrace_rho_mean": float((rho_t * mf).sum().item()) / n_tok,
        "vtrace_c_mean": float((c_t * mf).sum().item()) / n_tok,
    }
    return (vs * mf).detach(), (advantages * mf).detach(), stats
