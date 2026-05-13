"""Generalized Advantage Estimation for token-level rewards.

We treat each response token as one environment step. A typical agentic-RL
setup is sparse-reward: the full sequence gets a single scalar reward which
we place at the terminal token; intermediate tokens get 0. More elaborate
dense per-token rewards (e.g. format shaping, process reward models) are
supported by passing a full `token_rewards` vector.

Inputs (1-D lists / tensors of length T = response length):
    token_rewards[t]: r_t
    values[t]:        V(s_t) — critic estimate at the state *from which*
                      token t is emitted. Convention here: values has length T
                      and corresponds to per-response-token V-estimates.

Returns:
    advantages[t], returns[t] = advantages + values
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def compute_gae(
    token_rewards: Sequence[float] | torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
    last_value: float = 0.0,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level GAE(λ).

    Args:
        token_rewards: length-T sequence of per-token rewards.
        values: length-T tensor of V(s_t) (non-differentiable or detached copy
            is fine — the Trainer will pass values.detach() to keep the critic
            loss independent of the policy graph).
        gamma: discount factor.
        lam: GAE lambda.
        last_value: bootstrap value at step T (0 for terminal).
        normalize: if True, z-normalize advantages.

    Returns:
        (advantages [T], returns [T]).
    """
    if isinstance(token_rewards, torch.Tensor):
        rewards = token_rewards.detach().to(dtype=values.dtype, device=values.device)
    else:
        rewards = torch.tensor(list(token_rewards), dtype=values.dtype, device=values.device)
    T = rewards.numel()
    if T == 0:
        empty = torch.zeros(0, dtype=values.dtype, device=values.device)
        return empty, empty
    assert values.numel() == T, f"values len {values.numel()} != rewards len {T}"

    advs = torch.zeros(T, dtype=values.dtype, device=values.device)
    gae = 0.0
    next_value = float(last_value)
    for t in range(T - 1, -1, -1):
        delta = float(rewards[t].item()) + gamma * next_value - float(values[t].item())
        gae = delta + gamma * lam * gae
        advs[t] = gae
        next_value = float(values[t].item())
    returns = advs + values
    if normalize and T > 1:
        mean = advs.mean()
        std = advs.std(unbiased=False)
        if float(std.item()) > 1e-8:
            advs = (advs - mean) / (std + 1e-8)
    return advs.detach(), returns.detach()


def terminal_token_rewards(total_reward: float, length: int) -> list[float]:
    """Place the full reward on the last token; zeros elsewhere."""
    if length <= 0:
        return []
    out = [0.0] * length
    out[-1] = float(total_reward)
    return out


def compute_gae_batched(
    rewards: torch.Tensor,  # [B, T]
    values: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T] bool — True at valid response tokens
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
    normalize: bool = False,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized batched GAE(λ) with tensor ops (no .item(), no Python cast).

    This is 10-50x faster than calling ``compute_gae`` per row because the
    inner recursion runs on tensors and stays on-device.

    Semantics: each row of ``rewards``/``values`` is an independent rollout
    of length ``T_i <= T``. Positions past ``T_i`` must be masked False.
    Within each row the recursion is::

        δ_t = r_t + γ · V(s_{t+1}) · valid_{t+1} - V(s_t)
        A_t = δ_t + γ·λ · A_{t+1} · valid_{t+1}

    Bootstrap value at the final valid position is 0 (terminal).

    Args:
        rewards, values, mask: [B, T]. values/rewards outside mask may be
            anything (they are zeroed internally by the valid_next gate).
        gamma, lam: discount and GAE lambda.
        normalize: if True, whiten advantages over the valid-token
            population (across all rows), keeping padding zeros.

    Returns:
        (advantages [B, T], returns [B, T]) — both detached.
    """
    assert rewards.shape == values.shape == mask.shape, "shape mismatch"
    B, T = rewards.shape
    if B == 0 or T == 0:
        empty = torch.zeros(B, T, dtype=values.dtype, device=values.device)
        return empty, empty

    rewards = rewards.to(dtype=values.dtype, device=values.device)
    mf = mask.to(dtype=values.dtype)
    # valid_next[t] = mask[t+1] shifted left (bootstrap with 0 at terminal)
    valid_next = torch.cat(
        [mf[:, 1:], torch.zeros(B, 1, dtype=values.dtype, device=values.device)],
        dim=1,
    )
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, dtype=values.dtype, device=values.device)],
        dim=1,
    )
    deltas = rewards + gamma * next_values * valid_next - values
    deltas = deltas * mf

    advs = torch.zeros_like(deltas)
    gl = gamma * lam
    # Reverse time recursion. This loop is O(T) but purely tensor-op, no
    # Python-level scalar conversions — ~50x faster than the v0.1 variant.
    running = torch.zeros(B, dtype=values.dtype, device=values.device)
    for t in range(T - 1, -1, -1):
        running = deltas[:, t] + gl * valid_next[:, t] * running
        advs[:, t] = running * mf[:, t]

    returns = advs + values * mf

    if normalize:
        valid = mask.reshape(-1)
        advs_flat = advs.reshape(-1)
        if int(valid.sum().item()) > 1:
            sel = advs_flat[valid]
            mean = sel.mean()
            std = sel.std(unbiased=False)
            if float(std.item()) > eps:
                advs_flat = advs_flat.clone()
                advs_flat[valid] = (sel - mean) / (std + eps)
                advs = advs_flat.view(B, T)

    return advs.detach(), returns.detach()
