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
