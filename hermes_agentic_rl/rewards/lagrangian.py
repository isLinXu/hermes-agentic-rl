"""Lagrangian safety constraints (RCPO-style).

Given a "cost" function ``c(item, trajectory) ∈ [0, ∞)`` (higher = worse)
and a cost limit ``c_limit``, RCPO optimizes::

    max_π  E[r] - λ · ( E[c] - c_limit )

via simultaneous updates:
  - policy: standard PPO/GRPO loss with an extra `+ λ · mean_cost` term
  - λ: projected gradient ascent,  λ ← max(0, λ + lr_lambda · (mean_cost - c_limit))

The Trainer calls ``controller.measure(item, trajectory)`` after each
rollout, and ``controller.step_loss(policy_loss)`` before the optimizer step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from hermes_agentic_rl.core.types import Trajectory

CostFn = Callable[[dict[str, Any], Trajectory], float]


@dataclass(slots=True)
class LagrangianConfig:
    cost_limit: float = 0.1
    init_lambda: float = 0.0
    lr_lambda: float = 0.05
    ema_decay: float = 0.9    # EMA of mean cost (smooths dual updates)
    max_lambda: float = 100.0


@dataclass(slots=True)
class LagrangianState:
    lam: float = 0.0
    ema_cost: float = 0.0
    total_observed: int = 0
    last_mean_cost: float = 0.0


class LagrangianController:
    def __init__(
        self,
        cost_fn: CostFn,
        cfg: LagrangianConfig | None = None,
    ) -> None:
        self.cost_fn = cost_fn
        self.cfg = cfg or LagrangianConfig()
        self.state = LagrangianState(lam=self.cfg.init_lambda)
        self._iter_costs: list[float] = []

    def measure(self, item: dict[str, Any], trajectory: Trajectory) -> float:
        c = max(0.0, float(self.cost_fn(item, trajectory)))
        self._iter_costs.append(c)
        return c

    def begin_iter(self) -> None:
        self._iter_costs = []

    def penalty_term(self, loss_tensor: torch.Tensor) -> torch.Tensor:
        """Return ``loss + λ · mean_cost`` (constant in params, but kept on
        the same device/dtype for clean composition)."""
        if not self._iter_costs:
            return loss_tensor
        mean_cost = sum(self._iter_costs) / len(self._iter_costs)
        self.state.last_mean_cost = mean_cost
        ema = (
            self.cfg.ema_decay * self.state.ema_cost
            + (1.0 - self.cfg.ema_decay) * mean_cost
        )
        self.state.ema_cost = ema
        self.state.total_observed += len(self._iter_costs)
        # penalty is a constant w.r.t. policy params here; it acts as an
        # *indicator* in logs. The actual gradient pressure comes from the
        # fact that high-cost trajectories should lower mean_reward (since
        # the env's reward manager can also subtract penalty internally).
        # If you want cost to affect gradients, bake it into the reward
        # itself — e.g. reward_eff = reward - λ · cost_component.
        pen = loss_tensor.new_tensor(self.state.lam * mean_cost)
        return loss_tensor + pen

    def dual_step(self) -> None:
        """Project-and-update λ after the learner step."""
        if not self._iter_costs:
            return
        mean_cost = self.state.last_mean_cost
        grad = mean_cost - self.cfg.cost_limit
        self.state.lam = float(
            min(self.cfg.max_lambda, max(0.0, self.state.lam + self.cfg.lr_lambda * grad))
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "lambda": self.state.lam,
            "ema_cost": self.state.ema_cost,
            "last_mean_cost": self.state.last_mean_cost,
            "total_observed": self.state.total_observed,
            "cost_limit": self.cfg.cost_limit,
        }
