"""Lagrangian controller + integration with OnPolicyTrainer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.rewards.lagrangian import (
    LagrangianConfig,
    LagrangianController,
)
from hermes_agentic_rl.trainers import GRPOTrainer, GRPOTrainerConfig


def _backend() -> TinyCausalLMBackend:
    return TinyCausalLMBackend(TinyBackendConfig(seed=0, dim=32, n_heads=4, n_layers=2))


def test_lagrangian_lambda_grows_when_cost_exceeds_limit():
    # cost always 1.0, limit 0.1 → λ must grow monotonically per dual step.
    ctl = LagrangianController(
        cost_fn=lambda item, traj: 1.0,
        cfg=LagrangianConfig(cost_limit=0.1, init_lambda=0.0, lr_lambda=0.1),
    )
    for _ in range(5):
        ctl.begin_iter()
        ctl.measure({}, _dummy_traj())  # registers one cost = 1.0
        # fake a loss tensor
        loss = torch.zeros((), requires_grad=False)
        _ = ctl.penalty_term(loss)
        ctl.dual_step()
    assert ctl.state.lam > 0.0
    assert ctl.state.last_mean_cost == pytest.approx(1.0)


def test_lagrangian_lambda_stays_zero_when_safe():
    ctl = LagrangianController(
        cost_fn=lambda item, traj: 0.0,
        cfg=LagrangianConfig(cost_limit=1.0, init_lambda=0.0, lr_lambda=1.0),
    )
    for _ in range(3):
        ctl.begin_iter()
        ctl.measure({}, _dummy_traj())
        loss = torch.zeros((), requires_grad=False)
        _ = ctl.penalty_term(loss)
        ctl.dual_step()
    assert ctl.state.lam == 0.0


def test_grpo_with_lagrangian_runs_and_records_snapshot():
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    ctl = LagrangianController(
        cost_fn=lambda item, traj: len(traj.final_output or "") * 0.05,
        cfg=LagrangianConfig(cost_limit=0.2, init_lambda=0.0, lr_lambda=0.2),
    )
    cfg = GRPOTrainerConfig(n_iters=3, group_size=3, prompts_per_iter=1, lr=0.01,
                            max_new_tokens=4, log_every=1000, seed=0)
    trainer = GRPOTrainer(_backend(), env, rm, cfg=cfg, lagrangian=ctl)
    stats = trainer.train()
    assert len(stats.iters) == 3
    # every recorded iter must carry a 'lagrangian' snapshot
    for rec in stats.iters:
        assert "lagrangian" in rec
        assert rec["lagrangian"]["cost_limit"] == pytest.approx(0.2)


# helper -------------------------------------------------------------------

def _dummy_traj():
    from hermes_agentic_rl.core.types import Trajectory
    return Trajectory(
        task_id="x", prompt="x", steps=[], final_output="", finished_naturally=True,
        turns_used=0, metadata={},
    )
