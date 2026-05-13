"""CLI integration: train-rl with a pre-trained RewardModel head attached.

Validates that:
  (a) `_build_reward_model_component` builds a working RewardModelComponent
  (b) Loading a `head_path` checkpoint into the RM succeeds.
  (c) The component produces a finite reward via evaluate() on a trajectory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.cli.train_rl import _build_reward_model_component
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.reward_model import RewardModel


def test_build_reward_model_component_disabled_by_default() -> None:
    assert _build_reward_model_component({}) is None
    assert _build_reward_model_component({"reward_model": {"enabled": False}}) is None


def test_build_reward_model_component_loads_head(tmp_path: Path) -> None:
    # Save a compatible RM head (bonus: verify round-trip).
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    rm = RewardModel(backend, freeze_base=True)
    head_path = tmp_path / "rm_head.pt"
    torch.save(rm.head.state_dict(), head_path)

    cfg = {
        "reward_model": {
            "enabled": True,
            "backend": "tiny",
            "dim": 16,
            "n_heads": 2,
            "n_layers": 1,
            "seed": 0,
            "head_path": str(head_path),
            "weight": 0.5,
            "freeze_base": True,
        }
    }
    component = _build_reward_model_component(cfg)
    assert component is not None
    assert component.weight == 0.5
    assert component.name == "reward_model"


def test_reward_model_component_evaluates_trajectory() -> None:
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    rm = RewardModel(backend, freeze_base=True)
    from hermes_agentic_rl.rewards.reward_model import RewardModelComponent

    comp = RewardModelComponent(rm, weight=1.0)
    traj = Trajectory(
        task_id="t1",
        prompt="hello",
        steps=[],
        final_output="test",
        finished_naturally=True,
        turns_used=1,
        metadata={
            "runtime": {
                "rl": {
                    "prompt_ids": [1, 2, 3],
                    "response_ids": [4, 5],
                    "old_logprobs": [-0.1, -0.2],
                }
            }
        },
    )
    result = asyncio.run(comp.evaluate({}, traj, None))
    assert result.name == "reward_model"
    # finite score
    assert result.score == result.score  # not NaN
    assert isinstance(result.score, float)


def test_reward_model_component_handles_missing_rl_metadata() -> None:
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    rm = RewardModel(backend, freeze_base=True)
    from hermes_agentic_rl.rewards.reward_model import RewardModelComponent

    comp = RewardModelComponent(rm, weight=1.0)
    traj = Trajectory(
        task_id="t1",
        prompt="hello",
        steps=[],
        final_output="test",
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )
    result = asyncio.run(comp.evaluate({}, traj, None))
    assert result.score == 0.0
