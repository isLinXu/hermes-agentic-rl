"""Eval harness + version manager + A/B tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.eval import (
    EvalConfig,
    EvalHarness,
    VersionManager,
    paired_welch_t,
    run_ab,
)
from hermes_agentic_rl.eval.harness import leaderboard_markdown


def _backend(seed: int = 0) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(TinyBackendConfig(seed=seed, dim=32, n_heads=4, n_layers=2))


def test_eval_harness_produces_report():
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    h = EvalHarness(
        _backend(), env, rm, cfg=EvalConfig(n_rollouts=6, max_new_tokens=6, temperature=0.0, seed_base=0),
        name="policy_v1",
    )
    rep = h.run()
    assert rep.n_rollouts == 6
    assert 0.0 <= rep.success_rate <= 1.0
    assert rep.mean_reward >= 0.0
    assert "echo_reward" in rep.component_means


def test_leaderboard_markdown_multi_policy():
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    r1 = EvalHarness(
        _backend(seed=0), env, rm, EvalConfig(n_rollouts=4, seed_base=0), name="A"
    ).run()
    r2 = EvalHarness(
        _backend(seed=1), env, rm, EvalConfig(n_rollouts=4, seed_base=0), name="B"
    ).run()
    md = leaderboard_markdown([r1, r2])
    assert "| A |" in md and "| B |" in md


def test_paired_welch_t_detects_difference():
    b = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
    c = [0.8, 0.9, 0.7, 0.9, 0.85, 0.95, 0.8, 0.9]
    res = paired_welch_t(b, c)
    assert res.winner == "candidate"
    assert res.approx_p < 0.05
    assert res.mean_diff > 0


def test_paired_welch_t_rejects_tiny_diff():
    b = [0.4, 0.5, 0.6, 0.45, 0.55]
    c = [0.41, 0.49, 0.61, 0.46, 0.54]
    res = paired_welch_t(b, c)
    assert res.winner == "tie"


def test_run_ab_returns_reports_and_verdict():
    env = EchoTaskEnv(build_default_echo_dataset())
    rm = RewardManager([EchoRewardComponent(weight=1.0)])
    h1 = EvalHarness(
        _backend(seed=0), env, rm, EvalConfig(n_rollouts=6, seed_base=0), name="base"
    )
    h2 = EvalHarness(
        _backend(seed=1), env, rm, EvalConfig(n_rollouts=6, seed_base=0), name="cand"
    )
    b, c, res = run_ab(h1, h2)
    assert b.name == "base" and c.name == "cand"
    assert res.winner in {"baseline", "candidate", "tie"}


def test_version_manager_save_load_and_tags(tmp_path: Path):
    vm = VersionManager(tmp_path / "vm")
    backend = _backend()
    state = {k: v.detach().cpu() for k, v in backend.model.state_dict().items()}
    info = vm.save_version("v1", state, algo="grpo", metrics={"mean_reward": 0.3})
    assert info.name == "v1"
    vm.save_version("v2", state, algo="dpo", parent="v1", metrics={"mean_reward": 0.5})
    vm.promote("v2")
    assert vm.find_by_tag("production")[0].name == "v2"
    # reload a version's state_dict
    loaded = vm.load_version("v1")
    assert set(loaded.keys()) == set(state.keys())
    # demoting by promoting a different version should flip exclusive tag
    vm.promote("v1")
    assert vm.find_by_tag("production")[0].name == "v1"
