"""Integration-ish tests for MultiTurnAgentLoop, SimToolEnv, CurriculumEnv."""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.agent_loop.multi_turn_loop import MultiTurnAgentLoop
from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.envs.curriculum import CurriculumEnv
from hermes_agentic_rl.envs.echo_task_env import (
    EchoRewardComponent,
    EchoTaskEnv,
    build_default_echo_dataset,
)
from hermes_agentic_rl.envs.sim_tool_env import (
    DEFAULT_TOOLS,
    SimToolEnv,
    SimToolRewardComponent,
    build_sim_tool_dataset,
    safe_eval,
)


def _tiny(seed: int = 0, with_value: bool = False) -> TinyCausalLMBackend:
    return TinyCausalLMBackend(
        TinyBackendConfig(
            seed=seed, with_value_head=with_value, dim=16, n_heads=2, n_layers=1, max_len=64
        )
    )


def test_safe_eval_small_subset_only():
    assert safe_eval("3 + 4") == "7"
    assert safe_eval("2 * (3 + -4)") == "-2"
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('ls')")
    with pytest.raises(ValueError):
        safe_eval("pow(2, 3)")


def test_multi_turn_loop_emits_per_turn_metadata():
    backend = _tiny()
    loop = MultiTurnAgentLoop(
        backend=backend,
        tools=DEFAULT_TOOLS,
        max_turns=2,
        max_new_tokens_per_turn=6,
        temperature=1.0,
        seed=1,
    )
    out = asyncio.run(loop.run("What is 1 + 2?"))
    rl = out["metadata"]["rl"]
    assert rl["multi_turn"] is True
    assert "turns" in rl
    assert len(rl["turns"]) >= 1
    # flat views are consistent with per-turn concatenation
    flat = [tok for t in rl["turns"] for tok in t["response_ids"]]
    assert flat == list(rl["response_ids"])


def test_sim_tool_reward_when_answer_matches():
    dataset = build_sim_tool_dataset(n=4, seed=0)
    env = SimToolEnv(dataset)
    rm = RewardManager([SimToolRewardComponent(weight=1.0)])

    async def _run():
        await env.setup()
        item = await env.get_next_item()
        # fabricate a "perfect" trajectory by using the env's expected answer
        from hermes_agentic_rl.core.types import RolloutStep, Trajectory
        traj = Trajectory(
            task_id=item["task_id"],
            prompt=item["instruction"],
            steps=[
                RolloutStep(
                    turn_index=0,
                    tool_calls=[{"name": "calc", "arg": item["expr"]}],
                    tool_results=[{"name": "calc", "result": item["target"]}],
                )
            ],
            final_output=f"answer={item['target']}",
            finished_naturally=True,
            turns_used=1,
            metadata={},
        )
        return await rm.evaluate(item, traj, tool_context=None)

    summary = asyncio.run(_run())
    assert summary.final_score == pytest.approx(1.0, abs=1e-6)


def test_curriculum_promotes_once_threshold_met():
    l0 = EchoTaskEnv(build_default_echo_dataset())
    l1 = SimToolEnv(build_sim_tool_dataset(n=4, seed=0))
    changes: list[tuple[int, int]] = []
    env = CurriculumEnv(
        levels=[l0, l1],
        window=4,
        promote_threshold=0.5,
        on_level_change=lambda a, b: changes.append((a, b)),
    )
    assert env.current_level == 0
    for _ in range(4):
        env.observe(0.8)
    assert env.current_level == 1
    assert changes == [(0, 1)]
    snap = env.snapshot()
    assert snap["promotions"] == 1


def test_curriculum_demote_optional():
    l0 = EchoTaskEnv(build_default_echo_dataset())
    l1 = SimToolEnv(build_sim_tool_dataset(n=4, seed=0))
    env = CurriculumEnv(
        levels=[l0, l1], window=4, promote_threshold=0.5, allow_demote=True,
        demote_threshold=0.1,
    )
    # promote
    for _ in range(4):
        env.observe(0.9)
    assert env.current_level == 1
    # demote
    for _ in range(4):
        env.observe(0.0)
    assert env.current_level == 0
