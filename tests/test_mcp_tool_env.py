"""MCP tool environment adapter tests: registry, tool execution, reward."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

from hermes_agentic_rl.core.types import RolloutStep, Trajectory
from hermes_agentic_rl.envs.mcp_tool_env import (
    MCPToolEnv,
    MCPToolRegistry,
    MCPToolReward,
    MCPToolSpec,
    MCPToolTask,
    build_mcp_tool_dataset,
)


# ── Test handlers ──

def _echo_handler(arg: str) -> str:
    return f"echo:{arg}"


def _upper_handler(arg: str) -> str:
    return arg.upper()


def _fail_handler(arg: str) -> str:
    raise RuntimeError("intentional failure")


# ── Registry tests ──


def test_registry_register_and_lookup():
    registry = MCPToolRegistry()
    registry.register(MCPToolSpec(
        name="echo",
        description="Echoes the input",
        handler=_echo_handler,
    ))
    assert "echo" in registry
    assert registry.num_tools == 1
    spec = registry.get_spec("echo")
    assert spec is not None
    assert spec.name == "echo"


def test_registry_register_duplicate_raises():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_simple("echo", "echo", _echo_handler)


def test_registry_to_tool_mapping():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    registry.register_simple("upper", "uppercase", _upper_handler)
    mapping = registry.to_tool_mapping()
    assert "echo" in mapping
    assert "upper" in mapping
    assert len(mapping) == 2


def test_registry_tool_fn_executes():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    tool_fn = registry["echo"]
    result = tool_fn("echo", "hello")
    assert result == "echo:hello"


def test_registry_tool_fn_error_captured():
    registry = MCPToolRegistry()
    registry.register_simple("fail", "always fails", _fail_handler)
    tool_fn = registry["fail"]
    result = tool_fn("fail", "anything")
    assert result.startswith("error:RuntimeError:")


def test_registry_tool_fn_json_arg():
    registry = MCPToolRegistry()
    registry.register_simple(
        "json_tool", "JSON tool",
        lambda arg: f"got:{arg}",
        param_schema={"type": "string"},
    )
    tool_fn = registry["json_tool"]
    # JSON arg path
    result = tool_fn("json_tool", '{"key": "value"}')
    assert "got:" in result
    # Non-JSON arg path
    result2 = tool_fn("json_tool", "plain_string")
    assert "got:" in result2


def test_registry_prompt_descriptions():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "Echoes input", _echo_handler)
    registry.register_simple("upper", "Uppercases input", _upper_handler)
    desc = registry.to_prompt_descriptions()
    assert "echo" in desc
    assert "upper" in desc
    assert "Echoes input" in desc


# ── Dataset builder tests ──


def test_build_mcp_tool_dataset():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    tasks = build_mcp_tool_dataset(registry, n=5, seed=0)
    assert len(tasks) == 5
    assert all(isinstance(t, MCPToolTask) for t in tasks)
    assert all(t.expected_tool == "echo" for t in tasks)
    # Expected result should come from handler.
    assert tasks[0].expected_result is not None
    assert tasks[0].expected_result.startswith("echo:")


def test_build_mcp_tool_dataset_empty_registry():
    registry = MCPToolRegistry()
    tasks = build_mcp_tool_dataset(registry, n=5)
    assert len(tasks) == 0


# ── Environment tests ──


def _make_trajectory(tool_calls: list[dict], tool_results: list[dict]) -> Trajectory:
    step = RolloutStep(
        turn_index=0,
        tool_calls=tool_calls,
        tool_results=tool_results,
    )
    return Trajectory(
        task_id="test",
        prompt="test prompt",
        steps=[step],
        final_output="test output",
        finished_naturally=True,
        turns_used=1,
    )


def test_mcp_tool_env_get_next_item():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    tasks = build_mcp_tool_dataset(registry, n=3, seed=0)
    env = MCPToolEnv(registry, tasks)

    item = asyncio.run(env.get_next_item())
    assert "instruction" in item
    assert "expected_tool" in item
    assert "tools_description" in item
    assert "echo" in item["tools_description"]


def test_mcp_tool_env_format_prompt():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "Echoes input", _echo_handler)
    tasks = [MCPToolTask(
        task_id="t1",
        instruction="Echo hello",
        expected_tool="echo",
        expected_arg="hello",
        expected_result="echo:hello",
    )]
    env = MCPToolEnv(registry, tasks)
    item = asyncio.run(env.get_next_item())
    prompt = env.format_prompt(item)
    assert "Available tools" in prompt
    assert "Echo hello" in prompt


def test_mcp_tool_env_reward_correct_tool():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    tasks = [MCPToolTask(
        task_id="t1",
        instruction="Echo hello",
        expected_tool="echo",
        expected_arg="hello",
        expected_result="echo:hello",
    )]
    env = MCPToolEnv(registry, tasks)

    traj = _make_trajectory(
        tool_calls=[{"name": "echo", "arg": "hello"}],
        tool_results=[{"result": "echo:hello"}],
    )
    rewards = asyncio.run(env.compute_reward(dataclasses.asdict(tasks[0]), traj, None))
    assert len(rewards) == 3
    total = sum(r.score * r.weight for r in rewards)
    assert total == pytest.approx(1.0)  # full score


def test_mcp_tool_env_reward_wrong_tool():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    tasks = [MCPToolTask(
        task_id="t1",
        instruction="Echo hello",
        expected_tool="echo",
        expected_arg="hello",
        expected_result="echo:hello",
    )]
    env = MCPToolEnv(registry, tasks)

    traj = _make_trajectory(
        tool_calls=[{"name": "upper", "arg": "hello"}],
        tool_results=[{"result": "HELLO"}],
    )
    rewards = asyncio.run(env.compute_reward(dataclasses.asdict(tasks[0]), traj, None))
    total = sum(r.score * r.weight for r in rewards)
    assert total == pytest.approx(0.0)  # zero score


def test_mcp_tool_env_reward_correct_tool_wrong_arg():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    tasks = [MCPToolTask(
        task_id="t1",
        instruction="Echo hello",
        expected_tool="echo",
        expected_arg="hello",
        expected_result="echo:hello",
    )]
    env = MCPToolEnv(registry, tasks)

    traj = _make_trajectory(
        tool_calls=[{"name": "echo", "arg": "world"}],
        tool_results=[{"result": "echo:world"}],
    )
    rewards = asyncio.run(env.compute_reward(dataclasses.asdict(tasks[0]), traj, None))
    # tool called (0.4) + arg wrong (0.0) + result wrong (0.0) = 0.4
    total = sum(r.score * r.weight for r in rewards)
    assert total == pytest.approx(0.4)


def test_mcp_tool_env_supervised_samples():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    tasks = [MCPToolTask(
        task_id="t1",
        instruction="Echo hello",
        expected_tool="echo",
        expected_arg="hello",
        expected_result="echo:hello",
    )]
    env = MCPToolEnv(registry, tasks)
    item = {
        "instruction": "Echo hello",
        "expected_tool": "echo",
        "expected_arg": "hello",
        "tools_description": registry.to_prompt_descriptions(),
    }
    samples = env.build_supervised_samples(item)
    assert len(samples) == 1
    assert "echo" in samples[0].response
    assert "hello" in samples[0].response


# ── MCPToolReward (BaseReward) tests ──


def test_mcp_tool_reward_correct():
    registry = MCPToolRegistry()
    registry.register_simple("echo", "echo", _echo_handler)
    reward = MCPToolReward()
    item = {
        "expected_tool": "echo",
        "expected_arg": "hello",
        "expected_result": "echo:hello",
    }
    traj = _make_trajectory(
        tool_calls=[{"name": "echo", "arg": "hello"}],
        tool_results=[{"result": "echo:hello"}],
    )
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert result.score == pytest.approx(1.0)
    assert "tool_called" in result.reason


def test_mcp_tool_reward_no_tool():
    reward = MCPToolReward()
    item = {"expected_tool": "echo", "expected_arg": "x", "expected_result": "y"}
    traj = _make_trajectory(tool_calls=[], tool_results=[])
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert result.score == pytest.approx(0.0)
    assert "tool_not_called" in result.reason


def test_mcp_tool_reward_partial():
    reward = MCPToolReward()
    item = {
        "expected_tool": "echo",
        "expected_arg": "hello",
        "expected_result": "echo:hello",
    }
    traj = _make_trajectory(
        tool_calls=[{"name": "echo", "arg": "wrong"}],
        tool_results=[{"result": "echo:wrong"}],
    )
    result = asyncio.run(reward.evaluate(item, traj, None))
    # tool called (0.4) but arg wrong and result wrong
    assert result.score == pytest.approx(0.4)
