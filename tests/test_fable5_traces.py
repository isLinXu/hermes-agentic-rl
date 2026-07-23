"""Tests for Fable5TraceEnv, Fable5TraceReward, and data loading."""

from __future__ import annotations

import asyncio
import json

import pytest

from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.envs.fable5_traces import (
    Fable5TraceConfig,
    Fable5TraceEnv,
    Fable5TraceReward,
    _extract_tool_call_from_completion,
    _flat_row_to_item,
    _normalize_text,
    _text_similarity,
)

# --- Fixtures ---

SYNTHETIC_ITEMS = [
    {
        "task_id": "fable5::test-1",
        "instruction": "Read the configuration file at /etc/app/config.yaml",
        "target_response": (
            '{"name": "Read", "arguments": {"file_path": "/etc/app/config.yaml"}}'
        ),
        "target_response_full": (
            '{"name": "Read", "arguments": {"file_path": "/etc/app/config.yaml"}}'
        ),
        "output_type": "tool_use",
        "tool_name": "Read",
        "tool_arguments": {"file_path": "/etc/app/config.yaml"},
        "cot": "",
        "completion": '{"name": "Read", "arguments": {"file_path": "/etc/app/config.yaml"}}',
        "model": "claude-fable-5",
        "session": "test-session",
        "source_file": "",
        "origin": "test",
    },
    {
        "task_id": "fable5::test-2",
        "instruction": "Summarize the build system",
        "target_response": "The project uses CMake with Ninja generator.",
        "target_response_full": "The project uses CMake with Ninja generator.",
        "output_type": "text",
        "tool_name": None,
        "tool_arguments": None,
        "cot": "",
        "completion": "The project uses CMake with Ninja generator.",
        "model": "claude-fable-5",
        "session": "test-session",
        "source_file": "",
        "origin": "test",
    },
]

FLAT_ROW_TOOL_USE = {
    "uid": "test-uid-1",
    "source_file": "test.jsonl",
    "session": "test-session",
    "model": "claude-fable-5",
    "context": "Read the file at /tmp/test.py",
    "cot": "I need to read the file first.",
    "output_type": "tool_use",
    "output": {"tool": "Read", "input": {"file_path": "/tmp/test.py"}},
    "completion": (
        'I need to read the file first.\n'
        '{"tool": "Read", "input": {"file_path": "/tmp/test.py"}}'
    ),
    "origin": "local",
}

FLAT_ROW_TEXT = {
    "uid": "test-uid-2",
    "source_file": "test.jsonl",
    "session": "test-session",
    "model": "claude-fable-5",
    "context": "What is the project structure?",
    "cot": "Let me summarize.",
    "output_type": "text",
    "output": "The project has src/ and tests/ directories.",
    "completion": "Let me summarize.\nThe project has src/ and tests/ directories.",
    "origin": "local",
}


def _make_trajectory(output: str) -> Trajectory:
    return Trajectory(
        task_id="test",
        prompt="test",
        steps=[],
        final_output=output,
        finished_naturally=True,
        turns_used=1,
    )


# --- Helper tests ---

def test_normalize_text():
    assert _normalize_text("  hello   world  ") == "hello world"
    assert _normalize_text("hello\nworld") == "hello world"


def test_text_similarity_exact():
    assert _text_similarity("hello world", "hello world") == 1.0


def test_text_similarity_partial():
    score = _text_similarity("hello world", "hello there")
    assert 0.0 < score < 1.0


def test_text_similarity_empty_target():
    assert _text_similarity("prediction", "") == 0.0


def test_extract_tool_call_json():
    pred = '{"name": "Read", "arguments": {"file_path": "/tmp/test"}}'
    result = _extract_tool_call_from_completion(pred)
    assert result is not None
    assert result["name"] == "Read"


def test_extract_tool_call_embedded():
    pred = 'Some reasoning...\n{"tool": "Bash", "input": {"command": "ls"}}'
    result = _extract_tool_call_from_completion(pred)
    assert result is not None
    assert result.get("tool") == "Bash"


def test_extract_tool_call_none():
    assert _extract_tool_call_from_completion("just plain text") is None


# --- Flat row converter tests ---

def test_flat_row_to_item_tool_use():
    cfg = Fable5TraceConfig()
    item = _flat_row_to_item(FLAT_ROW_TOOL_USE, cfg)
    assert item["task_id"] == "fable5::test-uid-1"
    assert item["output_type"] == "tool_use"
    assert item["tool_name"] == "Read"
    assert item["tool_arguments"] == {"file_path": "/tmp/test.py"}
    assert "Read" in item["target_response"]
    assert item["instruction"] == "Read the file at /tmp/test.py"


def test_flat_row_to_item_text():
    cfg = Fable5TraceConfig()
    item = _flat_row_to_item(FLAT_ROW_TEXT, cfg)
    assert item["output_type"] == "text"
    assert item["tool_name"] is None
    assert "src/ and tests/" in item["target_response"]


def test_flat_row_to_item_skip_short():
    cfg = Fable5TraceConfig(min_target_chars=1000)
    item = _flat_row_to_item(FLAT_ROW_TEXT, cfg)
    assert item == {}


# --- Reward tests ---

def test_reward_init_valid_modes():
    for mode in ("similarity", "tool_call", "hybrid"):
        r = Fable5TraceReward(reward_mode=mode)
        assert r.reward_mode == mode


def test_reward_init_invalid_mode():
    with pytest.raises(ValueError, match="reward_mode"):
        Fable5TraceReward(reward_mode="invalid")


def test_reward_exact_match():
    reward = Fable5TraceReward()
    item = SYNTHETIC_ITEMS[0]
    traj = _make_trajectory(item["target_response"])
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert result.score == 1.0
    assert result.metadata["exact_match"] is True


def test_reward_tool_use_partial_match():
    reward = Fable5TraceReward(reward_mode="hybrid")
    item = SYNTHETIC_ITEMS[0]
    # Correct tool name, wrong arguments
    traj = _make_trajectory('{"name": "Read", "arguments": {"file_path": "/wrong/path"}}')
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert 0.0 < result.score < 1.0
    assert result.metadata["name_match"] == 1.0
    assert result.metadata["args_match"] < 1.0


def test_reward_tool_use_wrong_name():
    reward = Fable5TraceReward(reward_mode="tool_call", output_type_bonus=0.0)
    item = SYNTHETIC_ITEMS[0]
    traj = _make_trajectory('{"name": "Write", "arguments": {"file_path": "/etc/app/config.yaml"}}')
    result = asyncio.run(reward.evaluate(item, traj, None))
    # name_match=0, args_match=1.0 -> tool_score=0.5
    assert result.score == 0.5
    assert result.metadata["name_match"] == 0.0
    assert result.metadata["args_match"] == 1.0


def test_reward_text_similarity_mode():
    reward = Fable5TraceReward(reward_mode="similarity")
    item = SYNTHETIC_ITEMS[1]
    traj = _make_trajectory("The project uses CMake.")
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert 0.0 < result.score < 1.0


def test_reward_missing_target():
    reward = Fable5TraceReward()
    item = {"target_response": "", "output_type": "text"}
    traj = _make_trajectory("some output")
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert result.score == 0.0


def test_reward_output_type_bonus():
    reward = Fable5TraceReward(
        reward_mode="hybrid",
        output_type_bonus=0.2,
    )
    item = SYNTHETIC_ITEMS[0]
    # Prediction is a proper tool call -> should get type_bonus
    traj = _make_trajectory('{"name": "Read", "arguments": {"file_path": "/wrong"}}')
    result = asyncio.run(reward.evaluate(item, traj, None))
    assert result.metadata["type_bonus"] == 0.2


# --- Environment tests ---

def test_env_init():
    env = Fable5TraceEnv(SYNTHETIC_ITEMS)
    assert len(env.items) == 2
    assert env.reward_component.name == "fable5_trace_reward"


def test_env_init_empty():
    with pytest.raises(ValueError, match="at least one item"):
        Fable5TraceEnv([])


def test_env_get_next_item():
    env = Fable5TraceEnv(SYNTHETIC_ITEMS)
    item0 = asyncio.run(env.get_next_item())
    item1 = asyncio.run(env.get_next_item())
    item2 = asyncio.run(env.get_next_item())  # wraps around
    assert item0["task_id"] == "fable5::test-1"
    assert item1["task_id"] == "fable5::test-2"
    assert item2["task_id"] == "fable5::test-1"


def test_env_format_prompt():
    env = Fable5TraceEnv(SYNTHETIC_ITEMS)
    prompt = env.format_prompt(SYNTHETIC_ITEMS[0])
    assert "Read the configuration file" in prompt


def test_env_compute_reward():
    env = Fable5TraceEnv(SYNTHETIC_ITEMS)
    item = SYNTHETIC_ITEMS[0]
    traj = _make_trajectory(item["target_response"])
    results = asyncio.run(env.compute_reward(item, traj, None))
    assert len(results) == 1
    assert results[0].score == 1.0


def test_env_build_supervised_samples():
    env = Fable5TraceEnv(SYNTHETIC_ITEMS)
    samples = env.build_supervised_samples(SYNTHETIC_ITEMS[0])
    assert len(samples) == 1
    assert "Read" in samples[0].response
    assert samples[0].metadata["output_type"] == "tool_use"
    assert samples[0].metadata["tool_name"] == "Read"


def test_env_build_supervised_samples_empty():
    env = Fable5TraceEnv(SYNTHETIC_ITEMS)
    samples = env.build_supervised_samples({"instruction": "", "target_response": ""})
    assert samples == []


def test_env_setup_resets_cursor():
    env = Fable5TraceEnv(SYNTHETIC_ITEMS)
    asyncio.run(env.get_next_item())
    asyncio.run(env.get_next_item())
    asyncio.run(env.setup())
    item = asyncio.run(env.get_next_item())
    assert item["task_id"] == "fable5::test-1"


# --- Config tests ---

def test_config_defaults():
    cfg = Fable5TraceConfig()
    assert cfg.repo_id == "Glint-Research/Fable-5-traces"
    assert cfg.config_name == "pi_agent"
    assert cfg.use_flat_jsonl is True
    assert cfg.reward_mode == "hybrid"
    assert cfg.limit == 500


def test_env_from_config():
    cfg_dict = {
        "limit": 2,
        "shuffle": False,
        "use_flat_jsonl": True,
    }
    env = Fable5TraceEnv.from_config(cfg_dict)
    assert len(env.items) > 0
    assert isinstance(env, Fable5TraceEnv)
