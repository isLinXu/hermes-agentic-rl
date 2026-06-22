from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType

import pytest

from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.envs.hermes_reasoning_traces import (
    HermesReasoningTraceEnv,
    HermesReasoningTraceReward,
    load_hermes_reasoning_trace_turns,
)


class _FakeDataset(list):
    def shuffle(self, seed: int = 0):
        _ = seed
        return _FakeDataset(list(reversed(self)))


class _FakeDatasets(ModuleType):
    def __init__(self, rows: list[dict]) -> None:
        super().__init__("datasets")
        self._rows = rows
        self.calls: list[dict] = []

    def load_dataset(self, *args, **kwargs):
        self.calls.append({"args": args, **kwargs})
        return _FakeDataset(self._rows)


def _sample_trace() -> dict:
    return {
        "task_id": "trace-1",
        "category": "browser",
        "subcategory": "search",
        "task": "Find the weather in SF",
        "tools": [{"name": "search", "description": "Search the web"}],
        "conversations": [
            {"from": "system", "value": "You are Hermes."},
            {"from": "human", "value": "Find the weather in SF."},
            {"from": "gpt", "value": "<tool_call>search(weather sf)</tool_call>"},
            {"from": "tool", "value": "72F and sunny", "name": "search"},
            {"from": "gpt", "value": "It is 72F and sunny in San Francisco."},
        ],
    }


def test_load_hermes_reasoning_trace_turns_expands_assistant_messages(monkeypatch) -> None:
    fake_datasets = _FakeDatasets([_sample_trace()])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    items = load_hermes_reasoning_trace_turns(
        repo_id="lambda/hermes-agent-reasoning-traces",
        config_name="kimi",
        split="train",
        limit=1,
    )

    assert len(items) == 2
    assert items[0]["task_id"] == "trace-1::assistant::0"
    assert items[0]["target_response"] == "<tool_call>search(weather sf)</tool_call>"
    assert "Available tools:" in items[0]["instruction"]
    assert items[0]["instruction"].endswith("Assistant:")
    assert "Tool[search]:" in items[1]["instruction"]
    assert "72F and sunny" in items[1]["instruction"]
    assert items[1]["assistant_turn_index"] == 1
    assert fake_datasets.calls[0]["name"] == "kimi"
    assert fake_datasets.calls[0]["args"][0] == "lambda/hermes-agent-reasoning-traces"


def test_hermes_reasoning_trace_env_builds_supervised_sample_and_reward(monkeypatch) -> None:
    fake_datasets = _FakeDatasets([_sample_trace()])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    env = HermesReasoningTraceEnv.from_hf_dataset(
        {
            "dataset_name": "lambda/hermes-agent-reasoning-traces",
            "dataset_config": "kimi",
            "dataset_split": "train",
            "dataset_limit": 1,
        }
    )

    item = asyncio.run(env.get_next_item())
    samples = env.build_supervised_samples(item)
    assert len(samples) == 1
    assert samples[0].response == item["target_response"]

    reward = HermesReasoningTraceReward(weight=1.0)
    summary = asyncio.run(
        reward.evaluate(
            item,
            Trajectory(
                task_id=item["task_id"],
                prompt=item["instruction"],
                steps=[],
                final_output=item["target_response"],
                finished_naturally=True,
                turns_used=1,
            ),
            tool_context=None,
        )
    )
    assert summary.score == 1.0
    assert summary.metadata["exact_match"] is True


def test_load_hermes_reasoning_trace_turns_truncates_long_prompt(monkeypatch) -> None:
    long_trace = _sample_trace()
    long_trace["conversations"][0]["value"] = "System prompt " + ("A" * 400)
    fake_datasets = _FakeDatasets([long_trace])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    items = load_hermes_reasoning_trace_turns(
        repo_id="lambda/hermes-agent-reasoning-traces",
        config_name="kimi",
        split="train",
        limit=1,
        max_prompt_chars=120,
    )

    assert items
    assert len(items[0]["instruction"]) <= 120
    assert "[Earlier context truncated]" in items[0]["instruction"]


def test_load_hermes_reasoning_trace_turns_from_local_dataset_path(tmp_path) -> None:
    dataset_path = tmp_path / "traces.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "trace-1",
                        "category": "browser",
                        "subcategory": "search",
                        "task": "Find the weather in SF",
                        "tools": [{"name": "search"}],
                        "conversations": [
                            {"from": "human", "value": "Find the weather in SF."},
                            {
                                "from": "gpt",
                                "value": "<tool_call>search(weather sf)</tool_call>",
                            },
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_hermes_reasoning_trace_turns(
        dataset_path=str(dataset_path),
        limit=1,
    )

    assert len(items) == 1
    assert items[0]["task_id"] == "trace-1::assistant::0"
    assert items[0]["target_response"] == "<tool_call>search(weather sf)</tool_call>"


def test_load_hermes_reasoning_trace_turns_from_local_parquet_path(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    parquet_path = tmp_path / "traces.parquet"
    row = _sample_trace()
    row["tools"] = json.dumps(row["tools"])
    row["conversations"] = row["conversations"][:3]
    table = pa.Table.from_pylist([row])
    pq.write_table(table, parquet_path)

    items = load_hermes_reasoning_trace_turns(
        dataset_path=str(parquet_path),
        limit=1,
    )

    assert len(items) == 1
    assert items[0]["task_id"] == "trace-1::assistant::0"
    assert items[0]["target_response"] == "<tool_call>search(weather sf)</tool_call>"
    assert '"name": "search"' in items[0]["instruction"]
    assert isinstance(items[0]["tools"], list)


def test_load_hermes_reasoning_trace_turns_filters_targets(monkeypatch) -> None:
    trace = _sample_trace()
    trace["conversations"].append({"from": "gpt", "value": "A very long final answer" * 50})
    fake_datasets = _FakeDatasets([trace])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    items = load_hermes_reasoning_trace_turns(
        repo_id="lambda/hermes-agent-reasoning-traces",
        config_name="kimi",
        split="train",
        limit=1,
        max_target_chars=80,
        require_target_substring="<tool_call",
        max_assistant_turn_index=0,
    )

    assert len(items) == 1
    assert items[0]["assistant_turn_index"] == 0
    assert items[0]["target_response"] == "<tool_call>search(weather sf)</tool_call>"


def test_load_hermes_reasoning_trace_turns_adds_tool_call_hint_and_response_prefix(
    monkeypatch,
) -> None:
    target = (
        "<think>\n</think>\n<tool_call>\n"
        '{"name":"terminal","arguments":{"command":"python clean.py"}}'
        "\n</tool_call>"
    )
    trace = _sample_trace()
    trace["tools"] = [{"name": "terminal", "description": "Run shell commands"}]
    trace["conversations"] = [
        {"from": "human", "value": "Clean the file."},
        {"from": "gpt", "value": target},
    ]
    fake_datasets = _FakeDatasets([trace])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    prefix = "<think>\n</think>\n<tool_call>\n"
    items = load_hermes_reasoning_trace_turns(
        repo_id="lambda/hermes-agent-reasoning-traces",
        config_name="kimi",
        split="train",
        limit=1,
        tool_call_format_hint=True,
        assistant_response_prefix=prefix,
    )
    env = HermesReasoningTraceEnv(items, assistant_response_prefix=prefix)
    samples = env.build_supervised_samples(items[0])

    assert "Response format guidance:" in items[0]["instruction"]
    assert "terminal" in items[0]["instruction"]
    assert items[0]["response_prefix"] == prefix
    assert items[0]["target_response"].startswith('{"name":"terminal"')
    assert env.format_prompt(items[0]).endswith(prefix)
    assert samples[0].instruction.endswith(prefix)
    assert samples[0].response.startswith('{"name":"terminal"')


def test_terminal_command_adapter_targets_command_only(monkeypatch) -> None:
    target = (
        "<think>\n</think>\n<tool_call>\n"
        '{"name": "terminal", "arguments": {"command": "python clean.py"}}'
        "\n</tool_call>"
    )
    trace = _sample_trace()
    trace["tools"] = [{"name": "terminal", "description": "Run shell commands"}]
    trace["conversations"] = [
        {"from": "human", "value": "Clean the file."},
        {"from": "gpt", "value": target},
    ]
    fake_datasets = _FakeDatasets([trace])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    items = load_hermes_reasoning_trace_turns(
        repo_id="lambda/hermes-agent-reasoning-traces",
        config_name="kimi",
        split="train",
        limit=1,
        tool_call_format_hint=True,
        assistant_response_adapter="terminal_command_tool_call",
    )
    env = HermesReasoningTraceEnv(
        items,
        assistant_response_adapter="terminal_command_tool_call",
    )
    samples = env.build_supervised_samples(items[0])

    assert len(items) == 1
    assert "Return only the terminal command string" in items[0]["instruction"]
    assert items[0]["target_response"] == "python clean.py"
    assert items[0]["target_response_full"] == target
    assert items[0]["response_adapter"] == "terminal_command_tool_call"
    assert samples[0].response == "python clean.py"


def test_terminal_command_adapter_can_train_stop_suffix(monkeypatch) -> None:
    target = (
        '<tool_call>{"name": "terminal", "arguments": {"command": "python clean.py"}}</tool_call>'
    )
    trace = _sample_trace()
    trace["tools"] = [{"name": "terminal", "description": "Run shell commands"}]
    trace["conversations"] = [
        {"from": "human", "value": "Clean the file."},
        {"from": "gpt", "value": target},
    ]
    fake_datasets = _FakeDatasets([trace])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    items = load_hermes_reasoning_trace_turns(
        repo_id="lambda/hermes-agent-reasoning-traces",
        config_name="kimi",
        split="train",
        limit=1,
        assistant_response_adapter="terminal_command_tool_call",
        assistant_response_suffix="\n",
    )
    env = HermesReasoningTraceEnv(
        items,
        assistant_response_adapter="terminal_command_tool_call",
        assistant_response_suffix="\n",
    )
    samples = env.build_supervised_samples(items[0])

    assert items[0]["target_response"] == "python clean.py\n"
    assert samples[0].response == "python clean.py\n"


def _trajectory_for_output(item: dict, final_output: str) -> Trajectory:
    return Trajectory(
        task_id=item["task_id"],
        prompt=item.get("instruction", ""),
        steps=[],
        final_output=final_output,
        finished_naturally=True,
        turns_used=1,
    )


def test_hermes_reasoning_trace_reward_structured_tool_call_json_match() -> None:
    target = (
        "<think>\n</think>\n<tool_call>\n"
        '{"name":"terminal","arguments":{"command":"python csv_cleaner.py data.csv"}}'
        "\n</tool_call>"
    )
    item = {
        "task_id": "trace-json::assistant::0",
        "instruction": "Clean the CSV.\n\nAssistant:",
        "target_response": target,
    }
    reward = HermesReasoningTraceReward(weight=1.0, reward_mode="tool_call")

    result = asyncio.run(
        reward.evaluate(item, _trajectory_for_output(item, target), tool_context=None)
    )

    assert result.score == 1.0
    assert result.metadata["exact_match"] is True
    assert result.metadata["tool_name_match"] == 1.0
    assert result.metadata["argument_key_overlap"] == 1.0


def test_hermes_reasoning_trace_reward_penalizes_missing_tool_call_block() -> None:
    target = (
        "<tool_call>"
        '{"name":"terminal","arguments":{"command":"python csv_cleaner.py data.csv"}}'
        "</tool_call>"
    )
    item = {
        "task_id": "trace-json::assistant::0",
        "instruction": "Clean the CSV.\n\nAssistant:",
        "target_response": target,
    }
    reward = HermesReasoningTraceReward(weight=1.0, reward_mode="tool_call")

    result = asyncio.run(
        reward.evaluate(
            item,
            _trajectory_for_output(item, "I will run python csv_cleaner.py data.csv."),
            tool_context=None,
        )
    )

    assert result.score == 0.0
    assert result.metadata["prediction_tool_call_count"] == 0


def test_hermes_reasoning_trace_reward_penalizes_wrong_tool_name() -> None:
    target = (
        "<tool_call>"
        '{"name":"terminal","arguments":{"command":"python csv_cleaner.py data.csv"}}'
        "</tool_call>"
    )
    prediction = (
        "<tool_call>"
        '{"name":"browser","arguments":{"command":"python csv_cleaner.py data.csv"}}'
        "</tool_call>"
    )
    item = {
        "task_id": "trace-json::assistant::0",
        "instruction": "Clean the CSV.\n\nAssistant:",
        "target_response": target,
    }
    reward = HermesReasoningTraceReward(weight=1.0, reward_mode="tool_call")

    result = asyncio.run(
        reward.evaluate(item, _trajectory_for_output(item, prediction), tool_context=None)
    )

    assert 0.0 < result.score < 1.0
    assert result.metadata["tool_name_match"] == 0.0
    assert result.metadata["argument_value_similarity"] == 1.0


def test_hermes_reasoning_trace_reward_legacy_exact_match_stays_one() -> None:
    target = "<tool_call>search(weather sf)</tool_call>"
    item = {
        "task_id": "trace-legacy::assistant::0",
        "instruction": "Find the weather.\n\nAssistant:",
        "target_response": target,
    }
    reward = HermesReasoningTraceReward(weight=1.0, reward_mode="tool_call")

    result = asyncio.run(
        reward.evaluate(item, _trajectory_for_output(item, target), tool_context=None)
    )

    assert result.score == 1.0
    assert result.metadata["exact_match"] is True


def test_hermes_reasoning_trace_reward_scores_prefilled_suffix_against_full_target() -> None:
    prefix = "<think>\n</think>\n<tool_call>\n"
    suffix = '{"name":"terminal","arguments":{"command":"python clean.py"}}\n</tool_call>'
    item = {
        "task_id": "trace-prefix::assistant::0",
        "instruction": "Clean the file.\n\nAssistant:",
        "target_response": suffix,
        "target_response_full": prefix + suffix,
        "response_prefix": prefix,
    }
    reward = HermesReasoningTraceReward(weight=1.0, reward_mode="tool_call")

    result = asyncio.run(
        reward.evaluate(item, _trajectory_for_output(item, suffix), tool_context=None)
    )

    assert result.score == 1.0
    assert result.metadata["exact_match"] is True
    assert result.metadata["tool_name_match"] == 1.0


def test_hermes_reasoning_trace_reward_wraps_terminal_command_adapter() -> None:
    target = (
        "<think>\n</think>\n<tool_call>\n"
        '{"name": "terminal", "arguments": {"command": "python clean.py"}}'
        "\n</tool_call>"
    )
    item = {
        "task_id": "trace-command-adapter::assistant::0",
        "instruction": "Clean the file.\n\nAssistant:",
        "target_response": "python clean.py",
        "target_response_full": target,
        "response_adapter": "terminal_command_tool_call",
    }
    reward = HermesReasoningTraceReward(weight=1.0, reward_mode="tool_call")

    result = asyncio.run(
        reward.evaluate(
            item,
            _trajectory_for_output(item, "python clean.py"),
            tool_context=None,
        )
    )

    assert result.score == 1.0
    assert result.metadata["tool_call_parse_ok"] == 1.0
    assert result.metadata["tool_name_match"] == 1.0
    assert result.metadata["argument_key_overlap"] == 1.0
    assert result.metadata["argument_value_similarity"] == 1.0
    assert result.metadata["response_adapter"] == "terminal_command_tool_call"


def test_hermes_reasoning_trace_reward_gives_partial_structure_credit() -> None:
    item = {
        "task_id": "trace-partial::assistant::0",
        "instruction": "Clean the file.\n\nAssistant:",
        "target_response": '<tool_call>{"name":"terminal","arguments":{}}</tool_call>',
    }
    reward = HermesReasoningTraceReward(weight=1.0, reward_mode="tool_call")

    result = asyncio.run(
        reward.evaluate(
            item,
            _trajectory_for_output(item, '<tool_call>{"name":"terminal"'),
            tool_context=None,
        )
    )

    assert 0.0 < result.score < 1.0
    assert result.metadata["tool_call_present"] == 1.0
    assert result.metadata["partial_tool_call_score"] > 0.0


def test_hermes_reasoning_trace_reward_penalizes_invalid_argument_json() -> None:
    target = '<tool_call>{"name":"terminal","arguments":{"command":"python clean.py"}}</tool_call>'
    prediction = (
        "<tool_call>"
        '{"name":"terminal","arguments":"{\\"command\\": \\"python clean.py\\""}'
        "</tool_call>"
    )
    item = {
        "task_id": "trace-invalid-args::assistant::0",
        "instruction": "Clean the file.\n\nAssistant:",
        "target_response": target,
    }
    reward = HermesReasoningTraceReward(weight=1.0, reward_mode="tool_call")

    result = asyncio.run(
        reward.evaluate(item, _trajectory_for_output(item, prediction), tool_context=None)
    )

    assert 0.0 < result.score < 1.0
    assert result.metadata["tool_call_json_valid"] == 1.0
    assert result.metadata["argument_json_valid"] == 0.0
    assert result.metadata["argument_schema_ok"] == 0.0
