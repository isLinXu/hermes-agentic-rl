from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.cli.main import main


class _FakeDataset(list):
    def shuffle(self, seed: int = 0):
        _ = seed
        return self


class _FakeDatasets(ModuleType):
    def __init__(self, rows: list[dict]) -> None:
        super().__init__("datasets")
        self._rows = rows
        self.calls: list[dict] = []

    def load_dataset(self, *args, **kwargs):
        self.calls.append({"args": args, **kwargs})
        return _FakeDataset(self._rows)


def _trace_row() -> dict:
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


def test_train_rl_cli_runs_on_hf_reasoning_traces(tmp_path: Path, monkeypatch) -> None:
    fake_datasets = _FakeDatasets([_trace_row()])
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    output_dir = tmp_path / "train_hf_reasoning_traces_out"
    config_path = tmp_path / "train_hf_reasoning_traces.yaml"
    config_path.write_text(
        (
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "  max_len: 512\n"
            "  seed: 0\n"
            "environment:\n"
            "  type: hermes_reasoning_traces\n"
            "  dataset_name: lambda/hermes-agent-reasoning-traces\n"
            "  dataset_config: kimi\n"
            "  dataset_split: train\n"
            "  dataset_limit: 1\n"
            "  shuffle: false\n"
            "  history_window_messages: 8\n"
            "train_rl:\n"
            "  n_iters: 1\n"
            "  group_size: 4\n"
            "  prompts_per_iter: 1\n"
            "  lr: 0.005\n"
            "  max_new_tokens: 16\n"
            "  temperature: 1.0\n"
            "  update_epochs: 1\n"
            "  minibatch_size: 4\n"
            "  log_every: 1\n"
            "  bootstrap_sft_rounds: 1\n"
            "  bootstrap_sft_samples: 1\n"
            "  bootstrap_sft_lr: 0.0005\n"
            "  seed: 0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "train-rl",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
    )

    assert main() == 0
    summary_path = output_dir / "train_rl_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["algo"] == "grpo"
    assert len(summary["iters"]) == 2
    assert summary["iters"][0]["algo"] == "sft_bootstrap"
    assert summary["iters"][1]["algo"] == "grpo"
    assert fake_datasets.calls[0]["args"][0] == "lambda/hermes-agent-reasoning-traces"
