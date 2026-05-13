from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from hermes_agentic_rl.cli.main import main


class _FakeWandbRun:
    def __init__(self) -> None:
        self.summary: dict[str, float | bool] = {}


class _FakeWandb(ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.run = None
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, int]] = []
        self.finish_calls = 0
        self.finished_summaries: list[dict[str, float | bool]] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run = _FakeWandbRun()
        return self.run

    def log(self, payload, step=None):
        self.log_calls.append((dict(payload), step))

    def define_metric(self, *_args, **_kwargs):
        return None

    def finish(self):
        if self.run is not None:
            self.finished_summaries.append(dict(self.run.summary))
        self.finish_calls += 1
        self.run = None


def test_train_rl_cli_runs_a_small_grpo_update(tmp_path: Path, monkeypatch, capsys):
    output_dir = tmp_path / "train_rl_out"
    config_path = tmp_path / "train_rl.yaml"
    config_path.write_text(
        (
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "  max_len: 64\n"
            "  seed: 0\n"
            "environment:\n"
            "  type: echo\n"
            "train_rl:\n"
            "  n_iters: 2\n"
            "  group_size: 4\n"
            "  prompts_per_iter: 1\n"
            "  lr: 0.005\n"
            "  max_new_tokens: 6\n"
            "  temperature: 1.0\n"
            "  update_epochs: 1\n"
            "  minibatch_size: 4\n"
            "  log_every: 1\n"
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
    out = capsys.readouterr().out
    summary_path = output_dir / "train_rl_summary.json"

    assert "[train-rl] DONE algo=grpo" in out
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["algo"] == "grpo"
    assert len(summary["iters"]) == 2
    assert isinstance(summary["last_mean_reward"], float)


def test_train_rl_cli_wandb_smoke(tmp_path: Path, monkeypatch, capsys):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    output_dir = tmp_path / "train_rl_wandb_out"
    config_path = tmp_path / "train_rl_wandb.yaml"
    config_path.write_text(
        (
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "  max_len: 64\n"
            "  seed: 0\n"
            "environment:\n"
            "  type: echo\n"
            "metrics:\n"
            "  wandb:\n"
            "    project: hermes-agentic-rl-tests\n"
            "    prefix: train\n"
            "    mode: offline\n"
            "train_rl:\n"
            "  n_iters: 1\n"
            "  group_size: 4\n"
            "  prompts_per_iter: 1\n"
            "  lr: 0.005\n"
            "  max_new_tokens: 6\n"
            "  temperature: 1.0\n"
            "  update_epochs: 1\n"
            "  minibatch_size: 4\n"
            "  log_every: 1\n"
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
    _out = capsys.readouterr().out
    summary = json.loads((output_dir / "train_rl_summary.json").read_text(encoding="utf-8"))

    assert fake_wandb.init_calls
    init_kwargs = fake_wandb.init_calls[0]
    assert init_kwargs["project"] == "hermes-agentic-rl-tests"
    assert init_kwargs["name"] == output_dir.name
    assert init_kwargs["job_type"] == "train-rl"
    assert init_kwargs["config"]["config_path"] == str(config_path)
    assert init_kwargs["config"]["environment"]["type"] == "echo"
    assert fake_wandb.log_calls
    assert any("train/mean_reward" in payload for payload, _step in fake_wandb.log_calls)
    assert any("train/prompt_tokens/mean" in payload for payload, _step in fake_wandb.log_calls)
    assert any("train/response_tokens/mean" in payload for payload, _step in fake_wandb.log_calls)
    assert any("train/reward_min" in payload for payload, _step in fake_wandb.log_calls)
    assert any("train/grad_norm" in payload for payload, _step in fake_wandb.log_calls)
    assert fake_wandb.finish_calls == 1
    assert fake_wandb.finished_summaries
    assert fake_wandb.finished_summaries[0]["last_mean_reward"] == summary["last_mean_reward"]
    assert summary["algo"] == "grpo"
