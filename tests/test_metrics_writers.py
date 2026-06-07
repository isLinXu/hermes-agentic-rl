"""Metrics writers — JSONL / stdout / Multi / build-from-config."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from hermes_agentic_rl.monitor.writers import (
    JsonlMetricsWriter,
    MultiMetricsWriter,
    StdoutMetricsWriter,
    build_writer_from_config,
)
from hermes_agentic_rl.monitor import UnifiedObservable


def test_jsonl_writer_appends(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    w = JsonlMetricsWriter(path)
    w({"iter": 0, "loss": 0.5, "algo": "grpo"})
    w({"iter": 1, "loss": 0.3, "algo": "grpo"})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert rec0["iter"] == 0 and rec1["iter"] == 1
    assert rec1["loss"] == 0.3


def test_jsonl_writer_handles_non_json_values(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    w = JsonlMetricsWriter(path)

    class Custom:
        def __repr__(self) -> str:
            return "Custom()"

    w({"iter": 0, "obj": Custom(), "nested": {"k": [1, 2, 3]}})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["obj"] == "Custom()"
    assert rec["nested"]["k"] == [1, 2, 3]


def test_multi_writer_isolates_failures(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    calls: list[dict] = []

    def good(record: dict) -> None:
        calls.append(record)

    def bad(record: dict) -> None:
        raise RuntimeError("boom")

    multi = MultiMetricsWriter([bad, JsonlMetricsWriter(path), good])
    multi({"iter": 0, "x": 1.0})
    multi({"iter": 1, "x": 2.0})
    multi.close()

    assert len(calls) == 2
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_build_writer_from_config_defaults(tmp_path: Path) -> None:
    w = build_writer_from_config(
        {"jsonl": True, "stdout": True},
        output_dir=tmp_path,
    )
    assert w is not None
    w({"iter": 0, "loss": 0.1})
    w.close()
    assert (tmp_path / "metrics.jsonl").exists()


def test_unified_observable_defaults_to_jsonl(tmp_path: Path) -> None:
    with UnifiedObservable(tmp_path) as observable:
        observable({"iter": 0, "loss": 0.1})
        observable.log({"iter": 1}, loss=0.2)

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [{"iter": 0, "loss": 0.1}, {"iter": 1, "loss": 0.2}]


def test_build_writer_from_config_no_config_returns_none() -> None:
    assert build_writer_from_config(None) is None
    assert build_writer_from_config({}) is None


def test_build_writer_extra_sinks_composed(tmp_path: Path) -> None:
    calls: list[dict] = []

    def extra(r: dict) -> None:
        calls.append(r)

    w = build_writer_from_config(
        {"jsonl": True},
        output_dir=tmp_path,
        extra=[extra],
    )
    assert w is not None
    w({"iter": 0, "loss": 0.1})
    assert len(calls) == 1
    assert (tmp_path / "metrics.jsonl").exists()


def test_stdout_writer_no_exception(capsys) -> None:
    w = StdoutMetricsWriter()
    w({"iter": 5, "mean_reward": 0.1234, "loss": -0.001, "algo": "grpo"})
    out = capsys.readouterr().out
    assert "[metrics]" in out
    assert "iter=5" in out


class _FakeWandbRun:
    def __init__(self) -> None:
        self.summary: dict[str, float | bool] = {}


class _FakeWandb(ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.run = None
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, int]] = []
        self.define_metric_calls: list[tuple[tuple, dict]] = []
        self.finish_calls = 0

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run = _FakeWandbRun()
        return self.run

    def log(self, payload, step=None):
        self.log_calls.append((dict(payload), step))

    def define_metric(self, *args, **kwargs):
        self.define_metric_calls.append((args, kwargs))

    def finish(self):
        self.finish_calls += 1
        self.run = None


def test_build_writer_from_config_auto_inits_wandb_and_logs_nested_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    writer = build_writer_from_config(
        {
            "wandb": {
                "project": "unit-test-project",
                "prefix": "rl",
                "mode": "offline",
            }
        },
        output_dir=tmp_path,
        wandb_context={
            "name": "writer-smoke",
            "job_type": "train-rl",
            "tags": ["grpo", "echo"],
            "command": "train-rl",
            "config_path": "/tmp/train.yaml",
            "config": {
                "algo": "grpo",
                "train_rl": {"n_iters": 2},
            },
        },
    )

    assert writer is not None
    writer(
        {
            "iter": 3,
            "mean_reward": 0.75,
            "loss": -0.25,
            "lagrangian": {"lambda": 0.1},
            "algo": "grpo",
        }
    )
    writer.update_summary(
        {
            "last_mean_reward": 0.75,
            "curriculum_snapshot": {"level": 2},
        }
    )
    writer.close()

    assert fake_wandb.init_calls
    init_kwargs = fake_wandb.init_calls[0]
    assert init_kwargs["project"] == "unit-test-project"
    assert init_kwargs["name"] == "writer-smoke"
    assert init_kwargs["job_type"] == "train-rl"
    assert init_kwargs["dir"] == str(tmp_path / "wandb")
    assert init_kwargs["config"]["algo"] == "grpo"
    assert init_kwargs["config"]["train_rl"]["n_iters"] == 2
    assert init_kwargs["config"]["command"] == "train-rl"

    assert fake_wandb.log_calls == [
        (
            {
                "rl/iter": 3.0,
                "rl/mean_reward": 0.75,
                "rl/loss": -0.25,
                "rl/lagrangian/lambda": 0.1,
            },
            3,
        )
    ]
    assert fake_wandb.run is None
    assert fake_wandb.finish_calls == 1


def test_build_writer_from_config_wandb_summary_updates_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    writer = build_writer_from_config(
        {"wandb": True},
        output_dir=tmp_path,
        wandb_context={"config": {"algo": "ppo"}},
    )
    assert writer is not None
    writer.update_summary({"best_mean_reward": 1.2, "snapshot": {"level": 4}})
    assert fake_wandb.run is not None
    assert fake_wandb.run.summary["best_mean_reward"] == 1.2
    assert fake_wandb.run.summary["snapshot/level"] == 4.0


def test_build_writer_from_config_wandb_clamps_negative_log_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    writer = build_writer_from_config(
        {"wandb": {"prefix": "train", "mode": "offline"}},
        output_dir=tmp_path,
    )
    assert writer is not None

    writer({"iter": -1, "algo": "sft_bootstrap", "loss": 0.5})

    assert fake_wandb.log_calls == [
        (
            {
                "train/iter": -1.0,
                "train/loss": 0.5,
            },
            0,
        )
    ]


def test_build_writer_from_config_wandb_can_be_disabled(tmp_path: Path) -> None:
    writer = build_writer_from_config(
        {"wandb": {"enabled": False}},
        output_dir=tmp_path,
    )
    assert writer is None
