"""Metrics writers — JSONL / stdout / Multi / build-from-config."""

from __future__ import annotations

import json
from pathlib import Path

from hermes_agentic_rl.monitor.writers import (
    JsonlMetricsWriter,
    MultiMetricsWriter,
    StdoutMetricsWriter,
    build_writer_from_config,
)


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
