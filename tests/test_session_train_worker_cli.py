import json
from pathlib import Path
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

import hermes_agentic_rl.cli.session_train_worker_cli as worker_cli
from hermes_agentic_rl.cli.main import main


def test_session_train_worker_cli_consumes_replay_and_saves_policy(
    tmp_path: Path, monkeypatch, capsys
):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "prompt_ids": [1, 10, 11],
                "response_ids": [12, 13],
                "reward": 1.0,
                "metadata": {"session_id": "sess-1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "policy.pt"
    state_path = tmp_path / "worker_state.json"
    config_path = tmp_path / "session_train_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  min_reward: 0.5\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "[session-train-worker]" in out
    assert save_path.exists()
    assert state_path.exists()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["trained_samples"] == 1
    assert state["records_seen"] == 1
    loaded = torch.load(save_path, map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict)
    assert loaded


def test_session_train_worker_config_accepts_in_memory_config(tmp_path: Path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "prompt_ids": [1, 10, 11],
                "response_ids": [12, 13],
                "reward": 1.0,
                "metadata": {"session_id": "sess-memory"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "policy.pt"
    state_path = tmp_path / "state.json"

    exit_code = worker_cli.run_session_train_worker_config(
        {
            "algo": "bc",
            "input_path": str(replay_path),
            "save_path": str(save_path),
            "state_path": str(state_path),
            "backend": {
                "name": "tiny",
                "dim": 16,
                "n_heads": 2,
                "n_layers": 2,
            },
            "train": {
                "n_epochs": 1,
                "batch_size": 1,
                "lr": 0.001,
                "min_reward": 0.0,
            },
        },
        once=True,
    )

    assert exit_code == 0
    assert save_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1


def test_session_train_worker_cli_respects_min_reward_filter(tmp_path: Path, monkeypatch):
    replay_path = tmp_path / "low_reward_replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "prompt_ids": [1, 10, 11],
                "response_ids": [12, 13],
                "reward": -0.25,
                "metadata": {"session_id": "sess-2"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "low_reward_policy.pt"
    state_path = tmp_path / "low_reward_state.json"
    config_path = tmp_path / "low_reward_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  min_reward: 0.0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["records_seen"] == 1
    assert state["trained_samples"] == 0
    assert state["updates"] == 0
    assert not save_path.exists()


def test_session_train_worker_cli_filters_by_replay_mining_metadata(
    tmp_path: Path,
    monkeypatch,
):
    replay_path = tmp_path / "mining_replay.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [12, 13],
                        "reward": 1.0,
                        "metadata": {
                            "session_id": "sess-tool",
                            "capability_axes": ["tool_use_reliability"],
                            "replay_mining": {
                                "axes": ["tool_use_reliability"],
                                "recommended_uses": ["tool_reliability_replay"],
                                "skill_candidate": False,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 20, 21],
                        "response_ids": [22, 23],
                        "reward": 1.0,
                        "metadata": {
                            "session_id": "sess-task",
                            "capability_axes": ["task_success"],
                            "replay_mining": {
                                "axes": ["task_success"],
                                "recommended_uses": ["positive_replay"],
                                "skill_candidate": False,
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "mining_policy.pt"
    state_path = tmp_path / "mining_state.json"
    quarantine_path = tmp_path / "mining_quarantine.jsonl"
    config_path = tmp_path / "mining_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            f"quarantine_path: {quarantine_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  min_reward: 0.0\n"
            "  replay_filter:\n"
            "    require_any_capability_axes: [tool_use_reliability]\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["trained_samples"] == 1
    assert state["metadata_filtered_records"] == 1
    quarantine_rows = [
        json.loads(line)
        for line in quarantine_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert quarantine_rows[0]["kind"] == "metadata_filtered_replay_record"
    assert quarantine_rows[0]["reason"] == "missing_any_capability_axis"


def test_session_train_worker_cli_bc_recovers_after_checkpoint_loss(
    tmp_path: Path, monkeypatch, capsys
):
    replay_path = tmp_path / "checkpoint_replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "prompt_ids": [1, 10, 11],
                "response_ids": [12, 13],
                "reward": 1.0,
                "metadata": {"session_id": "sess-ckpt"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "checkpoint_policy.pt"
    state_path = tmp_path / "checkpoint_state.json"
    config_path = tmp_path / "checkpoint_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  min_reward: 0.5\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    save_path.unlink()

    assert main() == 0
    out = capsys.readouterr().out
    assert "checkpoint_missing=true" in out
    assert save_path.exists()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["records_seen"] == 1
    assert state["trained_samples"] == 1


def test_session_train_worker_cli_bc_recovers_from_truncated_replay(
    tmp_path: Path, monkeypatch, capsys
):
    replay_path = tmp_path / "truncated_replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "prompt_ids": [1, 10, 11],
                "response_ids": [12, 13, 14, 15, 16, 17],
                "reward": 1.0,
                "metadata": {"session_id": "sess-a", "note": "longer-first-record"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "truncated_policy.pt"
    state_path = tmp_path / "truncated_state.json"
    config_path = tmp_path / "truncated_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  min_reward: 0.5\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    replay_path.write_text(
        json.dumps(
            {
                "prompt_ids": [2, 20],
                "response_ids": [21],
                "reward": 0.9,
                "metadata": {"session_id": "sess-b"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "source_reset=truncated" in out or "source_reset=rewritten" in out

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 2
    assert state["records_seen"] == 2
    assert state["trained_samples"] == 2


def test_session_train_worker_cli_bc_waits_for_partial_jsonl_line(tmp_path: Path, monkeypatch):
    replay_path = tmp_path / "partial_replay.jsonl"
    first_record = json.dumps(
        {
            "prompt_ids": [1, 10, 11],
            "response_ids": [12, 13],
            "reward": 1.0,
            "metadata": {"session_id": "sess-a"},
        }
    )
    second_record = json.dumps(
        {
            "prompt_ids": [2, 20, 21],
            "response_ids": [22, 23],
            "reward": 0.8,
            "metadata": {"session_id": "sess-b"},
        }
    )
    split_at = max(1, len(second_record) // 2)
    replay_path.write_text(
        first_record + "\n" + second_record[:split_at],
        encoding="utf-8",
    )

    save_path = tmp_path / "partial_policy.pt"
    state_path = tmp_path / "partial_state.json"
    config_path = tmp_path / "partial_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  min_reward: 0.5\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["records_seen"] == 1
    assert state["trained_samples"] == 1

    with replay_path.open("a", encoding="utf-8") as handle:
        handle.write(second_record[split_at:] + "\n")

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 2
    assert state["records_seen"] == 2
    assert state["trained_samples"] == 2


def test_session_train_worker_cli_quarantines_invalid_records_and_writes_metrics(
    tmp_path: Path, monkeypatch
):
    replay_path = tmp_path / "invalid_replay.jsonl"
    quarantine_path = tmp_path / "invalid_replay_quarantine.jsonl"
    metrics_path = tmp_path / "worker_metrics.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [12, 13],
                        "reward": 1.0,
                        "metadata": {"session_id": "sess-good-1"},
                    }
                ),
                "{bad json",
                json.dumps([1, 2, 3]),
                json.dumps(
                    {
                        "prompt_ids": [2, 20, 21],
                        "response_ids": [],
                        "reward": 0.5,
                        "metadata": {"session_id": "sess-bad-empty"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [3, 30],
                        "response_ids": [31],
                        "reward": 0.8,
                        "metadata": {"session_id": "sess-good-2"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "invalid_policy.pt"
    state_path = tmp_path / "invalid_state.json"
    config_path = tmp_path / "invalid_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            f"quarantine_path: {quarantine_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  min_reward: 0.0\n"
            "metrics:\n"
            f"  jsonl: {metrics_path}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["records_seen"] == 2
    assert state["trained_samples"] == 2
    assert state["invalid_json_lines"] == 2
    assert state["invalid_records"] == 1
    assert state["quarantined_records"] == 3

    quarantine_lines = [
        json.loads(line)
        for line in quarantine_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(quarantine_lines) == 3
    assert {row["kind"] for row in quarantine_lines} == {
        "invalid_json_line",
        "non_dict_json_line",
        "invalid_replay_record",
    }

    metric_lines = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert metric_lines
    last_metric = metric_lines[-1]
    assert last_metric["event"] == "update"
    assert last_metric["records"] == 2
    assert last_metric["invalid_json_lines"] == 2
    assert last_metric["invalid_records"] == 1
    assert last_metric["quarantined_records"] == 3


def test_session_train_worker_cli_applies_quality_filters_and_dedupes(tmp_path: Path, monkeypatch):
    replay_path = tmp_path / "quality_replay.jsonl"
    quarantine_path = tmp_path / "quality_quarantine.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 10],
                        "response_ids": [11, 12],
                        "reward": 1.0,
                        "metadata": {"session_id": "sess-good-1"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 10],
                        "response_ids": [11, 12],
                        "reward": 1.0,
                        "metadata": {"session_id": "sess-dup"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [2],
                        "response_ids": [21, 22],
                        "reward": 0.2,
                        "metadata": {"session_id": "sess-short-prompt"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [3, 30],
                        "response_ids": [31, 32],
                        "reward": 99.0,
                        "metadata": {"session_id": "sess-high-reward"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [4, 40],
                        "response_ids": [41, 42],
                        "reward": 0.3,
                        "metadata": {"session_id": "sess-good-2"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "quality_policy.pt"
    state_path = tmp_path / "quality_state.json"
    config_path = tmp_path / "quality_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            f"quarantine_path: {quarantine_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  min_reward: 0.0\n"
            "  data_quality:\n"
            "    min_prompt_tokens: 2\n"
            "    min_response_tokens: 2\n"
            "    max_abs_reward: 5.0\n"
            "    dedupe_within_scan: true\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["records_seen"] == 2
    assert state["trained_samples"] == 2
    assert state["quality_filtered_records"] == 3
    assert state["quarantined_records"] == 3

    quarantine_rows = [
        json.loads(line)
        for line in quarantine_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(quarantine_rows) == 3
    assert {row["reason"] for row in quarantine_rows} == {
        "duplicate_record",
        "prompt_too_short",
        "reward_out_of_range",
    }


def test_session_train_worker_cli_can_publish_metrics_to_dashboard_sink(
    tmp_path: Path, monkeypatch, capsys
):
    replay_path = tmp_path / "dashboard_replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "prompt_ids": [1, 10, 11],
                "response_ids": [12, 13],
                "reward": 1.0,
                "metadata": {"session_id": "sess-dashboard"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeDashboard:
        instances: ClassVar[list["FakeDashboard"]] = []

        def __init__(self, *, host: str, port: int, maxlen: int = 5000) -> None:
            self.host = host
            self.port = port
            self.maxlen = maxlen
            self.records: list[dict] = []
            self.started = False
            self.stopped = False
            self.__class__.instances.append(self)

        def record(self, rec: dict[str, object]) -> None:
            self.records.append(dict(rec))

        def start(self) -> str:
            self.started = True
            return f"http://{self.host}:{self.port}/"

        def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(worker_cli, "LiveDashboard", FakeDashboard)

    save_path = tmp_path / "dashboard_policy.pt"
    state_path = tmp_path / "dashboard_state.json"
    config_path = tmp_path / "dashboard_worker.yaml"
    config_path.write_text(
        (
            "algo: bc\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  min_reward: 0.0\n"
            "dashboard:\n"
            "  enabled: true\n"
            "  host: 127.0.0.1\n"
            "  port: 9988\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "dashboard live at http://127.0.0.1:9988/" in out
    assert FakeDashboard.instances
    dash = FakeDashboard.instances[-1]
    assert dash.started is True
    assert dash.stopped is True
    assert dash.records
    assert dash.records[-1]["event"] == "update"
    assert dash.records[-1]["records_seen"] == 1


def test_session_train_worker_cli_supports_dpo_from_scored_replay(
    tmp_path: Path, monkeypatch, capsys
):
    replay_path = tmp_path / "dpo_replay.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [12, 13],
                        "reward": 1.0,
                        "metadata": {"session_id": "sess-a"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [14, 15],
                        "reward": 0.0,
                        "metadata": {"session_id": "sess-b"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "dpo_policy.pt"
    state_path = tmp_path / "dpo_worker_state.json"
    config_path = tmp_path / "session_dpo_worker.yaml"
    config_path.write_text(
        (
            "algo: dpo\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  beta: 0.1\n"
            "  preference:\n"
            "    min_reward_gap: 0.5\n"
            "    max_pairs_per_prompt: 1\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "algo=dpo" in out
    assert save_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["trained_pairs"] == 1


def test_session_train_worker_cli_dpo_drains_backlog_with_pair_limit(tmp_path: Path, monkeypatch):
    replay_path = tmp_path / "dpo_backlog_replay.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [12, 13],
                        "reward": 2.0,
                        "metadata": {"session_id": "sess-a"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [14, 15],
                        "reward": 1.0,
                        "metadata": {"session_id": "sess-b"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [16, 17],
                        "reward": 0.0,
                        "metadata": {"session_id": "sess-c"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "dpo_backlog_policy.pt"
    state_path = tmp_path / "dpo_backlog_state.json"
    config_path = tmp_path / "dpo_backlog_worker.yaml"
    config_path.write_text(
        (
            "algo: dpo\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  beta: 0.1\n"
            "  train_only_new_pairs: true\n"
            "  max_pairs_per_update: 1\n"
            "  preference:\n"
            "    min_reward_gap: 0.5\n"
            "    max_pairs_per_prompt: 3\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["candidate_pairs"] == 3
    assert state["seen_pairs"] == 1
    assert state["pending_pairs"] == 2

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 2
    assert state["seen_pairs"] == 2
    assert state["pending_pairs"] == 1

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 3
    assert state["seen_pairs"] == 3
    assert state["pending_pairs"] == 0
    assert state["trained_pairs"] == 1

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 3
    assert state["pending_pairs"] == 0


def test_session_train_worker_cli_dpo_does_not_double_count_rejected_records_in_backlog_mode(
    tmp_path: Path, monkeypatch
):
    replay_path = tmp_path / "dpo_invalid_backlog_replay.jsonl"
    quarantine_path = tmp_path / "dpo_invalid_backlog_quarantine.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [12, 13],
                        "reward": 2.0,
                        "metadata": {"session_id": "sess-a"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [14, 15],
                        "reward": 1.0,
                        "metadata": {"session_id": "sess-b"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [16, 17],
                        "reward": 0.0,
                        "metadata": {"session_id": "sess-c"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [9, 90],
                        "response_ids": [],
                        "reward": 0.2,
                        "metadata": {"session_id": "sess-invalid"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "dpo_invalid_backlog_policy.pt"
    state_path = tmp_path / "dpo_invalid_backlog_state.json"
    config_path = tmp_path / "dpo_invalid_backlog_worker.yaml"
    config_path.write_text(
        (
            "algo: dpo\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            f"quarantine_path: {quarantine_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  beta: 0.1\n"
            "  train_only_new_pairs: true\n"
            "  max_pairs_per_update: 1\n"
            "  preference:\n"
            "    min_reward_gap: 0.5\n"
            "    max_pairs_per_prompt: 3\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert first_state["invalid_records"] == 1
    assert first_state["quarantined_records"] == 1
    assert first_state["pending_pairs"] == 2

    assert main() == 0
    second_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert second_state["invalid_records"] == 1
    assert second_state["quarantined_records"] == 1
    assert second_state["pending_pairs"] == 1

    quarantine_lines = [
        json.loads(line)
        for line in quarantine_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(quarantine_lines) == 1
    assert quarantine_lines[0]["kind"] == "invalid_replay_record"


def test_session_train_worker_cli_dpo_skips_previously_trained_pairs_after_rescan(
    tmp_path: Path, monkeypatch, capsys
):
    replay_path = tmp_path / "dpo_repeat_replay.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [12, 13],
                        "reward": 1.0,
                        "metadata": {"session_id": "sess-a"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [14, 15],
                        "reward": 0.0,
                        "metadata": {"session_id": "sess-b"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "dpo_repeat_policy.pt"
    state_path = tmp_path / "dpo_repeat_state.json"
    config_path = tmp_path / "dpo_repeat_worker.yaml"
    config_path.write_text(
        (
            "algo: dpo\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  beta: 0.1\n"
            "  train_only_new_pairs: true\n"
            "  preference:\n"
            "    min_reward_gap: 0.5\n"
            "    max_pairs_per_prompt: 1\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    with replay_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "prompt_ids": [1, 10, 11],
                    "response_ids": [14, 15],
                    "reward": 0.0,
                    "metadata": {"session_id": "sess-c"},
                }
            )
            + "\n"
        )

    assert main() == 0
    out = capsys.readouterr().out
    assert "new_pairs=0" in out

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["candidate_pairs"] == 1
    assert state["trained_pairs"] == 0
    assert state["pending_pairs"] == 0
    assert state["seen_pairs"] == 1


def test_session_train_worker_cli_dpo_rescans_full_replay_when_source_grows(
    tmp_path: Path, monkeypatch
):
    replay_path = tmp_path / "dpo_rescan_replay.jsonl"
    first_record = {
        "prompt_ids": [1, 10, 11],
        "response_ids": [12, 13],
        "reward": 1.0,
        "metadata": {"session_id": "sess-a"},
    }
    replay_path.write_text(json.dumps(first_record) + "\n", encoding="utf-8")

    save_path = tmp_path / "dpo_rescan_policy.pt"
    state_path = tmp_path / "dpo_rescan_worker_state.json"
    config_path = tmp_path / "session_dpo_rescan_worker.yaml"
    config_path.write_text(
        (
            "algo: dpo\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  beta: 0.1\n"
            "  preference:\n"
            "    min_reward_gap: 0.5\n"
            "    max_pairs_per_prompt: 1\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )
    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["records_seen"] == 1
    assert state["trained_samples"] == 1
    assert state["trained_pairs"] == 0
    assert state["updates"] == 0
    assert not save_path.exists()

    second_record = {
        "prompt_ids": [1, 10, 11],
        "response_ids": [14, 15],
        "reward": 0.0,
        "metadata": {"session_id": "sess-b"},
    }
    with replay_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(second_record) + "\n")

    assert main() == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["records_seen"] == 2
    assert state["trained_samples"] == 2
    assert state["candidate_pairs"] == 1
    assert state["pending_pairs"] == 0
    assert state["trained_pairs"] == 1
    assert state["seen_pairs"] == 1
    assert state["updates"] == 1
    assert save_path.exists()


def test_session_train_worker_cli_supports_rm_from_scored_replay(
    tmp_path: Path, monkeypatch, capsys
):
    replay_path = tmp_path / "rm_replay.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [12, 13],
                        "reward": 1.0,
                        "metadata": {"session_id": "sess-a"},
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 10, 11],
                        "response_ids": [14, 15],
                        "reward": 0.0,
                        "metadata": {"session_id": "sess-b"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_path = tmp_path / "rm_head.pt"
    state_path = tmp_path / "rm_worker_state.json"
    config_path = tmp_path / "session_rm_worker.yaml"
    config_path.write_text(
        (
            "algo: rm\n"
            f"input_path: {replay_path}\n"
            f"save_path: {save_path}\n"
            f"state_path: {state_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "train:\n"
            "  n_epochs: 1\n"
            "  batch_size: 1\n"
            "  lr: 0.001\n"
            "  freeze_base: true\n"
            "  preference:\n"
            "    min_reward_gap: 0.5\n"
            "    max_pairs_per_prompt: 1\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-train-worker",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "algo=rm" in out
    assert save_path.exists()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["records_seen"] == 2
    assert state["trained_samples"] == 2
    assert state["candidate_pairs"] == 1
    assert state["pending_pairs"] == 0
    assert state["trained_pairs"] == 1
    assert state["seen_pairs"] == 1

    loaded = torch.load(save_path, map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict)
    assert set(loaded) == {"weight", "bias"}
