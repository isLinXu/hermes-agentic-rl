import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main
from hermes_agentic_rl.cli.session_eval_export_cli import (
    run_session_eval_export_config,
)


def test_session_eval_export_cli_writes_self_evolution_dataset(tmp_path: Path, monkeypatch, capsys):
    session_path = tmp_path / "sessions.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "session_id": "sess-1",
                        "task_id": "task-1",
                        "turns_used": 2,
                        "final_output": "done",
                        "reward": 0.8,
                        "metadata": {
                            "prompt": "Create a file",
                            "runtime": {"integration": "hermes"},
                        },
                        "messages": [
                            {"role": "user", "content": "Create a file"},
                            {"role": "assistant", "content": "done"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "session_id": "sess-2",
                        "task_id": "task-2",
                        "turns_used": 4,
                        "final_output": "updated docs",
                        "reward": 0.6,
                        "metadata": {
                            "runtime": {"integration": "hermes"},
                        },
                        "messages": [
                            {"role": "user", "content": "Update the docs"},
                            {"role": "assistant", "content": "updated docs"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "session_id": "sess-3",
                        "task_id": "task-3",
                        "turns_used": 6,
                        "final_output": "fixed bug",
                        "reward": 0.3,
                        "metadata": {
                            "runtime": {"integration": "hermes"},
                        },
                        "messages": [
                            {"role": "user", "content": "Fix the bug"},
                            {"role": "assistant", "content": "fixed bug"},
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "dataset"
    config_path = tmp_path / "export.yaml"
    config_path.write_text(
        (
            f"input_path: {session_path}\n"
            f"output_path: {output_dir}\n"
            "source: hermes-session\n"
            "train_ratio: 0.5\n"
            "val_ratio: 0.25\n"
            "seed: 0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-eval-export",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "[session-eval-export]" in out

    train = (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val = (output_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
    holdout = (output_dir / "holdout.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(train) == 1
    assert len(val) == 1
    assert len(holdout) == 1

    sample = json.loads(train[0])
    assert sample["source"] == "hermes-session"
    assert sample["task_input"] == "Create a file"
    assert "final_output" in sample
    assert "expected_behavior" in sample
    assert sample["difficulty"] in {"easy", "medium", "hard"}
    assert sample["category"] == "hermes"

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["samples"]["total"] == 3


def test_session_eval_export_config_accepts_in_memory_config(tmp_path: Path):
    session_path = tmp_path / "sessions.jsonl"
    session_path.write_text(
        json.dumps(
            {
                "session_id": "sess-1",
                "task_id": "task-1",
                "turns_used": 1,
                "final_output": "done",
                "reward": 0.9,
                "metadata": {"prompt": "Do work"},
                "messages": [
                    {"role": "user", "content": "Do work"},
                    {"role": "assistant", "content": "done"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "examples.jsonl"
    exit_code = run_session_eval_export_config(
        {
            "input_path": str(session_path),
            "output_path": str(output_path),
            "source": "direct-config",
        }
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["source"] == "direct-config"
