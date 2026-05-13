import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_train_negative_example_produces_content_mismatch_failure_top(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    workdir_base = tmp_path / "workdirs"
    (workdir_base / "task-ok").mkdir(parents=True, exist_ok=True)
    (workdir_base / "task-ok" / "x.txt").write_text("hello", encoding="utf-8")

    (workdir_base / "task-bad").mkdir(parents=True, exist_ok=True)
    (workdir_base / "task-bad" / "x.txt").write_text("WRONG", encoding="utf-8")

    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task-ok",
                        "instruction": "Create x.txt and write hello",
                        "expected_output": "done",
                        "expected_files": [{"path": "x.txt", "equals": "hello"}],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task-bad",
                        "instruction": "Create x.txt and write hello",
                        "expected_output": "done",
                        "expected_files": [{"path": "x.txt", "equals": "hello"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    export_path = tmp_path / "train.jsonl"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "reward:\n"
            "  aggregator: weighted_sum\n"
            "  components:\n"
            "    - name: filesystem_verifier_reward\n"
            "      weight: 1.0\n"
            "trainer:\n"
            f"  export_training_path: {export_path}\n"
            f"  workdir_base: {workdir_base}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "train",
            "--config",
            str(config_path),
            "--limit",
            "2",
            "--overwrite",
            "--min-verifier-pass-ratio",
            "1.0",
        ],
    )

    exit_code = main()
    out = capsys.readouterr().out
    assert exit_code == 4
    assert "content mismatch (equals)=1" in out
