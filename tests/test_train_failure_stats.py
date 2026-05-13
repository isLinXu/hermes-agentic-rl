import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_train_prints_verifier_failure_reasons_and_gate_can_fail(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    # dataset: 2 items, both expect a file that does NOT exist -> verifier should fail
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task-1",
                        "instruction": "Create hello.txt and write hello",
                        "expected_output": "done",
                        "expected_files": [{"path": "hello.txt", "equals": "hello"}],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task-2",
                        "instruction": "Create todo.txt and write buy milk",
                        "expected_output": "done",
                        "expected_files": [{"path": "todo.txt", "contains": "buy milk"}],
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
    captured = capsys.readouterr()

    # gate should fail because verifier_pass_ratio=0
    assert exit_code == 4
    assert "verifier_pass_ratio=0.0000" in captured.out
    assert "verifier_failure_top" in captured.out
    assert "verifier_failed_samples=2" in captured.out
    assert "verifier_failures_total=2" in captured.out
