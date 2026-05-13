import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_train_uses_per_item_workdir_base_for_filesystem_verifier(
    tmp_path: Path,
    monkeypatch,
):
    # Prepare deterministic per-item workdirs with expected files.
    workdir_base = tmp_path / "workdirs"
    (workdir_base / "task-1").mkdir(parents=True, exist_ok=True)
    (workdir_base / "task-1" / "hello.txt").write_text("hello", encoding="utf-8")

    (workdir_base / "task-2").mkdir(parents=True, exist_ok=True)
    (workdir_base / "task-2" / "todo.txt").write_text("buy milk", encoding="utf-8")

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
                        "expected_files": [{"path": "todo.txt", "equals": "buy milk"}],
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

    # Ensure the repo/root cwd doesn't contain the expected files. If train doesn't chdir,
    # the filesystem verifier will fail.
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

    assert main() == 0
