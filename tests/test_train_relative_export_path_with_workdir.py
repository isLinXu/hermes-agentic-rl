import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_train_allows_relative_export_path_when_using_workdir_base(
    tmp_path: Path,
    monkeypatch,
):
    workdir_base = tmp_path / "workdirs"
    (workdir_base / "task-1").mkdir(parents=True, exist_ok=True)
    (workdir_base / "task-1" / "hello.txt").write_text("hello", encoding="utf-8")

    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create hello.txt and write hello",
                "expected_output": "done",
                "expected_files": [{"path": "hello.txt", "equals": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # IMPORTANT: relative path
    export_path = Path("outputs/train.jsonl")
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
            "1",
            "--overwrite",
            "--min-verifier-pass-ratio",
            "1.0",
        ],
    )

    assert main() == 0
    assert (tmp_path / "outputs" / "train.jsonl").exists()
