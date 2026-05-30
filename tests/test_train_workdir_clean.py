import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_workdir_clean_removes_task_dirs_under_outputs(tmp_path: Path, monkeypatch):
    workdir_base = tmp_path / "outputs" / "workdirs" / "hermes"
    (workdir_base / "old-task").mkdir(parents=True, exist_ok=True)
    (workdir_base / "old-task" / "sentinel.txt").write_text("x", encoding="utf-8")

    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create x.txt and write hello",
                "expected_output": "done",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    export_path = tmp_path / "outputs" / "out.jsonl"
    config = tmp_path / "config.yaml"
    config.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset}\n"
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
            str(config),
            "--workdir-clean",
            "--limit",
            "1",
            "--overwrite",
        ],
    )

    assert main() == 0
    assert workdir_base.exists()
    assert not (workdir_base / "old-task").exists()
    assert (workdir_base / "task-1").exists()


def test_workdir_clean_refuses_to_delete_outside_outputs(tmp_path: Path, monkeypatch):
    workdir_base = tmp_path / "evil"
    (workdir_base / "old-task").mkdir(parents=True, exist_ok=True)

    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create x.txt and write hello",
                "expected_output": "done",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    export_path = tmp_path / "outputs" / "out.jsonl"
    config = tmp_path / "config.yaml"
    config.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset}\n"
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
            str(config),
            "--workdir-clean",
        ],
    )

    assert main() != 0
    assert (workdir_base / "old-task").exists()
