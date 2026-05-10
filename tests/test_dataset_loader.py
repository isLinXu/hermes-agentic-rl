import json
from pathlib import Path

from hermes_agentic_rl.datasets.jsonl_loader import load_jsonl_dataset


def test_load_jsonl_dataset_reads_items_and_preserves_task_id(tmp_path: Path):
    dataset_path = tmp_path / "tasks.jsonl"
    rows = [
        {"task_id": "task-1", "instruction": "Create a.txt", "expected_output": "done"},
        {"task_id": "task-2", "instruction": "Create b.txt", "expected_output": "done"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    items = load_jsonl_dataset(dataset_path)

    assert len(items) == 2
    assert items[0]["task_id"] == "task-1"
    assert items[1]["instruction"] == "Create b.txt"


def test_load_jsonl_dataset_generates_task_id_when_missing(tmp_path: Path):
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps({"instruction": "Create c.txt", "expected_output": "done"}),
        encoding="utf-8",
    )

    items = load_jsonl_dataset(dataset_path)

    assert len(items) == 1
    assert items[0]["task_id"] == "generated-0"
