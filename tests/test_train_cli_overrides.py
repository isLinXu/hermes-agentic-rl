import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_train_cli_overrides_dataset_and_export(tmp_path: Path, monkeypatch):
    # dataset A (config default) has 1 sample
    dataset_a = tmp_path / "a.jsonl"
    dataset_a.write_text(
        json.dumps(
            {
                "task_id": "a",
                "instruction": "Create x.txt and write hello",
                "expected_output": "done",
                "expected_files": [{"path": "x.txt", "equals": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # dataset B (CLI override) has 2 samples
    dataset_b = tmp_path / "b.jsonl"
    dataset_b.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": f"b{i}",
                        "instruction": "Create x.txt and write hello",
                        "expected_output": "done",
                        "expected_files": [{"path": "x.txt", "equals": "hello"}],
                    }
                )
                for i in range(1, 3)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Prepare per-item workdir with files so verifier passes (fake runtime doesn't write files)
    workdir_base = tmp_path / "outputs" / "workdirs"
    for tid in ["b1", "b2"]:
        d = workdir_base / tid
        d.mkdir(parents=True, exist_ok=True)
        (d / "x.txt").write_text("hello", encoding="utf-8")

    export_override = tmp_path / "outputs" / "export.jsonl"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "environment:\n"
            f"  dataset_path: {dataset_a}\n"
            "reward:\n"
            "  aggregator: weighted_sum\n"
            "  components:\n"
            "    - name: filesystem_verifier_reward\n"
            "      weight: 1.0\n"
            "trainer:\n"
            f"  export_training_path: {tmp_path/'outputs'/'default.jsonl'}\n"
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
            "--dataset",
            str(dataset_b),
            "--export",
            str(export_override),
            "--limit",
            "2",
            "--overwrite",
            "--min-verifier-pass-ratio",
            "1.0",
        ],
    )

    assert main() == 0
    lines = [l for l in export_override.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2

