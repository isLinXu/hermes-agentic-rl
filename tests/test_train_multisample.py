import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_cli_train_supports_multisample_export_with_fake_runtime(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    # 让 filesystem verifier 有真实文件可验证（fake runtime 不会真的写文件）
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "notes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "notes" / "todo.txt").write_text("buy milk", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": f"task-{i}",
                        "instruction": "Create hello.txt and write hello",
                        "expected_output": "done",
                        "expected_files": [{"path": "hello.txt", "equals": "hello"}],
                    }
                )
                for i in range(1, 3)
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
            "    - name: outcome_reward\n"
            "      weight: 0.2\n"
            "    - name: toolcall_reward\n"
            "      weight: 0.2\n"
            "    - name: filesystem_verifier_reward\n"
            "      weight: 0.6\n"
            "trainer:\n"
            f"  export_training_path: {export_path}\n"
        ),
        encoding="utf-8",
    )

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

    assert exit_code == 0
    assert export_path.exists()
    assert "samples=2" in captured.out
    assert "verifier_pass_ratio=1.0000" in captured.out

    lines = [line for line in export_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    payloads = [json.loads(line) for line in lines]
    assert all(p["reward"] == 1.0 for p in payloads)
    assert all(
        any(
            c["name"] == "filesystem_verifier_reward" and c["score"] == 1.0
            for c in p["reward_components"]
        )
        for p in payloads
    )
