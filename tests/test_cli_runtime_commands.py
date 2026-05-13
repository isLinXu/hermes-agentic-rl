import importlib
import json
import types
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_cli_rollout_writes_trajectory_json(tmp_path: Path, monkeypatch, capsys):
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create hello.txt",
                "expected_output": "done",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "  model: demo\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "trainer:\n"
            f"  export_training_path: {tmp_path / 'train.jsonl'}\n"
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "trajectory.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "rollout",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "trajectory saved" in captured.out
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["task_id"] == "task-1"
    assert payload["final_output"] == "done"
    assert payload["metadata"]["runtime"]["runtime"] == "fake"


def test_cli_train_exports_training_jsonl(tmp_path: Path, monkeypatch, capsys):
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create hello.txt",
                "expected_output": "done",
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "train.jsonl"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: fake\n"
            "  model: demo\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "reward:\n"
            "  aggregator: weighted_sum\n"
            "trainer:\n"
            f"  export_training_path: {output_path}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "train", "--config", str(config_path)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "training sample saved" in captured.out
    assert output_path.exists()


def test_cli_rollout_reports_clear_error_when_hermes_runtime_is_unavailable(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create hello.txt",
                "expected_output": "done",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: hermes\n"
            f"  repo_path: {tmp_path / 'missing-hermes-agent'}\n"
            "  model: demo\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "trainer:\n"
            f"  export_training_path: {tmp_path / 'train.jsonl'}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "rollout", "--config", str(config_path)],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "runtime unavailable" in captured.out.lower()


def test_cli_rollout_works_with_hermes_when_entrypoint_is_injected(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    # 注入 run_agent.AIAgent（模拟“已安装 pip 包的 hermes-agent”）
    run_agent_mod = types.ModuleType("run_agent")

    class FakeAIAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_conversation(self, user_message: str, task_id: str | None = None, system_message: str | None = None):
            del task_id, system_message
            return {
                "final_response": "done",
                "messages": [
                    {"role": "user", "content": user_message},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "write_file", "arguments": {"path": "x.txt"}}],
                    },
                    {"role": "tool", "name": "write_file", "content": "{\"ok\": true}"},
                    {"role": "assistant", "content": "done"},
                ],
            }

    run_agent_mod.AIAgent = FakeAIAgent
    monkeypatch.setitem(importlib.sys.modules, "run_agent", run_agent_mod)

    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create x.txt",
                "expected_output": "done",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: hermes\n"
            "  model: demo\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "trainer:\n"
            f"  export_training_path: {tmp_path / 'train.jsonl'}\n"
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "trajectory.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "rollout",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "trajectory saved" in captured.out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["runtime"]["runtime"] == "hermes"


def test_cli_help_still_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["hermes-agentic-rl", "help"])
    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out.lower()
