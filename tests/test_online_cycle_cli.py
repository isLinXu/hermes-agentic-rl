import importlib
import json
import types
from pathlib import Path

import hermes_agentic_rl.cli.session_train_worker_cli as worker_cli
from hermes_agentic_rl.cli.main import main


def test_online_cycle_cli_runs_rollout_replay_worker_and_eval_export(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    run_agent_mod = types.ModuleType("run_agent")

    class FakeAIAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_conversation(
            self,
            user_message: str,
            task_id: str | None = None,
            system_message: str | None = None,
        ):
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

    def fake_run_session_train_worker_config(cfg, *, once=False):  # type: ignore[no-untyped-def]
        del once
        Path(cfg["save_path"]).write_text("stub-policy", encoding="utf-8")
        Path(cfg["state_path"]).write_text(
            json.dumps(
                {
                    "updates": 1,
                    "trained_samples": 1,
                    "records_seen": 1,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        worker_cli,
        "run_session_train_worker_config",
        fake_run_session_train_worker_config,
    )

    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Create x.txt",
                "expected_output": "done",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    session_log_path = tmp_path / "sessions.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    policy_path = tmp_path / "policy.pt"
    state_path = tmp_path / "worker_state.json"
    eval_output_path = tmp_path / "eval_dataset"
    config_path = tmp_path / "online_cycle.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  integration: hermes\n"
            "  model: demo\n"
            "  session_sidecar:\n"
            "    enabled: true\n"
            f"    session_log_path: {session_log_path}\n"
            "environment:\n"
            f"  dataset_path: {dataset_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "online_cycle:\n"
            "  cycles: 1\n"
            "  limit: 1\n"
            "  session_replay:\n"
            f"    output_path: {replay_path}\n"
            "  session_train_worker:\n"
            "    algo: bc\n"
            f"    save_path: {policy_path}\n"
            f"    state_path: {state_path}\n"
            "    train:\n"
            "      n_epochs: 1\n"
            "      batch_size: 1\n"
            "      lr: 0.001\n"
            "      min_reward: 0.0\n"
            "  session_eval_export:\n"
            f"    output_path: {eval_output_path}\n"
            "    source: hermes-session\n"
            "    train_ratio: 0.5\n"
            "    val_ratio: 0.25\n"
            "    seed: 0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "online-cycle",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "[online-cycle]" in out
    assert session_log_path.exists()
    assert replay_path.exists()
    assert policy_path.exists()
    assert state_path.exists()
    assert (eval_output_path / "manifest.json").exists()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["updates"] == 1
    assert state["trained_samples"] == 1

    manifest = json.loads((eval_output_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["samples"]["total"] == 1
