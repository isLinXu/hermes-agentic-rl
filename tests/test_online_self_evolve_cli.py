import importlib
import json
import types
from pathlib import Path

import hermes_agentic_rl.cli.session_train_worker_cli as worker_cli
from hermes_agentic_rl.cli.main import main


def test_online_self_evolve_cli_runs_cycle_and_skill_export(
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
                        "content": "I created x.txt with the file tool.",
                        "tool_calls": [
                            {
                                "name": "write_file",
                                "arguments": {"path": "x.txt"},
                            }
                        ],
                    },
                    {"role": "tool", "name": "write_file", "content": '{"ok": true}'},
                    {"role": "user", "content": "great, thanks"},
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

    output_dir = tmp_path / "online_self_evolve"
    session_log_path = output_dir / "sessions.jsonl"
    replay_path = output_dir / "replay.jsonl"
    policy_path = output_dir / "policy.pt"
    state_path = output_dir / "worker_state.json"
    eval_output_path = output_dir / "eval_dataset"
    skill_output_path = output_dir / "skill_candidates"
    config_path = tmp_path / "online_self_evolve.yaml"
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
            "    replay_mining:\n"
            "      min_skill_reward: 0.0\n"
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
            "online_self_evolve:\n"
            f"  output_dir: {output_dir}\n"
            "  stages:\n"
            "    online_cycle: true\n"
            "    skill_export: true\n"
            "    eval_gate: false\n"
            "  skill_export:\n"
            f"    input_path: {replay_path}\n"
            f"    output_dir: {skill_output_path}\n"
            "    require_skill_candidate: true\n"
            "    min_reward: 0.0\n"
            "    quality_min_examples: 1\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "online-self-evolve",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "[online-self-evolve]" in out
    assert (output_dir / "online_self_evolve_summary.json").exists()
    assert (output_dir / "online_self_evolve_report.md").exists()
    assert (skill_output_path / "summary.json").exists()

    summary = json.loads(
        (output_dir / "online_self_evolve_summary.json").read_text(encoding="utf-8")
    )
    assert summary["signals"]["sessions"] == 1
    assert summary["signals"]["replay_records"] == 1
    assert summary["signals"]["skills_exported"] == 1
    assert summary["signals"]["skill_quality"]["status_counts"]["ready_for_review"] == 1
    assert summary["stages"]["online_cycle"]["exit_code"] == 0
    assert summary["stages"]["skill_export"]["skills_exported"] == 1
    assert summary["stages"]["skill_export"]["status_counts"]["ready_for_review"] == 1
