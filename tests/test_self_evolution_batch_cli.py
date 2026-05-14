import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from hermes_agentic_rl.cli.main import main


def _write_session(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_self_evolution_batch_cli_runs_directional_training_pipeline(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    session_path = tmp_path / "sessions.jsonl"
    _write_session(
        session_path,
        [
            {
                "session_id": "sess-tool-1",
                "task_id": "task-tool-1",
                "turns_used": 2,
                "final_output": "created file",
                "reward": 0.9,
                "metadata": {
                    "prompt": "Create a file",
                    "runtime": {"integration": "hermes"},
                },
                "messages": [
                    {"role": "user", "content": "Create a file"},
                    {"role": "assistant", "content": "I created it."},
                    {"role": "tool", "content": "success: file written"},
                    {"role": "user", "content": "thanks"},
                ],
            },
            {
                "session_id": "sess-doc-1",
                "task_id": "task-doc-1",
                "turns_used": 3,
                "final_output": "updated docs",
                "reward": 0.8,
                "metadata": {"runtime": {"integration": "hermes"}},
                "messages": [
                    {"role": "user", "content": "Update the docs"},
                    {"role": "assistant", "content": "Updated the README."},
                    {"role": "user", "content": "looks good"},
                ],
            },
            {
                "session_id": "sess-fix-1",
                "task_id": "task-fix-1",
                "turns_used": 4,
                "final_output": "fixed bug",
                "reward": 0.7,
                "metadata": {"runtime": {"integration": "hermes"}},
                "messages": [
                    {"role": "user", "content": "Fix the bug"},
                    {"role": "assistant", "content": "I found and fixed it."},
                    {"role": "tool", "content": "tests passed"},
                    {"role": "user", "content": "great"},
                ],
            },
        ],
    )

    output_dir = tmp_path / "batch"
    config_path = tmp_path / "self_evolution_batch.yaml"
    config_path.write_text(
        (
            "backend:\n"
            "  name: tiny\n"
            "  dim: 16\n"
            "  n_heads: 2\n"
            "  n_layers: 2\n"
            "self_evolution_batch:\n"
            f"  input_path: {session_path}\n"
            f"  output_dir: {output_dir}\n"
            "  directions:\n"
            "    - name: tool_use_reliability\n"
            "      objective:\n"
            "        target_metrics: [tool_call_parse_ok, tool_name_match]\n"
            "      session_replay:\n"
            "        data_quality:\n"
            "          min_prompt_tokens: 1\n"
            "          min_response_tokens: 1\n"
            "      session_train_worker:\n"
            "        algo: bc\n"
            "        train:\n"
            "          n_epochs: 1\n"
            "          batch_size: 1\n"
            "          lr: 0.001\n"
            "          min_reward: 0.0\n"
            "      session_eval_export:\n"
            "        train_ratio: 0.5\n"
            "        val_ratio: 0.25\n"
            "        seed: 0\n"
            "    - name: recovery_and_completion\n"
            "      objective:\n"
            "        target_metrics: [success_rate, finished_naturally_rate]\n"
            "      session_replay:\n"
            "        data_quality:\n"
            "          min_prompt_tokens: 1\n"
            "          min_response_tokens: 1\n"
            "      session_train_worker:\n"
            "        algo: bc\n"
            "        train:\n"
            "          n_epochs: 1\n"
            "          batch_size: 1\n"
            "          lr: 0.001\n"
            "          min_reward: 0.0\n"
            "      session_eval_export:\n"
            "        train_ratio: 0.5\n"
            "        val_ratio: 0.25\n"
            "        seed: 0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "self-evolution-batch",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "[self-evolution-batch]" in out

    summary_path = output_dir / "batch_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["totals"]["directions"] == 2
    assert summary["totals"]["worker_updates"] == 2
    assert summary["totals"]["replay_samples"] == 6

    first = summary["directions"][0]
    assert first["slug"] == "tool_use_reliability"
    assert first["objective"]["target_metrics"] == [
        "tool_call_parse_ok",
        "tool_name_match",
    ]
    assert first["capability_plan"]["axes"] == ["tool_use_reliability"]
    assert "tool_use_reliability" in summary["capability_axes"]
    assert "task_success" in summary["capability_axes"]
    assert first["replay"]["samples_written"] == 3
    assert first["worker"]["updates"] == 1

    direction_dir = output_dir / "tool_use_reliability"
    assert (direction_dir / "replay.jsonl").exists()
    assert (direction_dir / "policy.pt").exists()
    assert (direction_dir / "worker_state.json").exists()
    assert (direction_dir / "self_evolution_dataset" / "manifest.json").exists()
    direction_summary = json.loads(
        (direction_dir / "direction_summary.json").read_text(encoding="utf-8")
    )
    assert direction_summary["self_evolution_dataset"]["samples"]["total"] == 3
