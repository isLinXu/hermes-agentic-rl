from __future__ import annotations

import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main


def test_skill_export_cli_writes_candidate_skill_assets(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_ids": [1, 2, 3],
                        "response_ids": [4, 5, 6],
                        "reward": 0.9,
                        "metadata": {
                            "session_id": "sess-tool",
                            "task_id": "task-tool",
                            "turn_index": 0,
                            "capability_axes": [
                                "tool_use_reliability",
                                "skill_learning",
                                "self_evolution_signal",
                            ],
                            "source_turn": {
                                "prompt_messages": [
                                    {
                                        "role": "user",
                                        "content": "Create a report file with the file tool.",
                                    }
                                ],
                                "assistant_message": {
                                    "role": "assistant",
                                    "content": "I will call the file tool with the target path.",
                                    "tool_calls": [
                                        {
                                            "name": "write_file",
                                            "arguments": {"path": "report.md"},
                                        }
                                    ],
                                },
                                "feedback_messages": [
                                    {"role": "tool", "content": "success: file written"},
                                    {"role": "user", "content": "great"},
                                ],
                            },
                            "replay_mining": {
                                "axes": [
                                    "tool_use_reliability",
                                    "skill_learning",
                                    "self_evolution_signal",
                                ],
                                "reasons": ["assistant_tool_call", "skill_candidate"],
                                "recommended_uses": [
                                    "tool_reliability_replay",
                                    "skill_candidate",
                                ],
                                "skill_candidate": True,
                                "usefulness_score": 1.1,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [1, 2, 9],
                        "response_ids": [4, 5, 10],
                        "reward": 0.8,
                        "metadata": {
                            "session_id": "sess-tool-2",
                            "task_id": "task-tool-2",
                            "turn_index": 0,
                            "capability_axes": [
                                "tool_use_reliability",
                                "skill_learning",
                                "self_evolution_signal",
                            ],
                            "source_turn": {
                                "prompt_messages": [
                                    {
                                        "role": "user",
                                        "content": "Create a changelog file with the file tool.",
                                    }
                                ],
                                "assistant_message": {
                                    "role": "assistant",
                                    "content": "I will write the changelog with the file tool.",
                                    "tool_calls": [
                                        {
                                            "name": "write_file",
                                            "arguments": {"path": "CHANGELOG.md"},
                                        }
                                    ],
                                },
                                "feedback_messages": [
                                    {"role": "tool", "content": "success: file written"},
                                    {"role": "user", "content": "great"},
                                ],
                            },
                            "replay_mining": {
                                "axes": [
                                    "tool_use_reliability",
                                    "skill_learning",
                                    "self_evolution_signal",
                                ],
                                "reasons": ["assistant_tool_call", "skill_candidate"],
                                "recommended_uses": [
                                    "tool_reliability_replay",
                                    "skill_candidate",
                                ],
                                "skill_candidate": True,
                                "usefulness_score": 1.0,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "prompt_ids": [7],
                        "response_ids": [8],
                        "reward": 0.2,
                        "metadata": {
                            "replay_mining": {
                                "axes": ["task_success"],
                                "recommended_uses": ["positive_replay"],
                                "skill_candidate": False,
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "skills"
    config_path = tmp_path / "skill_export.yaml"
    config_path.write_text(
        (
            "skill_export:\n"
            f"  input_path: {replay_path}\n"
            f"  output_dir: {output_dir}\n"
            "  min_reward: 0.5\n"
            "  max_examples_per_skill: 4\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "skill-export", "--config", str(config_path)],
    )

    assert main() == 0
    assert "[skill-export]" in capsys.readouterr().out

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["records_read"] == 3
    assert summary["candidate_records"] == 2
    assert summary["skills_exported"] == 1
    assert summary["quality"]["status_counts"]["ready_for_review"] == 1
    skill_path = output_dir / "hermes-tool_use_reliability-candidate"
    skill_md = (skill_path / "SKILL.md").read_text(encoding="utf-8")
    manifest = json.loads((skill_path / "manifest.json").read_text(encoding="utf-8"))
    quality_report = json.loads((output_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert "Select tools only when they are necessary" in skill_md
    assert "Create a report file" in skill_md
    assert "Quality status: `ready_for_review`" in skill_md
    assert manifest["quality"]["status"] == "ready_for_review"
    assert quality_report["status_counts"]["ready_for_review"] == 1

    validation_rows = [
        json.loads(line)
        for line in (skill_path / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert validation_rows[0]["task_input"] == "Create a report file with the file tool."
    assert validation_rows[0]["metadata"]["axes"] == [
        "self_evolution_signal",
        "skill_learning",
        "tool_use_reliability",
    ]


def test_skill_export_quality_gate_detects_source_turn_negative_feedback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    replay_path = tmp_path / "negative_replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "prompt_ids": [1],
                "response_ids": [2],
                "reward": 0.9,
                "metadata": {
                    "capability_axes": ["tool_use_reliability"],
                    "source_turn": {
                        "prompt_messages": [
                            {"role": "user", "content": "Create a report file."}
                        ],
                        "assistant_message": {
                            "role": "assistant",
                            "content": "I created the file.",
                        },
                        "feedback_messages": [
                            {"role": "user", "content": "wrong file path, failed"}
                        ],
                    },
                    "replay_mining": {
                        "axes": ["tool_use_reliability"],
                        "recommended_uses": ["skill_candidate"],
                        "skill_candidate": True,
                        "usefulness_score": 1.0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "skills"
    config_path = tmp_path / "skill_export.yaml"
    config_path.write_text(
        (
            "skill_export:\n"
            f"  input_path: {replay_path}\n"
            f"  output_dir: {output_dir}\n"
            "  min_reward: 0.0\n"
            "  quality_min_examples: 1\n"
            "  quality_max_negative_signal_ratio: 0.0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["hermes-agentic-rl", "skill-export", "--config", str(config_path)],
    )

    assert main() == 0
    quality_report = json.loads((output_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert quality_report["status_counts"]["draft"] == 1
    skill_quality = quality_report["skills"][0]["quality"]
    assert skill_quality["metrics"]["negative_signal_ratio"] == 1.0
    assert "max_negative_signal_ratio" in skill_quality["blockers"]
