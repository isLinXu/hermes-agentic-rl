import json
from pathlib import Path

from hermes_agentic_rl.cli.main import main
from hermes_agentic_rl.offline.replay_buffer import ReplayBuffer


def test_session_replay_cli_exports_replay_buffer_jsonl(tmp_path: Path, monkeypatch, capsys):
    session_path = tmp_path / "sessions.jsonl"
    session_path.write_text(
        json.dumps(
            {
                "session_id": "sess-1",
                "task_id": "task-1",
                "messages": [
                    {"role": "user", "content": "Create a file"},
                    {"role": "assistant", "content": "I created it."},
                    {"role": "tool", "content": "success: file written"},
                    {"role": "user", "content": "great, thanks"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "replay.jsonl"
    config_path = tmp_path / "session_replay.yaml"
    config_path.write_text(
        (
            f"input_path: {session_path}\n"
            f"output_path: {output_path}\n"
            "backend:\n"
            "  name: tiny\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-replay",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "[session-replay]" in out
    assert output_path.exists()

    buffer = ReplayBuffer.load_jsonl(output_path)
    assert len(buffer.samples) == 1
    assert buffer.samples[0].reward > 0
    assert buffer.samples[0].metadata["session_id"] == "sess-1"


def test_session_replay_cli_accepts_trajectory_payload(tmp_path: Path, monkeypatch):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "task_id": "traj-1",
                "prompt": "Do work",
                "steps": [],
                "final_output": "done",
                "finished_naturally": True,
                "turns_used": 1,
                "metadata": {
                    "messages": [
                        {"role": "user", "content": "Do work"},
                        {"role": "assistant", "content": "Done"},
                        {"role": "user", "content": "wrong, try again"},
                    ],
                    "runtime": {"task_id": "hermes-task-9"},
                },
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "trajectory_replay.jsonl"
    config_path = tmp_path / "trajectory_replay.yaml"
    config_path.write_text(
        (
            f"input_path: {trajectory_path}\n"
            f"output_path: {output_path}\n"
            "backend:\n"
            "  name: tiny\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-replay",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    buffer = ReplayBuffer.load_jsonl(output_path)
    assert len(buffer.samples) == 1
    assert buffer.samples[0].reward < 0


def test_session_replay_cli_supports_judge_component_config(tmp_path: Path, monkeypatch):
    session_path = tmp_path / "judge_sessions.jsonl"
    session_path.write_text(
        json.dumps(
            {
                "session_id": "sess-j",
                "task_id": "task-j",
                "messages": [
                    {"role": "user", "content": "Create a file"},
                    {
                        "role": "assistant",
                        "content": "I created it.",
                        "tool_calls": [{"name": "write_file"}],
                    },
                    {"role": "tool", "content": "success: file written"},
                    {"role": "user", "content": "great"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "judge_replay.jsonl"
    config_path = tmp_path / "judge_replay.yaml"
    config_path.write_text(
        (
            f"input_path: {session_path}\n"
            f"output_path: {output_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "judge:\n"
            "  components:\n"
            "    - name: session_toolcall_reward\n"
            "      weight: 1.0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-replay",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    buffer = ReplayBuffer.load_jsonl(output_path)
    assert len(buffer.samples) == 1
    components = buffer.samples[0].metadata["reward_components"]
    assert len(components) == 1
    assert components[0]["name"] == "session_toolcall_reward"


def test_session_replay_cli_adds_direction_aware_replay_mining(
    tmp_path: Path,
    monkeypatch,
):
    session_path = tmp_path / "mining_sessions.jsonl"
    report_path = tmp_path / "mining_report.json"
    session_path.write_text(
        json.dumps(
            {
                "session_id": "sess-mining",
                "task_id": "task-mining",
                "messages": [
                    {"role": "user", "content": "Create a report file"},
                    {
                        "role": "assistant",
                        "content": "I created the report file.",
                        "tool_calls": [
                            {
                                "name": "write_file",
                                "arguments": {"path": "report.md"},
                            }
                        ],
                    },
                    {"role": "tool", "content": "success: file written"},
                    {"role": "user", "content": "great, thanks"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "mining_replay.jsonl"
    config_path = tmp_path / "mining_replay.yaml"
    config_path.write_text(
        (
            f"input_path: {session_path}\n"
            f"output_path: {output_path}\n"
            f"quality_report_path: {report_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "replay_mining:\n"
            "  min_skill_reward: 0.0\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-replay",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    buffer = ReplayBuffer.load_jsonl(output_path)
    mining = buffer.samples[0].metadata["replay_mining"]
    assert mining["skill_candidate"] is True
    assert "tool_use_reliability" in mining["axes"]
    assert "skill_learning" in mining["axes"]
    assert "tool_reliability_replay" in mining["recommended_uses"]
    assert "skill_candidate" in mining["recommended_uses"]
    assert buffer.samples[0].metadata["capability_axes"] == mining["axes"]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["replay_mining"]["samples_scored"] == 1
    assert report["replay_mining"]["skill_candidates"] == 1
    assert report["replay_mining"]["by_axis"]["tool_use_reliability"] == 1
    assert report["replay_mining"]["by_recommended_use"]["skill_candidate"] == 1


def test_session_replay_cli_can_disable_replay_mining(
    tmp_path: Path,
    monkeypatch,
):
    session_path = tmp_path / "no_mining_sessions.jsonl"
    report_path = tmp_path / "no_mining_report.json"
    session_path.write_text(
        json.dumps(
            {
                "session_id": "sess-no-mining",
                "task_id": "task-no-mining",
                "messages": [
                    {"role": "user", "content": "Create a file"},
                    {"role": "assistant", "content": "Created the file."},
                    {"role": "user", "content": "great"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "no_mining_replay.jsonl"
    config_path = tmp_path / "no_mining_replay.yaml"
    config_path.write_text(
        (
            f"input_path: {session_path}\n"
            f"output_path: {output_path}\n"
            f"quality_report_path: {report_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "replay_mining: false\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-replay",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    sample = ReplayBuffer.load_jsonl(output_path).samples[0]
    assert "replay_mining" not in sample.metadata
    assert "capability_axes" not in sample.metadata

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["replay_mining"]["enabled"] is False
    assert report["replay_mining"]["samples_scored"] == 0


def test_session_replay_cli_applies_quality_filters_and_writes_report(
    tmp_path: Path, monkeypatch, capsys
):
    session_path = tmp_path / "mixed_sessions.jsonl"
    quarantine_path = tmp_path / "session_replay_quarantine.jsonl"
    report_path = tmp_path / "session_replay_quality_report.json"

    session_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "session_id": "sess-good-1",
                        "task_id": "task-1",
                        "messages": [
                            {"role": "user", "content": "Create a file"},
                            {"role": "assistant", "content": "I created it."},
                            {"role": "tool", "content": "success: file written"},
                            {"role": "user", "content": "great, thanks"},
                        ],
                    }
                ),
                "{bad json",
                json.dumps(
                    {
                        "session_id": "sess-invalid",
                        "task_id": "task-invalid",
                    }
                ),
                json.dumps(
                    {
                        "session_id": "sess-dup",
                        "task_id": "task-dup",
                        "messages": [
                            {"role": "user", "content": "Create a file"},
                            {"role": "assistant", "content": "I created it."},
                            {"role": "tool", "content": "success: file written"},
                            {"role": "user", "content": "great, thanks"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "session_id": "sess-short",
                        "task_id": "task-short",
                        "messages": [
                            {"role": "user", "content": "Patch this"},
                            {"role": "assistant", "content": "ok"},
                            {"role": "user", "content": "thanks"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "session_id": "sess-good-2",
                        "task_id": "task-2",
                        "messages": [
                            {"role": "user", "content": "Update the docs"},
                            {"role": "assistant", "content": "Updated the docs."},
                            {"role": "user", "content": "great"},
                        ],
                    }
                ),
            ]
        )
        + "\n"
        + '{"session_id":"partial"',
        encoding="utf-8",
    )

    output_path = tmp_path / "quality_replay.jsonl"
    config_path = tmp_path / "quality_replay.yaml"
    config_path.write_text(
        (
            f"input_path: {session_path}\n"
            f"output_path: {output_path}\n"
            f"quarantine_path: {quarantine_path}\n"
            f"quality_report_path: {report_path}\n"
            "backend:\n"
            "  name: tiny\n"
            "data_quality:\n"
            "  min_response_tokens: 5\n"
            "  dedupe_within_scan: true\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "hermes-agentic-rl",
            "session-replay",
            "--config",
            str(config_path),
        ],
    )

    assert main() == 0
    out = capsys.readouterr().out
    assert "quality_filtered=2" in out
    assert "quarantined=5" in out

    buffer = ReplayBuffer.load_jsonl(output_path)
    assert len(buffer.samples) == 2
    assert {sample.metadata["session_id"] for sample in buffer.samples} == {
        "sess-good-1",
        "sess-good-2",
    }

    quarantine_rows = [
        json.loads(line)
        for line in quarantine_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(quarantine_rows) == 5
    assert {row["kind"] for row in quarantine_rows} == {
        "invalid_json_line",
        "invalid_session_record",
        "partial_json_line",
        "quality_filtered_replay_record",
    }
    assert {
        row["reason"]
        for row in quarantine_rows
        if row["kind"] == "quality_filtered_replay_record"
    } == {"duplicate_record", "response_too_short"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["sessions_loaded"] == 5
    assert report["sessions_exported"] == 4
    assert report["turns_seen"] == 4
    assert report["raw_samples_generated"] == 4
    assert report["samples_written"] == 2
    assert report["input_rejections"]["count"] == 2
    assert report["session_rejections"]["count"] == 1
    assert report["replay_invalid"]["count"] == 0
    assert report["quality_filtered"]["count"] == 2
    assert report["quality_filtered"]["by_reason"] == {
        "duplicate_record": 1,
        "response_too_short": 1,
    }
    assert report["quarantined"]["count"] == 5
    assert report["reward_distribution"]["positive"] == 2
    assert report["replay_mining"]["enabled"] is True
    assert report["replay_mining"]["samples_scored"] == 2
    assert report["data_quality"] == {
        "min_response_tokens": 5,
        "dedupe_within_scan": True,
    }
