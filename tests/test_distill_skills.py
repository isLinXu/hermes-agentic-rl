from __future__ import annotations

import json
from pathlib import Path

from hermes_agentic_rl.collectors.distill_skills import distill_skills


def _write_mock_sessions(path: Path) -> None:
    sessions = [
        {
            "session_id": "s1",
            "task_id": "fix",
            "reward": 0.9,
            "messages": [
                {"role": "user", "content": "Fix the divide-by-zero bug in calc.py."},
                {
                    "role": "assistant",
                    "content": "I'll patch it.\n<tool_call>{\"name\": \"write_file\", "
                    '"arguments": {"path": "calc.py"}}</tool_call>',
                },
                {"role": "tool", "content": "tests passed: 5/5"},
                {"role": "user", "content": "Great, that works. Thanks!"},
            ],
        },
        {
            "session_id": "s2",
            "task_id": "readme",
            "reward": 0.8,
            "messages": [
                {"role": "user", "content": "Create a README and run the build."},
                {
                    "role": "assistant",
                    "content": "Creating it.\n<tool_call>{\"name\": \"write_file\", "
                    '"arguments": {"path": "README.md"}}</tool_call>',
                },
                {"role": "tool", "content": "file written: README.md"},
                {"role": "user", "content": "correct, done"},
            ],
        },
        {
            "session_id": "s3",
            "task_id": "danger",
            "reward": -0.2,
            "messages": [
                {"role": "user", "content": "Delete temp files."},
                {
                    "role": "assistant",
                    "content": "<tool_call>{\"name\": \"run\", "
                    '"arguments": {"cmd": "rm -rf /"}}</tool_call>',
                },
                {"role": "tool", "content": "error: refused"},
                {"role": "user", "content": "That was wrong and dangerous, failed."},
            ],
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(json.dumps(s) + "\n")


def test_distill_skills_end_to_end(tmp_path: Path) -> None:
    traces = tmp_path / "sessions.jsonl"
    _write_mock_sessions(traces)
    out = tmp_path / "skills"

    summary = distill_skills(trace_paths=[traces], output_dir=out, min_reward=0.0)

    # Auto-mining ran: 3 sessions split into per-turn records.
    assert summary["mining"]["sessions_mined"] == 3
    assert summary["mining"]["turns_mined"] >= 3

    # At least one installable skill package was emitted.
    assert summary["skills_exported"] >= 1

    # The dangerous rm -rf turn must NOT become a skill candidate.
    assert summary["rejected_records"] >= 1

    # Standard artifacts exist.
    assert (out / "DISTILL_REPORT.md").exists()
    assert (out / "summary.json").exists()
    assert (out / "quality_report.json").exists()

    # Each exported skill is a self-contained, installable package.
    for skill in summary["skills"]:
        skill_dir = Path(skill["path"])
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "validation.jsonl").exists()
        assert (skill_dir / "manifest.json").exists()
        md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert md.startswith("---")  # frontmatter for use_skill loading
        assert "## Procedure" in md


def test_distill_skills_accepts_bare_message_list(tmp_path: Path) -> None:
    # A single anonymous session given as a bare list of OpenAI messages.
    traces = tmp_path / "bare.json"
    traces.write_text(
        json.dumps(
            [
                {"role": "user", "content": "Write a hello world script."},
                {
                    "role": "assistant",
                    "content": "<tool_call>{\"name\": \"write_file\", "
                    '"arguments": {"path": "hello.py"}}</tool_call>',
                },
                {"role": "tool", "content": "file written"},
                {"role": "user", "content": "works, thanks"},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "skills"
    summary = distill_skills(trace_paths=[traces], output_dir=out, min_reward=0.0)
    assert summary["mining"]["sessions_mined"] == 1
    assert summary["mining"]["turns_mined"] >= 1
