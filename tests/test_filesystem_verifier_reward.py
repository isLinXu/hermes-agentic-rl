import asyncio
from pathlib import Path

from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.filesystem_verifier_reward import FileSystemVerifierReward


def _empty_traj() -> Trajectory:
    return Trajectory(
        task_id="t",
        prompt="p",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )


def test_verifier_skips_when_no_expected_files(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reward = FileSystemVerifierReward(weight=0.6)
    result = asyncio.run(reward.evaluate(item={}, trajectory=_empty_traj(), tool_context=None))

    assert result.weight == 0.0
    assert result.score == 0.0


def test_verifier_fails_when_file_missing(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "hello.txt", "equals": "hello"}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 0.0
    assert result.metadata["checked_files"] == 1
    assert result.metadata["passed_files"] == 0


def test_verifier_passes_on_equals_match(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")

    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "hello.txt", "equals": "hello"}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 1.0
    assert result.metadata["passed_files"] == 1


def test_verifier_passes_on_contains_match(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "todo.txt").write_text("buy milk\nand eggs", encoding="utf-8")

    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "todo.txt", "contains": "buy milk"}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 1.0


def test_verifier_fails_when_min_bytes_not_met(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")  # 5 bytes

    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "hello.txt", "min_bytes": 6}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 0.0


def test_verifier_fails_when_max_bytes_exceeded(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")  # 5 bytes

    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "hello.txt", "max_bytes": 4}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 0.0


def test_verifier_passes_when_regex_matches(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")

    reward = FileSystemVerifierReward(weight=0.6)
    item = {"expected_files": [{"path": "hello.txt", "regex": r"he..o"}]}
    result = asyncio.run(reward.evaluate(item=item, trajectory=_empty_traj(), tool_context=None))

    assert result.score == 1.0
