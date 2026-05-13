import json
from pathlib import Path

from hermes_agentic_rl.collectors import sidecar as sidecar_module
from hermes_agentic_rl.collectors.sidecar import (
    LocalSessionSidecarConfig,
    close_all_sidecars,
    get_or_create_sidecar,
)
from hermes_agentic_rl.offline.replay_buffer import ReplayBuffer


def test_local_session_sidecar_writes_session_and_replay_files(tmp_path: Path):
    session_log_path = tmp_path / "sidecar_sessions.jsonl"
    replay_output_path = tmp_path / "sidecar_replay.jsonl"
    sidecar = get_or_create_sidecar(
        LocalSessionSidecarConfig(
            session_log_path=str(session_log_path),
            replay_output_path=str(replay_output_path),
            backend={"name": "tiny"},
            flush_interval_sec=0.01,
            max_batch_size=2,
        )
    )

    try:
        accepted = sidecar.submit(
            {
                "session_id": "sess-1",
                "task_id": "task-1",
                "messages": [
                    {"role": "user", "content": "Create a file"},
                    {"role": "assistant", "content": "I created it"},
                    {"role": "tool", "content": "success: file written"},
                    {"role": "user", "content": "great"},
                ],
            }
        )
        assert accepted is True
        sidecar.flush()
    finally:
        close_all_sidecars()

    assert session_log_path.exists()
    assert replay_output_path.exists()
    session_record = json.loads(session_log_path.read_text(encoding="utf-8").splitlines()[0])
    assert session_record["task_id"] == "task-1"
    buffer = ReplayBuffer.load_jsonl(replay_output_path)
    assert len(buffer.samples) == 1
    assert buffer.samples[0].reward > 0


def test_local_session_sidecar_uses_judge_config_for_breakdown(tmp_path: Path):
    session_log_path = tmp_path / "judge_sidecar_sessions.jsonl"
    replay_output_path = tmp_path / "judge_sidecar_replay.jsonl"
    sidecar = get_or_create_sidecar(
        LocalSessionSidecarConfig(
            session_log_path=str(session_log_path),
            replay_output_path=str(replay_output_path),
            backend={"name": "tiny"},
            judge={"components": [{"name": "session_toolcall_reward", "weight": 1.0}]},
            flush_interval_sec=0.01,
            max_batch_size=1,
        )
    )

    try:
        accepted = sidecar.submit(
            {
                "session_id": "sess-judge",
                "task_id": "task-judge",
                "messages": [
                    {"role": "user", "content": "Create a file"},
                    {
                        "role": "assistant",
                        "content": "I created it",
                        "tool_calls": [{"name": "write_file"}],
                    },
                    {"role": "tool", "content": "success: file written"},
                ],
            }
        )
        assert accepted is True
        sidecar.flush()
    finally:
        close_all_sidecars()

    buffer = ReplayBuffer.load_jsonl(replay_output_path)
    assert len(buffer.samples) == 1
    assert len(buffer.samples[0].metadata["reward_components"]) == 1
    assert buffer.samples[0].metadata["reward_components"][0]["name"] == "session_toolcall_reward"


def test_local_session_sidecar_snapshot_exposes_health_metrics(tmp_path: Path):
    session_log_path = tmp_path / "health_sidecar_sessions.jsonl"
    replay_output_path = tmp_path / "health_sidecar_replay.jsonl"
    sidecar = get_or_create_sidecar(
        LocalSessionSidecarConfig(
            session_log_path=str(session_log_path),
            replay_output_path=str(replay_output_path),
            backend={"name": "tiny"},
            flush_interval_sec=0.01,
            max_batch_size=1,
            queue_maxsize=4,
        )
    )

    try:
        accepted = sidecar.submit(
            {
                "session_id": "sess-health",
                "task_id": "task-health",
                "messages": [
                    {"role": "user", "content": "Create a file"},
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "thanks"},
                ],
            }
        )
        assert accepted is True
        sidecar.flush()
        snap = sidecar.snapshot()
    finally:
        close_all_sidecars()

    assert snap["worker_alive"] is True
    assert snap["closed"] is False
    assert snap["submitted"] == 1
    assert snap["written_sessions"] == 1
    assert snap["written_samples"] == 1
    assert snap["flushes"] >= 1
    assert snap["flush_errors"] == 0
    assert snap["queue_maxsize"] == 4
    assert snap["queue_size"] == 0
    assert "last_flush_ts" in snap
    assert "last_submit_ts" in snap


def test_local_session_sidecar_survives_flush_error(tmp_path: Path, monkeypatch):
    session_log_path = tmp_path / "error_sidecar_sessions.jsonl"
    replay_output_path = tmp_path / "error_sidecar_replay.jsonl"
    original_append = sidecar_module.append_jsonl
    calls = {"count": 0}

    def flaky_append(path, payloads):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        return original_append(path, payloads)

    monkeypatch.setattr(sidecar_module, "append_jsonl", flaky_append)
    sidecar = get_or_create_sidecar(
        LocalSessionSidecarConfig(
            session_log_path=str(session_log_path),
            replay_output_path=str(replay_output_path),
            backend={"name": "tiny"},
            flush_interval_sec=0.01,
            max_batch_size=1,
        )
    )

    try:
        assert sidecar.submit(
            {
                "session_id": "sess-error-1",
                "task_id": "task-error-1",
                "messages": [
                    {"role": "user", "content": "Create a file"},
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "thanks"},
                ],
            }
        )
        sidecar.flush()
        first_snap = sidecar.snapshot()

        assert sidecar.submit(
            {
                "session_id": "sess-error-2",
                "task_id": "task-error-2",
                "messages": [
                    {"role": "user", "content": "Create another file"},
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "thanks"},
                ],
            }
        )
        sidecar.flush()
        second_snap = sidecar.snapshot()
    finally:
        close_all_sidecars()

    assert first_snap["flush_errors"] == 1
    assert first_snap["worker_alive"] is True
    assert "RuntimeError: boom" in first_snap["last_error"]
    assert second_snap["flush_errors"] == 1
    assert second_snap["written_sessions"] == 1
    assert second_snap["written_samples"] == 1
    assert second_snap["worker_alive"] is True
