import asyncio
import json
from pathlib import Path

from hermes_agentic_rl.collectors.sidecar import close_all_sidecars
from hermes_agentic_rl.core.trajectory import trajectory_to_dict
from hermes_agentic_rl.framework import build_framework
from hermes_agentic_rl.offline.replay_buffer import ReplayBuffer
from hermes_agentic_rl.trainers.base import BaseTrainer


class RecorderTrainer(BaseTrainer):
    async def submit(self, item, trajectory, reward_summary):  # type: ignore[no-untyped-def]
        return {
            "task_id": item["task_id"],
            "reward": reward_summary.final_score,
            "trajectory": trajectory_to_dict(trajectory),
        }


def test_env_training_pipeline_collects_judges_and_exports():
    config = {
        "runtime": {"integration": "fake"},
        "backend": {"name": "tiny"},
        "reward": {
            "components": [
                {"name": "outcome_reward", "weight": 0.5},
                {"name": "toolcall_reward", "weight": 0.5},
            ]
        },
    }
    framework = build_framework(
        config,
        trainer=RecorderTrainer(),
        build_sidecar=False,
    )
    item = {
        "task_id": "task-1",
        "instruction": "create hello.txt",
        "expected_output": "done",
    }

    trajectory, summary = asyncio.run(framework.env.collect_and_judge(item))
    payload = asyncio.run(framework.env.export(item, trajectory, summary))
    description = framework.describe()

    assert trajectory.final_output == "done"
    assert summary.final_score == 1.0
    assert payload["reward"] == 1.0
    assert description["env"]["runtime_integration"] == "fake"
    assert description["env"]["has_trainer_bridge"] is True
    assert description["session"]["tokenizer"] == "TinyTokenizer"


def test_session_training_pipeline_uses_configured_judge_for_replay_conversion():
    config = {
        "runtime": {"integration": "fake"},
        "backend": {"name": "tiny"},
        "judge": {
            "components": [
                {"name": "session_toolcall_reward", "weight": 1.0},
            ]
        },
    }
    framework = build_framework(config, build_sidecar=False)
    record = {
        "session_id": "sess-1",
        "task_id": "task-1",
        "messages": [
            {"role": "user", "content": "Fix it"},
            {
                "role": "assistant",
                "content": "Updated",
                "tool_calls": [{"name": "write_file"}],
            },
            {"role": "tool", "content": "success: file written"},
            {"role": "user", "content": "great, thanks"},
        ],
    }

    samples = framework.session.samples_from_record(record)
    train_samples = framework.session.train_samples_from_record(record)
    replay_buffer = framework.session.replay_buffer_from_record(record)
    replay_payloads = framework.session.replay_payloads_from_record(record)
    description = framework.session.describe()

    assert len(samples) == 1
    assert train_samples[0].reward == 0.4
    assert replay_buffer.samples[0].reward == 0.4
    assert replay_buffer.samples[0].metadata["session_id"] == "sess-1"
    assert replay_payloads[0]["reward"] == 0.4
    assert description["judge_components"] == ["session_toolcall_reward"]


def test_session_training_pipeline_builds_preference_pairs_from_replay_records():
    config = {
        "runtime": {"integration": "fake"},
        "backend": {"name": "tiny"},
    }
    framework = build_framework(config, build_sidecar=False)
    replay_records = [
        {
            "prompt_ids": [1, 2, 3],
            "response_ids": [10, 11],
            "reward": 1.0,
            "metadata": {"session_id": "sess-pos"},
        },
        {
            "prompt_ids": [1, 2, 3],
            "response_ids": [20, 21],
            "reward": -0.5,
            "metadata": {"session_id": "sess-neg"},
        },
    ]

    train_samples = framework.session.replay_train_samples_from_records(
        replay_records,
        min_reward=-1.0,
    )
    replay_buffer = framework.session.replay_buffer_from_replay_records(
        replay_records,
        min_reward=-1.0,
    )
    pairs = framework.session.preference_pairs_from_replay_records(
        replay_records,
        min_reward=-1.0,
        min_reward_gap=0.25,
        max_pairs_per_prompt=1,
    )

    assert len(train_samples) == 2
    assert len(replay_buffer.samples) == 2
    assert len(pairs) == 1
    assert pairs[0].metadata["chosen_reward"] == 1.0
    assert pairs[0].metadata["rejected_reward"] == -0.5


def test_session_training_pipeline_can_submit_to_sidecar(tmp_path: Path):
    session_log_path = tmp_path / "framework_sessions.jsonl"
    replay_output_path = tmp_path / "framework_replay.jsonl"
    config = {
        "runtime": {
            "integration": "fake",
            "session_sidecar": {
                "enabled": True,
                "session_log_path": str(session_log_path),
                "replay_output_path": str(replay_output_path),
                "backend": {"name": "tiny"},
                "judge": {
                    "components": [
                        {"name": "session_toolcall_reward", "weight": 1.0},
                    ]
                },
            },
        },
        "backend": {"name": "tiny"},
    }
    framework = build_framework(config, build_sidecar=True)
    record = {
        "session_id": "sess-sidecar",
        "task_id": "task-sidecar",
        "messages": [
            {"role": "user", "content": "Fix it"},
            {
                "role": "assistant",
                "content": "Updated",
                "tool_calls": [{"name": "write_file"}],
            },
            {"role": "tool", "content": "success: file written"},
            {"role": "user", "content": "great"},
        ],
    }

    try:
        assert framework.session.submit_record_to_sidecar(record) is True
        framework.session.flush_sidecar()
    finally:
        framework.session.close_sidecar()
        close_all_sidecars()

    assert session_log_path.exists()
    session_rows = [
        json.loads(line)
        for line in session_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(session_rows) == 1
    assert session_rows[0]["session_id"] == "sess-sidecar"

    buffer = ReplayBuffer.load_jsonl(replay_output_path)
    assert len(buffer.samples) == 1
    assert buffer.samples[0].reward == 0.4
