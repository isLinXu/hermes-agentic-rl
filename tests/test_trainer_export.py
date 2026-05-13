import asyncio
import json
from pathlib import Path

from hermes_agentic_rl.core.trainer_bridge import TrainerBridge
from hermes_agentic_rl.core.types import RewardResult, RewardSummary, Trajectory
from hermes_agentic_rl.trainers.atropos_grpo import AtroposGrpoTrainer


def build_trajectory() -> Trajectory:
    return Trajectory(
        task_id="task-1",
        prompt="create a file",
        steps=[],
        final_output="done",
        finished_naturally=True,
        turns_used=1,
    )


def build_summary() -> RewardSummary:
    return RewardSummary(
        final_score=0.8,
        components=[RewardResult(name="outcome_reward", score=0.8, reason="ok")],
        metadata={"aggregator": "weighted_sum"},
    )


def test_trainer_export_includes_verifier_fields_when_present(tmp_path: Path):
    trajectory = build_trajectory()
    summary = RewardSummary(
        final_score=1.0,
        components=[
            RewardResult(name="filesystem_verifier_reward", score=0.0, reason="failed", metadata={"checked_files": 1, "passed_files": 0, "failures": [{"path": "x.txt", "reason": "file missing"}]}),
        ],
        metadata={"aggregator": "weighted_sum"},
    )
    output_path = tmp_path / "train.jsonl"
    trainer = AtroposGrpoTrainer(output_path=output_path)

    payload = trainer.submit_sync({"task_id": "task-1"}, trajectory, summary)

    assert payload["verifier_passed"] is False
    assert payload["verifier_checked_files"] == 1
    assert payload["verifier_passed_files"] == 0
    assert payload["verifier_failures"][0]["reason"] == "file missing"

def test_atropos_grpo_trainer_writes_jsonl(tmp_path: Path):
    trajectory = build_trajectory()
    summary = build_summary()
    output_path = tmp_path / "train.jsonl"
    trainer = AtroposGrpoTrainer(output_path=output_path)

    payload = trainer.submit_sync({"task_id": "task-1"}, trajectory, summary)

    assert payload["reward"] == 0.8
    written = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert written["task_id"] == "task-1"
    assert written["trajectory"]["final_output"] == "done"
    assert written["metadata"]["aggregator"] == "weighted_sum"


def test_trainer_bridge_submits_with_async_interface(tmp_path: Path):
    trajectory = build_trajectory()
    summary = build_summary()
    bridge = TrainerBridge(AtroposGrpoTrainer(output_path=tmp_path / "train.jsonl"))

    payload = asyncio.run(bridge.submit({"task_id": "task-1"}, trajectory, summary))

    assert payload["task_id"] == "task-1"
    assert payload["reward_components"][0]["name"] == "outcome_reward"
