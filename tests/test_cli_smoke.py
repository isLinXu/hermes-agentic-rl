import asyncio
from pathlib import Path

from hermes_agentic_rl import __version__
from hermes_agentic_rl.cli.main import build_parser
from hermes_agentic_rl.config import load_config
from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.rollout_manager import RolloutManager
from hermes_agentic_rl.core.trainer_bridge import TrainerBridge
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.outcome_reward import OutcomeReward
from hermes_agentic_rl.rewards.toolcall_reward import ToolcallReward
from hermes_agentic_rl.trainers.atropos_grpo import AtroposGrpoTrainer


def test_package_version_and_cli_parser():
    # Track current release version (bumped every release); just assert shape.
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2 and all(p[:1].isdigit() for p in parts[:2])
    parser = build_parser()
    assert parser.prog == "hermes-agentic-rl"
    assert parser.parse_args(["rollout"]).command == "rollout"
    assert parser.parse_args(["train"]).command == "train"


def test_load_config_reads_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  model: demo\n"
            "environment:\n"
            "  type: terminal_task_env\n"
            "reward:\n"
            "  aggregator: weighted_sum\n"
            "trainer:\n"
            "  type: atropos_grpo\n"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["runtime"]["model"] == "demo"
    assert config["trainer"]["type"] == "atropos_grpo"


def test_load_config_reads_runtime_integration_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "runtime:\n"
            "  model: demo\n"
            "environment:\n"
            "  dataset_path: data/tasks.jsonl\n"
            "trainer:\n"
            "  export_training_path: outputs/train.jsonl\n"
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["runtime"]["integration"] == "fake"
    assert config["environment"]["dataset_path"] == "data/tasks.jsonl"


class FakeAgentLoop:
    async def run(self, prompt: str) -> dict:
        return {
            "messages": [{"role": "assistant", "content": "done"}],
            "tool_calls": [[{"name": "write_file", "arguments": {"path": "a.txt"}}]],
            "tool_results": [[{"ok": True}]],
            "final_output": "done",
            "finished_naturally": True,
            "turns_used": 1,
        }


def test_rollout_manager_collects_trajectory():
    manager = RolloutManager(agent_loop=FakeAgentLoop())

    trajectory = asyncio.run(manager.collect({"task_id": "t1"}, "create a file"))

    assert isinstance(trajectory, Trajectory)
    assert trajectory.final_output == "done"


class FakeEndToEndLoop:
    async def run(self, prompt: str) -> dict:
        return {
            "messages": [{"role": "assistant", "content": "done"}],
            "tool_calls": [[{"name": "write_file", "arguments": {"path": "hello.txt"}}]],
            "tool_results": [[{"ok": True, "path": "hello.txt"}]],
            "final_output": "done",
            "finished_naturally": True,
            "turns_used": 1,
        }


class FakeLoopWithMetadata:
    async def run(self, prompt: str) -> dict:
        return {
            "messages": [{"role": "assistant", "content": "done"}],
            "tool_calls": [[{"name": "write_file", "arguments": {"path": "meta.txt"}}]],
            "tool_results": [[{"ok": True}]],
            "final_output": "done",
            "finished_naturally": True,
            "turns_used": 1,
            "metadata": {"runtime": "fake", "session_id": "demo-session"},
        }


def test_end_to_end_minimal_pipeline(tmp_path: Path):
    item = {"task_id": "task-1", "expected_output": "done"}
    rollout_manager = RolloutManager(agent_loop=FakeEndToEndLoop())
    trajectory = asyncio.run(rollout_manager.collect(item, "create hello.txt"))

    reward_manager = RewardManager(rewards=[OutcomeReward(weight=0.7), ToolcallReward(weight=0.3)])
    summary = asyncio.run(reward_manager.evaluate(item, trajectory, tool_context=None))

    bridge = TrainerBridge(AtroposGrpoTrainer(output_path=tmp_path / "train.jsonl"))
    payload = asyncio.run(bridge.submit(item, trajectory, summary))

    assert summary.final_score == 1.0
    assert payload["trajectory"]["final_output"] == "done"


def test_rollout_manager_preserves_runtime_metadata():
    manager = RolloutManager(agent_loop=FakeLoopWithMetadata())

    trajectory = asyncio.run(manager.collect({"task_id": "t-meta"}, "create meta.txt"))

    assert trajectory.metadata["runtime"]["runtime"] == "fake"
    assert trajectory.metadata["runtime"]["session_id"] == "demo-session"
