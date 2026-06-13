from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_agentic_rl.trainers.profiling import (
    StepProfiler,
    append_jsonl,
    format_json_record,
)


def test_step_profiler_accumulates_named_durations() -> None:
    profiler = StepProfiler(enabled=True)

    with profiler.measure("rollout.collect"):
        time.sleep(0.001)
    with profiler.measure("rollout.collect"):
        time.sleep(0.001)

    metrics = profiler.as_metrics()
    assert "time_rollout_collect_ms" in metrics
    assert metrics["time_rollout_collect_ms"] > 0.0


def test_disabled_step_profiler_emits_no_metrics() -> None:
    profiler = StepProfiler(enabled=False)

    with profiler.measure("algo.compute_loss"):
        pass

    assert profiler.as_metrics() == {}


def test_json_record_formatting_handles_paths(tmp_path: Path) -> None:
    line = format_json_record(
        {
            "iter": 1,
            "output_dir": tmp_path,
            "nested": {"path": tmp_path / "profile.jsonl"},
        }
    )
    payload = json.loads(line)

    assert payload["iter"] == 1
    assert payload["output_dir"] == str(tmp_path)
    assert payload["nested"]["path"] == str(tmp_path / "profile.jsonl")


def test_append_jsonl_writes_safe_record(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "profile.jsonl"

    append_jsonl(path, {"iter": 2, "path": tmp_path})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"iter": 2, "path": str(tmp_path)}


def test_trainer_profile_metrics_are_recorded(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    from hermes_agentic_rl.backends.tiny import TinyBackendConfig, TinyCausalLMBackend
    from hermes_agentic_rl.core.reward_manager import RewardManager
    from hermes_agentic_rl.envs.echo_task_env import (
        EchoRewardComponent,
        EchoTaskEnv,
        build_default_echo_dataset,
    )
    from hermes_agentic_rl.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    profile_path = tmp_path / "profile.jsonl"
    backend = TinyCausalLMBackend(TinyBackendConfig(dim=16, n_heads=2, n_layers=1, seed=0))
    trainer = GRPOTrainer(
        policy=backend,
        env=EchoTaskEnv(build_default_echo_dataset()),
        reward_manager=RewardManager([EchoRewardComponent(weight=1.0)]),
        cfg=GRPOTrainerConfig(
            n_iters=1,
            group_size=2,
            prompts_per_iter=1,
            max_new_tokens=2,
            log_every=0,
            profile=True,
            profile_output_path=profile_path,
            seed=0,
        ),
    )

    stats = trainer.train()

    record = stats.iters[-1]
    assert record["time_iter_total_ms"] > 0.0
    assert record["time_rollout_collect_ms"] > 0.0
    assert record["time_update_total_ms"] > 0.0

    profile_record = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile_record["iter"] == 0
    assert profile_record["algo"] == "grpo"
    assert profile_record["time_iter_total_ms"] > 0.0
