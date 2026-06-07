from __future__ import annotations

import asyncio

import pytest

from hermes_agentic_rl.cli.train_rl import _build_env_and_rewards
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.envs.context_benchmark import (
    ContextBenchmarkConfig,
    ContextBenchmarkEnv,
    ContextBenchmarkReward,
    build_context_benchmark_dataset,
)


def test_context_benchmark_reward_scores_retention_and_distractors() -> None:
    asyncio.run(_test_retention_and_distractors())


async def _test_retention_and_distractors() -> None:
    item = build_context_benchmark_dataset(n=1, seed=0, noise_blocks=3)[0]
    output = (
        "Project atlas uses label blue. The relevant component is CSV parser; "
        "the tool found latency regression and owner team-0. ship summary first; "
        "do not mention stale context."
    )
    reward = await ContextBenchmarkReward(ContextBenchmarkConfig()).evaluate(
        item,
        Trajectory(
            task_id=str(item["task_id"]),
            prompt=str(item["prompt"]),
            steps=[],
            final_output=output,
            finished_naturally=True,
            turns_used=1,
        ),
        tool_context=None,
    )

    assert reward.score > 0.90
    assert reward.metadata["context_required_fact_recall"] == 1.0
    assert reward.metadata["context_tool_summary_retention"] == 1.0
    assert reward.metadata["context_distractor_avoidance"] == 1.0
    assert reward.metadata["context_compression_ok"] == 1.0


def test_context_benchmark_reward_penalizes_stale_context() -> None:
    asyncio.run(_test_penalizes_stale_context())


async def _test_penalizes_stale_context() -> None:
    item = build_context_benchmark_dataset(n=1, seed=0, noise_blocks=3)[0]
    stale = item["forbidden_facts"][0]
    reward = await ContextBenchmarkReward(ContextBenchmarkConfig()).evaluate(
        item,
        Trajectory(
            task_id=str(item["task_id"]),
            prompt=str(item["prompt"]),
            steps=[],
            final_output=f"Project atlas uses label blue, but {stale}.",
            finished_naturally=True,
            turns_used=1,
        ),
        tool_context=None,
    )

    assert reward.metadata["context_distractor_avoidance"] < 1.0
    assert reward.metadata["context_forbidden_hits"] == 1
    assert reward.score < 0.80


def test_context_benchmark_env_builds_prompts_and_supervised_samples() -> None:
    item = build_context_benchmark_dataset(n=1, seed=1, noise_blocks=2)[0]
    env = ContextBenchmarkEnv([item])

    prompt = env.format_prompt(item)
    samples = env.build_supervised_samples(item)

    assert "Relevant memory facts" in prompt
    assert "Distractor or stale context" in prompt
    assert samples[0].metadata["capability_axis"] == "prompt_context"
    assert samples[0].response == item["target_response"]


def test_train_rl_factory_supports_context_benchmark_env() -> None:
    env, reward_manager = _build_env_and_rewards(
        {
            "environment": {
                "type": "context_benchmark",
                "dataset_size": 2,
                "dataset_seed": 3,
                "noise_blocks": 2,
            }
        }
    )

    assert isinstance(env, ContextBenchmarkEnv)
    assert reward_manager.rewards[0].name == "context_benchmark_reward"
