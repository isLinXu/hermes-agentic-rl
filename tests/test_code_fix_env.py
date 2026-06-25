from __future__ import annotations

import asyncio
import textwrap

import pytest

from hermes_agentic_rl.core.reward_manager import RewardManager
from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.envs.code_fix import CodeBug, CodeFixEnv, CodeFixReward


def _inprocess_sandbox(fixed_code: str, test_code: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Fast in-process substitute for subprocess sandbox in unit tests."""
    del timeout
    namespace: dict[str, object] = {}
    try:
        exec(fixed_code, namespace)
        exec(test_code, namespace)
        return True, ""
    except AssertionError as exc:
        return False, f"TEST_FAIL: {exc}"
    except Exception as exc:
        return False, f"TEST_ERROR: {type(exc).__name__}: {exc}"


@pytest.fixture(autouse=True)
def _fast_codefix_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_agentic_rl.envs.code_fix._run_tests_in_sandbox",
        _inprocess_sandbox,
    )


def _single_add_env() -> CodeFixEnv:
    bug = CodeBug(
        bug_id="add_bug",
        level=0,
        description="Fix add.",
        buggy_code="def add(a, b):\n    return a - b\n",
        test_code=textwrap.dedent(
            """\
            assert add(2, 3) == 5
            assert add(-1, 1) == 0
            assert add(0, 0) == 0
            """
        ),
        expected_keywords=["return", "add"],
    )
    return CodeFixEnv(bugs=[bug], seed=1, n_samples=1, max_level=0)


def test_code_fix_env_scores_passing_code_block() -> None:
    env = _single_add_env()
    item = asyncio.run(env.get_next_item())
    response = "<code>\ndef add(a, b):\n    return a + b\n</code>"

    assert env.score_response(item, response) == 1.0
    summary = env.compute_sequence_reward(item, response)
    assert summary == {"reward": 1.0, "passed": True, "partial": False}


def test_code_fix_env_scores_partial_test_passes() -> None:
    env = _single_add_env()
    item = asyncio.run(env.get_next_item())
    response = "<code>\ndef add(a, b):\n    return 0\n</code>"

    score = env.score_response(item, response)

    assert score == 2 / 3


def test_code_fix_env_keyword_bonus_without_code_block() -> None:
    env = _single_add_env()
    item = asyncio.run(env.get_next_item())

    assert env.score_response(item, "I would return from add") == 0.2


def test_code_fix_env_compute_reward_matches_base_env_contract() -> None:
    env = _single_add_env()
    item = asyncio.run(env.get_next_item())
    trajectory = Trajectory(
        task_id="add_bug",
        prompt=env.format_prompt(item),
        steps=[],
        final_output="<code>\ndef add(a, b):\n    return a + b\n</code>",
        finished_naturally=True,
        turns_used=1,
    )

    results = asyncio.run(env.compute_reward(item, trajectory, tool_context=None))

    assert len(results) == 1
    assert results[0].name == "codefix_reward"
    assert results[0].score == 1.0
    assert results[0].reason == "all tests passed"


def test_code_fix_reward_component_aggregates_with_reward_manager() -> None:
    env = _single_add_env()
    item = asyncio.run(env.get_next_item())
    trajectory = Trajectory(
        task_id="add_bug",
        prompt=env.format_prompt(item),
        steps=[],
        final_output="<code>\ndef add(a, b):\n    return a + b\n</code>",
        finished_naturally=True,
        turns_used=1,
    )
    manager = RewardManager([CodeFixReward(weight=0.5)])

    summary = asyncio.run(manager.evaluate(item, trajectory, tool_context=None))

    assert summary.final_score == 1.0
    assert summary.components[0].weight == 0.5
    assert summary.components[0].metadata["all_pass"] is True
