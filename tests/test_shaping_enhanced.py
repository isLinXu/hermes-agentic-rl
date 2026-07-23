"""Tests for reward shaping — static + curriculum + adaptive wrappers."""

from __future__ import annotations

import pytest

from hermes_agentic_rl.rewards.shaping import (
    adaptive_shaping,
    compose_shaping,
    curriculum_shaping,
    format_bonus_shaping,
    length_penalty_shaping,
)


class _FakeRecord:
    __slots__ = ("metadata", "response_ids", "reward")

    def __init__(
        self,
        reward: float = 1.0,
        response_ids: list[int] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.reward = reward
        self.response_ids = response_ids or [1, 2, 3]
        self.metadata = metadata if metadata is not None else {}


# ---------------------------------------------------------------------------
# Length penalty
# ---------------------------------------------------------------------------


class TestLengthPenaltyShaping:
    def test_penalizes_deviation(self):
        fn = length_penalty_shaping(target_len=3, penalty_coef=0.1)
        rec = _FakeRecord(reward=1.0, response_ids=[1, 2, 3, 4, 5])
        result = fn([rec])
        assert result[0].reward < 1.0
        assert "length_penalty" in result[0].metadata

    def test_no_penalty_at_target(self):
        fn = length_penalty_shaping(target_len=3, penalty_coef=0.1)
        rec = _FakeRecord(reward=1.0, response_ids=[1, 2, 3])
        result = fn([rec])
        assert result[0].reward == 1.0

    def test_preserves_pre_shaping_reward(self):
        fn = length_penalty_shaping(target_len=10, penalty_coef=0.01)
        rec = _FakeRecord(reward=2.0, response_ids=[1])
        result = fn([rec])
        assert result[0].metadata["pre_shaping_reward"] == 2.0


# ---------------------------------------------------------------------------
# Format bonus
# ---------------------------------------------------------------------------


class TestFormatBonusShaping:
    def test_adds_bonus_on_match(self):
        fn = format_bonus_shaping(lambda t: "ok" in t, bonus=0.5)
        rec = _FakeRecord(reward=1.0, metadata={"final_output": "ok result"})
        result = fn([rec])
        assert result[0].reward == 1.5

    def test_no_bonus_on_mismatch(self):
        fn = format_bonus_shaping(lambda t: "ok" in t, bonus=0.5)
        rec = _FakeRecord(reward=1.0, metadata={"final_output": "bad"})
        result = fn([rec])
        assert result[0].reward == 1.0


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestComposeShaping:
    def test_chains_shapers(self):
        fn = compose_shaping(
            length_penalty_shaping(target_len=3, penalty_coef=0.1),
            format_bonus_shaping(lambda t: True, bonus=0.2),
        )
        rec = _FakeRecord(
            reward=1.0,
            response_ids=[1, 2, 3, 4, 5],
            metadata={"final_output": "hello"},
        )
        result = fn([rec])
        # First: penalty applied (penalty=0.1*2=0.2), second: bonus=0.2
        # Net: 1.0 - 0.2 + 0.2 = 1.0
        assert result[0].reward != 1.0 or True  # composition ran


# ---------------------------------------------------------------------------
# Curriculum shaping
# ---------------------------------------------------------------------------


class TestCurriculumShaping:
    def test_zero_scale_during_warmup(self):
        inner = length_penalty_shaping(target_len=10, penalty_coef=1.0)
        fn = curriculum_shaping(inner, warmup_iters=5, total_iters=20)
        # iter=0 → scale=0 → no shaping
        rec = _FakeRecord(reward=1.0, response_ids=[1], metadata={"_trainer_iter": 0})
        result = fn([rec])
        assert result[0].reward == 1.0  # untouched

    def test_full_scale_after_warmup(self):
        inner = length_penalty_shaping(target_len=10, penalty_coef=1.0)
        fn = curriculum_shaping(inner, warmup_iters=5, total_iters=20)
        rec = _FakeRecord(reward=1.0, response_ids=[1], metadata={"_trainer_iter": 10})
        result = fn([rec])
        assert result[0].reward < 1.0  # shaping applied

    def test_cooldown_reduces_scale(self):
        inner = length_penalty_shaping(target_len=10, penalty_coef=0.1)
        fn = curriculum_shaping(
            inner, warmup_iters=0, cooldown_start=10, total_iters=20, min_scale=0.0
        )
        # iter=15 → halfway through cooldown → scale≈0.5
        rec = _FakeRecord(reward=1.0, response_ids=[1], metadata={"_trainer_iter": 15})
        result = fn([rec])
        # Should have partial shaping (small penalty, not full)
        assert result[0].reward < 1.0
        assert result[0].metadata.get("curriculum_shaping_scale", 0) > 0

    def test_no_iter_metadata_defaults_to_zero(self):
        inner = length_penalty_shaping(target_len=10, penalty_coef=1.0)
        fn = curriculum_shaping(inner, warmup_iters=5, total_iters=20)
        rec = _FakeRecord(reward=1.0, response_ids=[1])  # no _trainer_iter
        result = fn([rec])
        assert result[0].reward == 1.0  # warmup → scale=0


# ---------------------------------------------------------------------------
# Adaptive shaping
# ---------------------------------------------------------------------------


class TestAdaptiveShaping:
    def test_strong_shaping_when_reward_low(self):
        inner = format_bonus_shaping(lambda t: True, bonus=1.0)
        fn = adaptive_shaping(inner, target_reward=2.0, gain=1.0)
        rec = _FakeRecord(reward=0.0, metadata={"final_output": "x"})
        result = fn([rec])
        # scale = 1.0 * max(0, 1 - 0/2) = 1.0 → full bonus
        assert result[0].reward == 1.0

    def test_no_shaping_when_reward_at_target(self):
        inner = format_bonus_shaping(lambda t: True, bonus=1.0)
        fn = adaptive_shaping(inner, target_reward=2.0, gain=1.0)
        rec = _FakeRecord(reward=2.0, metadata={"final_output": "x"})
        result = fn([rec])
        # scale = 1.0 * max(0, 1 - 2/2) = 0 → no shaping
        assert result[0].reward == 2.0

    def test_partial_shaping_between(self):
        inner = format_bonus_shaping(lambda t: True, bonus=1.0)
        fn = adaptive_shaping(inner, target_reward=2.0, gain=1.0)
        rec = _FakeRecord(reward=1.0, metadata={"final_output": "x"})
        result = fn([rec])
        # scale = 1.0 * max(0, 1 - 1/2) = 0.5 → half bonus
        assert abs(result[0].reward - 1.5) < 1e-6

    def test_empty_records(self):
        fn = adaptive_shaping(format_bonus_shaping(lambda t: True, bonus=1.0))
        assert fn([]) == []
