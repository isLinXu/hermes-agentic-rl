"""Tests for curriculum scheduler module.

Covers:
  - CurriculumStage: observe, is_mastered, rolling_mean, reset
  - CurriculumScheduler: observe_batch_stats, should_advance, advance,
    get_stage_weights, snapshot, state_dict/load_state_dict
  - create_default_curriculum: factory function
  - compute_turn_reward: tool call extraction and scoring
  - compute_outcome_reward: answer correctness
"""

from __future__ import annotations

import pytest

from hermes_agentic_rl.curriculum import (
    CurriculumScheduler,
    CurriculumSchedulerConfig,
    CurriculumStage,
    compute_outcome_reward,
    compute_turn_reward,
    create_default_curriculum,
)


# ---------------------------------------------------------------------------
# CurriculumStage
# ---------------------------------------------------------------------------


class TestCurriculumStage:
    def test_observe_and_rolling_mean(self):
        stage = CurriculumStage(name="test", mastery_threshold=0.7, sustain_evals=3)
        stage.observe(0.5)
        stage.observe(0.8)
        stage.observe(0.9)
        assert abs(stage.rolling_mean() - 0.7333) < 0.01

    def test_is_mastered_below_threshold(self):
        stage = CurriculumStage(name="test", mastery_threshold=0.7, sustain_evals=3)
        stage.observe(0.5)
        assert not stage.is_mastered()

    def test_is_mastered_consecutive_above(self):
        stage = CurriculumStage(name="test", mastery_threshold=0.7, sustain_evals=3)
        stage.observe(0.8)
        stage.observe(0.8)
        assert not stage.is_mastered()  # only 2 consecutive
        stage.observe(0.8)
        assert stage.is_mastered()  # 3 consecutive above threshold

    def test_consecutive_reset_on_below(self):
        stage = CurriculumStage(name="test", mastery_threshold=0.7, sustain_evals=3)
        stage.observe(0.8)
        stage.observe(0.8)
        stage.observe(0.5)  # resets consecutive counter
        assert not stage.is_mastered()
        stage.observe(0.8)
        stage.observe(0.8)
        assert not stage.is_mastered()  # only 2 consecutive after reset

    def test_reset(self):
        stage = CurriculumStage(name="test", mastery_threshold=0.7, sustain_evals=3)
        stage.observe(0.8)
        stage.observe(0.8)
        stage.reset()
        assert stage.rolling_mean() == 0.0
        assert not stage.is_mastered()

    def test_rolling_window(self):
        stage = CurriculumStage(name="test", mastery_threshold=0.7, sustain_evals=3)
        for _ in range(25):
            stage.observe(0.8)
        # Should keep only last 20
        assert len(stage._rolling_scores) <= 20


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------


class TestCurriculumScheduler:
    def _make_scheduler(self, **kwargs):
        stages = [
            CurriculumStage(name="stage_a", reward_weight=0.5, mastery_threshold=0.7, sustain_evals=3, primary_metric="metric_a"),
            CurriculumStage(name="stage_b", reward_weight=0.7, mastery_threshold=0.6, sustain_evals=3, primary_metric="metric_b"),
            CurriculumStage(name="stage_c", reward_weight=1.0, mastery_threshold=0.5, sustain_evals=3, primary_metric="metric_c"),
        ]
        cfg_kwargs = {"auto_advance": True, "min_iters_per_stage": 2}
        cfg_kwargs.update(kwargs)
        cfg = CurriculumSchedulerConfig(**cfg_kwargs)
        return CurriculumScheduler(stages=stages, cfg=cfg)

    def test_initial_state(self):
        sched = self._make_scheduler()
        assert sched.current_stage_index == 0
        assert sched.get_current_stage().name == "stage_a"
        assert not sched.is_complete()

    def test_observe_and_advance(self):
        sched = self._make_scheduler()
        # Need min_iters_per_stage=2 before advancing
        sched.observe_batch_stats(0, {"metric_a": 0.8})
        assert not sched.should_advance()  # only 1 iter in stage

        sched.observe_batch_stats(1, {"metric_a": 0.8})
        assert not sched.should_advance()  # only 2 consecutive, need 3

        sched.observe_batch_stats(2, {"metric_a": 0.8})
        assert sched.should_advance()  # 3 consecutive above threshold + 3 iters

        new_stage = sched.advance()
        assert new_stage is not None
        assert new_stage.name == "stage_b"
        assert sched.current_stage_index == 1

    def test_no_advance_below_threshold(self):
        sched = self._make_scheduler()
        for i in range(5):
            sched.observe_batch_stats(i, {"metric_a": 0.5})  # below threshold
        assert not sched.should_advance()

    def test_auto_advance_disabled(self):
        sched = self._make_scheduler(auto_advance=False)
        for i in range(5):
            sched.observe_batch_stats(i, {"metric_a": 0.9})
        assert not sched.should_advance()

    def test_complete_curriculum(self):
        sched = self._make_scheduler()
        # Advance through all stages
        for stage_idx in range(3):
            for i in range(5):
                sched.observe_batch_stats(i, {f"metric_{'abc'[stage_idx]}": 0.9})
            if sched.should_advance():
                sched.advance()
        assert sched.is_complete()

    def test_advance_returns_none_when_complete(self):
        sched = self._make_scheduler()
        sched._current_idx = 3  # past all stages
        assert sched.advance() is None

    def test_get_stage_weights(self):
        sched = self._make_scheduler()
        weights = sched.get_stage_weights()
        assert weights["stage_a"] == 0.5  # current stage
        assert weights["stage_b"] == 0.0  # future
        assert weights["stage_c"] == 0.0  # future

    def test_get_stage_weights_after_advance(self):
        sched = self._make_scheduler()
        sched._current_idx = 1
        weights = sched.get_stage_weights()
        assert weights["stage_a"] == 0.25  # completed: 0.5 * 0.5
        assert weights["stage_b"] == 0.7  # current
        assert weights["stage_c"] == 0.0  # future

    def test_snapshot(self):
        sched = self._make_scheduler()
        snap = sched.snapshot()
        assert "curriculum/current_stage" in snap
        assert snap["curriculum/current_stage"] == "stage_a"

    def test_state_dict_roundtrip(self):
        sched = self._make_scheduler()
        sched.observe_batch_stats(0, {"metric_a": 0.8})
        sched.observe_batch_stats(1, {"metric_a": 0.9})

        state = sched.state_dict()
        sched2 = self._make_scheduler()
        sched2.load_state_dict(state)

        assert sched2.current_stage_index == sched.current_stage_index
        assert sched2._iter_count == sched._iter_count

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="at least one stage"):
            CurriculumScheduler(stages=[])


# ---------------------------------------------------------------------------
# create_default_curriculum
# ---------------------------------------------------------------------------


class TestCreateDefaultCurriculum:
    def test_returns_three_stages(self):
        stages = create_default_curriculum()
        assert len(stages) == 3
        assert stages[0].name == "tool_call_structure"
        assert stages[1].name == "tool_call_content"
        assert stages[2].name == "summary_quality"

    def test_stages_have_primary_metrics(self):
        stages = create_default_curriculum()
        for stage in stages:
            assert stage.primary_metric  # non-empty


# ---------------------------------------------------------------------------
# compute_turn_reward
# ---------------------------------------------------------------------------


class TestComputeTurnReward:
    def test_no_tool_calls_valid_summary(self):
        reward, is_valid = compute_turn_reward("This is a valid summary of the results.")
        assert is_valid
        assert reward == 0.3

    def test_no_tool_calls_short_response(self):
        reward, is_valid = compute_turn_reward("hi")
        assert not is_valid
        assert reward == 0.0

    def test_with_json_tool_call(self):
        response = '```json\n{"name": "calculator", "arguments": {"expr": "2+2"}}\n```'
        reward, is_valid = compute_turn_reward(response)
        assert reward > 0.0

    def test_with_expected_tool_names(self):
        response = '```json\n{"name": "calculator", "arguments": {"expr": "2+2"}}\n```'
        reward, _ = compute_turn_reward(response, expected_tool_names=["calculator"])
        assert reward > 0.3  # should be higher with name match


# ---------------------------------------------------------------------------
# compute_outcome_reward
# ---------------------------------------------------------------------------


class TestCurriculumOutcomeReward:
    def test_correct_answer(self):
        reward, is_correct = compute_outcome_reward("42", "42")
        assert is_correct
        assert reward >= 0.9

    def test_wrong_answer(self):
        reward, is_correct = compute_outcome_reward("37", "42")
        assert not is_correct
