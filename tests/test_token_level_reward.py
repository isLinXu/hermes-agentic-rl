"""Tests for token-level reward module.

Covers:
  - compute_token_advantages: per-token advantage normalization
  - assign_tool_call_token_scores: per-token score assignment
  - compute_kl_penalty: KL divergence penalty (k1/k2/k3)
  - compute_length_penalty: cosine and linear length penalty
  - compute_combined_reward: reward composition
  - compute_outcome_reward: exact + partial credit
  - compute_trajectory_token_rewards: end-to-end
"""

from __future__ import annotations

import math

import pytest

from hermes_agentic_rl.rewards.token_level_reward import (
    TokenRewardConfig,
    assign_tool_call_token_scores,
    compute_combined_reward,
    compute_kl_penalty,
    compute_length_penalty,
    compute_outcome_reward,
    compute_token_advantages,
    compute_trajectory_token_rewards,
)

# ---------------------------------------------------------------------------
# compute_token_advantages
# ---------------------------------------------------------------------------


class TestComputeTokenAdvantages:
    def test_empty_input(self):
        assert compute_token_advantages([], []) == []

    def test_all_prompt_tokens(self):
        scores = [0.0, 0.0, 0.0]
        masks = [0, 0, 0]
        advs = compute_token_advantages(scores, masks)
        assert all(a == 0.0 for a in advs)

    def test_single_completion_token(self):
        scores = [0.0, 1.0]
        masks = [0, 1]
        advs = compute_token_advantages(scores, masks)
        # Single completion token: std=1.0 (fallback), advantage = (1.0 - 1.0) / (1.0 + eps) = 0
        assert len(advs) == 2
        assert advs[0] == 0.0  # prompt token

    def test_multiple_completion_tokens(self):
        scores = [0.0, 1.0, 2.0, 3.0]
        masks = [0, 1, 1, 1]
        advs = compute_token_advantages(scores, masks)
        assert len(advs) == 4
        assert advs[0] == 0.0  # prompt token
        # Completion tokens should be normalized
        completion_advs = advs[1:]
        assert all(isinstance(a, float) for a in completion_advs)

    def test_identical_scores(self):
        scores = [0.0, 5.0, 5.0, 5.0]
        masks = [0, 1, 1, 1]
        advs = compute_token_advantages(scores, masks)
        # All same score → std=0 → fallback to 1.0 → advantage = (5-5)/1 = 0
        for a in advs[1:]:
            assert abs(a) < 1e-4


# ---------------------------------------------------------------------------
# assign_tool_call_token_scores
# ---------------------------------------------------------------------------


class TestAssignToolCallTokenScores:
    def test_no_tool_calls(self):
        scores = assign_tool_call_token_scores(
            token_ids=[1, 2, 3],
            token_texts=["a", "b", "c"],
            tool_call_spans=[],
        )
        assert scores == [0.0, 0.0, 0.0]

    def test_single_tool_call_span(self):
        scores = assign_tool_call_token_scores(
            token_ids=[1, 2, 3, 4, 5],
            token_texts=["a", "b", "c", "d", "e"],
            tool_call_spans=[(1, 3, 0.8)],
        )
        assert scores[0] == 0.0  # before span
        assert scores[1] == 0.8  # in span
        assert scores[2] == 0.8  # in span
        assert scores[3] == 0.0  # after span
        assert scores[4] == 0.0  # after span

    def test_overlapping_spans(self):
        scores = assign_tool_call_token_scores(
            token_ids=[1, 2, 3, 4],
            token_texts=["a", "b", "c", "d"],
            tool_call_spans=[(0, 2, 0.5), (1, 3, 0.9)],
        )
        # Second span overwrites first at index 1
        assert scores[0] == 0.5
        assert scores[1] == 0.9  # overwritten by second span
        assert scores[2] == 0.9
        assert scores[3] == 0.0

    def test_custom_base_score(self):
        scores = assign_tool_call_token_scores(
            token_ids=[1, 2, 3],
            token_texts=["a", "b", "c"],
            tool_call_spans=[(1, 2, 0.7)],
            base_score=0.1,
        )
        assert scores[0] == 0.1
        assert scores[1] == 0.7
        assert scores[2] == 0.1


# ---------------------------------------------------------------------------
# compute_kl_penalty
# ---------------------------------------------------------------------------


class TestComputeKLPenalty:
    def test_zero_coef(self):
        result = compute_kl_penalty([0.1], [0.2], [1], kl_coef=0.0)
        assert result == 0.0

    def test_k3_estimator(self):
        # k3: 0.5 * (log_ratio)^2
        log_probs = [math.log(0.5)]
        ref_log_probs = [math.log(0.25)]
        masks = [1]
        result = compute_kl_penalty(log_probs, ref_log_probs, masks, kl_coef=1.0, estimator="k3")
        expected = 0.5 * (math.log(0.5) - math.log(0.25)) ** 2
        assert abs(result - expected) < 1e-6

    def test_k1_estimator(self):
        log_probs = [math.log(0.5)]
        ref_log_probs = [math.log(0.25)]
        masks = [1]
        result = compute_kl_penalty(log_probs, ref_log_probs, masks, kl_coef=1.0, estimator="k1")
        expected = math.log(0.5) - math.log(0.25)
        assert abs(result - expected) < 1e-6

    def test_prompt_tokens_masked(self):
        log_probs = [0.0, math.log(0.5)]
        ref_log_probs = [0.0, math.log(0.25)]
        masks = [0, 1]
        result = compute_kl_penalty(log_probs, ref_log_probs, masks, kl_coef=1.0, estimator="k3")
        # Only completion token (mask=1) should contribute
        expected = 0.5 * (math.log(0.5) - math.log(0.25)) ** 2
        assert abs(result - expected) < 1e-6

    def test_invalid_estimator(self):
        with pytest.raises(ValueError, match="Unknown KL estimator"):
            compute_kl_penalty([0.1], [0.2], [1], kl_coef=1.0, estimator="k4")


# ---------------------------------------------------------------------------
# compute_length_penalty
# ---------------------------------------------------------------------------


class TestComputeLengthPenalty:
    def test_zero_coef(self):
        assert compute_length_penalty(100, coef=0.0) == 0.0

    def test_zero_length(self):
        assert compute_length_penalty(0, coef=0.01) == 0.0

    def test_at_target_length_cosine(self):
        # At target length, cosine penalty should be minimal
        penalty = compute_length_penalty(512, target_length=512, coef=0.01, cosine_schedule=True)
        # cos(π * 512/1024) = cos(π/2) = 0, so penalty = 0.01 * (1 + 0) = 0.01
        assert abs(penalty - 0.01) < 1e-6

    def test_at_zero_length_cosine(self):
        # At length 0, cos(0) = 1, so penalty = 0.01 * (1 + 1) = 0.02
        penalty = compute_length_penalty(0, target_length=512, coef=0.01, cosine_schedule=True)
        assert penalty == 0.0  # length 0 returns 0

    def test_linear_penalty(self):
        penalty = compute_length_penalty(0, target_length=512, coef=0.01, cosine_schedule=False)
        assert penalty == 0.0  # length 0 returns 0

    def test_linear_penalty_deviation(self):
        penalty = compute_length_penalty(256, target_length=512, coef=0.01, cosine_schedule=False)
        expected = 0.01 * 256 / 512
        assert abs(penalty - expected) < 1e-6


# ---------------------------------------------------------------------------
# compute_combined_reward
# ---------------------------------------------------------------------------


class TestComputeCombinedReward:
    def test_basic(self):
        result = compute_combined_reward(1.0)
        assert result == 1.0

    def test_with_kl_penalty(self):
        result = compute_combined_reward(1.0, kl_penalty=0.2)
        assert abs(result - 0.8) < 1e-6

    def test_with_length_penalty(self):
        result = compute_combined_reward(1.0, length_penalty=0.1)
        assert abs(result - 0.9) < 1e-6

    def test_with_outcome_bonus(self):
        result = compute_combined_reward(0.5, outcome_bonus=0.25)
        assert abs(result - 0.75) < 1e-6

    def test_reward_floor(self):
        result = compute_combined_reward(-5.0, reward_floor=-1.0)
        assert result == -1.0

    def test_all_components(self):
        result = compute_combined_reward(
            1.0, kl_penalty=0.1, length_penalty=0.05, outcome_bonus=0.25, reward_floor=-1.0
        )
        expected = 1.0 + 0.25 - 0.1 - 0.05
        assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# compute_outcome_reward
# ---------------------------------------------------------------------------


class TestComputeOutcomeReward:
    def test_exact_match(self):
        assert compute_outcome_reward("42", "42") == 1.0

    def test_empty_response(self):
        assert compute_outcome_reward("", "42") == 0.0

    def test_empty_gold(self):
        assert compute_outcome_reward("42", "") == 0.0

    def test_gold_contained_in_response(self):
        assert compute_outcome_reward("The answer is 42", "42") == 1.0

    def test_boxed_answer(self):
        assert compute_outcome_reward(r"\boxed{42}", "42") == 1.0

    def test_numeric_match(self):
        assert compute_outcome_reward("The result is 3.14", "3.14") >= 0.9

    def test_no_partial_credit(self):
        result = compute_outcome_reward("something", "42", partial_credit=False)
        assert result == 0.0

    def test_partial_credit_close(self):
        # Should give some partial credit for overlapping content
        result = compute_outcome_reward("The answer is approximately 42", "42")
        assert result > 0.0


# ---------------------------------------------------------------------------
# compute_trajectory_token_rewards (end-to-end)
# ---------------------------------------------------------------------------


class TestComputeTrajectoryTokenRewards:
    def test_basic_trajectory(self):
        token_ids = [1, 2, 3, 4, 5]
        token_texts = ["a", "b", "c", "d", "e"]
        masks = [0, 1, 1, 1, 1]
        tool_call_spans = [(1, 3, 0.8)]
        trajectory_reward = 0.5

        advantages, combined = compute_trajectory_token_rewards(
            token_ids, token_texts, masks, tool_call_spans, trajectory_reward
        )

        assert len(advantages) == 5
        assert advantages[0] == 0.0  # prompt token
        assert combined > 0.0  # should be positive

    def test_no_tool_calls(self):
        token_ids = [1, 2, 3]
        token_texts = ["a", "b", "c"]
        masks = [0, 1, 1]
        tool_call_spans = []
        trajectory_reward = 0.0

        advantages, combined = compute_trajectory_token_rewards(
            token_ids, token_texts, masks, tool_call_spans, trajectory_reward
        )

        assert len(advantages) == 3
        # No tool calls + zero reward → combined should be at floor
        assert combined >= -1.0

    def test_with_kl_penalty(self):
        token_ids = [1, 2, 3]
        token_texts = ["a", "b", "c"]
        masks = [0, 1, 1]
        tool_call_spans = []
        trajectory_reward = 1.0
        log_probs = [0.0, -1.0, -1.5]
        ref_log_probs = [0.0, -0.5, -0.8]

        cfg = TokenRewardConfig(kl_coef=0.1)

        advantages, combined = compute_trajectory_token_rewards(
            token_ids,
            token_texts,
            masks,
            tool_call_spans,
            trajectory_reward,
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            cfg=cfg,
        )

        assert len(advantages) == 3
        # KL penalty should reduce combined reward
        assert combined < 1.0 + cfg.outcome_bonus
