"""Tests for multimodal reward components.

Covers:
- VisionMatchReward: text fallback, threshold logic, missing fields
- ImageAttributeReward: attribute matching, penalties, max cap
- MultimodalCompositeReward: weighted sum, max aggregation
- Config dataclasses: defaults, custom values
- Helper functions: tokenization, jaccard similarity
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hermes_agentic_rl.core.types import Trajectory
from hermes_agentic_rl.rewards.multimodal import (
    ImageAttributeConfig,
    ImageAttributeReward,
    MultimodalCompositeConfig,
    MultimodalCompositeReward,
    VisionMatchConfig,
    VisionMatchReward,
    _cosine_text_similarity,
    _jaccard_similarity,
    _tokenize_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(final_output: str = "") -> Trajectory:
    """Build a minimal Trajectory for testing."""
    return Trajectory(
        task_id="test-task",
        prompt="test prompt",
        steps=[],
        final_output=final_output,
        finished_naturally=True,
        turns_used=1,
        metadata={},
    )


def _run_async(coro):
    """Run an async coroutine in tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_tokenize_text(self):
        tokens = _tokenize_text("Hello World!")
        assert tokens == {"hello", "world"}

    def test_tokenize_empty(self):
        assert _tokenize_text("") == set()

    def test_jaccard_identical(self):
        assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        result = _jaccard_similarity({"a", "b"}, {"b", "c"})
        assert 0 < result < 1.0
        assert abs(result - 1 / 3) < 0.01

    def test_jaccard_both_empty(self):
        assert _jaccard_similarity(set(), set()) == 1.0

    def test_cosine_text_similarity(self):
        score = _cosine_text_similarity("a red circle", "a blue circle")
        assert 0 < score < 1.0

    def test_cosine_text_identical(self):
        assert _cosine_text_similarity("hello world", "hello world") == 1.0

    def test_cosine_text_no_overlap(self):
        assert _cosine_text_similarity("hello", "world") == 0.0


# ---------------------------------------------------------------------------
# VisionMatchReward tests
# ---------------------------------------------------------------------------


class TestVisionMatchReward:
    def test_missing_image_description_returns_zero(self):
        reward = VisionMatchReward()
        traj = _make_trajectory("a red car")
        result = _run_async(
            reward.evaluate({"task_id": "t1"}, traj, None)
        )
        assert result.score == 0.0
        assert "no image_description" in result.reason

    def test_high_similarity_positive_reward(self):
        cfg = VisionMatchConfig(
            similarity_threshold=0.3,
            positive_reward=1.0,
            negative_reward=0.0,
            use_clip=False,
        )
        reward = VisionMatchReward(cfg=cfg)
        traj = _make_trajectory("a red circle on white background")
        result = _run_async(
            reward.evaluate(
                {"task_id": "t1", "image_description": "red circle on white background"},
                traj,
                None,
            )
        )
        assert result.score == 1.0
        assert "threshold" in result.reason
        assert result.metadata["method"] == "text_jaccard"

    def test_low_similarity_negative_reward(self):
        cfg = VisionMatchConfig(
            similarity_threshold=0.9,
            positive_reward=1.0,
            negative_reward=0.0,
            use_clip=False,
        )
        reward = VisionMatchReward(cfg=cfg)
        traj = _make_trajectory("hello world")
        result = _run_async(
            reward.evaluate(
                {"task_id": "t1", "image_description": "a red circle"},
                traj,
                None,
            )
        )
        assert result.score == 0.0

    def test_custom_threshold_and_rewards(self):
        cfg = VisionMatchConfig(
            similarity_threshold=0.5,
            positive_reward=2.0,
            negative_reward=-1.0,
            use_clip=False,
        )
        reward = VisionMatchReward(cfg=cfg)
        traj = _make_trajectory("red circle shape")
        result = _run_async(
            reward.evaluate(
                {"task_id": "t1", "image_description": "red circle"},
                traj,
                None,
            )
        )
        assert result.score == 2.0

    def test_clip_disabled_by_default(self):
        reward = VisionMatchReward()
        assert reward.cfg.use_clip is False
        assert reward._clip_model is None

    def test_clip_load_failure_falls_back_to_text(self):
        cfg = VisionMatchConfig(use_clip=True, clip_model="nonexistent/model")
        reward = VisionMatchReward(cfg=cfg)
        traj = _make_trajectory("red circle")
        result = _run_async(
            reward.evaluate(
                {"task_id": "t1", "image_description": "red circle"},
                traj,
                None,
            )
        )
        # Should fall back to text similarity
        assert result.metadata["method"] == "text_jaccard"

    def test_visual_description_alias(self):
        reward = VisionMatchReward()
        traj = _make_trajectory("red circle")
        result = _run_async(
            reward.evaluate(
                {"task_id": "t1", "visual_description": "red circle"},
                traj,
                None,
            )
        )
        assert result.score > 0.0

    def test_weight_forwarded(self):
        reward = VisionMatchReward(weight=0.5)
        assert reward.weight == 0.5


# ---------------------------------------------------------------------------
# ImageAttributeReward tests
# ---------------------------------------------------------------------------


class TestImageAttributeReward:
    def test_matched_expected_attributes(self):
        cfg = ImageAttributeConfig(
            expected_attributes=["red", "circle", "large"],
            per_attribute_reward=0.33,
            max_reward=1.0,
        )
        reward = ImageAttributeReward(cfg=cfg)
        traj = _make_trajectory("a large red circle appears")
        result = _run_async(
            reward.evaluate({"task_id": "t1"}, traj, None)
        )
        assert result.score > 0.0
        assert "red" in result.metadata["matched_expected"]
        assert "circle" in result.metadata["matched_expected"]
        assert "large" in result.metadata["matched_expected"]

    def test_penalty_attributes_reduce_score(self):
        cfg = ImageAttributeConfig(
            expected_attributes=["red"],
            penalty_attributes=["blue"],
            per_attribute_reward=0.5,
            per_attribute_penalty=0.5,
        )
        reward = ImageAttributeReward(cfg=cfg)
        traj = _make_trajectory("a red and blue shape")
        result = _run_async(
            reward.evaluate({"task_id": "t1"}, traj, None)
        )
        # 0.5 (red) - 0.5 (blue) = 0.0
        assert result.score == 0.0
        assert "blue" in result.metadata["matched_penalties"]

    def test_no_attributes_matched(self):
        cfg = ImageAttributeConfig(
            expected_attributes=["red", "circle"],
        )
        reward = ImageAttributeReward(cfg=cfg)
        traj = _make_trajectory("hello world")
        result = _run_async(
            reward.evaluate({"task_id": "t1"}, traj, None)
        )
        assert result.score == 0.0
        assert "no attributes matched" in result.reason

    def test_max_reward_cap(self):
        cfg = ImageAttributeConfig(
            expected_attributes=["a", "b", "c", "d", "e"],
            per_attribute_reward=0.5,
            max_reward=1.0,
        )
        reward = ImageAttributeReward(cfg=cfg)
        traj = _make_trajectory("a b c d e")
        result = _run_async(
            reward.evaluate({"task_id": "t1"}, traj, None)
        )
        assert result.score == 1.0  # capped

    def test_item_overrides_config_attributes(self):
        cfg = ImageAttributeConfig(
            expected_attributes=["red"],
        )
        reward = ImageAttributeReward(cfg=cfg)
        traj = _make_trajectory("blue green")
        result = _run_async(
            reward.evaluate(
                {"task_id": "t1", "expected_attributes": ["blue", "green"]},
                traj,
                None,
            )
        )
        assert result.score > 0.0
        assert "blue" in result.metadata["matched_expected"]

    def test_case_insensitive_matching(self):
        cfg = ImageAttributeConfig(
            expected_attributes=["Red", "CIRCLE"],
        )
        reward = ImageAttributeReward(cfg=cfg)
        traj = _make_trajectory("a red circle")
        result = _run_async(
            reward.evaluate({"task_id": "t1"}, traj, None)
        )
        assert result.score > 0.0

    def test_empty_output(self):
        cfg = ImageAttributeConfig(
            expected_attributes=["red"],
        )
        reward = ImageAttributeReward(cfg=cfg)
        traj = _make_trajectory("")
        result = _run_async(
            reward.evaluate({"task_id": "t1"}, traj, None)
        )
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# MultimodalCompositeReward tests
# ---------------------------------------------------------------------------


class TestMultimodalCompositeReward:
    def test_weighted_sum_aggregation(self):
        composite_cfg = MultimodalCompositeConfig(
            vision_weight=0.5,
            attribute_weight=0.3,
            text_weight=0.2,
            aggregation="weighted_sum",
        )
        vision_cfg = VisionMatchConfig(similarity_threshold=0.1, use_clip=False)
        attr_cfg = ImageAttributeConfig(
            expected_attributes=["red", "circle"],
            per_attribute_reward=0.5,
        )
        reward = MultimodalCompositeReward(
            vision_cfg=vision_cfg,
            attr_cfg=attr_cfg,
            composite_cfg=composite_cfg,
        )
        traj = _make_trajectory("a red circle")
        result = _run_async(
            reward.evaluate(
                {
                    "task_id": "t1",
                    "image_description": "red circle",
                    "expected_output": "a red circle",
                },
                traj,
                None,
            )
        )
        assert result.score > 0.0
        assert result.metadata["aggregation"] == "weighted_sum"
        assert "vision_score" in result.metadata

    def test_max_aggregation(self):
        composite_cfg = MultimodalCompositeConfig(
            vision_weight=0.5,
            attribute_weight=0.3,
            text_weight=0.2,
            aggregation="max",
        )
        vision_cfg = VisionMatchConfig(similarity_threshold=0.1, use_clip=False)
        attr_cfg = ImageAttributeConfig(
            expected_attributes=["red"],
            per_attribute_reward=1.0,
        )
        reward = MultimodalCompositeReward(
            vision_cfg=vision_cfg,
            attr_cfg=attr_cfg,
            composite_cfg=composite_cfg,
        )
        traj = _make_trajectory("a red circle")
        result = _run_async(
            reward.evaluate(
                {
                    "task_id": "t1",
                    "image_description": "red circle",
                    "expected_output": "different text",
                },
                traj,
                None,
            )
        )
        assert result.score > 0.0
        assert result.metadata["aggregation"] == "max"

    def test_all_zero_components(self):
        composite_cfg = MultimodalCompositeConfig(
            vision_weight=0.5,
            attribute_weight=0.3,
            text_weight=0.2,
        )
        vision_cfg = VisionMatchConfig(similarity_threshold=0.99, use_clip=False)
        attr_cfg = ImageAttributeConfig(
            expected_attributes=["nonexistent"],
        )
        reward = MultimodalCompositeReward(
            vision_cfg=vision_cfg,
            attr_cfg=attr_cfg,
            composite_cfg=composite_cfg,
        )
        traj = _make_trajectory("hello world")
        result = _run_async(
            reward.evaluate(
                {
                    "task_id": "t1",
                    "image_description": "red circle",
                    "expected_output": "different",
                },
                traj,
                None,
            )
        )
        assert result.score == 0.0

    def test_default_config(self):
        reward = MultimodalCompositeReward()
        assert reward.composite_cfg.vision_weight == 0.5
        assert reward.composite_cfg.attribute_weight == 0.3
        assert reward.composite_cfg.text_weight == 0.2
        assert reward.composite_cfg.aggregation == "weighted_sum"

    def test_reason_contains_all_components(self):
        reward = MultimodalCompositeReward()
        traj = _make_trajectory("red circle")
        result = _run_async(
            reward.evaluate(
                {"task_id": "t1", "image_description": "red circle"},
                traj,
                None,
            )
        )
        assert "vision=" in result.reason
        assert "attr=" in result.reason
        assert "text=" in result.reason


# ---------------------------------------------------------------------------
# Config dataclass tests
# ---------------------------------------------------------------------------


class TestConfigs:
    def test_vision_match_config_defaults(self):
        cfg = VisionMatchConfig()
        assert cfg.similarity_threshold == 0.7
        assert cfg.positive_reward == 1.0
        assert cfg.negative_reward == 0.0
        assert cfg.use_clip is False
        assert "clip-vit" in cfg.clip_model

    def test_image_attribute_config_defaults(self):
        cfg = ImageAttributeConfig()
        assert cfg.expected_attributes == []
        assert cfg.penalty_attributes == []
        assert cfg.per_attribute_reward == 0.25
        assert cfg.per_attribute_penalty == 0.5
        assert cfg.max_reward == 1.0

    def test_composite_config_defaults(self):
        cfg = MultimodalCompositeConfig()
        assert cfg.vision_weight == 0.5
        assert cfg.attribute_weight == 0.3
        assert cfg.text_weight == 0.2
        assert cfg.aggregation == "weighted_sum"

    def test_image_attribute_config_with_attributes(self):
        cfg = ImageAttributeConfig(
            expected_attributes=["red", "blue"],
            penalty_attributes=["green"],
        )
        assert len(cfg.expected_attributes) == 2
        assert len(cfg.penalty_attributes) == 1
