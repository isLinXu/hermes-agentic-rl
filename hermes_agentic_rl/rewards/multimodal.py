"""Multimodal reward components for vision, image, and audio-based reward signals.

This module provides reward components that evaluate agent outputs against
visual and auditory criteria — e.g. checking if a generated image description
matches ground-truth visual attributes, or if a transcribed audio description
matches ground-truth audio attributes.

Design:
- :class:`VisionMatchReward` — compares text descriptions to image
  ground-truth using CLIP similarity (lazy import).
- :class:`ImageAttributeReward` — scores based on presence/absence of
  expected visual attributes in the agent's output.
- :class:`AudioMatchReward` — compares text descriptions to audio
  ground-truth using Wav2Vec2 similarity (lazy import) or text fallback.
- :class:`AudioAttributeReward` — scores based on presence/absence of
  expected audio attributes in the agent's output.
- :class:`MultimodalCompositeReward` — combines multiple modal reward
  components with configurable weights.

All components follow the :class:`BaseReward` interface so they integrate
seamlessly with :class:`RewardManager`.

When optional dependencies (torch, transformers/CLIP/Wav2Vec2) are missing, the
components gracefully degrade to text-matching heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize_text(text: str) -> set[str]:
    """Lowercase tokenization for text-matching fallback."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _cosine_text_similarity(text_a: str, text_b: str) -> float:
    """Text-level cosine similarity using bag-of-words (fallback for CLIP)."""
    tokens_a = _tokenize_text(text_a)
    tokens_b = _tokenize_text(text_b)
    return _jaccard_similarity(tokens_a, tokens_b)


# ---------------------------------------------------------------------------
# Vision match reward
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VisionMatchConfig:
    """Configuration for :class:`VisionMatchReward`.

    Attributes
    ----------
    similarity_threshold:
        Minimum CLIP/text similarity (0–1) for a positive reward.

    positive_reward:
        Reward when similarity >= threshold.

    negative_reward:
        Reward when similarity < threshold.

    use_clip:
        When True, use CLIP embeddings (requires transformers). When
        False, fall back to text-level Jaccard similarity.

    clip_model:
        CLIP model name for transformers (e.g. "openai/clip-vit-base-patch32").
    """

    similarity_threshold: float = 0.7
    positive_reward: float = 1.0
    negative_reward: float = 0.0
    use_clip: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"


class VisionMatchReward(BaseReward):
    """Reward based on similarity between agent output and visual ground-truth.

    When ``use_clip`` is True and transformers is available, computes CLIP
    cosine similarity between the agent's text output and the item's
    ``image_description`` field. Otherwise falls back to text-level Jaccard
    similarity.

    Item fields used:
    - ``image_description``: ground-truth description of the target image
    - ``image_path``: optional path to actual image (for CLIP image encoder)
    """

    name = "vision_match_reward"

    def __init__(
        self,
        weight: float = 1.0,
        cfg: VisionMatchConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weight = weight
        self.cfg = cfg or VisionMatchConfig()
        self._clip_model: Any = None
        self._clip_processor: Any = None

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        ground_truth = item.get("image_description") or item.get("visual_description")
        agent_output = trajectory.final_output or ""

        if not ground_truth:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no image_description in item; cannot compute vision match",
                weight=self.weight,
            )

        similarity = self._compute_similarity(agent_output, ground_truth, item)

        if similarity >= self.cfg.similarity_threshold:
            score = self.cfg.positive_reward
            reason = (
                f"similarity={similarity:.3f} >= threshold={self.cfg.similarity_threshold}"
            )
        else:
            score = self.cfg.negative_reward
            reason = (
                f"similarity={similarity:.3f} < threshold={self.cfg.similarity_threshold}"
            )

        return RewardResult(
            name=self.name,
            score=score,
            reason=reason,
            weight=self.weight,
            metadata={"similarity": similarity, "method": self._method_name()},
        )

    def _compute_similarity(
        self,
        text_a: str,
        text_b: str,
        item: dict[str, Any],
    ) -> float:
        if self.cfg.use_clip and self._try_load_clip():
            return self._clip_similarity(text_a, text_b, item)
        return _cosine_text_similarity(text_a, text_b)

    def _method_name(self) -> str:
        if self.cfg.use_clip and self._clip_model is not None:
            return "clip"
        return "text_jaccard"

    def _try_load_clip(self) -> bool:
        """Lazy-load CLIP model. Returns True if available."""
        if self._clip_model is not None:
            return True
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-untyped]

            self._clip_model = CLIPModel.from_pretrained(self.cfg.clip_model)
            self._clip_processor = CLIPProcessor.from_pretrained(self.cfg.clip_model)
            return True
        except Exception:
            return False

    def _clip_similarity(
        self,
        text_a: str,
        text_b: str,
        item: dict[str, Any],
    ) -> float:
        """Compute CLIP cosine similarity between two text descriptions."""
        import torch  # local import

        inputs_a = self._clip_processor(
            text=[text_a], return_tensors="pt", padding=True, truncation=True,
        )
        inputs_b = self._clip_processor(
            text=[text_b], return_tensors="pt", padding=True, truncation=True,
        )

        with torch.no_grad():
            feat_a = self._clip_model.get_text_features(**inputs_a)
            feat_b = self._clip_model.get_text_features(**inputs_b)

        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return float((feat_a @ feat_b.T).item())


# ---------------------------------------------------------------------------
# Image attribute reward
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ImageAttributeConfig:
    """Configuration for :class:`ImageAttributeReward`.

    Attributes
    ----------
    expected_attributes:
        List of attribute keywords that should appear in the agent output.

    penalty_attributes:
        List of attribute keywords that should NOT appear.

    per_attribute_reward:
        Reward per matched expected attribute.

    per_attribute_penalty:
        Penalty per matched forbidden attribute.

    max_reward:
        Cap on total positive reward.
    """

    expected_attributes: list[str] = field(default_factory=list)
    penalty_attributes: list[str] = field(default_factory=list)
    per_attribute_reward: float = 0.25
    per_attribute_penalty: float = 0.5
    max_reward: float = 1.0


class ImageAttributeReward(BaseReward):
    """Reward based on presence/absence of expected visual attributes.

    Checks the agent's text output for expected visual attribute keywords
    (e.g. "red", "circle", "large") and optionally penalizes forbidden ones.

    Item fields used:
    - ``expected_attributes``: list of attribute strings (overrides config)
    - ``penalty_attributes``: list of forbidden attribute strings (overrides config)
    """

    name = "image_attribute_reward"

    def __init__(
        self,
        weight: float = 1.0,
        cfg: ImageAttributeConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weight = weight
        self.cfg = cfg or ImageAttributeConfig()

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        expected = item.get("expected_attributes", self.cfg.expected_attributes)
        penalties = item.get("penalty_attributes", self.cfg.penalty_attributes)
        text = (trajectory.final_output or "").lower()

        matched_expected = [attr for attr in expected if attr.lower() in text]
        matched_penalties = [attr for attr in penalties if attr.lower() in text]

        positive = len(matched_expected) * self.cfg.per_attribute_reward
        negative = len(matched_penalties) * self.cfg.per_attribute_penalty
        score = max(0.0, min(self.cfg.max_reward, positive - negative))

        reason_parts: list[str] = []
        if matched_expected:
            reason_parts.append(f"matched: {', '.join(matched_expected)}")
        if matched_penalties:
            reason_parts.append(f"penalized: {', '.join(matched_penalties)}")
        if not reason_parts:
            reason_parts.append("no attributes matched")

        return RewardResult(
            name=self.name,
            score=score,
            reason="; ".join(reason_parts),
            weight=self.weight,
            metadata={
                "matched_expected": matched_expected,
                "matched_penalties": matched_penalties,
                "positive_score": positive,
                "negative_score": negative,
            },
        )


# ---------------------------------------------------------------------------
# Audio match reward
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AudioMatchConfig:
    """Configuration for :class:`AudioMatchReward`.

    Attributes
    ----------
    similarity_threshold:
        Minimum audio/text similarity (0–1) for a positive reward.

    positive_reward:
        Reward when similarity >= threshold.

    negative_reward:
        Reward when similarity < threshold.

    use_audio_model:
        When True, use Wav2Vec2 embeddings (requires transformers). When
        False, fall back to text-level Jaccard similarity.

    audio_model_name:
        Wav2Vec2 model name for transformers (e.g. "facebook/wav2vec2-base-960h").
    """

    similarity_threshold: float = 0.7
    positive_reward: float = 1.0
    negative_reward: float = 0.0
    use_audio_model: bool = False
    audio_model_name: str = "facebook/wav2vec2-base-960h"


class AudioMatchReward(BaseReward):
    """Reward based on similarity between agent output and audio ground-truth.

    When ``use_audio_model`` is True and transformers is available, computes
    Wav2Vec2 cosine similarity between the agent's text output and the item's
    ``audio_description`` field. Otherwise falls back to text-level Jaccard
    similarity.

    Item fields used:
    - ``audio_description``: ground-truth description of the target audio
    - ``audio_path``: optional path to actual audio (for Wav2Vec2 audio encoder)
    """

    name = "audio_match_reward"

    def __init__(
        self,
        weight: float = 1.0,
        cfg: AudioMatchConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weight = weight
        self.cfg = cfg or AudioMatchConfig()
        self._audio_model: Any = None
        self._audio_processor: Any = None

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        ground_truth = item.get("audio_description") or item.get("audio_attributes")
        agent_output = trajectory.final_output or ""

        if not ground_truth:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no audio_description in item; cannot compute audio match",
                weight=self.weight,
            )

        similarity = self._compute_similarity(agent_output, ground_truth, item)

        if similarity >= self.cfg.similarity_threshold:
            score = self.cfg.positive_reward
            reason = (
                f"similarity={similarity:.3f} >= threshold={self.cfg.similarity_threshold}"
            )
        else:
            score = self.cfg.negative_reward
            reason = (
                f"similarity={similarity:.3f} < threshold={self.cfg.similarity_threshold}"
            )

        return RewardResult(
            name=self.name,
            score=score,
            reason=reason,
            weight=self.weight,
            metadata={"similarity": similarity, "method": self._method_name()},
        )

    def _compute_similarity(
        self,
        text_a: str,
        text_b: str,
        item: dict[str, Any],
    ) -> float:
        if self.cfg.use_audio_model and self._try_load_audio_model():
            return self._audio_model_similarity(text_a, text_b, item)
        return _cosine_text_similarity(text_a, text_b)

    def _method_name(self) -> str:
        if self.cfg.use_audio_model and self._audio_model is not None:
            return "wav2vec2"
        return "text_jaccard"

    def _try_load_audio_model(self) -> bool:
        """Lazy-load Wav2Vec2 model. Returns True if available."""
        if self._audio_model is not None:
            return True
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore[import-untyped]

            self._audio_model = AutoModel.from_pretrained(self.cfg.audio_model_name)
            self._audio_processor = AutoProcessor.from_pretrained(self.cfg.audio_model_name)
            return True
        except Exception:
            return False

    def _audio_model_similarity(
        self,
        text_a: str,
        text_b: str,
        item: dict[str, Any],
    ) -> float:
        """Compute Wav2Vec2 cosine similarity between two text descriptions."""
        import torch  # local import

        inputs_a = self._audio_processor(
            text=[text_a], return_tensors="pt", padding=True, truncation=True,
        )
        inputs_b = self._audio_processor(
            text=[text_b], return_tensors="pt", padding=True, truncation=True,
        )

        with torch.no_grad():
            feat_a = self._audio_model(**inputs_a).last_hidden_state.mean(dim=1)
            feat_b = self._audio_model(**inputs_b).last_hidden_state.mean(dim=1)

        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return float((feat_a @ feat_b.T).item())


# ---------------------------------------------------------------------------
# Audio attribute reward
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AudioAttributeConfig:
    """Configuration for :class:`AudioAttributeReward`.

    Attributes
    ----------
    expected_attributes:
        List of audio attribute keywords that should appear in the agent output.
        Examples: "speech", "music", "noise", "silence".

    attribute_reward:
        Reward per matched expected attribute.

    missing_penalty:
        Penalty for each expected attribute that is missing.

    max_reward:
        Cap on total positive reward.
    """

    expected_attributes: list[str] = field(default_factory=list)
    attribute_reward: float = 0.25
    missing_penalty: float = 0.0
    max_reward: float = 1.0


class AudioAttributeReward(BaseReward):
    """Reward based on presence/absence of expected audio attributes.

    Checks the agent's text output for expected audio attribute keywords
    (e.g. "speech", "music", "noise", "silence").

    Item fields used:
    - ``expected_audio_attributes``: list of attribute strings (overrides config)
    """

    name = "audio_attribute_reward"

    def __init__(
        self,
        weight: float = 1.0,
        cfg: AudioAttributeConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weight = weight
        self.cfg = cfg or AudioAttributeConfig()

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        expected = item.get("expected_audio_attributes", self.cfg.expected_attributes)
        text = (trajectory.final_output or "").lower()

        matched = [attr for attr in expected if attr.lower() in text]
        missing = [attr for attr in expected if attr.lower() not in text]

        positive = len(matched) * self.cfg.attribute_reward
        negative = len(missing) * self.cfg.missing_penalty
        score = max(0.0, min(self.cfg.max_reward, positive - negative))

        reason_parts: list[str] = []
        if matched:
            reason_parts.append(f"matched: {', '.join(matched)}")
        if missing:
            reason_parts.append(f"missing: {', '.join(missing)}")
        if not reason_parts:
            reason_parts.append("no audio attributes expected")

        return RewardResult(
            name=self.name,
            score=score,
            reason="; ".join(reason_parts),
            weight=self.weight,
            metadata={
                "matched": matched,
                "missing": missing,
                "positive_score": positive,
                "negative_score": negative,
            },
        )


# ---------------------------------------------------------------------------
# Multimodal composite reward
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MultimodalCompositeConfig:
    """Configuration for :class:`MultimodalCompositeReward`.

    Attributes
    ----------
    vision_weight:
        Weight for the vision-match component (when using legacy constructor).

    attribute_weight:
        Weight for the image-attribute component (when using legacy constructor).

    text_weight:
        Weight for a text-exact-match fallback component (when using legacy constructor).

    aggregation:
        How to combine component scores: "weighted_sum" or "max".
    """

    vision_weight: float = 0.5
    attribute_weight: float = 0.3
    text_weight: float = 0.2
    aggregation: str = "weighted_sum"


class MultimodalCompositeReward(BaseReward):
    """Combines arbitrary :class:`BaseReward` components into a single score.

    Accepts any list of reward instances (vision, audio, text, etc.) and
    aggregates their scores using ``aggregation`` mode. When no components are
    provided, falls back to a built-in vision + image-attribute + text-match
    composite for backward compatibility.
    """

    name = "multimodal_composite_reward"

    def __init__(
        self,
        weight: float = 1.0,
        components: list[BaseReward] | None = None,
        vision_cfg: VisionMatchConfig | None = None,
        attr_cfg: ImageAttributeConfig | None = None,
        composite_cfg: MultimodalCompositeConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weight = weight
        self.composite_cfg = composite_cfg or MultimodalCompositeConfig()
        if components is not None:
            self._components = components
        else:
            # Legacy backward-compatible default: vision + image-attribute + text fallback
            self._components = [
                VisionMatchReward(weight=1.0, cfg=vision_cfg),
                ImageAttributeReward(weight=1.0, cfg=attr_cfg),
            ]

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        results: list[RewardResult] = []
        for comp in self._components:
            results.append(await comp.evaluate(item, trajectory, tool_context))

        # Text exact-match fallback (always included as an implicit component)
        expected = item.get("expected_output") or ""
        text_score = 1.0 if expected and trajectory.final_output == expected else 0.0

        if self.composite_cfg.aggregation == "max":
            score = max(
                *(r.score * r.weight for r in results),
                text_score * self.composite_cfg.text_weight,
            )
        else:
            score = (
                sum(r.score * r.weight for r in results)
                + text_score * self.composite_cfg.text_weight
            )

        reason_parts = [f"{r.name}={r.score:.3f} ({r.reason})" for r in results]
        reason_parts.append(f"text={text_score:.3f}")

        metadata: dict[str, Any] = {
            "aggregation": self.composite_cfg.aggregation,
            "text_score": text_score,
        }
        for r in results:
            metadata[f"{r.name}_score"] = r.score

        return RewardResult(
            name=self.name,
            score=score,
            reason=", ".join(reason_parts),
            weight=self.weight,
            metadata=metadata,
        )
