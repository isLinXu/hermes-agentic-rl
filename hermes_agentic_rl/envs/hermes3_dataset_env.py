"""Environment adapter for agentlans/NousResearch-Hermes-3-Dataset.

The dataset contains ~959K ShareGPT-style conversations with ``from`` / ``value``
keys.  Each row is turned into a single training item where the *last* human
message is the instruction and the *last* assistant message is the reference
response.  This makes the dataset suitable for:

* SFT warm-start (teacher-forced on the reference response)
* RL fine-tuning with a similarity reward against the reference
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.datasets.hf_loader import load_hf_dataset
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.rewards.base import BaseReward

DEFAULT_REPO_ID = "agentlans/NousResearch-Hermes-3-Dataset"
DEFAULT_SPLIT = "train"


@dataclass(slots=True)
class Hermes3DatasetConfig:
    repo_id: str = DEFAULT_REPO_ID
    split: str = DEFAULT_SPLIT
    limit: int | None = None
    shuffle: bool = True
    seed: int = 42
    streaming: bool = True
    cache_dir: str | None = None
    max_prompt_chars: int = 2048
    val_fraction: float = 0.05


def _role(message: dict[str, Any]) -> str:
    raw = str(message.get("role") or message.get("from") or "").strip().lower()
    mapping = {
        "gpt": "assistant",
        "assistant": "assistant",
        "human": "user",
        "user": "user",
        "system": "system",
    }
    return mapping.get(raw, raw or "unknown")


def _message_text(message: dict[str, Any]) -> str:
    for key in ("value", "content", "text", "message"):
        value = message.get(key)
        if value is not None:
            return str(value)
    return ""


def sharegpt_to_messages(
    conversations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Convert ShareGPT ``from/value`` format to role/content format."""
    return [
        {"role": _role(msg), "content": _message_text(msg)}
        for msg in conversations
        if isinstance(msg, dict)
    ]


def _extract_instruction_and_reference(
    conversations: list[dict[str, Any]],
) -> tuple[str, str, list[dict[str, str]]] | None:
    """Extract the last user→assistant pair from a conversation.

    Returns ``(instruction, reference_response, all_messages)`` or *None* if
    no valid pair is found.
    """
    messages = sharegpt_to_messages(conversations)
    if len(messages) < 2:
        return None

    # Find the last assistant turn
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            prompt_messages = messages[:i]
            reference = messages[i]["content"]
            if not reference.strip():
                continue
            instruction_parts: list[str] = []
            for msg in prompt_messages:
                role_label = msg["role"].capitalize()
                instruction_parts.append(f"{role_label}: {msg['content']}")
            instruction = "\n\n".join(instruction_parts).strip()
            if not instruction:
                continue
            return instruction, reference, messages

    return None


def _truncate_instruction(text: str, *, max_prompt_chars: int) -> str:
    if max_prompt_chars <= 0 or len(text) <= max_prompt_chars:
        return text
    marker = "\n\n[Earlier context truncated]\n\n"
    tail_budget = max(32, max_prompt_chars - len(marker))
    return marker + text[-tail_budget:]


class Hermes3DatasetReward(BaseReward):
    """Similarity reward against the reference response in the dataset."""

    name = "hermes3_dataset_similarity"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        reference = str(item.get("reference_response") or "").strip()
        prediction = str(trajectory.final_output or "").strip()

        if not reference:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="missing reference_response",
                weight=self.weight,
            )

        if not prediction:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="empty prediction",
                weight=self.weight,
            )

        from difflib import SequenceMatcher

        similarity = SequenceMatcher(
            None, prediction.lower(), reference.lower()
        ).ratio()

        return RewardResult(
            name=self.name,
            score=similarity,
            reason=f"sequence similarity = {similarity:.4f}",
            weight=self.weight,
            metadata={
                "prediction_chars": len(prediction),
                "reference_chars": len(reference),
            },
        )


class Hermes3DatasetEnv(BaseEnv):
    """Training environment backed by
    ``agentlans/NousResearch-Hermes-3-Dataset``.

    Each dataset row is turned into one training item.  The last user message
    becomes the instruction and the last assistant message becomes the
    reference response (used for SFT warm-start and similarity reward).
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        reward_weight: float = 1.0,
    ) -> None:
        if not items:
            raise ValueError(
                "Hermes3DatasetEnv requires at least one item"
            )
        self.items = items
        self._reward = Hermes3DatasetReward(weight=reward_weight)
        self._cursor = 0

    @property
    def reward_component(self) -> Hermes3DatasetReward:
        return self._reward

    async def setup(self) -> None:
        """No-op — dataset is already loaded in memory."""
        self._cursor = 0

    async def get_next_item(self) -> dict[str, Any]:
        item = self.items[self._cursor % len(self.items)]
        self._cursor += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        instruction = str(item.get("instruction") or "")
        max_chars = item.get("max_prompt_chars", 2048)
        return _truncate_instruction(instruction, max_prompt_chars=max_chars)

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        result = await self._reward.evaluate(item, trajectory, tool_context)
        return [result]

    def build_supervised_samples(
        self, item: dict[str, Any]
    ) -> list[SupervisedSample]:
        """Build SFT sample from the reference response."""
        instruction = str(item.get("instruction") or "")
        reference = str(item.get("reference_response") or "")
        if not instruction or not reference:
            return []
        return [
            SupervisedSample(
                instruction=instruction,
                response=reference,
                metadata={
                    "task_id": item.get("task_id", ""),
                    "source": "hermes3_dataset",
                },
            )
        ]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Hermes3DatasetEnv:
        """Build from a YAML-style config dict."""
        return cls.from_hf_dataset(
            dataset_name=str(cfg.get("dataset_name", DEFAULT_REPO_ID)),
            split=str(cfg.get("split", DEFAULT_SPLIT)),
            limit=cfg.get("limit"),
            shuffle=bool(cfg.get("shuffle", True)),
            seed=int(cfg.get("seed", 42)),
            streaming=bool(cfg.get("streaming", True)),
            cache_dir=cfg.get("cache_dir"),
            max_prompt_chars=int(cfg.get("max_prompt_chars", 2048)),
            val_fraction=float(cfg.get("val_fraction", 0.05)),
            reward_weight=float(cfg.get("reward_weight", 1.0)),
        )

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name: str = DEFAULT_REPO_ID,
        split: str = DEFAULT_SPLIT,
        limit: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
        streaming: bool = True,
        cache_dir: str | None = None,
        max_prompt_chars: int = 2048,
        val_fraction: float = 0.05,
        reward_weight: float = 1.0,
    ) -> Hermes3DatasetEnv:
        """Load from HuggingFace datasets and build items."""
        rows = load_hf_dataset(
            dataset_name,
            split=split,
            limit=limit,
            shuffle=shuffle,
            seed=seed,
            streaming=streaming,
            cache_dir=cache_dir,
        )

        items: list[dict[str, Any]] = []
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue

            conversations = row.get("conversations")
            if not isinstance(conversations, list):
                continue

            extracted = _extract_instruction_and_reference(conversations)
            if extracted is None:
                continue

            instruction, reference, messages = extracted
            instruction = _truncate_instruction(
                instruction, max_prompt_chars=max_prompt_chars
            )

            items.append(
                {
                    "task_id": f"hermes3::{dataset_name}::{split}::{row_idx}",
                    "instruction": instruction,
                    "reference_response": reference,
                    "messages": messages,
                    "max_prompt_chars": max_prompt_chars,
                    "dataset_name": dataset_name,
                    "split": split,
                }
            )

        if not items:
            raise ValueError(
                f"No valid items extracted from {dataset_name} {split}. "
                "Check dataset format."
            )

        # Split into train / val if val_fraction > 0
        if val_fraction > 0 and len(items) > 1:
            rng = random.Random(seed)
            shuffled = list(items)
            rng.shuffle(shuffled)
            val_size = max(1, int(len(shuffled) * val_fraction))
            train_items = shuffled[val_size:]
            val_items = shuffled[:val_size]
            for it in train_items:
                it["_split"] = "train"
            for it in val_items:
                it["_split"] = "val"
            items = train_items

        return cls(items, reward_weight=reward_weight)
