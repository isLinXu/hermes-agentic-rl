"""Letter-counting environment for hermes (standalone, no atropos ServerManager).

Generates questions like:
  - "How many e's are in the string 'elephant'?"
  - "Count the occurrences of the letters 'a', 'b', and 'c' in the string ..."

Answers must be in `<answer>N</answer>` (single) or `<answer>{...}</answer>` (multi).

Adaptive difficulty: 10 tiers, moving average promote/demote.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

try:
    import nltk
    from nltk.corpus import words

    try:
        words.words()
    except LookupError:
        nltk.download("words", quiet=True)
except Exception:
    words = None

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.rewards.base import BaseReward

# ---------------------------------------------------------------------------
# Difficulty tiers (copied from atropos)
# ---------------------------------------------------------------------------

DIFFICULTY_TIERS = {
    1: {"min_word_length": 3, "max_word_length": 8, "multi_letter_probability": 0.0,
        "min_letters_to_count": 1, "max_letters_to_count": 1, "use_random_string": False},
    2: {"min_word_length": 5, "max_word_length": 12, "multi_letter_probability": 0.5,
        "min_letters_to_count": 1, "max_letters_to_count": 2, "use_random_string": False},
    3: {"min_word_length": 8, "max_word_length": 16, "multi_letter_probability": 0.7,
        "min_letters_to_count": 1, "max_letters_to_count": 3, "use_random_string": False},
    4: {"min_word_length": 12, "max_word_length": 20, "multi_letter_probability": 0.9,
        "min_letters_to_count": 2, "max_letters_to_count": 4, "use_random_string": False},
    5: {"min_word_length": 16, "max_word_length": 30, "multi_letter_probability": 1.0,
        "min_letters_to_count": 2, "max_letters_to_count": 5, "use_random_string": False},
    6: {"min_word_length": 20, "max_word_length": 40, "multi_letter_probability": 1.0,
        "min_letters_to_count": 3, "max_letters_to_count": 6, "use_random_string": True},
    7: {"min_word_length": 30, "max_word_length": 60, "multi_letter_probability": 1.0,
        "min_letters_to_count": 4, "max_letters_to_count": 8, "use_random_string": True},
    8: {"min_word_length": 50, "max_word_length": 100, "multi_letter_probability": 1.0,
        "min_letters_to_count": 5, "max_letters_to_count": 10, "use_random_string": True},
    9: {"min_word_length": 80, "max_word_length": 200, "multi_letter_probability": 1.0,
        "min_letters_to_count": 8, "max_letters_to_count": 15, "use_random_string": True},
    10: {"min_word_length": 150, "max_word_length": 500, "multi_letter_probability": 1.0,
         "min_letters_to_count": 10, "max_letters_to_count": 50, "use_random_string": True},
}


def _load_word_list() -> list[str]:
    if words is not None:
        try:
            wl = [w.lower() for w in words.words() if w.isalpha() and len(w) >= 3]
            if wl:
                return wl
        except Exception:
            pass
    # fallback: hardcoded common words
    return [
        "hello", "world", "python", "elephant", "computer", "language",
        "mathematics", "algorithm", "structure", "development", "engineering",
        "artificial", "intelligence", "reinforcement", "learning", "training",
        "model", "parameter", "gradient", "optimization", "transformer",
        "attention", "mechanism", "encoding", "decoding", "embedding",
        "tokenizer", "vocabulary", "sentence", "paragraph", "document",
    ]


def _generate_random_string(min_len: int, max_len: int) -> str:
    import string

    length = random.randint(min_len, max_len)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


# ---------------------------------------------------------------------------
# LetterCountingEnv
# ---------------------------------------------------------------------------


@dataclass
class LetterCountingConfig:
    starting_level: int = 2
    min_level: int = 1
    max_level: int = 10
    promote_threshold: float = 0.7
    demote_threshold: float = 0.3
    window_size: int = 50
    seed: int = 42


class LetterCountingEnv(BaseEnv):
    """Standalone letter-counting env for hermes (no atropos dependency)."""

    def __init__(self, cfg: LetterCountingConfig | None = None) -> None:
        self.cfg = cfg or LetterCountingConfig()
        self._rng = random.Random(self.cfg.seed)
        self._words = _load_word_list()
        self._iter = 0
        self._level = self.cfg.starting_level
        self._recent: list[float] = []

    async def setup(self) -> None:
        pass

    async def get_next_item(self) -> dict[str, Any]:
        tier = DIFFICULTY_TIERS.get(self._level, DIFFICULTY_TIERS[1])
        min_len = int(tier["min_word_length"])
        max_len = int(tier["max_word_length"])
        multi_prob = float(tier["multi_letter_probability"])
        min_letters = int(tier.get("min_letters_to_count", 1))
        max_letters = int(tier["max_letters_to_count"])
        use_random = bool(tier.get("use_random_string", False))

        if use_random:
            text = _generate_random_string(min_len, max_len)
        else:
            candidates = [w for w in self._words if min_len <= len(w) <= max_len]
            if candidates:
                text = self._rng.choice(candidates)
            else:
                text = _generate_random_string(min_len, max_len)

        if self._rng.random() < multi_prob:
            num_letters = self._rng.randint(max(2, min_letters), max_letters)
        else:
            num_letters = 1

        # pick unique letters from the text
        available = list(set(text.lower()))
        if len(available) < num_letters:
            num_letters = max(1, len(available))
        target_letters = self._rng.sample(available, num_letters)

        correct_counts = {ch: text.lower().count(ch) for ch in target_letters}
        self._iter += 1

        # build instruction
        if len(target_letters) == 1:
            ch = target_letters[0]
            instruction = (
                f"How many '{ch}'s are in the following string?\n\n"
                f"String: {text}\n\n"
                f"Provide your answer in the format: <answer>number</answer>"
            )
        else:
            letters_str = ", ".join(f"'{ch}'" for ch in target_letters)
            example_json = json.dumps({ch: 0 for ch in target_letters})
            instruction = (
                f"Count the occurrences of the letters {letters_str} "
                f"in the following string.\n\n"
                f"String: {text}\n\n"
                f"Provide your answer as JSON in the format: "
                f"<answer>{example_json}</answer>"
            )

        return {
            "task_id": str(hash(text + "".join(target_letters)) % 2**32),
            "instruction": instruction,
            "text": text,
            "target_letters": target_letters,
            "correct_counts": correct_counts,
            "difficulty_level": self._level,
        }

    def format_prompt(self, item: dict[str, Any]) -> str:
        return item["instruction"]

    def observe(self, score: float) -> None:
        """Curriculum learning: update difficulty based on reward."""
        self._recent.append(score)
        if len(self._recent) > self.cfg.window_size:
            self._recent = self._recent[-self.cfg.window_size :]
        if len(self._recent) < 20:
            return
        avg = sum(self._recent[-20:]) / 20
        if avg >= self.cfg.promote_threshold and self._level < self.cfg.max_level:
            old = self._level
            self._level += 1
            print(f"  [letter_counting] level {old} → {self._level} (avg_reward={avg:.3f})")
        elif avg <= self.cfg.demote_threshold and self._level > self.cfg.min_level:
            old = self._level
            self._level -= 1
            print(f"  [letter_counting] level {old} → {self._level} (avg_reward={avg:.3f})")

    async def close(self) -> None:
        pass

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        """Default: delegate to a LetterCountingReward(weight=1.0)."""
        from hermes_agentic_rl.envs.letter_counting import LetterCountingReward as _LC
        reward = _LC(weight=1.0)
        return [await reward.evaluate(item, trajectory, tool_context)]

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        instruction = str(item.get("instruction", "")).strip()
        correct = item.get("correct_counts", {})
        targets = item.get("target_letters", [])
        if not instruction or not correct or not targets:
            return []
        if len(targets) == 1:
            answer = f"<answer>{correct[targets[0]]}</answer>"
        else:
            ordered = {ch: correct[ch] for ch in targets}
            answer = f"<answer>{json.dumps(ordered)}</answer>"
        return [SupervisedSample(instruction=instruction, response=answer)]

    def snapshot(self) -> dict[str, Any]:
        return {"level": self._level, "iter": self._iter}


# ---------------------------------------------------------------------------
# LetterCountingReward
# ---------------------------------------------------------------------------


class LetterCountingReward(BaseReward):
    """Extract <answer>...</answer> and compare to correct_counts."""

    name = "letter_counting"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        correct = item.get("correct_counts", {})
        targets = item.get("target_letters", [])
        if not correct or not targets:
            return RewardResult(name=self.name, score=0.0, weight=self.weight, reason="no data")

        response = trajectory.final_output or ""
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if m is None:
            return RewardResult(name=self.name, score=0.0, weight=self.weight, reason="no <answer>")

        content = m.group(1).strip()
        if len(targets) == 1:
            try:
                pred = int(content)
            except (ValueError, TypeError):
                return RewardResult(name=self.name, score=0.0, weight=self.weight, reason=f"not int: {content!r}")
            expected = correct[targets[0]]
            ok = pred == expected
            return RewardResult(
                name=self.name,
                score=self.weight if ok else 0.0,
                weight=self.weight,
                reason="correct" if ok else f"pred={pred} != expected={expected}",
            )
        else:
            try:
                pred_dict = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return RewardResult(name=self.name, score=0.0, weight=self.weight, reason=f"bad json: {content!r}")
            if not isinstance(pred_dict, dict):
                return RewardResult(name=self.name, score=0.0, weight=self.weight, reason=f"not dict: {content!r}")
            ok = all(pred_dict.get(ch, -1) == correct[ch] for ch in targets)
            return RewardResult(
                name=self.name,
                score=self.weight if ok else 0.0,
                weight=self.weight,
                reason="correct" if ok else "mismatch",
            )
