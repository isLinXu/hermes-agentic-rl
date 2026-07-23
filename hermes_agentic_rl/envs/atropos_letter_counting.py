"""Letter-counting atropos env adapter for hermes.

Maps atropos ``LetterCountingEnv`` → hermes ``BaseEnv`` protocol so the
hermes GRPO/PPO trainer can drive it without any HTTP or vLLM.

Usage::

    from hermes_agentic_rl.envs.atropos_letter_counting import (
        AtroposLetterCountingEnv,
        AtroposLetterCountingReward,
    )
    env = AtroposLetterCountingEnv(tokenizer, base_dir="subprojects/atropos")
    rm  = RewardManager([AtroposLetterCountingReward(env, weight=1.0)])

Key mapping:
  - ``get_next_item()`` → calls atropos env's async ``get_next_item()``
  - ``format_prompt(item)`` → apply_chat_template on the messages tuple
  - ``score()`` → calls atropos env's ``score()`` per-group
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv
from hermes_agentic_rl.rewards.base import BaseReward

# ---------------------------------------------------------------------------
# AtroposLetterCountingEnv
# ---------------------------------------------------------------------------


class AtroposLetterCountingEnv(BaseEnv):
    """Adapter: atropos LetterCountingEnv → hermes BaseEnv."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        base_dir: str | Path = "subprojects/atropos",
        config_path: str | None = None,
    ) -> None:
        base = Path(base_dir).resolve()
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))

        from environments.letter_counting_environment.letter_counting_environment import (  # type: ignore[import-untyped]
            LetterCountingConfig,
            LetterCountingEnv,
        )

        if config_path is None:
            config_path = str(base / "environments/letter_counting_environment/config.yaml")
        cfg = LetterCountingConfig.from_yaml(config_path)
        # Override: hermes runs its own generate, no server needed
        cfg.use_wandb = False
        # Use a dummy rollout URL so atropos doesn't try to connect
        cfg.rollout_server_url = ""
        self._tokenizer = tokenizer
        self._atropos = LetterCountingEnv(cfg, tokenizer)
        self._item_buffer: list[dict[str, Any]] = []

    async def setup(self) -> None:
        await self._atropos.setup()

    async def get_next_item(self) -> dict[str, Any]:
        item = await self._atropos.get_next_item()
        # item = (prompt_messages, correct_counts, text, target_letters, difficulty_level)
        prompt_msgs, correct_counts, text, target_letters, difficulty_level = item
        return {
            "prompt_msgs": prompt_msgs,
            "correct_counts": correct_counts,
            "text": text,
            "target_letters": target_letters,
            "difficulty_level": difficulty_level,
            "task_id": str(hash(text + "".join(target_letters)) % 2**32),
        }

    def format_prompt(self, item: dict[str, Any]) -> str:
        msgs = item["prompt_msgs"]
        # msgs is a tuple of frozensets from atropos
        messages: list[dict[str, str]] = []
        for fs in msgs:
            messages.append(dict(fs))
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def close(self) -> None:
        await self._atropos.close()

    # ------------------------------------------------------------------
    # Score (used by AtroposLetterCountingReward)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "difficulty_level": getattr(self._atropos, "current_difficulty_level", 0),
            "iter": getattr(self._atropos, "iter", 0),
        }


# ---------------------------------------------------------------------------
# AtroposLetterCountingReward
# ---------------------------------------------------------------------------


class AtroposLetterCountingReward(BaseReward):
    """Reward component that extracts answer from trajectory.final_output.

    Single letter: expects ``<answer>N</answer>``
    Multi letter:  expects ``<answer>{...}</answer>`` as JSON dict

    Reward = 1.0 if all counts match, 0.0 otherwise.
    """

    name = "letter_counting"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        correct_counts = item.get("correct_counts", {})
        target_letters = item.get("target_letters", [])
        if not correct_counts or not target_letters:
            return RewardResult(
                name=self.name,
                score=0.0,
                weight=self.weight,
                reason="no correct_counts or target_letters in item",
            )

        response_text = trajectory.final_output or ""

        m = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if m is None:
            return RewardResult(
                name=self.name,
                score=0.0,
                weight=self.weight,
                reason="no <answer> tag found",
            )
        content = m.group(1).strip()

        if len(target_letters) == 1:
            try:
                pred = int(content)
            except (ValueError, TypeError):
                return RewardResult(
                    name=self.name,
                    score=0.0,
                    weight=self.weight,
                    reason=f"expected int, got {content!r}",
                )
            expected = correct_counts[target_letters[0]]
            ok = pred == expected
            return RewardResult(
                name=self.name,
                score=self.weight if ok else 0.0,
                weight=self.weight,
                reason=f"pred={pred} expected={expected}" if not ok else "correct",
            )
        else:
            import json

            try:
                pred_dict = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return RewardResult(
                    name=self.name,
                    score=0.0,
                    weight=self.weight,
                    reason=f"invalid json: {content!r}",
                )
            if not isinstance(pred_dict, dict):
                return RewardResult(
                    name=self.name,
                    score=0.0,
                    weight=self.weight,
                    reason=f"not a dict: {content!r}",
                )
            ok = all(
                pred_dict.get(letter, -1) == correct_counts[letter] for letter in target_letters
            )
            return RewardResult(
                name=self.name,
                score=self.weight if ok else 0.0,
                weight=self.weight,
                reason="correct" if ok else "mismatch",
            )
