"""Minimal learnable environment for the RL MVP.

Task: each item has a short target string; the policy's final_output is
compared (fuzzy-match) against the target. Reward ∈ [0, 1] is a combination of
  - exact-substring match bonus
  - character-level overlap

Why this task exists: the original filesystem-verifier tasks require a real
Hermes runtime + real tool execution; the MVP must demonstrate gradient-based
policy improvement **without** external dependencies. The echo task is the
simplest MDP that makes GRPO advantages nonzero: a random policy gets ~0, a
trained policy can reach close to 1 within ~50 iterations on CPU.
"""

from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample
from hermes_agentic_rl.rewards.base import BaseReward


def _char_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = max(sum((ca | cb).values()), 1)
    return inter / union


class EchoRewardComponent(BaseReward):
    """Reward: how close is final_output to the target?"""

    name = "echo_reward"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        target = str(item.get("target", ""))
        final = str(trajectory.final_output or "")
        if not target:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no target provided",
                weight=self.weight,
            )
        exact = 1.0 if target in final else 0.0
        overlap = _char_overlap(target.lower(), final.lower())
        score = 0.6 * exact + 0.4 * overlap
        return RewardResult(
            name=self.name,
            score=float(score),
            reason=f"exact={exact} overlap={overlap:.3f}",
            weight=self.weight,
            metadata={"target": target, "final": final[:80]},
        )


class EchoTaskEnv(BaseEnv):
    """Round-robin echo task env.

    Each item shape:
        {
            "task_id": "echo-1",
            "instruction": "Say: hello",
            "target": "hello"
        }
    """

    def __init__(self, dataset: list[dict[str, Any]]) -> None:
        self.dataset = dataset
        self._index = 0
        self._reward = EchoRewardComponent(weight=1.0)

    async def setup(self) -> None:
        self._index = 0

    async def get_next_item(self) -> dict[str, Any]:
        item = self.dataset[self._index % len(self.dataset)]
        self._index += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        return item["instruction"]

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        return [await self._reward.evaluate(item, trajectory, tool_context)]

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        target = str(item.get("target", "")).strip()
        instruction = str(item.get("instruction", "")).strip()
        if not target or not instruction:
            return []
        return [SupervisedSample(instruction=instruction, response=target)]


def build_default_echo_dataset() -> list[dict[str, Any]]:
    """A tiny fixed dataset used by MVP tests and the example config."""
    pairs = [
        ("Say: hello", "hello"),
        ("Say: world", "world"),
        ("Say: abc", "abc"),
        ("Say: ok", "ok"),
    ]
    return [
        {"task_id": f"echo-{i}", "instruction": instr, "target": target}
        for i, (instr, target) in enumerate(pairs)
    ]
