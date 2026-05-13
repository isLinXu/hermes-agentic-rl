"""Curriculum wrapper env.

Wraps a list of sub-environments (levels L0, L1, ...) and advances to the next
level once the running mean reward over the last ``window`` updates stays
above ``promote_threshold``. Demotion (optional) is symmetric: if the running
mean drops below ``demote_threshold`` the env steps back one level.

Usage::

    env = CurriculumEnv(
        levels=[EchoTaskEnv(easy), SimToolEnv(mid), TerminalTaskEnv(hard)],
        window=20,
        promote_threshold=0.6,
    )

    trainer = GRPOTrainer(env=env, ...)
    # In the training loop, call env.observe(reward) after each rollout, or
    # wire it via the trainer's metrics_sink.

The wrapper exposes ``env.current_level`` and ``env.on_level_change`` so
callers can log level transitions.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.envs.base_env import BaseEnv, SupervisedSample


@dataclass(slots=True)
class CurriculumState:
    current_level: int = 0
    total_items: int = 0
    promotions: int = 0
    demotions: int = 0
    recent_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=32), repr=False)
    level_history: list[tuple[int, int]] = field(default_factory=list)
    # (item_index_global, level) — one entry per transition


class CurriculumEnv(BaseEnv):
    def __init__(
        self,
        levels: list[BaseEnv],
        *,
        window: int = 20,
        promote_threshold: float = 0.6,
        demote_threshold: float | None = None,
        allow_demote: bool = False,
        on_level_change: Callable[[int, int], None] | None = None,
    ) -> None:
        if not levels:
            raise ValueError("CurriculumEnv requires at least one level")
        self.levels = levels
        self.window = window
        self.promote_threshold = promote_threshold
        self.demote_threshold = (
            demote_threshold if demote_threshold is not None else promote_threshold * 0.5
        )
        self.allow_demote = allow_demote
        self.on_level_change = on_level_change
        self.state = CurriculumState()
        self.state.recent_rewards = deque(maxlen=window)

    @property
    def current_level(self) -> int:
        return self.state.current_level

    @property
    def active(self) -> BaseEnv:
        return self.levels[self.state.current_level]

    # --- BaseEnv ---

    async def setup(self) -> None:
        for lvl in self.levels:
            await lvl.setup()

    async def get_next_item(self) -> dict[str, Any]:
        item = await self.active.get_next_item()
        # tag so trainers/loggers know which level this rollout came from
        item = dict(item)
        item["_curriculum_level"] = self.state.current_level
        self.state.total_items += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        return self.active.format_prompt(item)

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        return await self.active.compute_reward(item, trajectory, tool_context)

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        return self.active.build_supervised_samples(item)

    # --- curriculum control ---

    def observe(self, reward: float) -> None:
        """Feed the latest rollout reward; the wrapper decides level changes."""
        self.state.recent_rewards.append(float(reward))
        if len(self.state.recent_rewards) < self.window:
            return
        mean = sum(self.state.recent_rewards) / len(self.state.recent_rewards)
        old = self.state.current_level
        if mean >= self.promote_threshold and old + 1 < len(self.levels):
            self._change_level(old + 1)
        elif (
            self.allow_demote
            and mean <= self.demote_threshold
            and old > 0
        ):
            self._change_level(old - 1)

    def _change_level(self, new_level: int) -> None:
        old = self.state.current_level
        if new_level == old:
            return
        self.state.current_level = new_level
        self.state.level_history.append((self.state.total_items, new_level))
        self.state.recent_rewards.clear()
        if new_level > old:
            self.state.promotions += 1
        else:
            self.state.demotions += 1
        if self.on_level_change is not None:
            try:
                self.on_level_change(old, new_level)
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        recent = list(self.state.recent_rewards)
        return {
            "current_level": self.state.current_level,
            "n_levels": len(self.levels),
            "promotions": self.state.promotions,
            "demotions": self.state.demotions,
            "total_items": self.state.total_items,
            "recent_mean": sum(recent) / len(recent) if recent else 0.0,
            "window_full": len(recent) >= self.window,
            "level_history": list(self.state.level_history),
        }
