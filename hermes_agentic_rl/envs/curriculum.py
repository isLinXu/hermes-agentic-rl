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

import math
import random
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

    def observe(self, reward: float, level: int | None = None) -> None:
        """Feed the latest rollout reward; the wrapper decides level changes.

        ``level`` is accepted for signature-compatibility with
        :class:`MixedCurriculumEnv` (and the trainer's ``_observe_env_reward``
        helper) but ignored: a sequential curriculum only tracks the *active*
        level.
        """
        del level
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


class MixedCurriculumEnv(BaseEnv):
    """Multi-stream unified training: sample several sub-envs *concurrently*.

    Unlike :class:`CurriculumEnv` (one active level at a time, sequential
    promotion), ``MixedCurriculumEnv`` keeps **all** streams live and draws each
    item from a *weighted mixture* — the foundation of OpenClaw-RL-style unified
    multi-stream training where heterogeneous task types (tool-use, reasoning,
    counting, …) are interleaved in a single optimizer step instead of trained
    in separate phases.

    Adaptive reweighting
    --------------------
    When ``observe(reward, level=i)`` is fed back, each stream tracks a windowed
    mean reward. Once a stream's window fills, its sampling weight is nudged
    multiplicatively toward **struggling** streams (mean reward below
    ``target_reward`` raises the weight) so optimization budget follows where the
    policy is weakest — a difficulty-prioritised mixture. Weights are floored at
    ``min_weight`` and renormalised to sum to 1.0.

    Determinism: stream selection uses a private seeded RNG so runs are
    reproducible independent of global random state.
    """

    def __init__(
        self,
        levels: list[BaseEnv],
        *,
        weights: list[float] | None = None,
        window: int = 20,
        adapt_lr: float = 0.1,
        min_weight: float = 0.01,
        target_reward: float = 0.5,
        seed: int | None = 0,
    ) -> None:
        if not levels:
            raise ValueError("MixedCurriculumEnv requires at least one stream")
        if weights is not None and len(weights) != len(levels):
            raise ValueError(
                f"weights length ({len(weights)}) must match levels ({len(levels)})"
            )
        if weights is not None and any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        self.levels = levels
        self.window = window
        self.adapt_lr = adapt_lr
        self.min_weight = min_weight
        self.target_reward = target_reward
        self._rng = random.Random(seed)
        # Raw (unnormalised) weights drive the multiplicative update; the public
        # ``weights`` list is always the normalised view.
        raw = list(weights) if weights is not None else [1.0] * len(levels)
        self._raw_w: list[float] = [float(w) for w in raw]
        self.weights: list[float] = []
        self._renormalize()
        self._recent: list[deque[float]] = [
            deque(maxlen=window) for _ in levels
        ]
        self._counts: list[int] = [0] * len(levels)
        self._last_level: int = 0
        self.total_items: int = 0

    # --- multi-stream helpers ---

    @property
    def current_level(self) -> int:
        """The most recently sampled stream (for reward-routing compatibility)."""
        return self._last_level

    @property
    def active(self) -> BaseEnv:
        return self.levels[self._last_level]

    def _renormalize(self) -> None:
        n = len(self._raw_w)
        total = sum(self._raw_w)
        if total <= 0:
            self.weights = [1.0 / n] * n
            return
        w = [x / total for x in self._raw_w]
        mw = self.min_weight
        if mw <= 0 or n * mw >= 1.0:
            self.weights = w
            return
        # Clamp any sub-floor weight to ``min_weight`` and rescale the remaining
        # (unpinned) weights to fill ``1 - pinned_mass``. Iterate to a fixpoint
        # (rescaling can push another weight below the floor). Weights already
        # above the floor are preserved exactly when nothing needs clamping.
        for _ in range(n):
            below = [i for i in range(n) if w[i] < mw - 1e-12]
            if not below:
                break
            free = [i for i in range(n) if i not in below]
            free_sum = sum(w[i] for i in free)
            target_free = max(0.0, 1.0 - mw * len(below))
            for i in below:
                w[i] = mw
            if free_sum > 0:
                scale = target_free / free_sum
                for i in free:
                    w[i] *= scale
            elif free:
                for i in free:
                    w[i] = target_free / len(free)
        self.weights = w

    def _sample_level(self) -> int:
        return self._rng.choices(range(len(self.levels)), weights=self.weights, k=1)[0]

    def _level_of(self, item: dict[str, Any]) -> int:
        lvl = item.get("_curriculum_level", self._last_level)
        try:
            lvl = int(lvl)
        except (TypeError, ValueError):
            lvl = self._last_level
        return lvl if 0 <= lvl < len(self.levels) else self._last_level

    # --- BaseEnv ---

    async def setup(self) -> None:
        for lvl in self.levels:
            await lvl.setup()

    async def get_next_item(self) -> dict[str, Any]:
        idx = self._sample_level()
        self._last_level = idx
        item = await self.levels[idx].get_next_item()
        item = dict(item)
        item["_curriculum_level"] = idx
        self.total_items += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        return self.levels[self._level_of(item)].format_prompt(item)

    async def compute_reward(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> list[RewardResult]:
        return await self.levels[self._level_of(item)].compute_reward(
            item, trajectory, tool_context
        )

    def build_supervised_samples(self, item: dict[str, Any]) -> list[SupervisedSample]:
        return self.levels[self._level_of(item)].build_supervised_samples(item)

    # --- adaptive mixture control ---

    def observe(self, reward: float, level: int | None = None) -> None:
        """Feed a rollout reward; optionally attributed to a specific stream.

        ``level=None`` records the reward globally without per-stream
        attribution (e.g. from the distributed path that cannot echo the item).
        """
        r = float(reward)
        if level is None:
            return
        idx = int(level)
        if not (0 <= idx < len(self.levels)):
            return
        self._recent[idx].append(r)
        self._counts[idx] += 1
        if len(self._recent[idx]) < self.window:
            return
        mean = sum(self._recent[idx]) / len(self._recent[idx])
        # Struggling streams (mean < target) get up-weighted; mastered streams
        # decay toward min_weight so budget follows the weakest skills.
        factor = math.exp(self.adapt_lr * (self.target_reward - mean))
        self._raw_w[idx] *= factor
        self._renormalize()

    def snapshot(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "n_levels": len(self.levels),
            "total_items": self.total_items,
            "last_level": self._last_level,
            "per_level": [
                {
                    "weight": self.weights[i],
                    "count": self._counts[i],
                    "recent_mean": (
                        sum(self._recent[i]) / len(self._recent[i])
                        if self._recent[i]
                        else 0.0
                    ),
                    "window_full": len(self._recent[i]) >= self.window,
                }
                for i in range(len(self.levels))
            ],
        }
