"""RewardManager — thin aggregator over a flat list of BaseReward components.

This module also provides a compatibility shim so that ``RewardComposer``
(see ``hermes_agentic_rl.rewards.composer``) can be used as a drop-in
replacement. Both classes expose:

    async evaluate(item, trajectory, tool_context) -> RewardSummary
    .components

The trainer duck-types on these two members, so either class works.
"""

from __future__ import annotations

from typing import Any

from hermes_agentic_rl.core.types import RewardSummary, Trajectory
from hermes_agentic_rl.rewards.aggregate import weighted_sum
from hermes_agentic_rl.rewards.base import BaseReward


class RewardManager:
    """Flat-list reward aggregator.

    Parameters
    ----------
    rewards:
        List of ``BaseReward`` components (positional, preferred).
    components:
        Alias for ``rewards`` — accepts ``RewardComposer``-style keyword.
        If both are given, ``rewards`` wins.
    composer:
        Optional ``RewardComposer`` to delegate to. When set, ``evaluate``
        forwards to the composer and ``components`` mirrors the composer's.
    """

    def __init__(
        self,
        rewards: list[BaseReward] | None = None,
        *,
        components: list[BaseReward] | None = None,
        composer: Any | None = None,
    ) -> None:
        if rewards is not None:
            self.rewards = rewards
        elif components is not None:
            self.rewards = components
        else:
            self.rewards = []
        self._composer = composer

    @property
    def components(self) -> list[BaseReward]:
        """Mirror RewardComposer's ``.components`` for duck-typing compatibility."""
        if self._composer is not None:
            return getattr(self._composer, "components", self.rewards)
        return self.rewards

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardSummary:
        if self._composer is not None:
            return await self._composer.evaluate(item, trajectory, tool_context)
        results = []
        for reward in self.rewards:
            results.append(await reward.evaluate(item, trajectory, tool_context))
        return weighted_sum(results)
