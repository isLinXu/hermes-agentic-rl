from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hermes_agentic_rl.rewards.base import BaseReward

RewardFactory = type[BaseReward]

REWARD_REGISTRY: dict[str, RewardFactory] = {}
_BUILTINS_LOADED = False


def register_reward(name: str) -> Callable[[RewardFactory], RewardFactory]:
    """Register a reward component class under a config-facing name."""

    normalized = _normalize_name(name)

    def _decorator(cls: RewardFactory) -> RewardFactory:
        REWARD_REGISTRY[normalized] = cls
        return cls

    return _decorator


def get_reward_class(name: str) -> RewardFactory:
    ensure_builtin_rewards_registered()
    normalized = _normalize_name(name)
    try:
        return REWARD_REGISTRY[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(REWARD_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown reward component {name!r}; known rewards: {known}"
        ) from exc


def build_reward_from_spec(spec: dict[str, Any]) -> BaseReward:
    """Instantiate a reward component from a YAML-style component spec."""

    name = str(spec.get("name", "")).strip()
    if not name:
        raise ValueError("reward component spec is missing 'name'")
    cls = get_reward_class(name)
    weight = float(spec.get("weight", 1.0))
    kwargs = {
        key: value for key, value in spec.items()
        if key not in {"name", "weight"}
    }
    try:
        return cls(weight=weight, **kwargs)
    except TypeError:
        if kwargs:
            return cls(weight=weight)
        raise


def ensure_builtin_rewards_registered() -> None:
    """Import built-in reward modules once so their decorators run."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    # Imports are intentionally local to avoid import cycles at module load.
    from hermes_agentic_rl.rewards import (  # noqa: F401
        filesystem_verifier_reward,
        next_turn_feedback,
        outcome_reward,
        similarity_reward,
        toolcall_reward,
    )


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()
