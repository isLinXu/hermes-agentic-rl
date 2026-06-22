"""RL algorithms and registry.

Each algorithm operates on a `RolloutBatch` (a list of `RolloutRecord`s that
carry prompt_ids, response_ids, old_logprobs, and a scalar reward) and returns
`AlgoUpdateStats` (loss, kl, entropy, etc.). The algorithm does NOT own the
optimizer; the Trainer does. Algorithms are pure torch functions so they are
easy to test.
"""

from typing import cast

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
)
from hermes_agentic_rl.algos.factored import FactoredConfig, FactoredGRPO
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.algos.gspo import GSPO, GSPOConfig
from hermes_agentic_rl.algos.hybrid import HybridAlgo, HybridConfig
from hermes_agentic_rl.algos.opd import OPDAlgo, OPDConfig
from hermes_agentic_rl.algos.ppo import PPO, PPOConfig
from hermes_agentic_rl.algos.rloo import RLOOAlgo, RLOOConfig
from hermes_agentic_rl.algos.simpo import SimPOAlgo, SimPOConfig

ALGO_REGISTRY: dict[str, type[BaseAlgo]] = {}


def _normalize_algo_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if not key:
        raise ValueError("algorithm name must be non-empty")
    return key


def register_algo(
    name: str,
    algo_class: type[BaseAlgo],
    *,
    overwrite: bool = False,
) -> type[BaseAlgo]:
    """Register an algorithm class under a stable lowercase name.

    The class is returned so callers can use this as a decorator.
    """
    key = _normalize_algo_name(name)
    existing = ALGO_REGISTRY.get(key)
    if existing is not None and existing is not algo_class and not overwrite:
        raise ValueError(f"algorithm {key!r} is already registered")
    ALGO_REGISTRY[key] = algo_class
    return algo_class


def get_algo(name: str, default: type[BaseAlgo] | None = GRPO) -> type[BaseAlgo]:
    """Resolve an algorithm class by name.

    Unknown names fall back to GRPO by default, matching the historical CLI
    behavior. Pass ``default=None`` to turn unknown names into ``KeyError``.
    """
    key = _normalize_algo_name(name)
    if key in ALGO_REGISTRY:
        return ALGO_REGISTRY[key]
    if default is not None:
        return default
    raise KeyError(f"algorithm {key!r} is not registered")


def list_algos() -> list[str]:
    """Return registered algorithm names."""
    return sorted(ALGO_REGISTRY)


for _name, _algo in {
    "grpo": GRPO,
    "ppo": PPO,
    "opd": OPDAlgo,
    "hybrid": HybridAlgo,
    "gspo": GSPO,
    "rloo": RLOOAlgo,
    "factored_grpo": FactoredGRPO,
    "simpo": SimPOAlgo,
}.items():
    register_algo(_name, cast(type[BaseAlgo], _algo))

__all__ = [
    "ALGO_REGISTRY",
    "GRPO",
    "GSPO",
    "PPO",
    "AlgoUpdateStats",
    "BaseAlgo",
    "FactoredConfig",
    "FactoredGRPO",
    "GRPOConfig",
    "GSPOConfig",
    "HybridAlgo",
    "HybridConfig",
    "OPDAlgo",
    "OPDConfig",
    "PPOConfig",
    "RLOOAlgo",
    "RLOOConfig",
    "RolloutBatch",
    "RolloutRecord",
    "SimPOAlgo",
    "SimPOConfig",
    "get_algo",
    "list_algos",
    "register_algo",
]
