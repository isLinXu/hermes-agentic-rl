"""API stability markers for hermes-agentic-rl.

Provides decorators and constants to communicate API maturity:

- ``@experimental`` — function/class may change or be removed before v1.0
- ``@stable`` — API is frozen; breaking changes require a deprecation cycle
- ``__stable_apis__`` — list of names considered stable

Usage::

    from hermes_agentic_rl._compat import experimental, stable

    @experimental
    class ProcessRewardModel:
        ...

    @stable
    class LLMBackend:
        ...
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def experimental(func: F) -> F:
    """Mark a function or class as experimental.

    Emits a ``FutureWarning`` on first use. The API may change or be
    removed in any version before v1.0.
    """
    if isinstance(func, type):
        # Class decorator — wrap __init__
        original_init = func.__init__

        @functools.wraps(original_init)
        def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                f"{func.__name__} is experimental and may change before v1.0.",
                FutureWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        func.__init__ = wrapped_init  # type: ignore[method-assign]
        func._is_experimental = True  # type: ignore[attr-defined]
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{func.__name__} is experimental and may change before v1.0.",
            FutureWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    wrapper._is_experimental = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def stable(func: F) -> F:
    """Mark a function or class as stable (frozen API).

    Stable APIs will not have breaking changes without a deprecation
    cycle of at least one minor version.
    """
    func._is_stable = True  # type: ignore[attr-defined]
    return func


# ---------------------------------------------------------------------------
# Stable API registry
# ---------------------------------------------------------------------------

# Names that are considered stable (frozen) as of v0.11.
# New features should NOT be added here until they've been stable for
# at least one release cycle.
__stable_apis__: list[str] = [
    # Core protocols
    "LLMBackend",
    "BaseEnv",
    "BaseAlgo",
    "BaseReward",
    # Core data types
    "Trajectory",
    "RolloutStep",
    "RolloutRecord",
    "RolloutBatch",
    # Core config
    "OnPolicyTrainerConfig",
    "GRPOTrainerConfig",
    # Core trainer
    "OnPolicyTrainer",
    "GRPOTrainer",
]


def is_stable(name: str) -> bool:
    """Check if a name is in the stable API registry."""
    return name in __stable_apis__


def is_experimental(obj: Any) -> bool:
    """Check if an object has been marked as experimental."""
    return getattr(obj, "_is_experimental", False)
