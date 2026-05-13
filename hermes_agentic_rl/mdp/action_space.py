from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TokenAction:
    """Low-level action: the sampled token sequence from the policy.

    Carries the `old_logprobs` so the GRPO/PPO trainer can compute the ratio
    `exp(logπ_new - logπ_old)` without re-running the sampler.
    """

    token_ids: list[int]
    logprobs: list[float]
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class TextAction:
    """Higher-level action: the decoded text + optional tool call structure.

    The MVP task is "emit a string that mentions the answer", so the agent-loop
    returns a TextAction for the env and reward to inspect, while the TokenAction
    is what the trainer consumes.
    """

    text: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


# Alias: some callers just want "an action" without caring about the flavor.
Action = TokenAction
