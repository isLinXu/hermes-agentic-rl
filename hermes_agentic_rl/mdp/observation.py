from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Observation:
    """What the policy sees at a given step.

    For the MVP, observation = tokenized prompt. As the framework matures, this
    grows to include tool-state, memory, partial messages, etc.
    """

    prompt_ids: list[int]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
