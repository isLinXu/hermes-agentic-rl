from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RolloutStep:
    turn_index: int
    assistant_message: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class Trajectory:
    task_id: str
    prompt: str
    steps: list[RolloutStep]
    final_output: str | None
    finished_naturally: bool
    turns_used: int
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class RewardResult:
    name: str
    score: float
    reason: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class RewardSummary:
    final_score: float
    components: list[RewardResult]
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class TrainSample:
    task_id: str
    prompt: str
    final_output: str | None
    reward: float
    trajectory: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
