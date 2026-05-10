from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hermes_agentic_rl.backends.base import LLMBackend


@dataclass(slots=True)
class RolloutRecord:
    """One (prompt, sampled-response, reward) tuple with frozen old logprobs."""

    prompt_ids: list[int]
    response_ids: list[int]
    old_logprobs: list[float]
    reward: float
    group_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutBatch:
    records: list[RolloutRecord]

    def __len__(self) -> int:
        return len(self.records)

    def by_group(self) -> dict[str, list[RolloutRecord]]:
        out: dict[str, list[RolloutRecord]] = {}
        for r in self.records:
            out.setdefault(r.group_id, []).append(r)
        return out


@dataclass(slots=True)
class AlgoUpdateStats:
    loss: float
    policy_loss: float
    kl: float
    entropy: float
    mean_reward: float
    mean_advantage: float
    clip_frac: float
    n_records: int
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "policy_loss": self.policy_loss,
            "kl": self.kl,
            "entropy": self.entropy,
            "mean_reward": self.mean_reward,
            "mean_advantage": self.mean_advantage,
            "clip_frac": self.clip_frac,
            "n_records": self.n_records,
            **self.extra,
        }


class BaseAlgo(ABC):
    """Pure-function algorithm: takes a policy + batch, returns loss tensor + stats.

    The Trainer owns the optimizer and calls this per update step.
    """

    @abstractmethod
    def compute_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        batch: RolloutBatch,
    ) -> tuple[Any, AlgoUpdateStats]:
        """Return (loss_tensor, stats). `loss_tensor.backward()` is the learner step."""
        raise NotImplementedError
