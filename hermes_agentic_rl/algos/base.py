from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from hermes_agentic_rl.backends.base import LLMBackend


@dataclass(slots=True)
class RolloutRecord:
    """One (prompt, sampled-response, reward) tuple with frozen old logprobs."""

    prompt_ids: list[int]
    response_ids: list[int]
    old_logprobs: list[float]
    reward: float
    group_id: str
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class RolloutBatch:
    records: list[RolloutRecord]
    # Optional cache populated by compound algos (e.g. HybridAlgo) so child
    # branches can reuse one ``policy.score_batch`` forward. Keys are
    # ``id(RolloutRecord)``; values are 1-D differentiable logprob rows.
    shared_new_logprobs: dict[int, torch.Tensor] | None = None
    shared_logprobs_temperature: float | None = None

    def __len__(self) -> int:
        return len(self.records)

    def by_group(self) -> dict[str, list[RolloutRecord]]:
        out: dict[str, list[RolloutRecord]] = {}
        for r in self.records:
            out.setdefault(r.group_id, []).append(r)
        return out

    def has_shared_logprobs_for(self, records: list[RolloutRecord]) -> bool:
        if not records or self.shared_new_logprobs is None:
            return False
        cache = self.shared_new_logprobs
        return all(id(rec) in cache for rec in records)


def stack_cached_logprobs(
    cache: dict[int, torch.Tensor],
    records: list[RolloutRecord],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-record cached logprob rows into a padded (B, T_max) batch.

    Preserves the autograd graph of the cached tensors so a single hybrid
    forward can feed multiple algorithm branches.
    """
    if not records:
        empty = torch.zeros(0, 0)
        return empty, empty

    lengths = [len(rec.response_ids) for rec in records]
    t_max = max(lengths) if lengths else 0
    if t_max == 0:
        empty = torch.zeros(len(records), 0)
        return empty, empty.to(dtype=torch.bool)

    ref = next(iter(cache.values()))
    dtype = ref.dtype
    device = ref.device
    logp = torch.zeros(len(records), t_max, dtype=dtype, device=device)
    mask = torch.zeros(len(records), t_max, dtype=torch.bool, device=device)

    for i, rec in enumerate(records):
        row = cache[id(rec)]
        n = min(int(row.numel()), lengths[i], t_max)
        if n <= 0:
            continue
        logp[i, :n] = row[:n]
        mask[i, :n] = True

    return logp, mask


def old_logprobs_tensor(
    record: RolloutRecord,
    *,
    length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Materialize rollout old logprobs, preferring a cached tensor in metadata."""
    cached = record.metadata.get("_old_logprobs_tensor")
    if isinstance(cached, torch.Tensor):
        src = cached.to(dtype=dtype, device=device)
        if src.numel() >= length:
            return src[-length:]
        out = torch.zeros(length, dtype=dtype, device=device)
        if src.numel() > 0:
            out[-src.numel() :] = src
        return out

    vals = record.old_logprobs[-length:] if record.old_logprobs else []
    out = torch.zeros(length, dtype=dtype, device=device)
    if vals:
        n = min(len(vals), length)
        out[:n] = torch.tensor(vals[:n], dtype=dtype, device=device)
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
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

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
