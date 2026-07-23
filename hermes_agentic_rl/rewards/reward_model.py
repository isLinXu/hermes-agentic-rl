"""Learned scalar reward model (Bradley-Terry).

Given a base ``LLMBackend`` (tokenizer + trunk), we build a small reward
head that maps the final hidden state of ``prompt + response`` to a scalar
score ``r_φ(x, y)``.

Training objective on a preference pair (y_w ≻ y_l | x)::

    L = -log σ( r_φ(x, y_w) - r_φ(x, y_l) )

This is the standard Bradley-Terry loss used by OpenAI / Anthropic RLHF.

Implementation details:
  - The reward head is a single Linear(D → 1) over the trunk's last hidden
    state at the terminal position. For ``TinyCausalLM`` we reuse the
    internal ``_trunk`` method; for HF we fall back to
    ``output_hidden_states=True``.
  - The base model is NOT frozen by default so the full stack can adapt;
    set ``freeze_base=True`` to train only the head (much cheaper).
  - Inference (``score_pair(x, y)``) is batch-free / side-effect-free so
    the RM can be plugged into ``RewardManager`` as a ``BaseReward``.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.backends.tiny import TinyCausalLMBackend
from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.offline.replay_buffer import DPOPair, ReplayBuffer


@dataclass(slots=True)
class RewardModelConfig:
    lr: float = 1e-3
    n_epochs: int = 3
    batch_size: int = 4
    score_batch_size: int = 0
    freeze_base: bool = True
    grad_clip: float = 1.0
    shuffle: bool = True
    seed: int | None = 0
    log_every: int = 1


@dataclass(slots=True)
class RewardModelStats:
    steps: list[dict[str, Any]]


class RewardModel(nn.Module):
    """Scalar-output reward head built on top of any LLMBackend."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        freeze_base: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self._is_tiny = isinstance(backend, TinyCausalLMBackend)
        if self._is_tiny:
            hidden = int(backend.cfg.dim)  # type: ignore[attr-defined]
        else:
            model = getattr(backend, "model", None)
            config = getattr(model, "config", None)
            hidden = int(getattr(config, "hidden_size", 0) or getattr(config, "n_embd", 0) or 0)
            if hidden == 0:
                raise RuntimeError(
                    "RewardModel cannot infer hidden size from backend; pass a "
                    "TinyCausalLMBackend or an HF backend with config.hidden_size"
                )
        self.head = nn.Linear(hidden, 1, bias=True)
        model = getattr(backend, "model", None)
        if freeze_base and model is not None:
            for p in model.parameters():
                p.requires_grad_(False)
        if device is not None:
            target = torch.device(device)
            if model is not None:
                model.to(target)
            self.head.to(target)

    def _backbone_device(self) -> torch.device:
        return next(self.head.parameters()).device

    # Helper: get last hidden state at the terminal position of prompt+response.
    def _last_hidden(self, prompt_ids: list[int], response_ids: list[int]) -> torch.Tensor:
        full = list(prompt_ids) + list(response_ids)
        if len(full) < 1:
            raise ValueError("empty prompt+response")
        if self._is_tiny:
            model = self.backend.model  # type: ignore[attr-defined]
            model.train(self.training)
            device = next(model.parameters()).device
            ids = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
            h = model._trunk(ids)  # [1, T, D]
            return h[0, -1, :]  # [D]
        # HF path
        model = getattr(self.backend, "model", None)
        if model is None:
            raise RuntimeError("RewardModel requires backend.model for non-tiny backends")
        device = next(model.parameters()).device
        out = model(
            torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0),
            output_hidden_states=True,
        )
        h = out.hidden_states[-1][0, -1, :]
        return h

    def score_pair(self, prompt_ids: list[int], response_ids: list[int]) -> torch.Tensor:
        """Return a scalar reward (differentiable)."""
        h = self._last_hidden(prompt_ids, response_ids)
        return self.head(h).squeeze(-1)

    def score_batch(
        self,
        prompt_ids_list: list[list[int]],
        response_ids_list: list[list[int]],
    ) -> torch.Tensor:
        """Batched scalar rewards for aligned prompt/response pairs."""
        if len(prompt_ids_list) != len(response_ids_list):
            raise ValueError("prompt_ids_list and response_ids_list length mismatch")
        if not prompt_ids_list:
            return torch.zeros(0, device=self._backbone_device())
        if self._is_tiny:
            model = self.backend.model  # type: ignore[attr-defined]
            model.train(self.training)
            device = self._backbone_device()
            sequences = [
                list(prompt_ids_list[i]) + list(response_ids_list[i])
                for i in range(len(prompt_ids_list))
            ]
            max_len = max(len(seq) for seq in sequences)
            ids = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
            last_indices = torch.zeros(len(sequences), dtype=torch.long, device=device)
            for i, seq in enumerate(sequences):
                ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                last_indices[i] = len(seq) - 1
            h = model._trunk(ids)
            last_h = h[torch.arange(len(sequences), device=device), last_indices]
            return self.head(last_h).squeeze(-1)
        return torch.stack(
            [
                self.score_pair(prompt_ids_list[i], response_ids_list[i])
                for i in range(len(prompt_ids_list))
            ]
        )

    def trainable_parameters(self) -> Any:
        return [p for p in self.parameters() if p.requires_grad]


class RewardModelTrainer:
    def __init__(
        self,
        reward_model: RewardModel,
        buffer: ReplayBuffer,
        cfg: RewardModelConfig | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.rm = reward_model
        self.buffer = buffer
        self.cfg = cfg or RewardModelConfig()
        params = list(reward_model.trainable_parameters())
        if not params:
            raise RuntimeError("reward model has no trainable parameters")
        self.optim = torch.optim.AdamW(params, lr=self.cfg.lr)
        self.logger = logger or (lambda rec: print(self._format_log(rec)))

    def _format_log(self, r: dict[str, Any]) -> str:
        return (
            f"[rm] step={r['step']} epoch={r['epoch']} "
            f"loss={r['loss']:.4f} acc={r['acc']:.3f} n={r['n']}"
        )

    def _iter_batches(self, pairs: list[DPOPair]) -> list[list[DPOPair]]:
        rng = random.Random(self.cfg.seed)
        order = list(range(len(pairs)))
        if self.cfg.shuffle:
            rng.shuffle(order)
        return [
            [pairs[i] for i in order[j : j + self.cfg.batch_size]]
            for j in range(0, len(order), self.cfg.batch_size)
        ]

    def train(self) -> RewardModelStats:
        steps: list[dict[str, Any]] = []
        pairs = list(self.buffer.iter_dpo_pairs())
        if not pairs:
            raise RuntimeError("RewardModelTrainer: buffer has no DPOPair")
        step_idx = 0
        for epoch in range(self.cfg.n_epochs):
            self.rm.train()
            for batch in self._iter_batches(pairs):
                self.optim.zero_grad()
                if self.cfg.score_batch_size > 0:
                    prompts = [p.prompt_ids for p in batch]
                    chosen = [p.chosen_ids for p in batch]
                    rejected = [p.rejected_ids for p in batch]
                    r_w = self.rm.score_batch(prompts, chosen)
                    r_l = self.rm.score_batch(prompts, rejected)
                    diff = r_w - r_l
                    correct = int((diff.detach() > 0.0).sum().item())
                    loss = F.softplus(-diff).mean()
                    n_pairs = len(batch)
                else:
                    logits: list[torch.Tensor] = []
                    correct = 0
                    for p in batch:
                        r_w = self.rm.score_pair(p.prompt_ids, p.chosen_ids)
                        r_l = self.rm.score_pair(p.prompt_ids, p.rejected_ids)
                        diff = r_w - r_l
                        logits.append(diff)
                        if float(diff.detach().item()) > 0.0:
                            correct += 1
                    if not logits:
                        continue
                    stacked = torch.stack(logits)
                    loss = F.softplus(-stacked).mean()
                    n_pairs = len(logits)
                if n_pairs == 0:
                    continue
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.rm.trainable_parameters()), max_norm=self.cfg.grad_clip
                    )
                self.optim.step()
                rec = {
                    "step": step_idx,
                    "epoch": epoch,
                    "loss": float(loss.detach().item()),
                    "acc": correct / max(1, n_pairs),
                    "n": n_pairs,
                }
                steps.append(rec)
                if self.cfg.log_every and step_idx % self.cfg.log_every == 0:
                    self.logger(rec)
                step_idx += 1
        return RewardModelStats(steps=steps)

    def save_head(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.rm.head.state_dict(), p)


# ---------------------------------------------------------------------------
# Plug into RewardManager as a component
# ---------------------------------------------------------------------------


class RewardModelComponent:
    """Reward component that queries a trained RewardModel.

    Used by ``RewardManager([RewardModelComponent(rm)])`` to replace hand-
    crafted reward functions with a learned scalar during RL fine-tuning.
    """

    name = "reward_model"

    def __init__(self, reward_model: RewardModel, weight: float = 1.0) -> None:
        self.reward_model = reward_model
        self.weight = weight

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del tool_context
        runtime_block = trajectory.metadata.get("runtime") or {}
        rl_meta = runtime_block.get("rl") if isinstance(runtime_block, dict) else None
        if not rl_meta:
            return RewardResult(
                name=self.name, score=0.0, reason="no rl metadata", weight=self.weight
            )
        prompt_ids = list(rl_meta.get("prompt_ids") or [])
        response_ids = list(rl_meta.get("response_ids") or [])
        if not response_ids:
            return RewardResult(
                name=self.name, score=0.0, reason="empty response", weight=self.weight
            )
        with torch.no_grad():
            score = float(self.reward_model.score_pair(prompt_ids, response_ids).item())
        return RewardResult(
            name=self.name,
            score=score,
            reason=f"rm_score={score:.4f}",
            weight=self.weight,
        )
