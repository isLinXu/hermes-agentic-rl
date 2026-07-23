"""Behavior Cloning trainer.

Objective: minimize NLL of response_ids under π, conditioned on prompt_ids.

Useful as:
  - A warm-start before RL (so the policy isn't purely random)
  - A baseline that DPO / GRPO must beat
  - A smoke test that the backend's score() gradient works
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.offline.replay_buffer import ReplayBuffer, TrainSample


@dataclass(slots=True)
class BCConfig:
    n_epochs: int = 3
    batch_size: int = 4
    lr: float = 5e-3
    grad_clip: float = 1.0
    shuffle: bool = True
    seed: int | None = 0
    log_every: int = 1
    min_response_tokens: int = 1  # skip samples shorter than this


@dataclass(slots=True)
class BCStats:
    steps: list[dict[str, Any]]


class BCTrainer:
    """NLL trainer over (prompt, response) pairs."""

    def __init__(
        self,
        policy: LLMBackend,
        buffer: ReplayBuffer,
        cfg: BCConfig | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.policy = policy
        self.buffer = buffer
        self.cfg = cfg or BCConfig()
        params = list(policy.trainable_parameters())
        if not params:
            raise RuntimeError("policy has no trainable parameters")
        self.optim = torch.optim.AdamW(params, lr=self.cfg.lr)
        self.logger = logger or (lambda rec: print(self._format_log(rec)))

    def _format_log(self, r: dict[str, Any]) -> str:
        return f"[bc] step={r['step']} epoch={r['epoch']} nll={r['nll']:.4f} n={r['n']}"

    def _iter_batches(self, samples: list[TrainSample]) -> list[list[TrainSample]]:
        rng = random.Random(self.cfg.seed)
        order = list(range(len(samples)))
        if self.cfg.shuffle:
            rng.shuffle(order)
        return [
            [samples[i] for i in order[j : j + self.cfg.batch_size]]
            for j in range(0, len(order), self.cfg.batch_size)
        ]

    def train(self) -> BCStats:
        steps: list[dict[str, Any]] = []
        samples = [
            s
            for s in self.buffer.iter_samples()
            if len(s.response_ids) >= self.cfg.min_response_tokens
        ]
        if not samples:
            raise RuntimeError("BCTrainer: buffer has no usable TrainSamples")
        step_idx = 0
        for epoch in range(self.cfg.n_epochs):
            for batch in self._iter_batches(samples):
                self.optim.zero_grad()
                losses: list[torch.Tensor] = []
                for s in batch:
                    logp = self.policy.score(s.prompt_ids, s.response_ids)
                    if logp.numel() == 0:
                        continue
                    # NLL = -mean logπ(a_t | s_t)
                    losses.append(-logp.mean())
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy.trainable_parameters()),
                        max_norm=self.cfg.grad_clip,
                    )
                self.optim.step()

                rec = {
                    "step": step_idx,
                    "epoch": epoch,
                    "nll": float(loss.detach().item()),
                    "n": len(losses),
                }
                steps.append(rec)
                if self.cfg.log_every and step_idx % self.cfg.log_every == 0:
                    self.logger(rec)
                step_idx += 1
        return BCStats(steps=steps)

    def save_policy(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.policy, "model"):
            torch.save(self.policy.model.state_dict(), p)  # type: ignore[attr-defined]
