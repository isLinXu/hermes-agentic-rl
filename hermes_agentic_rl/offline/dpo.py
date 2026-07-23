"""Direct Preference Optimization (Rafailov et al. 2023).

Loss (per preference pair (x, y_w, y_l))::

    L = -log σ( β · ( logπ_θ(y_w|x) - logπ_θ(y_l|x)
                      - logπ_ref(y_w|x) + logπ_ref(y_l|x) ) )

We use the sequence-sum of token logπ as logπ(y|x). The reference policy is
a frozen snapshot of the backend at trainer-init time (``clone_frozen`` when
available, otherwise raises).

Implementation choices:
  - ``loss`` is computed per pair, then averaged across the batch.
  - ``accuracy`` = fraction of pairs where π_θ prefers y_w over y_l
    (i.e. sum_logp_chosen - sum_logp_rejected > 0).
  - ``beta`` default = 0.1 (DPO paper range 0.01 – 1.0).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.offline.replay_buffer import DPOPair, ReplayBuffer


@dataclass(slots=True)
class DPOConfig:
    n_epochs: int = 3
    batch_size: int = 4
    lr: float = 5e-4
    beta: float = 0.1
    grad_clip: float = 1.0
    shuffle: bool = True
    seed: int | None = 0
    log_every: int = 1


@dataclass(slots=True)
class DPOStats:
    steps: list[dict[str, Any]]


class DPOTrainer:
    def __init__(
        self,
        policy: LLMBackend,
        buffer: ReplayBuffer,
        cfg: DPOConfig | None = None,
        ref_policy: LLMBackend | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.policy = policy
        self.buffer = buffer
        self.cfg = cfg or DPOConfig()
        params = list(policy.trainable_parameters())
        if not params:
            raise RuntimeError("policy has no trainable parameters")
        self.optim = torch.optim.AdamW(params, lr=self.cfg.lr)
        if ref_policy is None:
            clone = getattr(policy, "clone_frozen", None)
            if clone is None:
                raise RuntimeError("DPOTrainer needs ref_policy or a backend with clone_frozen()")
            ref_policy = clone()
        self.ref_policy = ref_policy
        self.logger = logger or (lambda rec: print(self._format_log(rec)))

    def _format_log(self, r: dict[str, Any]) -> str:
        return (
            f"[dpo] step={r['step']} epoch={r['epoch']} "
            f"loss={r['loss']:.4f} acc={r['acc']:.3f} n={r['n']}"
        )

    @staticmethod
    def _seq_logp(
        backend: LLMBackend, prompt_ids: list[int], resp_ids: list[int], grad: bool
    ) -> torch.Tensor:
        if grad:
            logp = backend.score(prompt_ids, resp_ids)
        else:
            with torch.no_grad():
                logp = backend.score(prompt_ids, resp_ids)
        if logp.numel() == 0:
            return torch.zeros((), device=logp.device if hasattr(logp, "device") else "cpu")
        return logp.sum()

    def _iter_batches(self, pairs: list[DPOPair]) -> list[list[DPOPair]]:
        rng = random.Random(self.cfg.seed)
        order = list(range(len(pairs)))
        if self.cfg.shuffle:
            rng.shuffle(order)
        return [
            [pairs[i] for i in order[j : j + self.cfg.batch_size]]
            for j in range(0, len(order), self.cfg.batch_size)
        ]

    def train(self) -> DPOStats:
        steps: list[dict[str, Any]] = []
        pairs = list(self.buffer.iter_dpo_pairs())
        if not pairs:
            raise RuntimeError("DPOTrainer: buffer has no DPOPair")
        step_idx = 0
        for epoch in range(self.cfg.n_epochs):
            for batch in self._iter_batches(pairs):
                self.optim.zero_grad()
                logits_list: list[torch.Tensor] = []
                correct = 0
                for p in batch:
                    logp_chosen = self._seq_logp(self.policy, p.prompt_ids, p.chosen_ids, grad=True)
                    logp_reject = self._seq_logp(
                        self.policy, p.prompt_ids, p.rejected_ids, grad=True
                    )
                    logp_chosen_ref = self._seq_logp(
                        self.ref_policy, p.prompt_ids, p.chosen_ids, grad=False
                    )
                    logp_reject_ref = self._seq_logp(
                        self.ref_policy, p.prompt_ids, p.rejected_ids, grad=False
                    )
                    pref_logit = self.cfg.beta * (
                        (logp_chosen - logp_reject) - (logp_chosen_ref - logp_reject_ref)
                    )
                    logits_list.append(pref_logit)
                    if float(pref_logit.detach().item()) > 0.0:
                        correct += 1
                if not logits_list:
                    continue
                logits = torch.stack(logits_list)
                # L = -log σ(pref_logit) = softplus(-pref_logit)
                loss = F.softplus(-logits).mean()
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
                    "loss": float(loss.detach().item()),
                    "acc": correct / max(1, len(logits_list)),
                    "n": len(logits_list),
                }
                steps.append(rec)
                if self.cfg.log_every and step_idx % self.cfg.log_every == 0:
                    self.logger(rec)
                step_idx += 1
        return DPOStats(steps=steps)

    def save_policy(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.policy, "model"):
            torch.save(self.policy.model.state_dict(), p)  # type: ignore[attr-defined]
