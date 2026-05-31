"""Process Reward Model (PRM) — step-level reward scoring.

Outcome Reward Models (ORM) give one scalar per completed rollout.
Process Reward Models (PRM) give a scalar at every reasoning step,
providing denser training signal for multi-step tasks.

Architecture:
  - Same backbone as RewardModel (LLMBackend trunk + Linear head).
  - Input: prompt + prefix (steps 0..t) → scalar score for step t.
  - Training: pair (prompt, [step_0, ..., step_T], label_T) where
    label_T ∈ {0, 1} (human / verifier annotation per step).

Usage patterns:
  1. Dense training signal: replace or augment terminal reward with
     per-step PRM scores in token_rewards dict.
  2. Best-of-N reranking: score N rollouts, pick the best prefix path.
  3. MCTS guidance: score partial trajectories during tree search.

Integration with RewardManager:
  Use ``PRMComponent`` exactly like ``RewardModelComponent``.

Integration with OnPolicyTrainer (token-level PRM rewards):
  Set ``record.metadata["token_rewards"]`` from PRM scores so PPO's
  GAE computes advantages per step rather than per terminal token.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.backends.tiny import TinyCausalLMBackend
from hermes_agentic_rl.core.types import RewardResult, Trajectory

# ---------------------------------------------------------------------------
# Step tokenization helper
# ---------------------------------------------------------------------------

STEP_SEPARATOR = "\n\n"  # default separator between reasoning steps


def split_steps(text: str, sep: str = STEP_SEPARATOR) -> list[str]:
    """Split a response into reasoning steps."""
    parts = [p.strip() for p in text.split(sep)]
    return [p for p in parts if p]


def steps_to_prefix(steps: list[str], t: int, sep: str = STEP_SEPARATOR) -> str:
    """Return the text prefix up to and including step t."""
    return sep.join(steps[: t + 1])


# ---------------------------------------------------------------------------
# PRM model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PRMConfig:
    lr: float = 1e-3
    n_epochs: int = 3
    batch_size: int = 4
    freeze_base: bool = True
    grad_clip: float = 1.0
    step_sep: str = STEP_SEPARATOR


class ProcessRewardModel(nn.Module):
    """Step-level scalar reward model.

    Scores a (prompt, prefix) pair by forwarding through the LLM backbone
    and projecting the final hidden state to a scalar.

    Compatible with both TinyCausalLMBackend and HFCausalLMBackend.
    """

    def __init__(self, backend: LLMBackend, *, freeze_base: bool = True) -> None:
        super().__init__()
        self.backend = backend
        self._is_tiny = isinstance(backend, TinyCausalLMBackend)
        if self._is_tiny:
            hidden = int(backend.cfg.dim)  # type: ignore[attr-defined]
        else:
            model = getattr(backend, "model", None)
            cfg = getattr(model, "config", None)
            hidden = int(
                getattr(cfg, "hidden_size", 0)
                or getattr(cfg, "n_embd", 0)
                or 0
            )
            if hidden == 0:
                raise RuntimeError(
                    "ProcessRewardModel: cannot infer hidden_size from backend. "
                    "Use TinyCausalLMBackend or HF backend with config.hidden_size."
                )
        self.head = nn.Linear(hidden, 1, bias=True)
        base_model = getattr(backend, "model", None)
        if freeze_base and base_model is not None:
            for p in base_model.parameters():
                p.requires_grad_(False)

    def _last_hidden(self, ids: list[int]) -> torch.Tensor:
        if not ids:
            raise ValueError("empty ids")
        if self._is_tiny:
            model = self.backend.model  # type: ignore[attr-defined]
            model.train(self.training)
            device = next(model.parameters()).device
            t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            h = model._trunk(t)
            return h[0, -1, :]
        model = getattr(self.backend, "model", None)
        if model is None:
            raise RuntimeError("ProcessRewardModel requires backend.model")
        device = next(model.parameters()).device
        out = model(
            torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0),
            output_hidden_states=True,
        )
        return out.hidden_states[-1][0, -1, :]

    def score_prefix(
        self,
        prompt_ids: list[int],
        prefix_ids: list[int],
    ) -> torch.Tensor:
        """Return a scalar reward for prompt + prefix (differentiable)."""
        full = list(prompt_ids) + list(prefix_ids)
        h = self._last_hidden(full)
        return self.head(h).squeeze(-1)

    def score_steps(
        self,
        prompt_ids: list[int],
        response_text: str,
        tokenizer: Any,
        step_sep: str = STEP_SEPARATOR,
    ) -> list[float]:
        """Return per-step PRM scores for a complete response.

        Args:
            prompt_ids: tokenized prompt.
            response_text: full decoded response.
            tokenizer: backend tokenizer (for encoding steps).
            step_sep: step separator string.

        Returns:
            List of scalar scores, one per step (length = n_steps).
        """
        steps = split_steps(response_text, sep=step_sep)
        scores: list[float] = []
        for t in range(len(steps)):
            prefix = steps_to_prefix(steps, t, sep=step_sep)
            prefix_ids = tokenizer.encode(prefix)
            with torch.no_grad():
                s = float(self.score_prefix(prompt_ids, prefix_ids).item())
            scores.append(s)
        return scores

    def trainable_parameters(self) -> Any:
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# PRM Trainer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PRMStepSample:
    """One training example: (prompt, prefix, label)."""

    prompt_ids: list[int]
    prefix_ids: list[int]   # prompt + steps 0..t
    label: float            # 1.0 = good step, 0.0 = bad step


class PRMTrainer:
    """Train a ProcessRewardModel on step-level binary labels."""

    def __init__(
        self,
        prm: ProcessRewardModel,
        cfg: PRMConfig | None = None,
        logger: Any = None,
    ) -> None:
        self.prm = prm
        self.cfg = cfg or PRMConfig()
        params = list(prm.trainable_parameters())
        if not params:
            raise RuntimeError("ProcessRewardModel has no trainable parameters")
        self.optim = torch.optim.AdamW(params, lr=self.cfg.lr)
        self.logger = logger or (lambda r: print(
            f"[prm] step={r['step']} loss={r['loss']:.4f} acc={r['acc']:.3f}"
        ))

    def train(self, samples: list[PRMStepSample]) -> dict[str, Any]:
        if not samples:
            return {"steps": []}
        records: list[dict[str, Any]] = []
        step_idx = 0
        for epoch in range(self.cfg.n_epochs):
            self.prm.train()
            batch_size = max(1, self.cfg.batch_size)
            for start in range(0, len(samples), batch_size):
                batch = samples[start: start + batch_size]
                self.optim.zero_grad()
                scores: list[torch.Tensor] = []
                labels: list[float] = []
                for s in batch:
                    sc = self.prm.score_prefix(s.prompt_ids, s.prefix_ids)
                    scores.append(sc)
                    labels.append(s.label)
                stacked = torch.stack(scores)
                label_t = torch.tensor(labels, dtype=stacked.dtype, device=stacked.device)
                loss = F.binary_cross_entropy_with_logits(stacked, label_t)
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.prm.trainable_parameters()), self.cfg.grad_clip
                    )
                self.optim.step()
                with torch.no_grad():
                    pred_pos = stacked > 0
                    acc = float((pred_pos == (label_t > 0.5)).float().mean().item())
                rec = {
                    "step": step_idx,
                    "epoch": epoch,
                    "loss": float(loss.detach().item()),
                    "acc": acc,
                    "n": len(batch),
                }
                records.append(rec)
                self.logger(rec)
                step_idx += 1
        return {"steps": records}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.prm.head.state_dict(), p)


# ---------------------------------------------------------------------------
# RewardManager component
# ---------------------------------------------------------------------------


class PRMComponent:
    """Plug a ProcessRewardModel into RewardManager as a dense reward.

    Computes per-step scores, aggregates them (mean / min / last / sum),
    and returns the result as a single RewardResult.
    Also stores per-step scores in trajectory metadata for PPO token_rewards.
    """

    name = "prm"

    def __init__(
        self,
        prm: ProcessRewardModel,
        weight: float = 1.0,
        agg: str = "mean",   # mean | min | last | sum
        step_sep: str = STEP_SEPARATOR,
        write_token_rewards: bool = True,
    ) -> None:
        self.prm = prm
        self.weight = weight
        self.agg = agg
        self.step_sep = step_sep
        self.write_token_rewards = write_token_rewards

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        del tool_context
        runtime_block = trajectory.metadata.get("runtime") or {}
        rl_meta = runtime_block.get("rl") if isinstance(runtime_block, dict) else None
        if rl_meta is None:
            return RewardResult(name=self.name, score=0.0, reason="no rl metadata", weight=self.weight)

        prompt_ids = list(rl_meta.get("prompt_ids") or [])
        response_ids = list(rl_meta.get("response_ids") or [])
        if not response_ids:
            return RewardResult(name=self.name, score=0.0, reason="empty response", weight=self.weight)

        response_text = trajectory.final_output or ""
        tokenizer = getattr(self.prm.backend, "tokenizer", None)
        if tokenizer is None:
            return RewardResult(name=self.name, score=0.0, reason="no tokenizer", weight=self.weight)

        step_scores = self.prm.score_steps(
            prompt_ids, response_text, tokenizer, step_sep=self.step_sep
        )
        if not step_scores:
            return RewardResult(name=self.name, score=0.0, reason="no steps", weight=self.weight)

        if self.agg == "mean":
            score = sum(step_scores) / len(step_scores)
        elif self.agg == "min":
            score = min(step_scores)
        elif self.agg == "last":
            score = step_scores[-1]
        elif self.agg == "sum":
            score = sum(step_scores)
        else:
            score = sum(step_scores) / len(step_scores)

        # Write per-step scores as token_rewards for PPO dense signal
        if self.write_token_rewards and len(response_ids) > 0:
            n_steps = len(step_scores)
            chunk = max(1, len(response_ids) // n_steps)
            token_rewards = [0.0] * len(response_ids)
            for i, s in enumerate(step_scores):
                pos = min(i * chunk + chunk - 1, len(response_ids) - 1)
                token_rewards[pos] = s
            trajectory.metadata.setdefault("runtime", {})
            if isinstance(trajectory.metadata["runtime"], dict):
                trajectory.metadata["runtime"].setdefault("rl", {})
                trajectory.metadata["runtime"]["rl"]["token_rewards"] = token_rewards

        return RewardResult(
            name=self.name,
            score=float(score) * self.weight,
            reason=f"prm_{self.agg}={score:.4f} n_steps={len(step_scores)}",
            weight=self.weight,
        )
