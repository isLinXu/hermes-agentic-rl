"""GRPO: Group Relative Policy Optimization.

Reference: DeepSeek-Math / DeepSeek-R1. Differences from PPO:
  - No value/critic head. Advantage is computed per-group as
    `A_i = (R_i - mean_g(R)) / (std_g(R) + eps)`, where each "group" is a set
    of rollouts sharing the same prompt.
  - The rest (clipped surrogate + optional KL to reference) matches PPO.

v0.5 enhancements:
  - Per-token advantage (REINFORCE++): optional token-level credit assignment
    via ``per_token_advantage`` flag, which constructs a [T] advantage tensor
    instead of a scalar. Tokens after <answer> tag get concentrated reward.
  - Batch-normalized advantage: alternative to group normalization via
    ``advantage_norm="batch"``. Useful when groups are small or zero-variance.
  - Whitened advantage: ``advantage_norm="whiten"`` clips to [-3, 3].

This implementation is intentionally small (~160 lines) and side-effect-free:
it takes a policy, an optional reference policy, and a batch; returns a loss
tensor plus stats. The outer Trainer handles the optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
)
from hermes_agentic_rl.algos.common.advantage import group_normalize_advantage
from hermes_agentic_rl.algos.common.kl import kl_from_logprobs
from hermes_agentic_rl.algos.common.loss import clipped_surrogate_loss
from hermes_agentic_rl.algos.common.reinforce_pp import (
    batch_normalize_advantage,
    reinforce_plusplus_advantage,
    whitened_advantage,
)
from hermes_agentic_rl.backends.base import LLMBackend


@dataclass(slots=True)
class GRPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.02          # β: KL(π_new || π_ref); 0 disables ref
    entropy_coef: float = 0.0      # small positive value encourages exploration
    advantage_eps: float = 1e-6
    # Loss aggregation — default "mean_token" matches DeepSeek GRPO.
    #   "mean_token": per-rollout arithmetic mean, then mean over rollouts.
    #   "dr_grpo": sum-over-tokens / MAX_LEN, then mean over rollouts (Liu 2024).
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token"
    max_len_for_dr_grpo: int = 256
    # --- v0.5: per-token advantage ---
    # "group": original GRPO group-normalized scalar advantage.
    # "batch": z-score normalize across ALL rollouts (not per-group).
    # "whiten": batch normalize + clip to [-3, 3].
    advantage_norm: Literal["group", "batch", "whiten"] = "group"
    # If True, construct per-token advantage via REINFORCE++ (answer-tag
    # heuristic). Tokenizer must be provided to the compute_loss call.
    per_token_advantage: bool = False
    # For per_token_advantage: the token id that marks the start of the
    # answer section. If None, uniform per-token advantage is used.
    answer_start_token_id: int | None = None
    # Gamma decay for tokens before the answer section.
    reinforce_gamma: float = 0.95
    # v0.6: KL estimator. "k1" = mean(logπ_new - logπ_ref) [legacy, unbiased but signed];
    # "k2" = mean(0.5 * r**2); "k3" = mean(exp(-r) - 1 + r) [unbiased, non-neg, recommended].
    kl_estimator: Literal["k1", "k2", "k3"] = "k1"


class GRPO(BaseAlgo):
    def __init__(self, cfg: GRPOConfig | None = None) -> None:
        self.cfg = cfg or GRPOConfig()

    def compute_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, AlgoUpdateStats]:
        cfg = self.cfg

        # 1) Advantage normalization
        all_records = batch.records

        if cfg.advantage_norm == "group":
            # Original GRPO: group-normalize within each prompt group.
            records_with_adv: list[tuple[RolloutRecord, list[float]]] = []
            for gid, recs in batch.by_group().items():
                rewards = [r.reward for r in recs]
                scalar_advs = group_normalize_advantage(rewards, eps=cfg.advantage_eps)
                for rec, adv in zip(recs, scalar_advs, strict=False):
                    records_with_adv.append((rec, [adv]))
        elif cfg.advantage_norm == "batch":
            rewards = [r.reward for r in all_records]
            scalar_advs = batch_normalize_advantage(rewards, eps=cfg.advantage_eps)
            records_with_adv = [(rec, [a]) for rec, a in zip(all_records, scalar_advs, strict=False)]
        elif cfg.advantage_norm == "whiten":
            rewards = [r.reward for r in all_records]
            scalar_advs = whitened_advantage(rewards, eps=cfg.advantage_eps)
            records_with_adv = [(rec, [a]) for rec, a in zip(all_records, scalar_advs, strict=False)]
        else:
            raise ValueError(f"Unknown advantage_norm: {cfg.advantage_norm}")

        # 2) Per-token advantage expansion (REINFORCE++)
        if cfg.per_token_advantage:
            records_with_adv = self._expand_per_token(records_with_adv)

        losses: list[torch.Tensor] = []
        policy_losses: list[float] = []
        kls: list[float] = []
        entropies: list[float] = []
        clip_fracs: list[float] = []
        advs: list[float] = []
        rewards: list[float] = []
        zero = torch.zeros((), dtype=torch.float32)

        for rec, adv_list in records_with_adv:
            rewards.append(rec.reward)
            if len(rec.response_ids) == 0:
                continue

            new_logp = policy.score(rec.prompt_ids, rec.response_ids)
            old_logp = torch.tensor(rec.old_logprobs, dtype=new_logp.dtype, device=new_logp.device)
            # safety: lengths might differ by 1 if truncation happened; align by min.
            n = min(new_logp.numel(), old_logp.numel())
            if n == 0:
                continue
            new_logp = new_logp[-n:]
            old_logp = old_logp[-n:]

            # Build advantage tensor
            if len(adv_list) == 1:
                # Scalar advantage (original GRPO path)
                adv = adv_list[0]
                advs.append(adv)
                if adv == 0.0:
                    continue
            else:
                # Per-token advantage [T]
                adv_tensor = torch.tensor(
                    adv_list[-n:], dtype=new_logp.dtype, device=new_logp.device
                )
                if adv_tensor.numel() == 0:
                    continue
                advs.append(float(adv_tensor.mean().item()))
                if adv_tensor.abs().sum() < 1e-8:
                    continue
                adv = adv_tensor

            pol_loss, loss_stats = clipped_surrogate_loss(
                new_logp,
                old_logp,
                advantage=adv,
                clip_eps=cfg.clip_eps,
                loss_agg=cfg.loss_agg,
            )
            if cfg.loss_agg == "dr_grpo":
                pol_loss = pol_loss / float(cfg.max_len_for_dr_grpo)

            total = pol_loss

            # KL(π_new || π_ref) via configured estimator (default k1 for v0.5
            # backward compatibility; k3 recommended for new configs).
            if ref_policy is not None and cfg.kl_coef > 0:
                with torch.no_grad():
                    ref_logp = ref_policy.score(rec.prompt_ids, rec.response_ids)[-n:]
                kl_tok = kl_from_logprobs(new_logp, ref_logp, estimator=cfg.kl_estimator)
                total = total + cfg.kl_coef * kl_tok
                kls.append(float(kl_tok.detach().item()))

            if cfg.entropy_coef > 0:
                ent = -new_logp.mean()
                total = total - cfg.entropy_coef * ent
                entropies.append(float(ent.detach().item()))

            losses.append(total)
            policy_losses.append(float(pol_loss.detach().item()))
            clip_fracs.append(loss_stats["clip_frac"])

        if not losses:
            loss = zero
        else:
            loss = torch.stack(losses).mean()

        stats = AlgoUpdateStats(
            loss=float(loss.detach().item()) if loss.requires_grad else float(loss.item()),
            policy_loss=(sum(policy_losses) / len(policy_losses)) if policy_losses else 0.0,
            kl=(sum(kls) / len(kls)) if kls else 0.0,
            entropy=(sum(entropies) / len(entropies)) if entropies else 0.0,
            mean_reward=(sum(rewards) / len(rewards)) if rewards else 0.0,
            mean_advantage=(sum(advs) / len(advs)) if advs else 0.0,
            clip_frac=(sum(clip_fracs) / len(clip_fracs)) if clip_fracs else 0.0,
            n_records=len(batch),
            extra={"n_updated": len(losses)},
        )
        return loss, stats

    def _expand_per_token(
        self,
        records_with_adv: list[tuple[RolloutRecord, list[float]]],
    ) -> list[tuple[RolloutRecord, list[float]]]:
        """Expand scalar advantages into per-token advantages via REINFORCE++."""
        cfg = self.cfg
        expanded: list[tuple[RolloutRecord, list[float]]] = []
        for rec, adv_list in records_with_adv:
            if len(rec.response_ids) == 0 or len(adv_list) == 0:
                expanded.append((rec, adv_list))
                continue
            reward = adv_list[0]
            per_tok = reinforce_plusplus_advantage(
                response_ids=rec.response_ids,
                old_logprobs=rec.old_logprobs,
                reward=reward,
                answer_start_id=cfg.answer_start_token_id,
                gamma=cfg.reinforce_gamma,
                eps=cfg.advantage_eps,
            )
            expanded.append((rec, per_tok))
        return expanded
