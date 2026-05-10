"""Proximal Policy Optimization (token-level, with GAE + clipped value loss).

Design choices, kept deliberately small:
  - Advantages computed with token-level GAE(γ, λ). By default, the total
    rollout reward is placed on the terminal response token (sparse reward
    setup matches most agentic-RL tasks); callers who have dense per-token
    rewards can pass a `token_rewards` list via ``RolloutRecord.metadata``.
  - Policy loss: standard PPO clipped surrogate (shared with GRPO via
    ``algos.common.loss.clipped_surrogate_loss``).
  - Value loss: clipped MSE (``clipped_value_loss``), optional.
  - Entropy bonus: uses the policy's per-token entropy (via
    ``score_with_value``); gracefully falls back to ``-logπ`` proxy if the
    backend can't emit entropy.
  - KL-to-reference (optional): same β·KL(π_new ‖ π_ref) mechanism as GRPO.

The algorithm is a pure function: Trainer owns the optimizer, calls
``compute_loss``, and drives .backward() / .step().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
)
from hermes_agentic_rl.algos.common.gae import compute_gae, terminal_token_rewards
from hermes_agentic_rl.algos.common.kl import kl_from_logprobs
from hermes_agentic_rl.algos.common.loss import (
    clipped_surrogate_loss,
    clipped_value_loss,
)
from hermes_agentic_rl.backends.base import LLMBackend


@dataclass(slots=True)
class PPOConfig:
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    vf_clip_eps: float = 0.2
    entropy_coef: float = 0.0
    kl_coef: float = 0.0
    gamma: float = 1.0
    lam: float = 0.95
    normalize_advantage: bool = True
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token"
    max_len_for_dr_grpo: int = 256  # denominator when loss_agg == "dr_grpo"
    # v0.6: KL estimator (see algos.common.kl for semantics).
    kl_estimator: Literal["k1", "k2", "k3"] = "k1"
    # v0.7: advantage whitening — mirrors GRPO's `advantage_norm="whiten"`.
    # When True AND normalize_advantage is True, the batch-normalized
    # advantage tensor is additionally clipped to ±advantage_clip. Mitigates
    # exploding updates on sparse-reward / zero-variance batches.
    whiten_advantage: bool = False
    advantage_clip: float = 3.0


class PPO(BaseAlgo):
    """Token-level PPO with a value-head backend."""

    def __init__(self, cfg: PPOConfig | None = None) -> None:
        self.cfg = cfg or PPOConfig()

    def compute_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, AlgoUpdateStats]:
        if not policy.supports_value_head():
            raise RuntimeError(
                "PPO requires a backend with a value head "
                "(TinyBackendConfig(with_value_head=True))"
            )
        cfg = self.cfg

        # --- pass 1: forward each record, compute raw (un-normalized) advantages
        per_rec: list[dict[str, torch.Tensor | Any]] = []
        for rec in batch.records:
            R = len(rec.response_ids)
            if R == 0:
                continue
            new_logp, ent, values = policy.score_with_value(rec.prompt_ids, rec.response_ids)
            n = min(new_logp.numel(), R)
            if n == 0:
                continue
            new_logp = new_logp[-n:]
            ent = ent[-n:]
            values = values[-n:]
            old_logp = torch.tensor(
                rec.old_logprobs[-n:], dtype=new_logp.dtype, device=new_logp.device
            )
            token_rewards = rec.metadata.get("token_rewards")
            if not token_rewards:
                token_rewards = terminal_token_rewards(float(rec.reward), n)
            else:
                token_rewards = list(token_rewards)[-n:]
            values_old = values.detach()
            advs_raw, returns = compute_gae(
                token_rewards,
                values_old,
                gamma=cfg.gamma,
                lam=cfg.lam,
                last_value=0.0,
                normalize=False,  # we normalize globally after this pass
            )
            per_rec.append(
                {
                    "rec": rec,
                    "n": n,
                    "new_logp": new_logp,
                    "old_logp": old_logp,
                    "ent": ent,
                    "values": values,
                    "values_old": values_old,
                    "advs_raw": advs_raw,
                    "returns": returns,
                }
            )

        # --- pass 2: batch-level advantage normalization (+ optional whiten)
        if cfg.normalize_advantage and per_rec:
            all_advs = torch.cat([p["advs_raw"] for p in per_rec])
            if all_advs.numel() > 1:
                mean = all_advs.mean()
                std = all_advs.std(unbiased=False)
                if float(std.item()) > 1e-8:
                    for p in per_rec:
                        norm = (p["advs_raw"] - mean) / (std + 1e-8)
                        if cfg.whiten_advantage and cfg.advantage_clip > 0:
                            norm = norm.clamp(-cfg.advantage_clip, cfg.advantage_clip)
                        p["advs"] = norm
                else:
                    for p in per_rec:
                        p["advs"] = p["advs_raw"]
            else:
                for p in per_rec:
                    p["advs"] = p["advs_raw"]
        else:
            for p in per_rec:
                p["advs"] = p["advs_raw"]

        # --- pass 3: per-record loss computation
        total_losses: list[torch.Tensor] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        kls: list[float] = []
        entropies: list[float] = []
        clip_fracs: list[float] = []
        adv_means: list[float] = []
        rewards: list[float] = [r.reward for r in batch.records]
        zero = torch.zeros((), dtype=torch.float32)

        for p in per_rec:
            rec = p["rec"]
            n = p["n"]
            new_logp = p["new_logp"]
            old_logp = p["old_logp"]
            ent = p["ent"]
            values = p["values"]
            values_old = p["values_old"]
            advs = p["advs"]
            returns = p["returns"]

            pol_loss, loss_stats = clipped_surrogate_loss(
                new_logp, old_logp, advantage=advs, clip_eps=cfg.clip_eps,
                loss_agg=cfg.loss_agg,
            )
            if cfg.loss_agg == "dr_grpo":
                pol_loss = pol_loss / float(cfg.max_len_for_dr_grpo)

            vf_loss, _ = clipped_value_loss(
                values, values_old, returns, clip_eps=cfg.vf_clip_eps
            )

            total = pol_loss + cfg.vf_coef * vf_loss

            if cfg.entropy_coef > 0:
                ent_term = ent.mean()
                total = total - cfg.entropy_coef * ent_term
                entropies.append(float(ent_term.detach().item()))

            if ref_policy is not None and cfg.kl_coef > 0:
                with torch.no_grad():
                    ref_logp = ref_policy.score(rec.prompt_ids, rec.response_ids)[-n:]
                kl_tok = kl_from_logprobs(new_logp, ref_logp, estimator=cfg.kl_estimator)
                total = total + cfg.kl_coef * kl_tok
                kls.append(float(kl_tok.detach().item()))

            total_losses.append(total)
            policy_losses.append(float(pol_loss.detach().item()))
            value_losses.append(float(vf_loss.detach().item()))
            clip_fracs.append(loss_stats["clip_frac"])
            adv_means.append(float(advs.mean().item()))

        if not total_losses:
            loss = zero
        else:
            loss = torch.stack(total_losses).mean()

        stats = AlgoUpdateStats(
            loss=float(loss.detach().item()) if loss.requires_grad else float(loss.item()),
            policy_loss=(sum(policy_losses) / len(policy_losses)) if policy_losses else 0.0,
            kl=(sum(kls) / len(kls)) if kls else 0.0,
            entropy=(sum(entropies) / len(entropies)) if entropies else 0.0,
            mean_reward=(sum(rewards) / len(rewards)) if rewards else 0.0,
            mean_advantage=(sum(adv_means) / len(adv_means)) if adv_means else 0.0,
            clip_frac=(sum(clip_fracs) / len(clip_fracs)) if clip_fracs else 0.0,
            n_records=len(batch),
            extra={
                "n_updated": len(total_losses),
                "value_loss": (sum(value_losses) / len(value_losses)) if value_losses else 0.0,
                "algo": "ppo",
            },
        )
        return loss, stats
