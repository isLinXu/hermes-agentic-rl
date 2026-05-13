"""Proximal Policy Optimization (token-level, batched).

Design choices:
  - Advantages: batched GAE(γ, λ) via ``compute_gae_batched``. By default the
    total rollout reward is placed on the terminal response token (sparse
    reward matches most agentic-RL tasks); callers with dense per-token
    rewards can pass ``token_rewards`` via ``RolloutRecord.metadata``.
  - Policy loss: PPO clipped surrogate (shared with GRPO).
  - Value loss: clipped MSE.
  - Entropy bonus: per-token entropy from ``score_with_value_batch``.
  - KL-to-reference (optional): same β·KL(π_new ‖ π_ref) as GRPO.

v0.8 improvements:
  - Single batched forward for the whole batch.
  - Vectorized GAE across [B, T_max].
  - ``_ppo_old_values`` frozen at rollout time (via ``PPOTrainer.
    _prepare_update_batch``), so ``value_clip`` actually takes effect on
    the first update epoch. Falls back to ``values.detach()`` only when
    the metadata is absent (legacy path).
  - ``approx_kl`` reported, used by Trainer for ratio-based early stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
)
from hermes_agentic_rl.algos.common.gae import compute_gae_batched
from hermes_agentic_rl.algos.common.loss import (
    clipped_surrogate_loss_batched,
    clipped_value_loss_batched,
)
from hermes_agentic_rl.algos.common.temperature import rollout_score_temperature
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
    max_len_for_dr_grpo: int = 256
    kl_estimator: Literal["k1", "k2", "k3"] = "k1"
    whiten_advantage: bool = False
    advantage_clip: float = 3.0


class PPO(BaseAlgo):
    """Token-level PPO with a value-head backend (batched)."""

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
        records = batch.records
        if not records:
            zero = torch.zeros((), dtype=torch.float32)
            return zero, AlgoUpdateStats(
                loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0,
                mean_reward=0.0, mean_advantage=0.0, clip_frac=0.0,
                n_records=0,
                extra={"algo": "ppo", "n_updated": 0, "value_loss": 0.0,
                       "value_clip_frac": 0.0, "approx_kl": 0.0},
            )

        prompt_ids_list = [r.prompt_ids for r in records]
        response_ids_list = [r.response_ids for r in records]
        score_temperature = rollout_score_temperature(records)

        # 1) Single batched forward.
        new_logp, ent, values_new, mask = policy.score_with_value_batch(
            prompt_ids_list,
            response_ids_list,
            temperature=score_temperature,
        )  # all [B, T_max]
        B, T_max = new_logp.shape
        if B == 0 or T_max == 0:
            zero = new_logp.new_zeros(())
            return zero, AlgoUpdateStats(
                loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0,
                mean_reward=float(sum(r.reward for r in records) / max(1, len(records))),
                mean_advantage=0.0, clip_frac=0.0,
                n_records=len(records),
                extra={"algo": "ppo", "n_updated": 0, "value_loss": 0.0,
                       "value_clip_frac": 0.0, "approx_kl": 0.0},
            )

        device = new_logp.device
        dtype = new_logp.dtype

        # 2) Stack old_logprobs, old_values, token_rewards into [B, T_max].
        old_logp = torch.zeros(B, T_max, dtype=dtype, device=device)
        old_values = torch.zeros(B, T_max, dtype=dtype, device=device)
        token_rewards = torch.zeros(B, T_max, dtype=dtype, device=device)
        old_values_present = False

        for i, rec in enumerate(records):
            R_i = int(mask[i].sum().item())
            if R_i == 0:
                continue
            olp = rec.old_logprobs[-R_i:] if rec.old_logprobs else []
            if olp:
                old_logp[i, :len(olp)] = torch.tensor(olp, dtype=dtype, device=device)

            # old_values: prefer metadata-frozen snapshot (set by
            # PPOTrainer._prepare_update_batch at rollout time).
            ov_raw = rec.metadata.get("_ppo_old_values")
            if isinstance(ov_raw, list) and ov_raw:
                ov = ov_raw[-R_i:]
                old_values[i, :len(ov)] = torch.tensor(ov, dtype=dtype, device=device)
                old_values_present = True

            tr = rec.metadata.get("token_rewards")
            if isinstance(tr, list) and tr:
                trs = list(tr)[-R_i:]
                token_rewards[i, :len(trs)] = torch.tensor(trs, dtype=dtype, device=device)
            else:
                # sparse terminal reward at position R_i-1
                token_rewards[i, R_i - 1] = float(rec.reward)

        # Fallback: if no _ppo_old_values were captured, use detached V_new.
        # This means the value clip is a no-op on the first update epoch,
        # which is the v0.7 behavior.
        if not old_values_present:
            old_values = values_new.detach().clone()

        # 3) Batched GAE.
        advs_raw, returns = compute_gae_batched(
            token_rewards, old_values, mask,
            gamma=cfg.gamma, lam=cfg.lam, normalize=False,
        )

        # 4) Advantage normalization + optional whitening.
        if cfg.normalize_advantage and int(mask.sum().item()) > 1:
            valid = mask.reshape(-1)
            flat = advs_raw.reshape(-1)
            sel = flat[valid]
            mean = sel.mean()
            std = sel.std(unbiased=False)
            if float(std.item()) > 1e-8:
                flat = flat.clone()
                flat[valid] = (sel - mean) / (std + 1e-8)
                if cfg.whiten_advantage and cfg.advantage_clip > 0:
                    flat[valid] = flat[valid].clamp(
                        -cfg.advantage_clip, cfg.advantage_clip
                    )
                advs = flat.view(B, T_max)
            else:
                advs = advs_raw
        else:
            advs = advs_raw

        # 5) Batched policy + value losses.
        pol_loss, pol_stats = clipped_surrogate_loss_batched(
            new_logp, old_logp, advs, mask,
            clip_eps=cfg.clip_eps,
            loss_agg=cfg.loss_agg,
            max_len_for_dr_grpo=cfg.max_len_for_dr_grpo,
        )
        vf_loss, vf_stats = clipped_value_loss_batched(
            values_new, old_values, returns, mask,
            clip_eps=cfg.vf_clip_eps,
        )

        total = pol_loss + cfg.vf_coef * vf_loss

        # 6) Entropy bonus.
        ent_val = 0.0
        if cfg.entropy_coef > 0:
            mf = mask.to(dtype)
            ent_per_row = (ent * mf).sum(dim=-1) / mf.sum(dim=-1).clamp(min=1)
            ent_scalar = ent_per_row.mean()
            total = total - cfg.entropy_coef * ent_scalar
            ent_val = float(ent_scalar.detach().item())

        # 7) KL-to-reference.
        kl_val = 0.0
        if ref_policy is not None and cfg.kl_coef > 0:
            with torch.no_grad():
                ref_logp, ref_mask = ref_policy.score_batch(
                    prompt_ids_list,
                    response_ids_list,
                    temperature=score_temperature,
                )
            common_T = min(new_logp.shape[1], ref_logp.shape[1])
            r_kl = (new_logp[:, :common_T] - ref_logp[:, :common_T]) * mask[:, :common_T].to(dtype)
            if cfg.kl_estimator == "k1":
                kl_per_tok = r_kl
            elif cfg.kl_estimator == "k2":
                kl_per_tok = 0.5 * r_kl.pow(2)
            else:
                r_c = r_kl.clamp(min=-20.0, max=20.0)
                kl_per_tok = torch.exp(-r_c) - 1.0 + r_c
            mf2 = mask[:, :common_T].to(dtype)
            kl_per_row = (kl_per_tok * mf2).sum(dim=-1) / mf2.sum(dim=-1).clamp(min=1)
            kl_scalar = kl_per_row.mean()
            total = total + cfg.kl_coef * kl_scalar
            kl_val = float(kl_scalar.detach().item())

        mean_r = sum(r.reward for r in records) / max(1, len(records))
        mean_a = float((advs * mask.to(dtype)).sum().item()) / max(
            1, int(mask.sum().item())
        )

        stats = AlgoUpdateStats(
            loss=float(total.detach().item()) if total.requires_grad else float(total.item()),
            policy_loss=float(pol_loss.detach().item()),
            kl=kl_val,
            entropy=ent_val,
            mean_reward=float(mean_r),
            mean_advantage=mean_a,
            clip_frac=pol_stats["clip_frac"],
            n_records=len(records),
            extra={
                "algo": "ppo",
                "n_updated": len(records),
                "value_loss": float(vf_loss.detach().item()),
                "value_clip_frac": vf_stats["value_clip_frac"],
                "approx_kl": pol_stats["approx_kl"],
                "ratio_mean": pol_stats["ratio_mean"],
                "n_tokens": pol_stats["n_tokens"],
                "score_temperature": float(score_temperature),
            },
        )
        return total, stats
