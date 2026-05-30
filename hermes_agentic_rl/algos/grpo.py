"""GRPO: Group Relative Policy Optimization.

Reference: DeepSeek-Math / DeepSeek-R1. Differences from PPO:
  - No value/critic head. Advantage is computed per-group as
    ``A_i = (R_i - mean_g(R)) / (std_g(R) + eps)``, where each "group" is a
    set of rollouts sharing the same prompt.
  - The rest (clipped surrogate + optional KL to reference) matches PPO.

v0.5 enhancements:
  - Per-token advantage (REINFORCE++): optional token-level credit assignment
    via ``per_token_advantage`` flag, which constructs a [T] advantage tensor
    instead of a scalar. Tokens after ``<answer>`` tag get concentrated reward.
  - Batch-normalized / whitened advantage.

v0.8 enhancements (PERFORMANCE):
  - Single batched forward for the whole batch via ``policy.score_batch``.
    On HF GPT-2 this is a ~5-10× speedup over the per-record loop.
  - Vectorized clipped surrogate + mask-aware KL.
  - ``approx_kl`` reported per-batch, used by the Trainer for ratio-based
    early stopping across ``update_epochs``.

This implementation is side-effect-free: takes a policy, an optional
reference policy, and a batch; returns a loss tensor plus stats. The outer
Trainer handles the optimizer step.
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
from hermes_agentic_rl.algos.common.loss import (
    clipped_surrogate_loss_batched,
)
from hermes_agentic_rl.algos.common.reinforce_pp import (
    batch_normalize_advantage,
    reinforce_plusplus_advantage,
    whitened_advantage,
)
from hermes_agentic_rl.algos.common.temperature import rollout_score_temperature
from hermes_agentic_rl.backends.base import LLMBackend


@dataclass(slots=True)
class GRPOConfig:
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28
    kl_coef: float = 0.02
    entropy_coef: float = 0.0
    advantage_eps: float = 1e-6
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token"
    max_len_for_dr_grpo: int = 256
    advantage_norm: Literal["group", "batch", "whiten"] = "group"
    per_token_advantage: bool = False
    answer_start_token_id: int | None = None
    reinforce_gamma: float = 0.95
    # k3 is the low-variance, always-nonnegative PPO/GRPO community standard
    # (TRL/DeepSeek/verl). Kept consistent with RLOO/OPD/GSPO so that the KL
    # signal is comparable across branches in HybridAlgo and feeds a single,
    # coherent AdaptiveKLController.
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"


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

        # 1) Build scalar or per-token advantage for each record.
        all_records = batch.records
        group_norm_batch_fallback = 0.0
        if not all_records:
            zero = torch.zeros((), dtype=torch.float32)
            return zero, AlgoUpdateStats(
                loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0,
                mean_reward=0.0, mean_advantage=0.0, clip_frac=0.0,
                n_records=0, extra={"algo": "grpo", "n_updated": 0, "approx_kl": 0.0},
            )

        if cfg.advantage_norm == "group":
            records_with_adv: list[tuple[RolloutRecord, list[float]]] = []
            grouped = batch.by_group()
            all_singleton_groups = grouped and all(
                len(recs) == 1 for recs in grouped.values()
            )
            if all_singleton_groups and len(all_records) > 1:
                # Every prompt produced a single rollout — group-relative
                # advantage is identically zero. Fall back to batch norm so
                # cross-prompt variance still yields a usable signal.
                rewards_all = [r.reward for r in all_records]
                scalar_advs = batch_normalize_advantage(
                    rewards_all, eps=cfg.advantage_eps
                )
                records_with_adv = [
                    (rec, [a])
                    for rec, a in zip(all_records, scalar_advs, strict=False)
                ]
                group_norm_batch_fallback = 1.0
            else:
                for _gid, recs in grouped.items():
                    rewards_g = [r.reward for r in recs]
                    scalar_advs = group_normalize_advantage(
                        rewards_g, eps=cfg.advantage_eps
                    )
                    for rec, adv in zip(recs, scalar_advs, strict=False):
                        records_with_adv.append((rec, [adv]))
                # Degenerate config guard: every group has a single rollout, so the
                # group-relative advantage is identically zero and no gradient
                # signal flows. This silently produces a no-op update; warn loudly
                # so users bump ``group_size`` instead of staring at a flat reward.
                if all_singleton_groups:
                    import warnings as _warnings

                    _warnings.warn(
                        "GRPO group-norm advantage is zero for every record: each "
                        f"prompt produced a single rollout (n_groups={len(grouped)}, "
                        "group_size=1). Increase group_size (>= 2) or switch "
                        "advantage_norm to 'batch'/'whiten' to get a usable signal.",
                        stacklevel=2,
                    )
            # Preserve original order
            order = {id(r): i for i, r in enumerate(all_records)}
            records_with_adv.sort(key=lambda p: order[id(p[0])])
        elif cfg.advantage_norm == "batch":
            rewards_all = [r.reward for r in all_records]
            scalar_advs = batch_normalize_advantage(rewards_all, eps=cfg.advantage_eps)
            records_with_adv = [
                (rec, [a]) for rec, a in zip(all_records, scalar_advs, strict=False)
            ]
        elif cfg.advantage_norm == "whiten":
            rewards_all = [r.reward for r in all_records]
            scalar_advs = whitened_advantage(rewards_all, eps=cfg.advantage_eps)
            records_with_adv = [
                (rec, [a]) for rec, a in zip(all_records, scalar_advs, strict=False)
            ]
        else:
            raise ValueError(f"Unknown advantage_norm: {cfg.advantage_norm}")

        # 2) Per-token advantage expansion (REINFORCE++).
        if cfg.per_token_advantage:
            records_with_adv = self._expand_per_token(records_with_adv)

        # 3) Batched forward (PERFORMANCE CRITICAL).
        prompt_ids_list = [rec.prompt_ids for rec, _ in records_with_adv]
        response_ids_list = [rec.response_ids for rec, _ in records_with_adv]
        score_temperature = rollout_score_temperature([rec for rec, _ in records_with_adv])
        new_logp, mask = policy.score_batch(
            prompt_ids_list,
            response_ids_list,
            temperature=score_temperature,
        )  # [B, T_max]
        B, T_max = new_logp.shape

        if B == 0 or T_max == 0:
            zero = new_logp.new_zeros(())
            return zero, AlgoUpdateStats(
                loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0,
                mean_reward=float(sum(r.reward for r in all_records) / max(1, len(all_records))),
                mean_advantage=0.0, clip_frac=0.0,
                n_records=len(all_records),
                extra={"algo": "grpo", "n_updated": 0, "approx_kl": 0.0},
            )

        device = new_logp.device
        dtype = new_logp.dtype

        # 4) Stack old_logprobs → [B, T_max] (right-aligned within R_i, 0-padded).
        #    We mirror the backend's "last R_i positions of the response"
        #    convention: score_batch puts each record's response in [0, R_i).
        old_logp = torch.zeros(B, T_max, dtype=dtype, device=device)
        for i, (rec, _) in enumerate(records_with_adv):
            olp = rec.old_logprobs
            R_i = min(len(olp), int(mask[i].sum().item()))
            if R_i > 0:
                old_logp[i, :R_i] = torch.tensor(
                    olp[-R_i:], dtype=dtype, device=device
                )

        # 5) Build advantage tensor [B, T_max].
        adv_tensor = torch.zeros(B, T_max, dtype=dtype, device=device)
        has_per_token = any(len(a) > 1 for _, a in records_with_adv)
        if has_per_token:
            for i, (rec, adv_list) in enumerate(records_with_adv):
                R_i = int(mask[i].sum().item())
                if R_i == 0 or not adv_list:
                    continue
                vals = adv_list[-R_i:] if len(adv_list) >= R_i else adv_list
                adv_tensor[i, :len(vals)] = torch.tensor(vals, dtype=dtype, device=device)
        else:
            # scalar advantage broadcast across response tokens
            scalars = torch.tensor(
                [float(a[0]) if a else 0.0 for _, a in records_with_adv],
                dtype=dtype, device=device,
            )  # [B]
            adv_tensor = scalars.unsqueeze(-1) * mask.to(dtype)  # [B, T_max]

        # 6) Batched clipped surrogate.
        pol_loss, loss_stats = clipped_surrogate_loss_batched(
            new_logp, old_logp, adv_tensor, mask,
            clip_eps=cfg.clip_eps,
            clip_eps_high=cfg.clip_eps_high,
            loss_agg=cfg.loss_agg,
            max_len_for_dr_grpo=cfg.max_len_for_dr_grpo,
        )
        total = pol_loss
        kl_val = 0.0

        # 7) KL-to-reference (batched).
        if ref_policy is not None and cfg.kl_coef > 0:
            with torch.no_grad():
                ref_logp, ref_mask = ref_policy.score_batch(
                    prompt_ids_list,
                    response_ids_list,
                    temperature=score_temperature,
                )
            # Align widths
            common_T = min(new_logp.shape[1], ref_logp.shape[1])
            r = (new_logp[:, :common_T] - ref_logp[:, :common_T]) * mask[:, :common_T].to(dtype)
            # estimator selection
            if cfg.kl_estimator == "k1":
                kl_per_tok = r
            elif cfg.kl_estimator == "k2":
                kl_per_tok = 0.5 * r.pow(2)
            else:  # k3
                r_c = r.clamp(min=-20.0, max=20.0)
                kl_per_tok = torch.exp(-r_c) - 1.0 + r_c
            mf = mask[:, :common_T].to(dtype)
            tokens_per_row = mf.sum(dim=-1).clamp(min=1)
            kl_per_row = (kl_per_tok * mf).sum(dim=-1) / tokens_per_row
            kl_scalar = kl_per_row.mean()
            total = total + cfg.kl_coef * kl_scalar
            kl_val = float(kl_scalar.detach().item())

        # 8) Entropy bonus (-logπ proxy).
        ent_val = 0.0
        if cfg.entropy_coef > 0:
            mf = mask.to(dtype)
            ent_per_row = -(new_logp * mf).sum(dim=-1) / mf.sum(dim=-1).clamp(min=1)
            ent_scalar = ent_per_row.mean()
            total = total - cfg.entropy_coef * ent_scalar
            ent_val = float(ent_scalar.detach().item())

        mean_r = sum(r.reward for r in all_records) / max(1, len(all_records))
        mean_a = float((adv_tensor * mask.to(dtype)).sum().item()) / max(
            1, int(mask.sum().item())
        )

        stats = AlgoUpdateStats(
            loss=float(total.detach().item()) if total.requires_grad else float(total.item()),
            policy_loss=float(pol_loss.detach().item()),
            kl=kl_val,
            entropy=ent_val,
            mean_reward=float(mean_r),
            mean_advantage=mean_a,
            clip_frac=loss_stats["clip_frac"],
            n_records=len(all_records),
            extra={
                "algo": "grpo",
                "n_updated": len(all_records),
                "approx_kl": loss_stats["approx_kl"],
                "ratio_mean": loss_stats["ratio_mean"],
                "n_tokens": loss_stats["n_tokens"],
                "score_temperature": float(score_temperature),
                "group_norm_batch_fallback": float(group_norm_batch_fallback),
            },
        )
        return total, stats

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
