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
from hermes_agentic_rl.algos.common.advantage import (
    dapo_group_advantage,
    group_normalize_advantage,
)
from hermes_agentic_rl.algos.common.batch_prepare import (
    build_advantage_tensor,
    compute_entropy_bonus,
    compute_kl_penalty,
    mean_advantage_from_tensors,
    mean_reward_from_records,
    stack_old_logprobs,
)
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
    advantage_norm: Literal["group", "batch", "whiten", "dapo"] = "group"
    per_token_advantage: bool = False
    answer_start_token_id: int | None = None
    reinforce_gamma: float = 0.95
    # k3 is the low-variance, always-nonnegative PPO/GRPO community standard
    # (TRL/DeepSeek/verl). Kept consistent with RLOO/OPD/GSPO so that the KL
    # signal is comparable across branches in HybridAlgo and feeds a single,
    # coherent AdaptiveKLController.
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"
    # --- Off-policy correction (async / stale-rollout training) ---
    # When > 0, apply Truncated Importance Sampling to the advantage so stale
    # rollouts (sampled under an older policy version) are bias-corrected. 0.0
    # disables it, making the synchronous/BSP path a strict no-op. The IS weight
    # is min(exp(logπ_new − logπ_behavior), tis_rho_clip), evaluated per token.
    tis_rho_clip: float = 0.0


class GRPO(BaseAlgo):
    """Group Relative Policy Optimization (DeepSeek-Math / DeepSeek-R1).

    Key differences from PPO:
      - No value/critic head. Advantage is computed per-group.
      - Supports batch-normalized or whitened advantage.
      - Optional per-token advantage via REINFORCE++.
      - Configurable off-policy correction (TIS/V-trace) via ``tis`` config.

    Usage::

        from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig

        cfg = GRPOConfig(clip_eps=0.2, kl_coef=0.02, entropy_coef=0.01)
        grpo = GRPO(cfg)
        loss, stats = grpo.compute_loss(policy, ref_policy, batch)
    """

    def __init__(self, cfg: GRPOConfig | None = None) -> None:
        self.cfg = cfg or GRPOConfig()

    def compute_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, AlgoUpdateStats]:
        """Compute the GRPO loss and return stats.

        Args:
            policy: The current policy (LLM backend).
            ref_policy: Optional reference policy for KL regularization. If
                None, no KL penalty is applied.
            batch: A batch of rollout records.

        Returns:
            A tuple of (loss_tensor, stats).
        """
        cfg = self.cfg

        # 1) Scalar or per-token advantage for each record.
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
        elif cfg.advantage_norm == "dapo":
            # DAPO: z-score when informative, discard group when all rewards equal.
            records_with_adv = []
            grouped = batch.by_group()
            for _gid, recs in grouped.items():
                rewards_g = [r.reward for r in recs]
                advs = dapo_group_advantage(rewards_g, eps=cfg.advantage_eps)
                if advs is None:
                    # Group has zero variance → skip entirely (DAPO filter).
                    continue
                for rec, adv in zip(recs, advs, strict=False):
                    records_with_adv.append((rec, [adv]))
            # Preserve original order
            order = {id(r): i for i, r in enumerate(all_records)}
            records_with_adv.sort(key=lambda p: order[id(p[0])])
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
                mean_reward=0.0, mean_advantage=0.0, clip_frac=0.0,
                n_records=0, extra={"algo": "grpo", "n_updated": 0, "approx_kl": 0.0},
            )

        device = new_logp.device
        dtype = new_logp.dtype

        # 4) Stack old_logprobs → [B, T_max] (right-aligned within R_i, 0-padded).
        old_logp = stack_old_logprobs(records_with_adv, B, T_max, dtype, device, mask)

        # 5) Build advantage tensor [B, T_max].
        adv_tensor = build_advantage_tensor(records_with_adv, B, T_max, dtype, device, mask)

        # 5b) Off-policy correction (TIS) for stale rollouts. No-op when
        #     tis_rho_clip <= 0 (synchronous/BSP). Multiplies the advantage by
        #     a per-token clipped importance weight min(π_new/π_behavior, c̄).
        tis_stats: dict[str, float] = {}
        if cfg.tis_rho_clip > 0:
            from hermes_agentic_rl.algos.common.vtrace import (
                TISConfig,
                tis_corrected_advantage,
            )

            adv_tensor, tis_stats = tis_corrected_advantage(
                adv_tensor,
                new_logp,
                old_logp,
                mask,
                TISConfig(rho_clip=cfg.tis_rho_clip, enabled=True),
            )

        # 6) Batched clipped surrogate.
        pol_loss, loss_stats = clipped_surrogate_loss_batched(
            new_logp, old_logp, adv_tensor, mask,
            clip_eps=cfg.clip_eps,
            clip_eps_high=cfg.clip_eps_high,
            loss_agg=cfg.loss_agg,
            max_len_for_dr_grpo=cfg.max_len_for_dr_grpo,
            kl_estimator=cfg.kl_estimator,
        )
        total = pol_loss
        kl_val = 0.0

        # 7) KL-to-reference (batched). Uses shared compute_kl_penalty
        #    so the estimator path is consistent with RLOO/OPD/PPO.
        kl_result = compute_kl_penalty(
            new_logp, mask,
            prompt_ids_list, response_ids_list,
            score_temperature, ref_policy, cfg.kl_coef,
            kl_estimator=cfg.kl_estimator,
        )
        if kl_result is not None:
            total = total + cfg.kl_coef * kl_result.kl_scalar
            kl_val = kl_result.kl_val

        # 8) Entropy bonus (-logπ proxy).
        ent_val, ent_scalar = compute_entropy_bonus(new_logp, mask, cfg.entropy_coef)
        if ent_scalar is not None:
            total = total - cfg.entropy_coef * ent_scalar

        mean_r = mean_reward_from_records(all_records)
        mean_a = mean_advantage_from_tensors(adv_tensor, mask)

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
                **tis_stats,
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
