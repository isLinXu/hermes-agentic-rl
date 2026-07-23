"""Factored Action Space loss for Hermes-style multi-head policies.

Hermes actions decompose into 5 independent heads:
    (action_type, tool_id, params, mem_op, mem_slot)

Each head has its own vocabulary, logprobs, and advantage weighting.
The total policy loss is a weighted sum across heads:

    L_policy = Σ_h  weight_h · L_clipped(π_h_new, π_h_old, A)

Advantages are shared across heads (same rollout reward → same advantage
estimate), but each head's gradient is scaled by its weight to reflect
its relative importance for the task.

Typical weight scheme for Hermes:
  action_type : 0.30  (choose tool-call vs direct-answer)
  tool_id     : 0.30  (which tool to call)
  params      : 0.25  (tool argument tokens)
  mem_op      : 0.10  (read/write/noop memory)
  mem_slot    : 0.05  (which memory slot)

Usage in GRPO trainer:

    algo = FactoredGRPO(
        grpo_cfg=GRPOConfig(),
        factored_cfg=FactoredConfig(head_weights={"action_type": 0.3, ...}),
    )
    loss, stats = algo.compute_loss(policy, ref_policy, batch)

The batch records must carry per-head logprobs in
``RolloutRecord.metadata["factored"]``:
    {
      "action_type": {"response_ids": [...], "old_logprobs": [...]},
      "tool_id":     {"response_ids": [...], "old_logprobs": [...]},
      ...
    }

When ``factored`` metadata is absent the loss falls back to the standard
flat GRPO on the full response (backward-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
)
from hermes_agentic_rl.algos.common.advantage import group_normalize_advantage
from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched
from hermes_agentic_rl.algos.common.loss import clipped_surrogate_loss_batched
from hermes_agentic_rl.algos.common.reinforce_pp import (
    batch_normalize_advantage,
    whitened_advantage,
)
from hermes_agentic_rl.algos.grpo import GRPOConfig
from hermes_agentic_rl.backends.base import LLMBackend

# Default head names and their relative weights.
DEFAULT_HEADS: list[str] = ["action_type", "tool_id", "params", "mem_op", "mem_slot"]
DEFAULT_WEIGHTS: dict[str, float] = {
    "action_type": 0.30,
    "tool_id": 0.30,
    "params": 0.25,
    "mem_op": 0.10,
    "mem_slot": 0.05,
}


@dataclass(slots=True)
class FactoredConfig:
    """Configuration for the Factored Action Space loss.

    head_weights: mapping from head name to loss weight. Weights are
        automatically normalized to sum to 1.0.
    heads: ordered list of head names. Must be consistent with how the
        runtime populates RolloutRecord.metadata["factored"].
    fallback_to_flat: when True (default), records without "factored"
        metadata are processed with the standard flat GRPO loss. When
        False, such records are skipped.
    """

    heads: list[str] = field(default_factory=lambda: list(DEFAULT_HEADS))
    head_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    fallback_to_flat: bool = True
    # KL estimator for the per-head reference penalty. Defaults to "k3" to
    # stay consistent with GRPO/RLOO/OPD/GSPO so that, under a Hybrid config,
    # the KL signal reaching the AdaptiveKLController is comparable across
    # branches. (Previously hard-coded to k2.)
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


class FactoredGRPO(BaseAlgo):
    """GRPO with a Factored Action Space.

    Computes one clipped-surrogate loss per head, weighted and summed.
    Falls back to standard flat GRPO for records without per-head metadata.
    """

    def __init__(
        self,
        grpo_cfg: GRPOConfig | None = None,
        factored_cfg: FactoredConfig | None = None,
    ) -> None:
        self.grpo_cfg = grpo_cfg or GRPOConfig()
        self.factored_cfg = factored_cfg or FactoredConfig()
        self._norm_weights = _normalize_weights(self.factored_cfg.head_weights)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, AlgoUpdateStats]:
        cfg = self.grpo_cfg
        all_records = batch.records
        if not all_records:
            zero = torch.zeros((), dtype=torch.float32)
            return zero, AlgoUpdateStats(
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                entropy=0.0,
                mean_reward=0.0,
                mean_advantage=0.0,
                clip_frac=0.0,
                n_records=0,
                extra={"algo": "factored_grpo", "n_updated": 0},
            )

        # 1) Compute advantages (shared across all heads).
        if cfg.advantage_norm == "group":
            adv_map: dict[int, float] = {}
            for _gid, recs in batch.by_group().items():
                rewards_g = [r.reward for r in recs]
                scalar_advs = group_normalize_advantage(rewards_g, eps=cfg.advantage_eps)
                for rec, adv in zip(recs, scalar_advs, strict=False):
                    adv_map[id(rec)] = adv
        elif cfg.advantage_norm == "batch":
            all_rewards = [r.reward for r in all_records]
            all_advs = batch_normalize_advantage(all_rewards, eps=cfg.advantage_eps)
            adv_map = {id(rec): adv for rec, adv in zip(all_records, all_advs, strict=False)}
        else:  # whiten
            all_rewards = [r.reward for r in all_records]
            all_advs = whitened_advantage(all_rewards, eps=cfg.advantage_eps)
            adv_map = {id(rec): adv for rec, adv in zip(all_records, all_advs, strict=False)}

        # 2) Split records into factored vs flat.
        factored_records: list[tuple[RolloutRecord, float]] = []
        flat_records: list[tuple[RolloutRecord, float]] = []
        for rec in all_records:
            adv = adv_map.get(id(rec), 0.0)
            if "factored" in rec.metadata:
                factored_records.append((rec, adv))
            elif self.factored_cfg.fallback_to_flat:
                flat_records.append((rec, adv))

        total_loss = torch.zeros((), dtype=torch.float32)
        total_kl = 0.0
        n_updated = 0

        # 3) Per-head loss for factored records.
        if factored_records:
            head_loss, head_kl, head_n = self._factored_head_loss(
                policy, ref_policy, factored_records
            )
            total_loss = total_loss + head_loss
            total_kl += head_kl
            n_updated += head_n

        # 4) Flat fallback.
        if flat_records:
            flat_loss, flat_kl, flat_n = self._flat_grpo_loss(policy, ref_policy, flat_records)
            flat_weight = len(flat_records) / max(1, len(all_records))
            total_loss = total_loss + flat_weight * flat_loss
            total_kl += flat_kl
            n_updated += flat_n

        mean_r = sum(r.reward for r in all_records) / max(1, len(all_records))
        mean_a = sum(adv_map.get(id(r), 0.0) for r in all_records) / max(1, len(all_records))

        stats = AlgoUpdateStats(
            loss=float(total_loss.detach().item())
            if total_loss.requires_grad
            else float(total_loss.item()),
            policy_loss=float(total_loss.detach().item()),
            kl=total_kl,
            entropy=0.0,
            mean_reward=float(mean_r),
            mean_advantage=float(mean_a),
            clip_frac=0.0,
            n_records=len(all_records),
            extra={
                "algo": "factored_grpo",
                "n_updated": n_updated,
                "n_factored": len(factored_records),
                "n_flat_fallback": len(flat_records),
                "head_weights": dict(self._norm_weights),
            },
        )
        return total_loss, stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _factored_head_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        records_with_adv: list[tuple[RolloutRecord, float]],
    ) -> tuple[torch.Tensor, float, int]:
        """Compute weighted sum of per-head clipped-surrogate losses."""
        cfg = self.grpo_cfg
        total: torch.Tensor = torch.zeros((), dtype=torch.float32)
        total_kl = 0.0
        n = len(records_with_adv)

        for head_name in self.factored_cfg.heads:
            w = self._norm_weights.get(head_name, 0.0)
            if w <= 0:
                continue

            # Collect per-head prompt/response/old_logprobs for this head.
            prompt_ids_list: list[list[int]] = []
            resp_ids_list: list[list[int]] = []
            old_lps: list[list[float]] = []
            advs_scalar: list[float] = []

            for rec, adv in records_with_adv:
                fac = rec.metadata.get("factored", {})
                head_data = fac.get(head_name)
                if head_data is None:
                    # Head absent for this record: use flat response
                    prompt_ids_list.append(rec.prompt_ids)
                    resp_ids_list.append(rec.response_ids)
                    old_lps.append(rec.old_logprobs)
                else:
                    prompt_ids_list.append(rec.prompt_ids)
                    resp_ids_list.append(list(head_data.get("response_ids") or rec.response_ids))
                    old_lps.append(list(head_data.get("old_logprobs") or rec.old_logprobs))
                advs_scalar.append(adv)

            if not prompt_ids_list:
                continue

            # Batched forward for this head.
            try:
                new_logp, mask = policy.score_batch(prompt_ids_list, resp_ids_list, temperature=1.0)
            except Exception:
                continue  # head scoring failed; skip this head

            B, T_max = new_logp.shape
            dtype = new_logp.dtype
            device = new_logp.device

            old_logp = torch.zeros(B, T_max, dtype=dtype, device=device)
            for i, olp in enumerate(old_lps):
                R_i = int(mask[i].sum().item())
                if R_i > 0 and olp:
                    chunk = olp[-R_i:]
                    old_logp[i, : len(chunk)] = torch.tensor(chunk, dtype=dtype, device=device)

            adv_t = torch.tensor(advs_scalar, dtype=dtype, device=device).unsqueeze(-1)
            adv_tensor = adv_t * mask.to(dtype)

            head_pol_loss, _ = clipped_surrogate_loss_batched(
                new_logp,
                old_logp,
                adv_tensor,
                mask,
                clip_eps=cfg.clip_eps,
                loss_agg=cfg.loss_agg,
                max_len_for_dr_grpo=cfg.max_len_for_dr_grpo,
            )
            total = total + w * head_pol_loss

            # KL per head — uses the shared, configurable estimator so the
            # signal is comparable with the other algorithms (default k3).
            if ref_policy is not None and cfg.kl_coef > 0:
                with torch.no_grad():
                    ref_logp, _ref_mask = ref_policy.score_batch(
                        prompt_ids_list, resp_ids_list, temperature=1.0
                    )
                kl_scalar = kl_from_logprobs_batched(
                    new_logp,
                    ref_logp,
                    mask,
                    estimator=self.factored_cfg.kl_estimator,
                )
                total = total + w * cfg.kl_coef * kl_scalar
                total_kl += float(kl_scalar.detach().item()) * w

        return total, total_kl, n

    def _flat_grpo_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        records_with_adv: list[tuple[RolloutRecord, float]],
    ) -> tuple[torch.Tensor, float, int]:
        """Standard flat GRPO loss for records without per-head metadata."""
        from hermes_agentic_rl.algos.base import RolloutBatch
        from hermes_agentic_rl.algos.grpo import GRPO

        flat_records = [rec for rec, _ in records_with_adv]
        # Override rewards so adv uses pre-computed values directly via batch
        flat_batch = RolloutBatch(records=flat_records)
        grpo = GRPO(self.grpo_cfg)
        loss, stats = grpo.compute_loss(policy, ref_policy, flat_batch)
        return loss, float(stats.kl), len(flat_records)
