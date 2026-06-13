"""GSPO: sequence-level Group Stable Policy Optimization.

GSPO keeps GRPO's group-relative advantage but applies the policy ratio at
the sequence level instead of per token. For each rollout ``i``:

    ratio_i = exp(sum_t log pi_new(a_t|s_t) - sum_t log pi_old(a_t|s_t))

The clipped surrogate then acts on one scalar ratio and one scalar advantage
per sequence. This is useful for long or variable-length responses where a
token-wise ratio can make optimization noisier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
    stack_cached_logprobs,
)
from hermes_agentic_rl.algos.common.advantage import (
    dapo_group_advantage,
    group_normalize_advantage,
)
from hermes_agentic_rl.algos.common.batch_prepare import (
    compute_entropy_bonus,
    compute_kl_penalty,
)
from hermes_agentic_rl.algos.common.reinforce_pp import (
    batch_normalize_advantage,
    whitened_advantage,
)
from hermes_agentic_rl.algos.common.temperature import rollout_score_temperature
from hermes_agentic_rl.backends.base import LLMBackend


@dataclass(slots=True)
class GSPOConfig:
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28
    kl_coef: float = 0.02
    entropy_coef: float = 0.0
    advantage_eps: float = 1e-6
    advantage_norm: Literal["group", "dapo", "batch", "whiten", "none"] = "group"
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"
    # Prevent exp overflow while keeping the unclipped PPO-style objective.
    # exp(60) ~= 1.1e26 overflows bf16 (max ~3.4e38 is fine for fp32 but the
    # *gradient* through exp(60) is already inf in bf16). Cap at 40 so the
    # surrogate ratio stays finite across fp32/bf16/fp16 autocast paths.
    log_ratio_clip: float = 40.0


class GSPO(BaseAlgo):
    """Sequence-level GRPO variant.

    Unlike token-wise GRPO, the policy ratio is computed once per rollout from
    the masked sum of response log-probs. The advantage remains scalar.
    """

    def __init__(self, cfg: GSPOConfig | None = None) -> None:
        self.cfg = cfg or GSPOConfig()

    def compute_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, AlgoUpdateStats]:
        cfg = self.cfg
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
                extra={"algo": "gspo", "n_updated": 0, "approx_kl": 0.0},
            )

        records_with_adv, n_filtered = _records_with_advantages(all_records, cfg)
        if not records_with_adv:
            zero = torch.zeros((), dtype=torch.float32)
            return zero, AlgoUpdateStats(
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                entropy=0.0,
                mean_reward=_mean_reward(all_records),
                mean_advantage=0.0,
                clip_frac=0.0,
                n_records=len(all_records),
                extra={
                    "algo": "gspo",
                    "n_updated": 0,
                    "approx_kl": 0.0,
                    "dapo_filtered_records": float(n_filtered),
                },
            )

        prompt_ids_list = [rec.prompt_ids for rec, _adv in records_with_adv]
        response_ids_list = [rec.response_ids for rec, _adv in records_with_adv]
        score_temperature = rollout_score_temperature(
            [rec for rec, _adv in records_with_adv]
        )

        # When a compound algorithm (HybridAlgo) has pre-computed
        # ``policy.score_batch`` for the same records and at a matching
        # temperature, reuse those rows. This mirrors the GRPO/OPD cache
        # path so wrapping GSPO inside a Hybrid configuration doesn't
        # double-forward the backbone.
        cache_records = [rec for rec, _adv in records_with_adv]
        used_cache = False
        if (
            batch.has_shared_logprobs_for(cache_records)
            and batch.shared_logprobs_temperature is not None
            and abs(float(batch.shared_logprobs_temperature) - float(score_temperature))
            < 1e-9
        ):
            new_logp, mask = stack_cached_logprobs(
                batch.shared_new_logprobs, cache_records  # type: ignore[arg-type]
            )
            used_cache = True
        else:
            new_logp, mask = policy.score_batch(
                prompt_ids_list,
                response_ids_list,
                temperature=score_temperature,
            )
        B, T_max = new_logp.shape
        if B == 0 or T_max == 0:
            zero = new_logp.new_zeros(())
            return zero, AlgoUpdateStats(
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                entropy=0.0,
                mean_reward=_mean_reward(all_records),
                mean_advantage=0.0,
                clip_frac=0.0,
                n_records=len(all_records),
                extra={"algo": "gspo", "n_updated": 0, "approx_kl": 0.0},
            )

        dtype = new_logp.dtype
        device = new_logp.device
        mask_f = mask.to(dtype)

        seq_logp_new = (new_logp * mask_f).sum(dim=-1)
        seq_logp_old = torch.tensor(
            [
                _old_seq_logprob(rec, int(mask[i].sum().item()))
                for i, (rec, _adv) in enumerate(records_with_adv)
            ],
            dtype=dtype,
            device=device,
        )
        adv = torch.tensor(
            [float(adv_i) for _rec, adv_i in records_with_adv],
            dtype=dtype,
            device=device,
        )

        log_ratio = seq_logp_new - seq_logp_old
        ratio = torch.exp(
            log_ratio.clamp(min=-cfg.log_ratio_clip, max=cfg.log_ratio_clip)
        )
        low = math.log(max(1e-8, 1.0 - cfg.clip_eps))
        high = math.log(max(1e-8, 1.0 + cfg.clip_eps_high))
        clipped_ratio = torch.exp(log_ratio.clamp(min=low, max=high))

        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        pol_terms = torch.minimum(surr1, surr2)
        pol_loss = -pol_terms.mean()
        total = pol_loss

        ratio_detached = ratio.detach()
        clipped_detached = clipped_ratio.detach()
        clip_frac = float(
            (ratio_detached.ne(clipped_detached)).to(torch.float32).mean().item()
        )
        # k3 estimator — consistent with GRPO/RLOO/OPD/PPO global default.
        log_r = log_ratio.detach().clamp(min=-20.0, max=20.0)
        approx_kl = float((torch.exp(-log_r) - 1.0 + log_r).mean().item())

        kl_val = 0.0
        kl_result = compute_kl_penalty(
            new_logp, mask,
            prompt_ids_list, response_ids_list,
            score_temperature, ref_policy, cfg.kl_coef,
            kl_estimator=cfg.kl_estimator,
        )
        if kl_result is not None:
            total = total + cfg.kl_coef * kl_result.kl_scalar
            kl_val = kl_result.kl_val

        ent_val, ent_scalar = compute_entropy_bonus(new_logp, mask, cfg.entropy_coef)
        if ent_scalar is not None:
            total = total - cfg.entropy_coef * ent_scalar

        stats = AlgoUpdateStats(
            loss=float(total.detach().item()) if total.requires_grad else float(total.item()),
            policy_loss=float(pol_loss.detach().item()),
            kl=kl_val,
            entropy=ent_val,
            mean_reward=_mean_reward(all_records),
            mean_advantage=float(adv.detach().mean().item()),
            clip_frac=clip_frac,
            n_records=len(all_records),
            extra={
                "algo": "gspo",
                "n_updated": len(records_with_adv),
                "approx_kl": approx_kl,
                "ratio_mean": float(ratio_detached.mean().item()),
                "n_tokens": int(mask.sum().item()),
                "score_temperature": float(score_temperature),
                "dapo_filtered_records": float(n_filtered),
                "shared_logprobs_cache_hit": 1.0 if used_cache else 0.0,
            },
        )
        return total, stats


def _records_with_advantages(
    records: list[RolloutRecord],
    cfg: GSPOConfig,
) -> tuple[list[tuple[RolloutRecord, float]], int]:
    if cfg.advantage_norm == "group":
        out: list[tuple[RolloutRecord, float]] = []
        for _gid, recs in RolloutBatch(records).by_group().items():
            advs = group_normalize_advantage(
                [rec.reward for rec in recs],
                eps=cfg.advantage_eps,
            )
            out.extend((rec, float(adv)) for rec, adv in zip(recs, advs, strict=False))
        return _restore_record_order(records, out), 0

    if cfg.advantage_norm == "dapo":
        out = []
        filtered = 0
        for _gid, recs in RolloutBatch(records).by_group().items():
            advs = dapo_group_advantage(
                [rec.reward for rec in recs],
                eps=cfg.advantage_eps,
            )
            if advs is None:
                filtered += len(recs)
                continue
            out.extend((rec, float(adv)) for rec, adv in zip(recs, advs, strict=False))
        return _restore_record_order(records, out), filtered

    if cfg.advantage_norm == "batch":
        advs = batch_normalize_advantage(
            [rec.reward for rec in records],
            eps=cfg.advantage_eps,
        )
    elif cfg.advantage_norm == "whiten":
        advs = whitened_advantage([rec.reward for rec in records], eps=cfg.advantage_eps)
    elif cfg.advantage_norm == "none":
        advs = [float(rec.reward) for rec in records]
    else:
        raise ValueError(f"Unknown advantage_norm: {cfg.advantage_norm}")
    return list(zip(records, [float(adv) for adv in advs], strict=False)), 0


def _restore_record_order(
    records: list[RolloutRecord],
    pairs: list[tuple[RolloutRecord, float]],
) -> list[tuple[RolloutRecord, float]]:
    order = {id(rec): i for i, rec in enumerate(records)}
    return sorted(pairs, key=lambda pair: order[id(pair[0])])


def _old_seq_logprob(record: RolloutRecord, response_len: int) -> float:
    raw = record.metadata.get("old_seq_logprob")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    if response_len <= 0:
        return 0.0
    return float(sum(float(v) for v in record.old_logprobs[-response_len:]))


def _mean_reward(records: list[RolloutRecord]) -> float:
    return float(sum(rec.reward for rec in records) / max(1, len(records)))
