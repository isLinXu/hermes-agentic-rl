"""RLOO: REINFORCE Leave-One-Out baseline (Kool et al. 2019; Williams 1992).

Standard GRPO uses group-mean as baseline:
    A_i = (R_i - mean_g) / std_g

RLOO uses a per-sample leave-one-out baseline:
    b_i = sum_{j ≠ i} R_j / (G - 1)
    A_i = R_i - b_i

This is provably unbiased (each A_i has E[A_i] = 0 under the baseline
distribution) and has *lower variance* than GRPO when group size G ≥ 3,
because the baseline only uses samples from the same prompt.

Key properties vs. GRPO group-norm:
  - GRPO divides by std_g → advantage scale varies per group.
  - RLOO keeps reward scale → combines naturally with PRM ±1 signals.
  - RLOO A_i ≡ 0 when G = 1 (no information) — safe fallback.
  - RLOO variance = O(σ²/G) while GRPO variance = O(1/G) after normalisation
    — both decrease with G but RLOO preserves absolute magnitude.

``RLOOAlgo`` reuses the GRPO clipped surrogate; the only difference is
how the advantage tensor is built.

Usage::

    algo = RLOOAlgo(RLOOConfig(clip_eps=0.2, kl_coef=0.01, normalize=False))
    loss, stats = algo.compute_loss(policy, ref_policy, batch)
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
from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched
from hermes_agentic_rl.algos.common.loss import clipped_surrogate_loss_batched
from hermes_agentic_rl.algos.common.reinforce_pp import expand_per_token_advantage
from hermes_agentic_rl.algos.common.temperature import rollout_score_temperature
from hermes_agentic_rl.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Pure-Python RLOO advantage computation (no torch dependency)
# ---------------------------------------------------------------------------


def rloo_advantage(
    rewards: list[float],
    *,
    eps: float = 1e-6,
    normalize: bool = True,
) -> list[float]:
    """Leave-one-out baseline advantage.

    Args:
        rewards: scalar rewards for one group (same prompt, different samples).
        eps: small constant added to std when normalize=True.
        normalize: if True, z-score-normalise after LOO subtraction (useful
            when combining RLOO with PRM signals of varying magnitude).

    Returns:
        Per-sample advantages in the same order as ``rewards``.
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        # Cannot form a leave-one-out estimate — return 0.
        return [0.0]

    total = sum(rewards)
    advs = [(rewards[i] - (total - rewards[i]) / (n - 1)) for i in range(n)]

    if normalize:
        mean_a = sum(advs) / n
        std_a = (sum((a - mean_a) ** 2 for a in advs) / n) ** 0.5
        if std_a > eps:
            advs = [(a - mean_a) / (std_a + eps) for a in advs]
        else:
            advs = [0.0] * n

    return advs


# ---------------------------------------------------------------------------
# Config + Algo
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RLOOConfig:
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28        # asymmetric upper clip (same default as GRPO)
    kl_coef: float = 0.01
    entropy_coef: float = 0.0
    advantage_eps: float = 1e-6
    # z-score after LOO subtraction. Defaults to True because RLOO preserves
    # the raw reward scale (A_i = R_i - b_i), so when PRM/ORM rewards span a
    # wide range (e.g. [-5, +5]) the un-normalised advantage variance can be
    # much larger than GRPO's group-normalised signal and destabilise the
    # update. Set normalize=False to keep absolute magnitudes (e.g. when
    # combining with bounded ±1 PRM signals).
    normalize: bool = True             # z-score after LOO subtraction
    loss_agg: Literal["mean_token", "sum_token", "dr_grpo"] = "mean_token"
    max_len_for_dr_grpo: int = 256
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"
    # v1.0: per-token advantage (REINFORCE++) — same semantics as GRPO.
    per_token_advantage: bool = False
    answer_start_token_id: int | None = None
    reinforce_gamma: float = 0.95


class RLOOAlgo(BaseAlgo):
    """REINFORCE Leave-One-Out algorithm.

    Identical to GRPO except the per-group advantage uses the LOO baseline
    instead of group mean normalisation.
    """

    def __init__(self, cfg: RLOOConfig | None = None) -> None:
        self.cfg = cfg or RLOOConfig()

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
                loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0,
                mean_reward=0.0, mean_advantage=0.0, clip_frac=0.0,
                n_records=0,
                extra={"algo": "rloo", "n_updated": 0, "approx_kl": 0.0},
            )

        # 1) Build per-group LOO advantages.
        rwa: list[tuple[RolloutRecord, list[float]]] = []
        for _gid, recs in batch.by_group().items():
            rewards_g = [r.reward for r in recs]
            advs_g = rloo_advantage(rewards_g, eps=cfg.advantage_eps, normalize=cfg.normalize)
            for rec, adv in zip(recs, advs_g, strict=False):
                rwa.append((rec, [adv]))

        # Preserve original insertion order.
        order = {id(r): i for i, r in enumerate(all_records)}
        rwa.sort(key=lambda p: order[id(p[0])])

        # v1.0: per-token advantage expansion (shared helper).
        if cfg.per_token_advantage:
            rwa = expand_per_token_advantage(
                rwa,
                answer_start_token_id=cfg.answer_start_token_id,
                gamma=cfg.reinforce_gamma,
                eps=cfg.advantage_eps,
            )

        records_with_adv = rwa

        # 2) Batched forward.
        prompt_ids_list = [rec.prompt_ids for rec, _ in records_with_adv]
        response_ids_list = [rec.response_ids for rec, _ in records_with_adv]
        score_temperature = rollout_score_temperature([rec for rec, _ in records_with_adv])

        new_logp, mask = policy.score_batch(
            prompt_ids_list, response_ids_list, temperature=score_temperature,
        )
        B, T_max = new_logp.shape
        if B == 0 or T_max == 0:
            zero = new_logp.new_zeros(())
            return zero, AlgoUpdateStats(
                loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0,
                mean_reward=float(sum(r.reward for r in all_records) / max(1, len(all_records))),
                mean_advantage=0.0, clip_frac=0.0,
                n_records=len(all_records),
                extra={"algo": "rloo", "n_updated": 0, "approx_kl": 0.0},
            )

        device, dtype = new_logp.device, new_logp.dtype

        # 3) Stack old logprobs and advantage tensor.
        old_logp = torch.zeros(B, T_max, dtype=dtype, device=device)
        for i, (rec, _) in enumerate(records_with_adv):
            olp = rec.old_logprobs
            R_i = min(len(olp), int(mask[i].sum().item()))
            if R_i > 0:
                old_logp[i, :R_i] = torch.tensor(olp[-R_i:], dtype=dtype, device=device)

        has_per_token = any(len(a) > 1 for _, a in records_with_adv)
        if has_per_token:
            adv_tensor = torch.zeros(B, T_max, dtype=dtype, device=device)
            for i, (rec, adv_list) in enumerate(records_with_adv):
                R_i = int(mask[i].sum().item())
                if R_i == 0 or not adv_list:
                    continue
                vals = adv_list[-R_i:] if len(adv_list) >= R_i else adv_list
                adv_tensor[i, :len(vals)] = torch.tensor(vals, dtype=dtype, device=device)
        else:
            scalars = torch.tensor(
                [float(a[0]) if a else 0.0 for _, a in records_with_adv],
                dtype=dtype, device=device,
            )
            adv_tensor = scalars.unsqueeze(-1) * mask.to(dtype)  # [B, T_max]

        # 4) Clipped surrogate (asymmetric clip).
        pol_loss, loss_stats = clipped_surrogate_loss_batched(
            new_logp, old_logp, adv_tensor, mask,
            clip_eps=cfg.clip_eps,
            clip_eps_high=cfg.clip_eps_high,
            loss_agg=cfg.loss_agg,
            max_len_for_dr_grpo=cfg.max_len_for_dr_grpo,
        )
        total = pol_loss
        kl_val = 0.0

        # 5) KL-to-reference — uses shared kl_from_logprobs_batched.
        if ref_policy is not None and cfg.kl_coef > 0:
            with torch.no_grad():
                ref_logp, _ = ref_policy.score_batch(
                    prompt_ids_list, response_ids_list, temperature=score_temperature,
                )
            kl_scalar = kl_from_logprobs_batched(
                new_logp, ref_logp, mask, estimator=cfg.kl_estimator
            )
            total = total + cfg.kl_coef * kl_scalar
            kl_val = float(kl_scalar.detach().item())

        # 6) Entropy bonus.
        ent_val = 0.0
        if cfg.entropy_coef > 0:
            mf = mask.to(dtype)
            ent_per_row = -(new_logp * mf).sum(-1) / mf.sum(-1).clamp(min=1)
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
                "algo": "rloo",
                "n_updated": len(all_records),
                "approx_kl": loss_stats["approx_kl"],
                "ratio_mean": loss_stats["ratio_mean"],
                "n_tokens": loss_stats["n_tokens"],
                "score_temperature": float(score_temperature),
            },
        )
        return total, stats
