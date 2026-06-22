"""SimPO + DAPO dynamic sampling algorithm.

SimPO (Simple Preference Optimization, Meng et al. 2024):
  - Reference-model-free: uses only the current policy.
  - Length-normalized reward: R̃(x, y) = (1/|y|) Σ_t log π_θ(y_t | x, y<t)
    This naturally penalizes verbose outputs and prefers concise, correct
    responses. Equivalent to adding a length penalty to the Bradley-Terry
    objective without tuning a separate coefficient.
  - Margin loss: L = -log σ(β · (R̃_w - R̃_l - γ)) where w/l = winner/loser.
    The margin γ ensures the winner is preferred by at least γ/β in log-prob.

DAPO dynamic sampling (Yu et al., 2024):
  - Filter groups where all rollouts have the same reward (zero variance)
    — these provide no contrastive signal.
  - Keep groups where at least one rollout is positive and one is negative
    (if ``require_mixed`` is True) for maximum contrastive gradient.

Integration with hermes-agentic-rl:
  ``SimPOAlgo.compute_loss()`` follows the same interface as GRPO — takes
  (policy, ref_policy, batch) and returns (loss, AlgoUpdateStats).
  ``ref_policy`` is accepted but ignored (SimPO is reference-free).

Reference:
  Meng et al. (2024) "SimPO: Simple Preference Optimization with a
  Reference-Free Reward", arXiv:2405.14734.
  Yu et al. (2024) "DAPO: Dynamic Sampling for Preference Optimization",
  arXiv:2410.11425.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
)
from hermes_agentic_rl.algos.common.advantage import dapo_group_advantage
from hermes_agentic_rl.algos.common.batch_prepare import mean_reward_from_records
from hermes_agentic_rl.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SimPOConfig:
    """Configuration for :class:`SimPOAlgo`.

    beta: inverse temperature / Bradley-Terry scaling factor.
        Higher β = sharper preference distinction.
    gamma: reward margin in the BT loss (``L = -log σ(β(R̃_w - R̃_l - γ))``).
        Pushes the model to prefer winners by at least γ/β log-prob per token.
    loss_agg: how to aggregate token log-probs into the length-normalized
        reward. ``mean_token`` (default) divides by actual response length;
        ``sum_token`` is un-normalized (original RLHF convention).
    dapo_filter: when True, filter out groups where all rewards are equal
        (zero-variance groups have no contrastive signal). Mirrors DAPO §3.1.
    require_mixed: when True (and ``dapo_filter`` is also True), additionally
        require that each group contains at least one positive and one negative
        reward. Groups that fail this check are dropped. Useful when rewards
        are binary (0/1) correct/incorrect.
    min_reward_gap: minimum absolute reward gap between winner and loser in a
        group to include a pair in the update. Pairs with ``|R_w - R_l| <
        min_reward_gap`` are skipped. Set 0 (default) to keep all pairs.
    entropy_coef: optional entropy regularization. Adds ``entropy_coef *
        H(π)`` to the loss. Most SimPO configs set this to 0.
    """

    beta: float = 2.5
    gamma: float = 1.0
    loss_agg: str = "mean_token"  # "mean_token" | "sum_token"
    dapo_filter: bool = True
    require_mixed: bool = False
    min_reward_gap: float = 0.0
    entropy_coef: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _length_normalized_logprob(
    logprobs: torch.Tensor,  # [T] valid response log-probs (no padding)
    loss_agg: str,
) -> torch.Tensor:
    """Compute length-normalized log-prob of a single response."""
    if logprobs.numel() == 0:
        return logprobs.new_zeros(())
    if loss_agg == "mean_token":
        return logprobs.mean()
    return logprobs.sum()  # "sum_token" = standard log P(y|x)


def _select_winner_loser(
    records: list[RolloutRecord],
    *,
    min_reward_gap: float,
) -> list[tuple[RolloutRecord, RolloutRecord]]:
    """Return (winner, loser) pairs from a group of rollouts.

    Simple strategy: for each (i, j) pair where R_i > R_j + min_reward_gap,
    yield (record_i, record_j). Duplicates are allowed (each record can
    appear in multiple pairs) — this naturally up-weights groups with a large
    within-group spread.

    For efficiency, instead of all O(n²) pairs, we pair the max-reward
    record against all others that are at least ``min_reward_gap`` below.
    """
    if len(records) < 2:
        return []
    sorted_recs = sorted(records, key=lambda r: r.reward, reverse=True)
    winner = sorted_recs[0]
    pairs: list[tuple[RolloutRecord, RolloutRecord]] = []
    for loser in sorted_recs[1:]:
        if winner.reward - loser.reward >= min_reward_gap:
            pairs.append((winner, loser))
    return pairs


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class SimPOAlgo(BaseAlgo):
    """Reference-free SimPO with optional DAPO dynamic-group filtering.

    The update rule for each (winner, loser) pair::

        R̃(x, y) = mean_t log π_θ(y_t | x, y<t)     # length-normalized
        L_pair   = -log σ(β · (R̃_w - R̃_l - γ))
        L        = mean over all pairs in batch

    DAPO filtering (when ``cfg.dapo_filter=True``):
        Groups where all rollouts have the same reward are dropped entirely.
        When ``cfg.require_mixed=True``, also drop groups that have no
        positive–negative contrast (all positive or all negative).
    """

    def __init__(self, cfg: SimPOConfig | None = None) -> None:
        self.cfg = cfg or SimPOConfig()

    def compute_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,  # accepted but not used (reference-free)
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, AlgoUpdateStats]:
        cfg = self.cfg
        records = list(batch.records)
        if not records:
            # Return a detached zero — no gradient computation needed for empty batches.
            zero = torch.zeros(())
            return zero, AlgoUpdateStats(
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                entropy=0.0,
                mean_reward=0.0,
                mean_advantage=0.0,
                clip_frac=0.0,
                n_records=0,
                extra={"algo": "simpo", "n_pairs": 0},
            )

        # ── 1. Group records by prompt ───────────────────────────────────
        groups: dict[str, list[RolloutRecord]] = {}
        for rec in records:
            # RolloutRecord.group_id is the canonical group identifier;
            # fall back to metadata["group_id"] for legacy callers, or prompt_ids
            # hash for records without an explicit group tag.
            gid = (
                str(rec.group_id)
                if rec.group_id
                else str(rec.metadata.get("group_id", hash(tuple(rec.prompt_ids))))
            )
            groups.setdefault(gid, []).append(rec)

        # ── 2. DAPO filtering ────────────────────────────────────────────
        filtered_groups: list[list[RolloutRecord]] = []
        n_dropped_zero_var = 0
        n_dropped_no_contrast = 0
        for recs in groups.values():
            rewards = [r.reward for r in recs]
            # dapo_group_advantage returns None when all rewards identical
            adv = dapo_group_advantage(rewards) if cfg.dapo_filter else rewards
            if adv is None:
                n_dropped_zero_var += 1
                continue
            if cfg.require_mixed and cfg.dapo_filter:
                has_pos = any(r > 0 for r in rewards)
                has_neg = any(r <= 0 for r in rewards)
                if not (has_pos and has_neg):
                    n_dropped_no_contrast += 1
                    continue
            filtered_groups.append(recs)

        if not filtered_groups:
            zero = torch.zeros(())
            return zero, AlgoUpdateStats(
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                entropy=0.0,
                mean_reward=sum(r.reward for r in records) / max(1, len(records)),
                mean_advantage=0.0,
                clip_frac=0.0,
                n_records=len(records),
                extra={
                    "algo": "simpo",
                    "n_pairs": 0,
                    "n_dropped_zero_var": float(n_dropped_zero_var),
                    "n_dropped_no_contrast": float(n_dropped_no_contrast),
                },
            )

        # ── 3. Collect (winner, loser) pairs ────────────────────────────
        all_pairs: list[tuple[RolloutRecord, RolloutRecord]] = []
        for recs in filtered_groups:
            all_pairs.extend(_select_winner_loser(recs, min_reward_gap=cfg.min_reward_gap))

        if not all_pairs:
            zero = torch.zeros(())
            return zero, AlgoUpdateStats(
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                entropy=0.0,
                mean_reward=sum(r.reward for r in records) / max(1, len(records)),
                mean_advantage=0.0,
                clip_frac=0.0,
                n_records=len(records),
                extra={"algo": "simpo", "n_pairs": 0},
            )

        # ── 4. Batch score winners + losers ─────────────────────────────
        winners = [w for w, _ in all_pairs]
        losers = [l for _, l in all_pairs]
        all_unique_recs = list({id(r): r for r in winners + losers}.values())

        # Score all unique records in one batched forward pass.
        prompt_ids_list = [rec.prompt_ids for rec in all_unique_recs]
        response_ids_list = [rec.response_ids for rec in all_unique_recs]
        new_logp_batch, mask_batch = policy.score_batch(
            prompt_ids_list, response_ids_list
        )  # [N, T], [N, T]

        dtype = new_logp_batch.dtype

        # Build a fast lookup: id(rec) → row index
        rec_to_idx: dict[int, int] = {id(r): i for i, r in enumerate(all_unique_recs)}

        # ── 5. Compute SimPO loss ────────────────────────────────────────
        pair_losses: list[torch.Tensor] = []
        reward_gaps: list[float] = []

        for winner, loser in all_pairs:
            w_idx = rec_to_idx[id(winner)]
            l_idx = rec_to_idx[id(loser)]

            w_lp = new_logp_batch[w_idx]  # [T]
            w_mask = mask_batch[w_idx].to(dtype)  # [T]
            l_lp = new_logp_batch[l_idx]
            l_mask = mask_batch[l_idx].to(dtype)

            # Mask-select valid (non-padding) token log-probs.
            w_valid = w_lp * w_mask
            l_valid = l_lp * l_mask

            r_w = _length_normalized_logprob(w_valid[w_mask.bool()], cfg.loss_agg)
            r_l = _length_normalized_logprob(l_valid[l_mask.bool()], cfg.loss_agg)

            # SimPO Bradley-Terry margin loss.
            # L = -log σ(β * (R̃_w - R̃_l - γ))
            margin = cfg.beta * (r_w - r_l - cfg.gamma)
            loss_pair = -F.logsigmoid(margin)
            pair_losses.append(loss_pair)
            reward_gaps.append(float(winner.reward - loser.reward))

        simpo_loss = torch.stack(pair_losses).mean()

        # ── 6. Entropy bonus ─────────────────────────────────────────────
        ent_val = 0.0
        total = simpo_loss
        if cfg.entropy_coef > 0:
            ent_vals: list[torch.Tensor] = []
            for rec in all_unique_recs:
                idx = rec_to_idx[id(rec)]
                lp = new_logp_batch[idx]
                mask = mask_batch[idx].to(dtype)
                n = mask.sum().clamp(min=1)
                ent_vals.append(-(lp * mask).sum() / n)
            ent_scalar = torch.stack(ent_vals).mean()
            total = total - cfg.entropy_coef * ent_scalar
            ent_val = float(ent_scalar.detach().item())

        # ── 7. Stats ─────────────────────────────────────────────────────
        mean_r = mean_reward_from_records(records)
        mean_gap = sum(reward_gaps) / max(1, len(reward_gaps))

        stats = AlgoUpdateStats(
            loss=float(total.detach().item()),
            policy_loss=float(simpo_loss.detach().item()),
            kl=0.0,  # reference-free — no KL term
            entropy=ent_val,
            mean_reward=mean_r,
            mean_advantage=mean_gap,
            clip_frac=0.0,  # no clipping in SimPO
            n_records=len(records),
            extra={
                "algo": "simpo",
                "n_pairs": float(len(all_pairs)),
                "n_filtered_groups": float(len(filtered_groups)),
                "n_dropped_zero_var": float(n_dropped_zero_var),
                "n_dropped_no_contrast": float(n_dropped_no_contrast),
                "mean_reward_gap": mean_gap,
                "simpo_beta": cfg.beta,
                "simpo_gamma": cfg.gamma,
            },
        )
        return total, stats
