"""Hybrid RL Objective — GRPO + OPD in one loss.

Paper: OpenClaw-RL §3.3 "Hybrid Method"

    L_i = w_RL * L_i^GRPO + w_OPD * L_i^OPD

The hybrid combines:
  - **Evaluative signal** (GRPO): scalar reward from PRM/ORM, dense because
    every scored turn contributes, including turns with implicit feedback.
  - **Directive signal** (OPD): token-level teacher log-prob advantage from
    hindsight hints, sparse but richer than any scalar.

Design in hermes-agentic-rl:
  - HybridAlgo.compute_loss() dispatches each record to the GRPO or OPD
    branch based on the presence of teacher_logprobs metadata.
  - Records that have BOTH scalar reward AND teacher logprobs contribute to
    BOTH branches simultaneously (true hybrid).
  - Records with only scalar reward → GRPO branch only.
  - Records with only teacher logprobs → OPD branch only.
  - w_RL / w_OPD can be set to 0 to ablate either branch.

Enabling the hybrid in the trainer YAML:
    algo:
      type: hybrid
      w_rl: 1.0
      w_opd: 1.0
      grpo:
        advantage_norm: group
        clip_eps: 0.2
        clip_eps_high: 0.28
        kl_coef: 0.02
      opd:
        kl_coef: 0.02
        clip_eps: 0.2
        clip_eps_high: 0.28
        adv_diff_clip: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
)
from hermes_agentic_rl.algos.grpo import GRPO, GRPOConfig
from hermes_agentic_rl.algos.opd import OPDAlgo, OPDConfig
from hermes_agentic_rl.backends.base import LLMBackend


@dataclass(slots=True)
class HybridConfig:
    """Config for the Hybrid GRPO + OPD objective.

    w_rl:  weight for the GRPO (evaluative) branch.
    w_opd: weight for the OPD  (directive)  branch.
    grpo:  GRPO sub-config (advantage normalisation, clipping, KL).
    opd:   OPD  sub-config (hint-adv clip, asymmetric clip, KL).
    """

    w_rl: float = 1.0
    w_opd: float = 1.0
    grpo: GRPOConfig = None  # type: ignore[assignment]
    opd: OPDConfig = None  # type: ignore[assignment]
    # When True, a cache-temperature mismatch (rollout temperature != 1.0 so
    # the OPD branch cannot consume the shared forward) raises instead of
    # silently re-forwarding. Useful in large runs where the warning is
    # easily drowned out and the silent double-forward inflates step time.
    strict_cache_temperature: bool = False

    def __post_init__(self) -> None:
        if self.grpo is None:
            self.grpo = GRPOConfig(
                clip_eps=0.2,
                clip_eps_high=0.28,  # asymmetric clip (OpenClaw-RL default)
                kl_coef=0.02,
            )
        if self.opd is None:
            self.opd = OPDConfig(
                clip_eps=0.2,
                clip_eps_high=0.28,
                adv_diff_clip=1.0,
                kl_coef=0.02,
            )


class HybridAlgo(BaseAlgo):
    """GRPO + OPD unified training objective (OpenClaw-RL Hybrid Method).

    Per-record dispatch:
        has reward + has teacher_logprobs → both branches (true hybrid)
        has reward only                   → GRPO branch
        has teacher_logprobs only         → OPD  branch

    The final loss is:
        L = w_rl * L_GRPO + w_opd * L_OPD

    where each branch loss is computed only over the records that feed it.
    """

    def __init__(self, cfg: HybridConfig | None = None) -> None:
        self.cfg = cfg or HybridConfig()
        self._grpo = GRPO(self.cfg.grpo)
        self._opd = OPDAlgo(self.cfg.opd)

    def compute_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, AlgoUpdateStats]:
        cfg = self.cfg
        records = batch.records
        if not records:
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
                extra={"algo": "hybrid", "n_grpo": 0, "n_opd": 0},
            )

        # Partition records into GRPO / OPD subsets. ``both_records`` are the
        # records that legitimately contribute to BOTH branches (true hybrid
        # records). They currently induce two independent forward passes via
        # the sub-algorithms; that overhead is reported below as
        # ``hybrid_double_forward_records`` so downstream callers can decide
        # whether the configuration is worth the extra compute.
        grpo_records: list[RolloutRecord] = []
        opd_records: list[RolloutRecord] = []
        both_records: list[RolloutRecord] = []
        for rec in records:
            has_reward = rec.reward != 0.0 or "reward" in rec.metadata
            has_hints = bool(rec.metadata.get("teacher_logprobs"))
            grpo_take = has_reward and cfg.w_rl > 0
            opd_take = has_hints and cfg.w_opd > 0
            if grpo_take:
                grpo_records.append(rec)
            if opd_take:
                opd_records.append(rec)
            if grpo_take and opd_take:
                both_records.append(rec)

        total_loss = torch.zeros((), dtype=torch.float32)
        grpo_stats: AlgoUpdateStats | None = None
        opd_stats: AlgoUpdateStats | None = None

        # ── v0.9.2: shared forward pass ──────────────────────────────────
        # When BOTH branches consume one or more of the same records, run
        # ``policy.score_batch`` ONCE over the union and inject the resulting
        # rows into each sub-batch's cache. GRPO and OPD then skip their own
        # forward.
        #
        # GRPO calls score_batch at the rollout-derived ``score_temperature``;
        # OPD at temperature 1.0. To keep both branches happy we pre-compute
        # the cache at the rollout temperature; OPD only consumes the cache
        # when the rollout temperature is 1.0. Outside that window the OPD
        # branch silently falls back to its own forward.
        from hermes_agentic_rl.algos.common.temperature import (
            rollout_score_temperature as _rollout_score_temperature,
        )

        cache_temperature: float | None = None
        cache_dict: dict[int, torch.Tensor] | None = None
        opd_cache_skipped = False  # whether OPD will silently re-forward
        if (grpo_records or opd_records) and len(both_records) > 0:
            # Build union preserving the records' identity (id()).
            seen: set[int] = set()
            union_records: list[RolloutRecord] = []
            for rec in (*grpo_records, *opd_records):
                if id(rec) in seen:
                    continue
                seen.add(id(rec))
                union_records.append(rec)
            if union_records:
                cache_temperature = float(_rollout_score_temperature(union_records))
                new_logp, mask = policy.score_batch(
                    [r.prompt_ids for r in union_records],
                    [r.response_ids for r in union_records],
                    temperature=cache_temperature,
                )
                cache_dict = {}
                for i, rec in enumerate(union_records):
                    R_i = int(mask[i].sum().item())
                    cache_dict[id(rec)] = new_logp[i, :R_i]
                # OPD only consumes the shared cache when the rollout
                # temperature is exactly 1.0 (its score temperature is
                # always 1.0). When that contract is violated the OPD
                # branch silently re-runs ``score_batch`` and the
                # "savings ledger" advertised in the stats becomes
                # misleading. Surface the miss explicitly so users can
                # either accept the second forward or align the rollout
                # temperature.
                if opd_records and abs(cache_temperature - 1.0) > 1e-9:
                    opd_cache_skipped = True
                    msg = (
                        "HybridAlgo: OPD branch cannot consume the shared "
                        f"forward cache (cache temperature={cache_temperature:.4f} "
                        "!= 1.0). The OPD branch will re-run policy.score_batch "
                        "at temperature=1.0. Set rollout_temperature=1.0 to "
                        "remove the extra forward, or accept the cost."
                    )
                    if cfg.strict_cache_temperature:
                        raise ValueError(msg + " (strict_cache_temperature=True)")
                    import warnings as _warnings

                    _warnings.warn(msg, stacklevel=2)

        # ── GRPO branch ──────────────────────────────────────────────────
        if grpo_records and cfg.w_rl > 0:
            grpo_batch = RolloutBatch(
                records=grpo_records,
                shared_new_logprobs=cache_dict,
                shared_logprobs_temperature=cache_temperature,
            )
            g_loss, grpo_stats = self._grpo.compute_loss(policy, ref_policy, grpo_batch)
            total_loss = total_loss + cfg.w_rl * g_loss

        # ── OPD branch ───────────────────────────────────────────────────
        if opd_records and cfg.w_opd > 0:
            opd_batch = RolloutBatch(
                records=opd_records,
                shared_new_logprobs=cache_dict,
                shared_logprobs_temperature=cache_temperature,
            )
            o_loss, opd_stats = self._opd.compute_loss(policy, ref_policy, opd_batch)
            total_loss = total_loss + cfg.w_opd * o_loss

        # ── Aggregate stats ──────────────────────────────────────────────
        mean_r = sum(r.reward for r in records) / max(1, len(records))

        # KL aggregation: weight each branch's KL by its branch weight × the
        # number of records that fed it. The previous implementation averaged
        # by raw record count, which under-counted the OPD branch when
        # w_opd > 1 (or over-counted when w_opd < 1).
        kl_num = 0.0
        kl_den = 0.0
        if grpo_stats is not None:
            kl_num += cfg.w_rl * grpo_stats.kl * len(grpo_records)
            kl_den += cfg.w_rl * len(grpo_records)
        if opd_stats is not None:
            kl_num += cfg.w_opd * opd_stats.kl * len(opd_records)
            kl_den += cfg.w_opd * len(opd_records)
        kl = (kl_num / kl_den) if kl_den > 0 else 0.0

        stats = AlgoUpdateStats(
            loss=float(total_loss.detach().item()),
            policy_loss=float(total_loss.detach().item()),
            kl=kl,
            entropy=grpo_stats.entropy if grpo_stats else 0.0,
            mean_reward=float(mean_r),
            mean_advantage=grpo_stats.mean_advantage if grpo_stats else 0.0,
            clip_frac=grpo_stats.clip_frac if grpo_stats else 0.0,
            n_records=len(records),
            extra={
                "algo": "hybrid",
                "w_rl": cfg.w_rl,
                "w_opd": cfg.w_opd,
                "n_grpo": len(grpo_records),
                "n_opd": len(opd_records),
                "n_both": len(both_records),
                # When ``shared_forward_active`` is 1.0 the two sub-algos
                # consumed a single backbone forward pass instead of two.
                # ``hybrid_double_forward_records`` retains the count of
                # records that *would* otherwise have been double-forwarded;
                # post-optimisation that's the savings ledger.
                "hybrid_double_forward_records": len(both_records),
                "shared_forward_active": 1.0 if cache_dict is not None else 0.0,
                "opd_cache_miss_temperature": 1.0 if opd_cache_skipped else 0.0,
                "cache_temperature": (
                    float(cache_temperature) if cache_temperature is not None else 0.0
                ),
                "grpo_loss": float(grpo_stats.loss) if grpo_stats else 0.0,
                "opd_loss": float(opd_stats.loss) if opd_stats else 0.0,
                # Branch-level KL so the AdaptiveKLController feedback can be
                # disentangled per branch (the aggregate ``kl`` above is a
                # weighted blend that hides per-branch divergence).
                "grpo_kl": float(grpo_stats.kl) if grpo_stats else 0.0,
                "opd_kl": float(opd_stats.kl) if opd_stats else 0.0,
            },
        )
        return total_loss, stats
