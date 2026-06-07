"""On-Policy Distillation (OPD) — ported from OpenClaw-RL.

Paper: "OpenClaw-RL: Train Any Agent Simply by Talking"
       arXiv:2603.10165, Wang et al. 2026

Core Idea (§3.2 of the paper):
  Every next-state signal (user reply, tool output, test result, …) contains
  a *directive* component: it implicitly tells the policy *how* the action
  should have been different.  OPD makes this directive explicit by asking a
  judge to extract a textual *hint*, then uses that hint to build a stronger
  teacher distribution.

  Token-level advantage for OPD (Eq. 4 in paper):

      A_t^OPD = log π_T(a_t | s + hint) − log π_θ(a_t | s)

  where π_T is the teacher (hint-augmented policy at the hint-conditioned
  prompt) and π_θ is the current student.

  This is *richer* than any scalar reward because it provides a separate
  directional signal at every token position.

Training objective (Eq. 5):
      L = L_pg + β_KL · L_KL

  where L_pg uses the token-level advantages above with a PPO-style clip.

Integration with hermes-agentic-rl:
  OPDAlgo.compute_loss() follows the same interface as GRPO.compute_loss()
  (policy, ref_policy, batch) → (loss, AlgoUpdateStats).

  The batch records must carry hint metadata:
      record.metadata["opd_hint"]          : str   — the extracted hint text
      record.metadata["teacher_logprobs"]  : list[float] — log π_T per token
                                                         (pre-computed by judge)

  When teacher_logprobs are absent the OPD term is skipped (graceful
  degradation to plain KL-regularised PG).

Hint extraction (judge interface):
  Use OPDJudge.extract_hint(response, next_state) → str | None.
  The judge wraps any callable: an LLM call, a rule-based extractor, or a
  pre-trained PRM.  It is intentionally decoupled so callers can swap it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from hermes_agentic_rl.algos.base import (
    AlgoUpdateStats,
    BaseAlgo,
    RolloutBatch,
    RolloutRecord,
    old_logprobs_tensor,
    stack_cached_logprobs,
)
from hermes_agentic_rl.algos.common.kl import kl_from_logprobs_batched
from hermes_agentic_rl.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Hint extraction — judge interface
# ---------------------------------------------------------------------------

HINT_START = "[HINT_START]"
HINT_END   = "[HINT_END]"


def wrap_hint(hint: str) -> str:
    """Wrap a hint in OpenClaw-RL canonical delimiters."""
    return f"{HINT_START}{hint}{HINT_END}"


def extract_hint_text(text: str) -> str | None:
    """Parse [HINT_START]...[HINT_END] from judge output."""
    import re
    m = re.search(
        r"\[HINT_START\](.*?)\[HINT_END\]", text, re.DOTALL
    )
    return m.group(1).strip() if m else None


class OPDJudge:
    """Wrapper that turns a next-state signal into a textual hint.

    Callers provide a ``judge_fn`` with signature:
        async judge_fn(response: str, next_state: str) -> str | None

    The function should return a hint string (wrapped or raw) or None if
    no useful hint can be extracted from the next state.

    Example rule-based judge for letter_counting:
        If next_state contains "Wrong answer", extract the correct count
        and format it as "The correct answer was <answer>N</answer>".
    """

    def __init__(self, judge_fn: Any) -> None:
        self._fn = judge_fn

    async def extract_hint(
        self, response: str, next_state: str
    ) -> str | None:
        raw = await self._fn(response, next_state)
        if raw is None:
            return None
        hint = extract_hint_text(raw) or raw.strip()
        return hint if hint else None


# ---------------------------------------------------------------------------
# OPD Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OPDConfig:
    """Configuration for OPD training.

    kl_coef: β_KL in L = L_pg + β_KL · L_KL.
    clip_eps: PPO clip for the policy gradient term (ε_low).
    clip_eps_high: asymmetric upper clip (ε_high = 0.28 in OpenClaw-RL paper).
    adv_diff_clip: bound on |log π_T - log π_old| per token (Eq. 6 in paper).
        OpenClaw-RL default = 1.0.  Prevents exploding OPD advantages when
        teacher and student diverge sharply.
    skip_missing_hints: if True, records without teacher_logprobs are skipped.
        If False, they fall back to a pure KL loss (no OPD term).
    kl_estimator: which Schulman estimator to use for KL(π_θ ‖ π_ref).
        Default ``"k3"`` matches GRPO/PPO/RLOO so a Hybrid configuration
        produces comparable, additive KL signal across branches. Prior
        versions hardcoded the (biased) ``"k2"`` estimator, which made
        adaptive-KL controllers see different magnitudes for OPD vs GRPO.
    """

    kl_coef: float = 0.02
    clip_eps: float = 0.2
    clip_eps_high: float = 0.28   # asymmetric upper clip (OpenClaw-RL §3.1)
    adv_diff_clip: float = 1.0    # OpenClaw-RL Eq.6 log-prob-diff clip
    skip_missing_hints: bool = False
    kl_estimator: Literal["k1", "k2", "k3"] = "k3"


# ---------------------------------------------------------------------------
# OPD Algorithm
# ---------------------------------------------------------------------------


class OPDAlgo(BaseAlgo):
    """On-Policy Distillation algorithm (OpenClaw-RL §3.2).

    Computes a token-level OPD advantage from pre-computed teacher logprobs
    and trains with a PPO-style clipped surrogate + KL regularisation.

    Record requirements:
        record.metadata["teacher_logprobs"] : list[float]  (required for OPD)
        record.metadata["opd_hint"]         : str          (optional, logging)
    """

    def __init__(self, cfg: OPDConfig | None = None) -> None:
        self.cfg = cfg or OPDConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
                loss=0.0, policy_loss=0.0, kl=0.0, entropy=0.0,
                mean_reward=0.0, mean_advantage=0.0, clip_frac=0.0,
                n_records=0,
                extra={"algo": "opd", "n_with_hints": 0},
            )

        # Split records that have teacher logprobs vs those that don't.
        with_hints: list[RolloutRecord] = []
        without_hints: list[RolloutRecord] = []
        for rec in records:
            if rec.metadata.get("teacher_logprobs"):
                with_hints.append(rec)
            elif not cfg.skip_missing_hints:
                without_hints.append(rec)

        total_loss = torch.zeros((), dtype=torch.float32)
        total_kl = 0.0
        clip_frac_sum = 0.0
        n_processed = 0

        # ── OPD loss for records with teacher logprobs ──────────────────
        cache_hit_opd = False
        if with_hints:
            opd_loss, kl, cf, n, cache_hit_opd = self._opd_loss(
                policy, ref_policy, with_hints, batch=batch,
            )
            total_loss = total_loss + opd_loss
            total_kl += kl
            clip_frac_sum += cf * n
            n_processed += n

        # ── KL-only loss for records without hints ───────────────────────
        if without_hints and ref_policy is not None:
            kl_loss, kl_v = self._kl_only_loss(
                policy, ref_policy, without_hints, batch=batch,
            )
            total_loss = total_loss + cfg.kl_coef * kl_loss
            total_kl += kl_v
            n_processed += len(without_hints)

        mean_r = sum(r.reward for r in records) / max(1, len(records))
        clip_frac = clip_frac_sum / max(1, n_processed)

        stats = AlgoUpdateStats(
            loss=float(total_loss.detach().item()),
            policy_loss=float(total_loss.detach().item()),
            kl=total_kl,
            entropy=0.0,
            mean_reward=float(mean_r),
            mean_advantage=0.0,
            clip_frac=clip_frac,
            n_records=len(records),
            extra={
                "algo": "opd",
                "n_with_hints": len(with_hints),
                "n_without_hints": len(without_hints),
                "shared_logprobs_cache_hit": 1.0 if cache_hit_opd else 0.0,
                "opd_adv_scale_mean": (
                    sum(float(r.metadata.get("opd_adv_scale", 1.0)) for r in with_hints)
                    / len(with_hints)
                    if with_hints
                    else 1.0
                ),
            },
        )
        return total_loss, stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _opd_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend | None,
        records: list[RolloutRecord],
        *,
        batch: RolloutBatch | None = None,
    ) -> tuple[torch.Tensor, float, float, int, bool]:
        """Compute OPD policy gradient + KL loss for records with hints.

        When ``batch.shared_new_logprobs`` covers ``records`` at temperature
        1.0, we reuse those cached differentiable rows instead of running a
        fresh ``policy.score_batch`` pass. The returned ``cache_hit`` flag
        is True iff the cache short-circuited the new-logprob forward.
        """
        cfg = self.cfg
        prompt_ids_list = [r.prompt_ids for r in records]
        resp_ids_list   = [r.response_ids for r in records]

        # New logprobs from current policy (or from the hybrid shared cache).
        cache_hit = False
        if (
            batch is not None
            and batch.has_shared_logprobs_for(records)
            and batch.shared_logprobs_temperature is not None
            and abs(float(batch.shared_logprobs_temperature) - 1.0) < 1e-9
        ):
            new_logp, mask = stack_cached_logprobs(
                batch.shared_new_logprobs, records  # type: ignore[arg-type]
            )
            cache_hit = True
        else:
            new_logp, mask = policy.score_batch(
                prompt_ids_list, resp_ids_list, temperature=1.0
            )
        B, T = new_logp.shape
        dtype = new_logp.dtype
        device = new_logp.device

        # Old logprobs (from rollout) — uses cached tensor when available.
        old_logp = torch.zeros(B, T, dtype=dtype, device=device)
        for i, rec in enumerate(records):
            R = int(mask[i].sum().item())
            if R > 0 and rec.old_logprobs:
                R_eff = min(R, len(rec.old_logprobs))
                old_logp[i, :R_eff] = old_logprobs_tensor(
                    rec, length=R_eff, dtype=dtype, device=device,
                )

        # Teacher logprobs (pre-computed by judge/teacher model)
        teacher_logp = torch.zeros(B, T, dtype=dtype, device=device)
        for i, rec in enumerate(records):
            t_lp = rec.metadata.get("teacher_logprobs") or []
            R = int(mask[i].sum().item())
            if R > 0 and t_lp:
                chunk = t_lp[-R:]
                teacher_logp[i, :len(chunk)] = torch.tensor(
                    chunk, dtype=dtype, device=device
                )

        # OPD token-level advantage: A_t = clip(log π_T - log π_old, ±adv_diff_clip)
        raw_adv = teacher_logp - old_logp
        adv = raw_adv.clamp(-cfg.adv_diff_clip, cfg.adv_diff_clip)

        # Capability-axis-aware OPD weighting (hermes extension beyond
        # OpenClaw-RL's single global w_opd). ``TeacherLogprobFiller`` may
        # stamp a per-record ``opd_adv_scale`` derived from the hint's target
        # capability axis; emphasise the directive signal on the axes a run
        # cares about. Absent the key the scale is a no-op (1.0).
        adv_scale = torch.tensor(
            [float(r.metadata.get("opd_adv_scale", 1.0)) for r in records],
            dtype=dtype, device=device,
        ).unsqueeze(1)
        if not torch.allclose(adv_scale, torch.ones_like(adv_scale)):
            adv = adv * adv_scale

        # PPO-style clipped surrogate with asymmetric clip (OpenClaw-RL §3.1)
        log_ratio = new_logp - old_logp
        ratio = torch.exp(log_ratio.clamp(-20, 20))
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps_high)
        pg = torch.max(pg_loss1, pg_loss2)
        m = mask.to(dtype)
        policy_loss = (pg * m).sum() / m.sum().clamp(min=1)

        clip_frac = float(
            ((ratio - 1.0).abs() > max(cfg.clip_eps, cfg.clip_eps_high))
            .float()
            .mean()
            .item()
        )

        # KL regularisation vs reference (uses the shared, configurable
        # estimator from ``algos.common.kl`` so OPD's KL is directly
        # comparable to GRPO/PPO/RLOO under the same ``kl_estimator``.
        # Default ``k3`` matches the rest of the algorithms in v1.0).
        kl_scalar = 0.0
        kl_loss = torch.zeros((), dtype=dtype, device=device)
        if ref_policy is not None and cfg.kl_coef > 0:
            with torch.no_grad():
                ref_logp, _ref_mask = ref_policy.score_batch(
                    prompt_ids_list, resp_ids_list, temperature=1.0
                )
            T2 = min(T, ref_logp.shape[1])
            kl_loss = kl_from_logprobs_batched(
                new_logp[:, :T2],
                ref_logp[:, :T2],
                mask[:, :T2],
                estimator=cfg.kl_estimator,
            )
            kl_scalar = float(kl_loss.detach().item())

        total = policy_loss + cfg.kl_coef * kl_loss
        return total, kl_scalar, clip_frac, len(records), cache_hit

    def _kl_only_loss(
        self,
        policy: LLMBackend,
        ref_policy: LLMBackend,
        records: list[RolloutRecord],
        *,
        batch: RolloutBatch | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Plain KL(π_θ ‖ π_ref) for records without teacher hints."""
        prompt_ids_list = [r.prompt_ids for r in records]
        resp_ids_list   = [r.response_ids for r in records]

        if (
            batch is not None
            and batch.has_shared_logprobs_for(records)
            and batch.shared_logprobs_temperature is not None
            and abs(float(batch.shared_logprobs_temperature) - 1.0) < 1e-9
        ):
            new_logp, mask = stack_cached_logprobs(
                batch.shared_new_logprobs, records  # type: ignore[arg-type]
            )
        else:
            new_logp, mask = policy.score_batch(
                prompt_ids_list, resp_ids_list, temperature=1.0
            )
        with torch.no_grad():
            ref_logp, _ = ref_policy.score_batch(
                prompt_ids_list, resp_ids_list, temperature=1.0
            )
        T = min(new_logp.shape[1], ref_logp.shape[1])
        kl_loss = kl_from_logprobs_batched(
            new_logp[:, :T],
            ref_logp[:, :T],
            mask[:, :T],
            estimator=self.cfg.kl_estimator,
        )
        return kl_loss, float(kl_loss.detach().item())
