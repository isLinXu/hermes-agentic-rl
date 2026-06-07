"""OPD teacher log-prob filler — closes the OpenClaw-RL §3.2 OPD loop.

Background
----------
``NextStatePRMComponent`` writes a *directive* hint into
``trajectory.metadata["runtime"]["rl"]["opd_hint"]`` whenever the next-state
PRM extracts one.  The OPD algorithm (``algos/opd.py``) then needs the
*teacher* log-probs

    log π_T(a_t | s + hint)      (Eq. 4)

to form the token-level directional advantage ``A_t = log π_T − log π_old``.

Until now nothing produced those teacher log-probs — ``next_state_prm.py``
only wrote an empty ``teacher_logprobs`` placeholder, so the OPD branch was a
no-op on real data and ``HybridAlgo`` silently degraded to plain GRPO.

This module fills that gap.  Following OpenClaw-RL §3.2 we use the *current
policy itself* as the teacher (self-distillation): we re-score the already
sampled response under a **hint-enhanced context** and write the resulting
per-token log-probs back onto each record.

Crucially the teacher pass:
  * runs under ``torch.no_grad()`` (the teacher is a target, not a learner), and
  * re-uses the rollout's ``response_ids`` verbatim so the teacher log-probs are
    token-aligned with ``old_logprobs`` / ``new_logprobs`` — otherwise the
    directional advantage is meaningless.

Capability-axis-aware weighting (the differentiator over OpenClaw-RL)
---------------------------------------------------------------------
hermes-agentic-rl already groups behaviour into ``capability_axes``
(tool_use_reliability, interaction_control, …).  When
``capability_axis_weights`` is configured we classify each hint into the most
likely axis (keyword match against ``OBJECTIVE_AXIS_KEYWORDS``) and stamp a
per-record ``opd_adv_scale`` multiplier.  ``OPDAlgo`` honours that scale, so the
directive signal can be emphasised on the axes a run cares about — something
OpenClaw-RL's single global OPD weight cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hermes_agentic_rl.algos.base import RolloutRecord
from hermes_agentic_rl.backends.base import LLMBackend
from hermes_agentic_rl.eval.capability_axes import OBJECTIVE_AXIS_KEYWORDS

# Canonical hint delimiter used by the PRM judge / OPD (mirrors algos/opd.py).
DEFAULT_HINT_TEMPLATE = "\n\n[HINT_START]{hint}[HINT_END]\n"


def classify_hint_axis(hint: str) -> str | None:
    """Return the capability axis a hint most likely targets, or None.

    Pure keyword scoring against ``OBJECTIVE_AXIS_KEYWORDS``. The axis with the
    most keyword hits wins; ties and zero-match hints return None so callers
    fall back to the neutral (1.0) weight.
    """
    if not hint:
        return None
    text = hint.lower()
    best_axis: str | None = None
    best_hits = 0
    for axis_name, keywords in OBJECTIVE_AXIS_KEYWORDS:
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_hits:
            best_hits = hits
            best_axis = axis_name
    return best_axis if best_hits > 0 else None


@dataclass(slots=True)
class TeacherFillConfig:
    """Configuration for :class:`TeacherLogprobFiller`.

    enabled: master switch. When False, ``fill`` is a no-op.
    hint_template: format string with a ``{hint}`` placeholder appended to the
        prompt to build the hint-enhanced teacher context.
    max_hint_tokens: hard cap on hint tokens so a verbose judge cannot blow up
        the teacher context length. <= 0 disables the cap.
    score_batch_size: chunk size for the (no-grad) teacher forward passes.
    capability_axis_weights: optional ``{axis_name: weight}`` map. When set,
        each hint is classified into an axis and the record receives
        ``metadata["opd_adv_scale"] = weight`` (default 1.0 for unmatched).
    default_axis_weight: multiplier for hints that match no configured axis.
    """

    enabled: bool = True
    hint_template: str = DEFAULT_HINT_TEMPLATE
    max_hint_tokens: int = 128
    score_batch_size: int = 16
    capability_axis_weights: dict[str, float] | None = None
    default_axis_weight: float = 1.0

    def __post_init__(self) -> None:
        if "{hint}" not in self.hint_template:
            raise ValueError("hint_template must contain a '{hint}' placeholder")


@dataclass(slots=True)
class TeacherFillStats:
    """Diagnostics returned by :meth:`TeacherLogprobFiller.fill`."""

    n_records: int = 0
    n_with_hint: int = 0
    n_filled: int = 0
    axis_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {
            "opd_teacher_n_records": float(self.n_records),
            "opd_teacher_n_with_hint": float(self.n_with_hint),
            "opd_teacher_n_filled": float(self.n_filled),
        }
        for axis, count in self.axis_counts.items():
            out[f"opd_teacher_axis/{axis}"] = float(count)
        return out


class TeacherLogprobFiller:
    """Fill ``teacher_logprobs`` (and optional ``opd_adv_scale``) onto records.

    Usage (in the trainer, after reward computation, before the update)::

        filler = TeacherLogprobFiller(policy, TeacherFillConfig())
        stats = filler.fill(batch_records)

    Only records carrying a non-empty ``metadata["opd_hint"]`` are touched.
    """

    def __init__(
        self,
        policy: LLMBackend,
        cfg: TeacherFillConfig | None = None,
    ) -> None:
        self.policy = policy
        self.cfg = cfg or TeacherFillConfig()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def fill(self, records: list[RolloutRecord]) -> TeacherFillStats:
        stats = TeacherFillStats(n_records=len(records))
        if not self.cfg.enabled or not records:
            return stats

        targets: list[RolloutRecord] = []
        aug_prompts: list[list[int]] = []
        for rec in records:
            hint = rec.metadata.get("opd_hint")
            if not isinstance(hint, str) or not hint.strip():
                continue
            stats.n_with_hint += 1
            aug_prompts.append(self._hint_enhanced_prompt(rec.prompt_ids, hint))
            targets.append(rec)
            self._maybe_stamp_axis_scale(rec, hint, stats)

        if not targets:
            return stats

        resp_ids = [rec.response_ids for rec in targets]
        # Teacher is a target distribution: no gradient should flow through it.
        with torch.no_grad():
            for start in range(0, len(targets), max(1, self.cfg.score_batch_size)):
                chunk = slice(start, start + max(1, self.cfg.score_batch_size))
                logp, mask = self.policy.score_batch(
                    aug_prompts[chunk], resp_ids[chunk], temperature=1.0
                )
                for j, rec in enumerate(targets[chunk]):
                    valid = int(mask[j].sum().item())
                    if valid <= 0:
                        # No alignable response tokens — drop the (now stale)
                        # placeholder so the OPD branch treats it as hint-less.
                        rec.metadata.pop("teacher_logprobs", None)
                        continue
                    row = logp[j, :valid].detach().to("cpu", dtype=torch.float32)
                    rec.metadata["teacher_logprobs"] = row.tolist()
                    stats.n_filled += 1
        return stats

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _hint_enhanced_prompt(self, prompt_ids: list[int], hint: str) -> list[int]:
        suffix = self.cfg.hint_template.format(hint=hint.strip())
        suffix_ids = self.policy.tokenizer.encode(suffix)
        if self.cfg.max_hint_tokens > 0 and len(suffix_ids) > self.cfg.max_hint_tokens:
            suffix_ids = suffix_ids[: self.cfg.max_hint_tokens]
        return list(prompt_ids) + list(suffix_ids)

    def _maybe_stamp_axis_scale(
        self,
        rec: RolloutRecord,
        hint: str,
        stats: TeacherFillStats,
    ) -> None:
        weights = self.cfg.capability_axis_weights
        if not weights:
            return
        axis = classify_hint_axis(hint)
        scale = (
            float(weights.get(axis, self.cfg.default_axis_weight))
            if axis is not None
            else float(self.cfg.default_axis_weight)
        )
        rec.metadata["opd_adv_scale"] = scale
        if axis is not None:
            rec.metadata["opd_hint_axis"] = axis
            stats.axis_counts[axis] = stats.axis_counts.get(axis, 0) + 1
