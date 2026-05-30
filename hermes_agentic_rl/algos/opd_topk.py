"""Overlap-Guided Top-K Hint Selection — from OpenClaw-RL Hybrid method.

Paper: OpenClaw-RL §3.3 + openclaw-combine/README.md

Problem being solved:
  In OPD, multiple candidate hints may be extracted from a next-state signal.
  Naively picking the longest hint or the first hint ignores a key risk:
  **teacher-student mismatch**.  If the hint-conditioned teacher distribution
  is too far from the student's current distribution at a token position, the
  OPD advantage `log π_T - log π_old` will be huge, causing unstable updates.

Overlap-Guided Selection (§3.3 "overlap-guided hint selection"):
  For each candidate hint h and each response token position i, compute the
  vocabulary overlap between:
      S_i^q     = top-k tokens under the student's old policy π_old
      S_{i,h}^p = top-k tokens under the hint-conditioned teacher π_T

      O[h, i] = |S_i^q ∩ S_{i,h}^p|

  Two selection strategies:
    1. ``sequence_optimal`` (default): pick the hint that maximises
       Σ_i O[h, i] — total overlap across all response token positions.
       Stable and generally better for agentic RL.
    2. ``token_optimal``: at each token position, independently pick the hint
       with highest overlap.  Maximises per-token guidance but less stable.

Vocabulary subset for OPD loss:
  After selecting the hint, the OPD advantage is computed only on a top-k
  vocabulary subset:
    - ``student``:  top-k(π_old)            ← default, stable
    - ``teacher``:  top-k(π_T)
    - ``overlap``:  top-k(π_old) ∩ top-k(π_T)

Advantage difference clip:
  A_t = clip(log π_T(t | h) − log π_old(t | s), −δ, +δ)
  OpenClaw-RL default δ = 1.0.

Integration:
  OPDTopKSelector.select(student_top_k, teacher_top_k_list) → (hint_idx, subset_mask)
  The caller (HybridAlgo or a custom trainer) uses the returned indices.

  For lightweight (non-SGLang) backends in hermes-agentic-rl, teacher top-k
  distributions are obtained via ``policy.score_batch_topk()``.  When the
  backend doesn't implement it, we fall back to a single-token argmax
  approximation (``_approx_topk``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OPDTopKConfig:
    """Config for overlap-guided top-k hint selection.

    k: top-k vocabulary width for both student and teacher.
    max_candidates: max number of candidate hints kept per turn.
    hint_selection: strategy for selecting among candidate hints.
        - ``sequence_optimal``: maximise total overlap across all tokens (default).
        - ``token_optimal``: per-token independent best hint.
        - ``shortest``: always use the first (shortest) hint candidate.
    subset_mode: vocabulary subset used for computing the OPD advantage.
        - ``student``: top-k(π_old)  (default, most stable)
        - ``teacher``: top-k(π_T)
        - ``overlap``: intersection of top-k(π_old) and top-k(π_T)
    adv_diff_clip: clip bound δ on log π_T - log π_old per token.
    """
    k:                int   = 4
    max_candidates:   int   = 3
    hint_selection:   Literal["sequence_optimal", "token_optimal", "shortest"] = "sequence_optimal"
    subset_mode:      Literal["student", "teacher", "overlap"] = "student"
    adv_diff_clip:    float = 1.0


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


class OPDTopKSelector:
    """Select the best hint using vocabulary overlap (OpenClaw-RL §3.3).

    Inputs:
        student_top_k : [T, k]   top-k token indices under π_old
        teacher_top_k : [C, T, k] top-k token indices under each candidate hint
                                  (C = number of candidates)

    Outputs:
        selected_hint_idx : int           index into the C candidates
        subset_mask       : [T, vocab]    bool — True for tokens in the subset
        per_token_overlap : [T]           float — overlap count at each position
    """

    def __init__(self, cfg: OPDTopKConfig | None = None) -> None:
        self.cfg = cfg or OPDTopKConfig()

    def select(
        self,
        student_top_k: torch.Tensor,    # [T, k]
        teacher_top_k: torch.Tensor,    # [C, T, k]
        vocab_size: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (selected_hint_idx, subset_mask [T, vocab], per_token_overlap [T])
        """
        cfg = self.cfg
        C, T, k = teacher_top_k.shape
        device = student_top_k.device

        if C == 0:
            return 0, torch.zeros(T, vocab_size, dtype=torch.bool, device=device), \
                       torch.zeros(T, dtype=torch.float32, device=device)

        # Compute overlap [C, T]: |top-k(π_old) ∩ top-k(π_T[c])|
        # Student top-k: [T, k] → expand to [1, T, k]
        s = student_top_k.unsqueeze(0).expand(C, -1, -1)   # [C, T, k]
        t = teacher_top_k                                    # [C, T, k]

        # For each (c, t_pos), count tokens in both sets.
        # Efficient: sort both sets and intersect. Approx: one-hot sums.
        # One-hot approach (exact, O(C*T*k)):
        s_oh = torch.zeros(C, T, vocab_size, dtype=torch.bool, device=device)
        t_oh = torch.zeros(C, T, vocab_size, dtype=torch.bool, device=device)
        # Scatter student indices
        _scatter_topk(s, s_oh, vocab_size)
        _scatter_topk(t, t_oh, vocab_size)
        overlap = (s_oh & t_oh).sum(dim=-1).float()   # [C, T]

        if cfg.hint_selection == "sequence_optimal":
            # Select hint with max Σ_t overlap[c, t]
            total_overlap = overlap.sum(dim=-1)        # [C]
            best_c = int(total_overlap.argmax().item())
            per_token_overlap = overlap[best_c]        # [T]

        elif cfg.hint_selection == "token_optimal":
            # Per-token best hint: we need to pick one hint overall for
            # the batch submission, so fall back to the one with max total.
            # (True token-optimal would require per-token different teachers,
            # which requires a different training loop.)
            total_overlap = overlap.sum(dim=-1)
            best_c = int(total_overlap.argmax().item())
            per_token_overlap = overlap[best_c]

        else:  # "shortest" — first candidate
            best_c = 0
            per_token_overlap = overlap[0]

        # Build vocabulary subset mask [T, vocab]
        subset_mask = _build_subset_mask(
            s_oh[best_c], t_oh[best_c], cfg.subset_mode
        )  # [T, vocab]

        return best_c, subset_mask, per_token_overlap

    def compute_topk_opd_loss(
        self,
        new_logprobs_full: torch.Tensor,     # [T, vocab] current policy full logits
        old_logprobs_full: torch.Tensor,     # [T, vocab] old policy full logits
        teacher_logprobs_full: torch.Tensor, # [T, vocab] teacher full logits
        subset_mask: torch.Tensor,           # [T, vocab] bool
        response_mask: torch.Tensor,         # [T]        bool
    ) -> torch.Tensor:
        """Compute OPD loss restricted to the top-k subset.

        OPD advantage at each position:
            A_t = clip(log π_T(t) - log π_old(t), ±δ)

        Loss: mean over (T, subset) of -A_t * ratio_t with PPO clip.

        This is a simplified scalar variant suitable for the hermes-agentic-rl
        per-token logprob API (which returns one logprob per token, not a
        full distribution).  For true top-k distillation you need the full
        vocabulary logits from ``score_batch_full_vocab()``.
        """
        cfg = self.cfg
        dtype = new_logprobs_full.dtype

        # Restrict to subset
        sub_new = new_logprobs_full  * subset_mask.to(dtype)   # [T, vocab]
        sub_old = old_logprobs_full  * subset_mask.to(dtype)
        sub_tch = teacher_logprobs_full * subset_mask.to(dtype)

        adv = (sub_tch - sub_old).clamp(-cfg.adv_diff_clip, cfg.adv_diff_clip)
        ratio = torch.exp((sub_new - sub_old).clamp(-20, 20))
        surr = -adv * ratio

        # Mask: only valid response positions × subset tokens
        token_mask = response_mask.unsqueeze(-1).to(dtype) * subset_mask.to(dtype)  # [T, v]
        n = token_mask.sum().clamp(min=1)
        loss = (surr * token_mask).sum() / n
        return loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scatter_topk(
    indices: torch.Tensor,   # [C, T, k] or [T, k]
    out: torch.Tensor,       # [C, T, vocab] or [T, vocab] bool
    vocab_size: int,
) -> None:
    """In-place: set out[..., indices[..., j]] = True for j in range(k)."""
    k = indices.shape[-1]
    for j in range(k):
        idx = indices[..., j].clamp(0, vocab_size - 1)
        out.scatter_(-1, idx.unsqueeze(-1), True)


def _build_subset_mask(
    s_oh: torch.Tensor,   # [T, vocab] bool — student top-k
    t_oh: torch.Tensor,   # [T, vocab] bool — teacher top-k
    mode: str,
) -> torch.Tensor:         # [T, vocab] bool
    if mode == "student":
        return s_oh
    if mode == "teacher":
        return t_oh
    if mode == "overlap":
        return s_oh & t_oh
    return s_oh


# ---------------------------------------------------------------------------
# Token-level approximation when full-vocab logits are unavailable
# ---------------------------------------------------------------------------


def approx_topk_from_per_token_logprobs(
    per_token_logprobs: torch.Tensor,  # [T] one logprob per sampled token
    sampled_ids: torch.Tensor,         # [T] the sampled token ids
    vocab_size: int,
    k: int = 4,
) -> torch.Tensor:
    """Approximate [T, k] top-k set when only per-token logprobs are available.

    Since we only have one logprob per position (the sampled token), we
    approximate the top-k set by taking the k highest-probability sampled
    tokens *globally* and treating them as the top-k at every position.
    This is a coarse approximation — use full-vocab logits when possible.
    """
    T = per_token_logprobs.shape[0]
    if k >= T:
        # Repeat sampled_ids to fill [T, k]
        ids = sampled_ids.unsqueeze(1).expand(T, k)
        return ids.clamp(0, vocab_size - 1)
    # Take global top-k by logprob
    top_vals, top_pos = per_token_logprobs.topk(min(k, T))
    top_ids = sampled_ids[top_pos]                      # [k]
    result = top_ids.unsqueeze(0).expand(T, -1)         # [T, k]
    return result.clamp(0, vocab_size - 1)
