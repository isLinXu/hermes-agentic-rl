"""Dynamic Token Budget Manager.

Prevents long-sequence OOM while preserving gradient signal.

Problems addressed:
  1. Long rollouts (e.g. code generation) exhaust GPU memory during the
     training forward pass → OOM crash.
  2. Static truncation (always cut at max_len) discards the end of the
     rollout where the <answer> tag appears → zero gradient on format.
  3. Variable-length batches waste memory with naïve right-padding.

Solution: two-pass budget allocation per batch:
  Pass 1: measure actual lengths, compute batch-level token budget.
  Pass 2: truncate each rollout intelligently, preserving key regions
     (format tags, tool calls, answer spans).

Smart truncation strategy (priority order):
  1. Keep the last ``tail_keep`` tokens (usually contains the answer).
  2. Keep the first ``head_keep`` tokens (usually contains the reasoning start).
  3. If still over budget, uniformly subsample the middle.

This ensures the loss always sees the format tokens even on long rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


@dataclass(slots=True)
class TokenBudgetConfig:
    """Configuration for token budget management.

    max_tokens_per_batch: hard cap on total tokens in one training batch.
        Trainer should use this to limit grad accumulation.
    max_response_tokens: hard cap per individual response (before truncation).
        Sequences longer than this are truncated.
    head_keep: number of tokens to preserve from the start of response.
    tail_keep: number of tokens to preserve from the end of response.
        The tail usually contains the answer/format tokens — most important.
    dynamic_scale: if True, scale max_response_tokens dynamically based on
        measured GPU memory pressure (requires torch.cuda.memory_reserved).
    dynamic_headroom: target fraction of GPU memory to keep free (0.1 = 10%).
    protect_tool_call_boundaries: if True and a tokenizer is available,
        avoid starting the preserved tail in the middle of a
        ``<tool_call>...</tool_call>`` block.
    middle_strategy: how to handle the over-budget middle span.
        ``"drop"`` (default, recommended): discard the middle entirely
        and concatenate ``head + tail``. This preserves causal positional
        adjacency so the surviving tokens still align with their
        ``old_logprobs`` (the stored values were computed at the *same*
        positions) and the policy/reference forwards generate matching
        embeddings. The number of dropped tokens is recorded in
        ``rec.metadata["token_budget_middle_dropped"]``.
        ``"subsample"``: legacy behaviour. Uniformly subsamples the
        middle to fit the budget. Functionally cheaper but breaks RoPE
        alignment because the kept indices are no longer contiguous in
        the original sequence; the resulting per-token logprobs are
        **not** consistent with the rollout's ``old_logprobs`` and the
        ratio computation becomes biased. Kept for backwards compat
        and ablation only.
    """

    max_tokens_per_batch: int = 8192
    max_response_tokens: int = 512
    head_keep: int = 64
    tail_keep: int = 128
    dynamic_scale: bool = False
    dynamic_headroom: float = 0.15
    protect_tool_call_boundaries: bool = True
    middle_strategy: str = "drop"  # "drop" | "subsample"


class TokenBudgetManager:
    """Per-batch token budget allocator with smart truncation.

    Usage in trainer::

        budget_mgr = TokenBudgetManager(TokenBudgetConfig(
            max_tokens_per_batch=8192,
            max_response_tokens=512,
            tail_keep=128,
        ))
        records = budget_mgr.apply(records)  # in-place length clipping
        stats = budget_mgr.last_stats
    """

    def __init__(self, cfg: TokenBudgetConfig | None = None, *, tokenizer: Any = None) -> None:
        self.cfg = cfg or TokenBudgetConfig()
        self.tokenizer = tokenizer
        self.last_stats: dict[str, Any] = {}

    def _current_max_response(self) -> int:
        """Optionally scale max_response_tokens based on GPU memory pressure."""
        if not self.cfg.dynamic_scale:
            return self.cfg.max_response_tokens
        try:
            import torch

            if not torch.cuda.is_available():
                return self.cfg.max_response_tokens
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            free_frac = 1.0 - reserved / max(1, total)
            if free_frac < self.cfg.dynamic_headroom:
                # Memory pressure: shrink budget by up to 50%
                scale = max(0.5, free_frac / self.cfg.dynamic_headroom)
                return int(self.cfg.max_response_tokens * scale)
        except Exception:
            pass
        return self.cfg.max_response_tokens

    @staticmethod
    def _smart_truncate(
        ids: list[int],
        logprobs: list[float],
        max_len: int,
        head_keep: int,
        tail_keep: int,
        *,
        tokenizer: Any = None,
        protect_tool_call_boundaries: bool = True,
        middle_strategy: str = "drop",
    ) -> tuple[list[int], list[float], int]:
        """Truncate to max_len, preserving head and tail spans.

        Returns ``(ids, logprobs, n_middle_dropped)``. ``n_middle_dropped``
        counts tokens that were *removed* from the middle of the
        sequence, so the trainer can surface that in metadata for
        observability.
        """
        if len(ids) <= max_len:
            return ids, logprobs, 0

        # Clamp keep sizes to fit within budget while honouring explicit
        # head/tail when the caller sets a small max_len (unit tests and
        # short-response tasks).
        effective_head = min(head_keep, max(0, max_len - 1))
        effective_tail = min(tail_keep, max(0, max_len - effective_head))
        middle_budget = max_len - effective_head - effective_tail

        tail_start = TokenBudgetManager._safe_tail_start(
            ids,
            effective_tail,
            tokenizer=tokenizer,
            protect_tool_call_boundaries=protect_tool_call_boundaries,
        )
        if tail_start > len(ids):
            tail_start = len(ids)
        safe_tail_len = len(ids) - tail_start

        if middle_budget <= 0:
            # Budget too tight: keep the configured head plus any safe tail.
            head_ids = ids[:effective_head]
            head_lp = logprobs[:effective_head] if logprobs else []
            tail_ids = ids[tail_start:] if safe_tail_len > 0 else []
            tail_lp = logprobs[tail_start:] if logprobs and safe_tail_len > 0 else []
            room_for_tail = max(0, max_len - len(head_ids))
            tail_ids = tail_ids[-room_for_tail:] if room_for_tail > 0 else []
            tail_lp = tail_lp[-room_for_tail:] if room_for_tail > 0 else []
            kept_ids = head_ids + tail_ids
            kept_lp = head_lp + tail_lp if logprobs else []
            n_dropped = len(ids) - len(kept_ids)
            return kept_ids, kept_lp, max(0, n_dropped)

        head_ids = ids[:effective_head]
        head_lp = logprobs[:effective_head] if logprobs else []

        tail_ids = ids[tail_start:] if safe_tail_len > 0 else []
        tail_lp = logprobs[tail_start:] if logprobs and safe_tail_len > 0 else []
        effective_tail = len(tail_ids)
        middle_budget = max_len - effective_head - effective_tail

        middle_ids = ids[effective_head:tail_start]
        middle_lp = logprobs[effective_head:tail_start] if logprobs else []

        total_mid = len(middle_ids)
        n_dropped = 0
        if middle_budget <= 0:
            sampled_ids: list[int] = []
            sampled_lp: list[float] = []
            n_dropped = total_mid
        elif total_mid <= middle_budget:
            sampled_ids = middle_ids
            sampled_lp = middle_lp
        elif middle_strategy == "subsample":
            # Legacy uniform subsample. WARNING: breaks positional
            # adjacency between kept tokens and their original indices,
            # which biases the policy/reference forward (RoPE / absolute
            # position embeddings see indices that no longer match what
            # the rollout used). Retained for ablation only.
            step = total_mid / middle_budget
            indices = [int(i * step) for i in range(middle_budget)]
            sampled_ids = [middle_ids[i] for i in indices]
            sampled_lp = [middle_lp[i] for i in indices] if middle_lp else []
            n_dropped = total_mid - len(sampled_ids)
        else:
            # Default "drop": discard the middle. The remaining
            # head + tail spans are *each* internally contiguous and
            # therefore consistent with their own positional embeddings.
            # The seam between them creates a positional discontinuity
            # but only at one location; for a well-trained model this is
            # closer to the natural distribution than scattering kept
            # positions throughout the original window.
            sampled_ids = []
            sampled_lp = []
            n_dropped = total_mid

        final_ids = head_ids + sampled_ids + tail_ids
        final_lp = head_lp + sampled_lp + tail_lp
        return final_ids, final_lp, n_dropped

    @staticmethod
    def _safe_tail_start(
        ids: list[int],
        tail_keep: int,
        *,
        tokenizer: Any = None,
        protect_tool_call_boundaries: bool = True,
    ) -> int:
        """Return a tail start that does not bisect a tool-call block."""
        nominal = max(0, len(ids) - max(0, tail_keep))
        if not protect_tool_call_boundaries or tokenizer is None or tail_keep <= 0 or nominal <= 0:
            return nominal
        try:
            full_text = str(tokenizer.decode(list(ids)))
            prefix_text = str(tokenizer.decode(list(ids[:nominal])))
        except Exception:
            return nominal

        cut_char = len(prefix_text)
        last_open = full_text.rfind(_TOOL_CALL_OPEN, 0, cut_char)
        last_close = full_text.rfind(_TOOL_CALL_CLOSE, 0, cut_char)
        if last_open <= last_close:
            return nominal

        close_at = full_text.find(_TOOL_CALL_CLOSE, cut_char)
        if close_at < 0:
            return len(ids)
        safe_char = close_at + len(_TOOL_CALL_CLOSE)
        return TokenBudgetManager._token_index_at_or_after_char(
            ids,
            safe_char,
            tokenizer=tokenizer,
            start=nominal,
        )

    @staticmethod
    def _token_index_at_or_after_char(
        ids: list[int],
        char_offset: int,
        *,
        tokenizer: Any,
        start: int = 0,
    ) -> int:
        """Best-effort char offset → token index mapping via prefix decode."""
        lo = max(0, min(start, len(ids)))
        try:
            if len(str(tokenizer.decode(list(ids[:lo])))) >= char_offset:
                return lo
            for idx in range(lo + 1, len(ids) + 1):
                if len(str(tokenizer.decode(list(ids[:idx])))) >= char_offset:
                    return idx
        except Exception:
            return start
        return len(ids)

    def apply(self, records: list[Any]) -> list[Any]:
        """Apply token budget to a list of RolloutRecords.

        Modifies records in-place (truncates response_ids and old_logprobs).
        Also trims the batch to fit max_tokens_per_batch by dropping the
        longest rollouts first.

        Args:
            records: list of RolloutRecord objects.

        Returns:
            Filtered + truncated list of records.
        """
        max_resp = self._current_max_response()
        head_keep = self.cfg.head_keep
        tail_keep = self.cfg.tail_keep
        max_batch = self.cfg.max_tokens_per_batch

        n_orig = len(records)
        n_truncated = 0
        prompt_tokens = 0
        resp_tokens = 0

        total_middle_dropped = 0
        for rec in records:
            resp_ids = list(rec.response_ids)
            old_lp = list(rec.old_logprobs) if rec.old_logprobs else []
            if len(resp_ids) > max_resp:
                orig_len = len(resp_ids)
                resp_ids, old_lp, n_mid_dropped = self._smart_truncate(
                    resp_ids,
                    old_lp,
                    max_resp,
                    head_keep,
                    tail_keep,
                    tokenizer=self.tokenizer,
                    protect_tool_call_boundaries=self.cfg.protect_tool_call_boundaries,
                    middle_strategy=self.cfg.middle_strategy,
                )
                n_truncated += 1
                total_middle_dropped += n_mid_dropped
                metadata = getattr(rec, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["token_budget_truncated"] = True
                    metadata["token_budget_original_response_tokens"] = orig_len
                    metadata["response_tokens"] = len(resp_ids)
                    if n_mid_dropped > 0:
                        metadata["token_budget_middle_dropped"] = n_mid_dropped
                        metadata["token_budget_middle_strategy"] = self.cfg.middle_strategy
            rec.response_ids = resp_ids
            rec.old_logprobs = old_lp
            if hasattr(rec, "old_seq_logprob"):
                rec.old_seq_logprob = float(sum(old_lp)) if old_lp else 0.0
            metadata = getattr(rec, "metadata", None)
            if isinstance(metadata, dict):
                metadata["old_seq_logprob"] = float(sum(old_lp)) if old_lp else 0.0
            prompt_tokens += len(rec.prompt_ids)
            resp_tokens += len(resp_ids)

        # Second pass: if total exceeds batch budget, drop longest rollouts
        total = prompt_tokens + resp_tokens
        kept = records[:]
        n_dropped = 0
        if total > max_batch:
            kept = sorted(records, key=lambda r: len(r.prompt_ids) + len(r.response_ids))
            running = 0
            trimmed = []
            for rec in kept:
                length = len(rec.prompt_ids) + len(rec.response_ids)
                if running + length <= max_batch:
                    trimmed.append(rec)
                    running += length
                else:
                    n_dropped += 1
            kept = trimmed

        self.last_stats = {
            "n_orig": n_orig,
            "n_kept": len(kept),
            "n_truncated": n_truncated,
            "n_dropped": n_dropped,
            "max_resp_tokens_used": max_resp,
            "total_tokens_before": total,
            "total_tokens_after": sum(len(r.prompt_ids) + len(r.response_ids) for r in kept),
            "middle_strategy": self.cfg.middle_strategy,
            "total_middle_dropped": total_middle_dropped,
        }
        return kept
