"""Similarity-based reward comparing generated responses to reference responses."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from hermes_agentic_rl.core.types import RewardResult, Trajectory
from hermes_agentic_rl.rewards.base import BaseReward

# ---------------------------------------------------------------------------
# Scorer helpers (pure-Python, zero heavy dependencies)
# ---------------------------------------------------------------------------


def _default_tokenizer(text: str) -> list[str]:
    return text.strip().split()


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Return a Counter of n-gram tuples from a token list."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Standard DP Longest Common Subsequence length for two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use two 1-D rows to keep O(min(m, n)) memory.
    if n > m:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


class RougeScorer:
    """Lightweight ROUGE scorer (ROUGE-1, ROUGE-2, ROUGE-L)."""

    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None) -> None:
        self.tokenizer = tokenizer or _default_tokenizer

    def _tokenize(self, text: str) -> list[str]:
        return self.tokenizer(text)

    def rouge_1(self, generated: str, reference: str) -> float:
        """Unigram overlap F1."""
        gen_tokens = self._tokenize(generated)
        ref_tokens = self._tokenize(reference)
        if not gen_tokens or not ref_tokens:
            return 0.0
        gen_counts = Counter(gen_tokens)
        ref_counts = Counter(ref_tokens)
        overlap = sum((gen_counts & ref_counts).values())
        precision = overlap / len(gen_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def rouge_2(self, generated: str, reference: str) -> float:
        """Bigram overlap F1."""
        gen_tokens = self._tokenize(generated)
        ref_tokens = self._tokenize(reference)
        if len(gen_tokens) < 2 or len(ref_tokens) < 2:
            return 0.0
        gen_counts = _get_ngrams(gen_tokens, 2)
        ref_counts = _get_ngrams(ref_tokens, 2)
        overlap = sum((gen_counts & ref_counts).values())
        precision = overlap / max(1, sum(gen_counts.values()))
        recall = overlap / max(1, sum(ref_counts.values()))
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def rouge_l(self, generated: str, reference: str) -> float:
        """LCS-based F1 (ROUGE-L)."""
        gen_tokens = self._tokenize(generated)
        ref_tokens = self._tokenize(reference)
        if not gen_tokens or not ref_tokens:
            return 0.0
        lcs = _lcs_length(gen_tokens, ref_tokens)
        if lcs == 0:
            return 0.0
        precision = lcs / len(gen_tokens)
        recall = lcs / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def score(self, metric: str, generated: str, reference: str) -> float:
        method = getattr(self, metric, None)
        if method is None:
            raise ValueError(f"Unknown ROUGE metric: {metric!r}")
        return method(generated, reference)


class BleuScorer:
    """Lightweight BLEU scorer (up to 4-grams) with brevity penalty."""

    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None, max_n: int = 4) -> None:
        self.tokenizer = tokenizer or _default_tokenizer
        self.max_n = max_n

    def score(self, generated: str, reference: str) -> float:
        cand = self.tokenizer(generated)
        ref = self.tokenizer(reference)
        if not cand or not ref:
            return 0.0

        # Brevity penalty
        c_len, r_len = len(cand), len(ref)
        if c_len > r_len:
            bp = 1.0
        elif c_len == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - r_len / c_len)

        precisions: list[float] = []
        for n in range(1, self.max_n + 1):
            if len(cand) < n:
                continue
            cand_ngrams = _get_ngrams(cand, n)
            ref_ngrams = _get_ngrams(ref, n)
            clipped = sum((cand_ngrams & ref_ngrams).values())
            total = sum(cand_ngrams.values())
            if total > 0:
                precisions.append(clipped / total)

        if not precisions:
            return 0.0

        # Geometric mean with simple smoothing for zeros
        log_sum = 0.0
        for p in precisions:
            log_sum += math.log(p) if p > 0 else math.log(1e-10)
        geo_mean = math.exp(log_sum / len(precisions))
        return max(0.0, min(1.0, bp * geo_mean))


class ExactMatchScorer:
    """Normalized exact-match scorer."""

    @staticmethod
    def score(generated: str, reference: str) -> float:
        return 1.0 if generated.strip().lower() == reference.strip().lower() else 0.0


class TokenOverlapScorer:
    """Jaccard similarity over token sets."""

    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None) -> None:
        self.tokenizer = tokenizer or _default_tokenizer

    def score(self, generated: str, reference: str) -> float:
        gen_tokens = set(self.tokenizer(generated))
        ref_tokens = set(self.tokenizer(reference))
        if not gen_tokens and not ref_tokens:
            return 1.0  # both empty → identical
        intersection = gen_tokens & ref_tokens
        union = gen_tokens | ref_tokens
        if not union:
            return 0.0
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Reward classes
# ---------------------------------------------------------------------------


class SimilarityReward(BaseReward):
    """Reward computed as similarity between generated and reference text.

    Supported metrics:
      - ``rouge_l``  – ROUGE-L (LCS-based F1)
      - ``rouge_1``  – unigram overlap F1
      - ``rouge_2``  – bigram overlap F1
      - ``bleu``     – BLEU-4 with brevity penalty
      - ``exact_match`` – case-insensitive, whitespace-stripped exact match
      - ``token_overlap`` – Jaccard similarity of token sets
    """

    name = "similarity_reward"

    _VALID_METRICS = frozenset(
        {"rouge_l", "rouge_1", "rouge_2", "bleu", "exact_match", "token_overlap"}
    )

    def __init__(
        self,
        weight: float = 1.0,
        metric: str = "rouge_l",
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.weight = weight
        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Choose from {sorted(self._VALID_METRICS)}."
            )
        self.metric = metric
        self.tokenizer = tokenizer or _default_tokenizer

        # Initialise scorers once.
        self._rouge = RougeScorer(tokenizer=self.tokenizer)
        self._bleu = BleuScorer(tokenizer=self.tokenizer)
        self._exact = ExactMatchScorer()
        self._overlap = TokenOverlapScorer(tokenizer=self.tokenizer)

    def _score(self, generated: str, reference: str) -> float:
        if self.metric.startswith("rouge"):
            return self._rouge.score(self.metric, generated, reference)
        if self.metric == "bleu":
            return self._bleu.score(generated, reference)
        if self.metric == "exact_match":
            return self._exact.score(generated, reference)
        if self.metric == "token_overlap":
            return self._overlap.score(generated, reference)
        return 0.0  # unreachable because of validation in __init__

    def compute(self, item: dict[str, Any], trajectory: Trajectory, **kwargs: Any) -> RewardResult:
        """Synchronous scoring entry-point."""
        reference = item.get("reference_response", "")
        generated = trajectory.final_output or ""

        if not reference:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no reference_response provided in item",
                weight=self.weight,
                metadata={"metric": self.metric, "generated": generated, "reference": reference},
            )

        if not generated:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="trajectory produced empty final_output",
                weight=self.weight,
                metadata={"metric": self.metric, "generated": generated, "reference": reference},
            )

        score = float(self._score(generated, reference))
        score = max(0.0, min(1.0, score))

        return RewardResult(
            name=self.name,
            score=score,
            reason=f"{self.metric} similarity = {score:.4f}",
            weight=self.weight,
            metadata={"metric": self.metric, "generated": generated, "reference": reference},
        )

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        """Async entry-point required by ``BaseReward``."""
        return self.compute(item, trajectory)


class CompositeSimilarityReward(BaseReward):
    """Weighted combination of multiple similarity metrics.

    Example::

        reward = CompositeSimilarityReward(
            metrics=[
                ("rouge_l", 0.5),
                ("bleu", 0.3),
                ("exact_match", 0.2),
            ]
        )
    """

    name = "composite_similarity_reward"

    def __init__(
        self,
        weight: float = 1.0,
        metrics: list[tuple[str, float]] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.weight = weight
        if metrics is None:
            metrics = [("rouge_l", 0.5), ("bleu", 0.3), ("exact_match", 0.2)]
        if not metrics:
            raise ValueError("metrics list must not be empty")

        # Build sub-rewards and normalise weights.
        self._rewards: list[tuple[SimilarityReward, float]] = []
        raw_weights = [w for _, w in metrics]
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("sum of metric weights must be > 0")

        for metric_name, w in metrics:
            reward = SimilarityReward(weight=1.0, metric=metric_name, tokenizer=tokenizer)
            self._rewards.append((reward, w / total))

    def compute(self, item: dict[str, Any], trajectory: Trajectory, **kwargs: Any) -> RewardResult:
        """Synchronous scoring entry-point."""
        reference = item.get("reference_response", "")
        generated = trajectory.final_output or ""

        if not reference:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="no reference_response provided in item",
                weight=self.weight,
                metadata={"generated": generated, "reference": reference, "components": []},
            )

        if not generated:
            return RewardResult(
                name=self.name,
                score=0.0,
                reason="trajectory produced empty final_output",
                weight=self.weight,
                metadata={"generated": generated, "reference": reference, "components": []},
            )

        weighted_sum = 0.0
        components: list[dict[str, Any]] = []
        for reward, norm_w in self._rewards:
            result = reward.compute(item, trajectory)
            weighted_sum += result.score * norm_w
            components.append(
                {
                    "metric": reward.metric,
                    "score": result.score,
                    "weight": norm_w,
                }
            )

        final_score = max(0.0, min(1.0, float(weighted_sum)))

        return RewardResult(
            name=self.name,
            score=final_score,
            reason=f"composite similarity = {final_score:.4f} over {len(components)} metrics",
            weight=self.weight,
            metadata={
                "generated": generated,
                "reference": reference,
                "components": components,
            },
        )

    async def evaluate(
        self,
        item: dict[str, Any],
        trajectory: Trajectory,
        tool_context: Any,
    ) -> RewardResult:
        """Async entry-point required by ``BaseReward``."""
        return self.compute(item, trajectory)
