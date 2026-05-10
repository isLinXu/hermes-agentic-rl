"""Paired Welch-style t-test for comparing two eval reports.

We treat each rollout's reward as an independent observation. Welch's t-test
is robust to unequal variances. For paired data (same prompts evaluated
against both models), subtract per-prompt and use one-sample t; for
independent data, use two-sample Welch.

Both functions are stdlib-only (no scipy).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hermes_agentic_rl.eval.harness import EvalHarness, EvalReport


@dataclass(slots=True)
class ABResult:
    baseline: str
    candidate: str
    n: int
    mean_diff: float
    t: float
    approx_p: float   # two-sided, approximated via normal CDF
    winner: str       # "candidate" / "baseline" / "tie"


def paired_welch_t(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    alpha: float = 0.05,
) -> ABResult:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired samples must have same non-zero length")
    n = len(baseline)
    diffs = [c - b for b, c in zip(baseline, candidate, strict=False)]
    mean_d = sum(diffs) / n
    if n == 1:
        return ABResult("baseline", "candidate", n, mean_d, 0.0, 1.0, "tie")
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n) if var_d > 0 else 1e-12
    t = mean_d / se
    # Two-sided p via standard normal approximation (valid for n ≳ 30)
    p = 2.0 * (1.0 - _phi(abs(t)))
    if p < alpha and mean_d > 0:
        winner = "candidate"
    elif p < alpha and mean_d < 0:
        winner = "baseline"
    else:
        winner = "tie"
    return ABResult("baseline", "candidate", n, mean_d, t, p, winner)


def _phi(z: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def run_ab(
    baseline_harness: EvalHarness,
    candidate_harness: EvalHarness,
) -> tuple[EvalReport, EvalReport, ABResult]:
    """Run both harnesses (they should share the same env/seed_base so the
    comparison is paired) and return reports + the A/B result."""
    b = baseline_harness.run()
    c = candidate_harness.run()
    n = min(len(b.rewards), len(c.rewards))
    res = paired_welch_t(b.rewards[:n], c.rewards[:n])
    return b, c, res
