# ADR 0004 — K3 KL estimator as a configurable option (v0.6)

Date: 2026-05-10
Status: Accepted

## Context

Both GRPO and PPO use `β · KL(π_new ‖ π_ref)` as a regularizer. Historically
we estimated it as `mean(logπ_new − logπ_ref)` — Schulman's K1. K1 is:
- unbiased ✓
- simple ✓
- **can be negative** in finite samples ✗

A negative KL adds *positive* reward (loss decreases), incentivizing the
policy to drift *further* from the reference, which is the opposite of the
intent. In practice this shows up as KL values in `[-0.05, 0.1]` and
unstable training on sparse-reward tasks.

## Decision

Add `kl_estimator ∈ {k1, k2, k3}` to both `GRPOConfig` and `PPOConfig`.
Implement in `algos/common/kl.py`:

```
r = logπ_new - logπ_ref

k1:  mean(r)                      # legacy; unbiased, signed
k2:  mean(0.5 * r²)               # biased; always ≥ 0
k3:  mean(exp(-r) - 1 + r)        # unbiased AND always ≥ 0   ← recommended
```

All three are differentiable. K3 clamps `r` to ±20 to prevent `exp(-r)`
overflow when the policy drifts far from the reference (which shouldn't
happen with a well-tuned `kl_coef`, but we guard anyway).

Default is `k1` to preserve v0.5 numeric equivalence. New configs that
want the fix set `kl_estimator: k3`.

## Consequences

**Pros:**
- Non-negative KL removes the pathological "drift for free" incentive.
- Low variance means smaller `kl_coef` can still keep the policy anchored.
- Matches what TRL, verl, DeepSeek-R1 now use in production.

**Cons:**
- Slight compute overhead (an exponential per token vs. a subtraction).
- `k3` is mathematically unbiased but uses the K3 estimator's clip range.
  For extreme drifts the clamp hides issues — monitor the KL value.

**Reference:** John Schulman, "Approximating KL Divergence," 2020.
http://joschu.net/blog/kl-approx.html
