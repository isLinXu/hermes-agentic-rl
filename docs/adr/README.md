# Architecture Decision Records

An ADR captures a high-impact design decision and the context that led to
it. New ADRs should go here (`docs/adr/NNNN-short-slug.md`), numbered
sequentially. Use the template at the bottom.

| # | Title | Status | Since |
|---|-------|--------|-------|
| [0001](0001-backend-protocol-over-inheritance.md) | Backend protocol over class inheritance | Accepted | v0.2 |
| [0002](0002-grpo-ppo-shared-trainer.md) | Single on-policy trainer for GRPO and PPO | Accepted | v0.3 |
| [0003](0003-resumable-checkpoints.md) | Resumable checkpoints as a first-class feature | Accepted | v0.6 |
| [0004](0004-kl-k3-estimator.md) | K3 KL estimator as a configurable option | Accepted | v0.6 |
| [0005](0005-pluggable-metrics-writers.md) | Pluggable metrics writers with graceful degradation | Accepted | v0.6 |

---

## Template

```markdown
# ADR NNNN — <one-line title>

Date: YYYY-MM-DD
Status: Proposed | Accepted | Deprecated | Superseded by ADR NNNN

## Context

What is the problem / constraint / discovery that prompted a decision?
Include the state of the world *before* the decision.

## Decision

What did we decide? Be concrete — include module paths, config keys,
function signatures.

## Consequences

**Pros:** …
**Cons:** …
**Alternatives considered:** …
```
