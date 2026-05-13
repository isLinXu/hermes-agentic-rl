# ADR 0002 — Single on-policy trainer for GRPO and PPO

Date: 2026-05-08
Status: Accepted

## Context

GRPO and PPO share the entire control flow surrounding the loss computation:
env setup → per-iter rollout collection (group-shaped) → optimizer zero_grad →
`algo.compute_loss` → backward → grad clip → step → logger / metrics /
checkpoint. The only differences are:
- PPO requires a value-head backend (validation check)
- PPO uses GAE; GRPO uses group-normalized scalar advantage
- Different config fields (vf_coef, lam, gamma vs. loss_agg, advantage_norm)

The initial v0.2 implementation had `GRPOTrainer` owning the control flow.
Adding PPO in v0.3 forced a refactor or else duplication; we went with the
refactor.

## Decision

- `OnPolicyTrainer` owns the full lifecycle, parameterized by a `BaseAlgo`
  object implementing `compute_loss(policy, ref_policy, batch) -> (loss, stats)`.
- `GRPOTrainer` and `PPOTrainer` become thin constructors: take their own
  config dataclass, build the matching `Algo`, delegate to
  `OnPolicyTrainer.__init__`.
- Algo objects are **pure functions with a config dataclass** — no internal
  state between calls, no optimizer, no side effects. Unit-testable against
  hand-crafted `RolloutBatch`.

## Consequences

**Pros:**
- Resumable-checkpoint work (ADR 0003) needed to be written once in
  `OnPolicyTrainer`, benefiting both algos immediately.
- Lagrangian safety constraints, multi-turn mode, distributed rollout pool,
  metrics sink, auto-resume — all implemented once.
- Adding a 3rd algo (e.g., RLOO, DPO-RL) is ~100 lines.

**Cons:**
- Config duplication between `GRPOTrainerConfig` and `PPOTrainerConfig` for
  shared fields (n_iters, group_size, lr, …). We accept this for IDE
  completion and explicit per-algo validation.
- `PPO._validate_backend` has to guard against a GRPO-style Tiny backend
  lacking a value head — runtime error, not construction error.

**Counterfactual:** Keeping separate trainers means each new feature has to be
implemented twice — we know this because the v0.2 → v0.3 refactor actually
backed out a ~200-line duplication that had already grown.
