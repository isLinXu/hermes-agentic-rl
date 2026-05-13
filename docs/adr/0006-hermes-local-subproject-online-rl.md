# ADR 0006 — Hermes local subproject loading for online RL

Date: 2026-05-12
Status: Accepted

## Context

We want `hermes-agentic-rl` to train against the real upstream `hermes-agent`
codebase without forcing a pip install, while keeping the existing online
session pipeline (`rollout -> sidecar -> replay -> worker`) intact.

The upstream `hermes-agent-self-evolution` project also assumes a local
Hermes checkout and treats session traces as the durable artifact for future
prompt / skill / policy refinement.

## Decision

Load Hermes from, in order:

1. `runtime.repo_path`
2. `HERMES_AGENT_REPO`
3. a local child checkout at `subprojects/hermes-agent`

The runtime adapter injects that path before importing `run_agent`, and the
pipeline lets `build_agent_loop(config)` perform the decisive availability
check so config-driven local paths work.

Session logs and replay exports remain the canonical handoff between online
rollouts and downstream training / evolution workers.

## Consequences

**Pros:**
- Real Hermes runs work from a checked-out child repo.
- CI and local development can use the same config surface.
- Self-evolution style trace mining can reuse the same session artifacts.

**Cons:**
- Hermes availability is now config/path sensitive, so a generic
  `is_available()` probe is less informative than a full build attempt.
- Submodule checkout adds one more repo lifecycle step.

## Alternatives considered

- Require pip installation only. Rejected because it makes local iteration
  harder and hides the upstream source tree from the training loop.
- Vendor the Hermes code directly into this package. Rejected because we want
  a clean upstream boundary and easier rebasing.
