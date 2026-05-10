# Contributing to hermes-agentic-rl

## Dev setup

```bash
# editable install + dev tooling
make install-dev
# or, explicitly:
pip install -e '.[rl,dev]'
pre-commit install
```

## Workflow

```bash
make format     # ruff --fix + ruff format
make lint       # ruff check + format --check (matches CI)
make typecheck  # mypy (non-blocking)
make test       # full pytest
make ci         # all three (what CI runs)
make smoke      # end-to-end GRPO on tiny backend
```

Before pushing:
```bash
make ci
```

## Conventions

- **One behavior change per PR.** Refactors and feature additions should
  be separate commits; aggregate them only if they share a rationale.
- **Add tests for every behavior change.** The bar is a test that would
  *have caught the bug* — not a test that simply exercises the code.
- **No new runtime dependencies without an ADR.** Runtime deps stay at
  `PyYAML`; `torch` + `transformers` are opt-in extras.
- **Keep config defaults backward-compatible.** Each v0.N introduces new
  config fields default-off, so v0.(N-1) configs still run verbatim.
- **Record high-impact decisions as ADRs.** Copy the template in
  `docs/adr/README.md`, append the next number, update the index table.

## Tests

- Fast: `make test` runs the whole suite (~40s).
- Parallel: `make test-fast` (requires `pytest-xdist`).
- Filter: `pytest tests -k grpo -v`.
- Docker: `make docker-test` runs the suite inside the CPU image.

## Commit style

Prefer short imperative subject lines (≤72 chars):

```
trainers: add auto_resume and checkpoint_every to OnPolicyTrainer
algos/grpo: switch kl_coef regularizer to K3 estimator
docs: ADR 0005 — pluggable metrics writers
```

When a commit changes the CLI surface or config schema, note it in the
first body paragraph and update `docs/configuration.md` in the same PR.

## Release

`pyproject.toml` version bumps follow
[SemVer](https://semver.org/) with a `devN` suffix during pre-release:

- `0.6.0.dev0` → `0.6.0` (drop suffix when branch-cut)
- Always update `README.md` top matter and the `docs/adr/README.md` table
  in the same commit as the version bump.
