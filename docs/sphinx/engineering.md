# Engineering Workflow

The project treats engineering checks as part of the product. The following
commands should all pass before merging:

## Lint & Type Check

```bash
# Ruff (fast linter + formatter)
python -m ruff check hermes_agentic_rl tests scripts/

# Mypy (static type checking)
python -m mypy --ignore-missing-imports hermes_agentic_rl
```

## Tests

```bash
# Full suite with coverage
python -m pytest tests -q --cov=hermes_agentic_rl \
  --cov-report=term-missing --cov-report=xml

# Affected tests only (fast iteration)
python -m pytest tests/test_config_validation.py tests/test_quantized_backend.py -q
```

## Benchmarks

```bash
# Algorithm micro-benchmarks (correctness + timing)
python -m pytest tests/benchmarks/ -m benchmark -q

# Performance suite
python -m pytest hermes_agentic_rl/benchmarks/ -q
```

## Documentation

```bash
# Build Sphinx docs (warnings as errors)
sphinx-build -W --keep-going -b html docs/sphinx docs/sphinx/_build/html
```

## Security

```bash
# Lock file audit
pip-audit -r requirements-lock.txt
```

## CI

CI runs on Python 3.11 and 3.12 with the following jobs:

1. **lint** — ruff + mypy + compile + pip check
2. **test** — pytest with coverage + eval-gate smoke
3. **benchmark** — performance benchmarks with timing
4. **docs** — Sphinx build + link check
5. **security** — secret scan + pip-audit
6. **package** — build + twine check

Or use Makefile shortcuts: `make ci`, `make benchmark`, `make smoke`.
