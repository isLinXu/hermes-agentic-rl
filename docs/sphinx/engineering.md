# Engineering Workflow

The project now treats engineering checks as part of the product:

1. `python -m ruff check hermes_agentic_rl tests scripts/check_real_hermes.py scripts/ci_eval_gate_smoke.py`
2. `python -m mypy --follow-imports=skip hermes_agentic_rl`
3. `python -m pytest tests -q --cov=hermes_agentic_rl --cov-report=term-missing --cov-report=xml`
4. `python scripts/ci_eval_gate_smoke.py`
5. `sphinx-build -W --keep-going -b html docs/sphinx docs/sphinx/_build/html`
6. `uv pip compile --universal pyproject.toml --extra dev --extra docs --output-file requirements-lock.txt`
7. `pip-audit -r requirements-lock.txt`

CI runs the same checks on Python 3.11 and 3.12. In addition to the pytest
suite, the workflow executes a standalone `eval-gate` smoke path so the
checkpoint-promotion CLI contract stays covered even when no real checkpoint
artifact is available.
