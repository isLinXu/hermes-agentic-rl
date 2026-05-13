# Contributing

Thanks for helping harden `hermes-agentic-rl`.

## Local setup

```bash
python -m pip install -e '.[dev]'
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e '.[rl,data,metrics,test]'
```

## Quality gates

Run these before opening a PR:

```bash
python -m ruff check hermes_agentic_rl tests scripts/check_real_hermes.py
python -m mypy --follow-imports=skip hermes_agentic_rl
python -m pytest tests -q --cov=hermes_agentic_rl --cov-report=term-missing --cov-report=xml
pip-audit -r requirements-lock.txt
```

## Lock refresh

```bash
uv pip compile --universal pyproject.toml --extra dev --extra docs --output-file requirements-lock.txt
```

## Docs

```bash
sphinx-build -W --keep-going -b html docs/sphinx docs/sphinx/_build/html
```

## Notes

- Keep secrets in environment variables, not in configs.
- Prefer held-out `eval-rl` checks over training reward when judging progress.
- Avoid changing `subprojects/hermes-agent` unless the change is explicitly
  about the subproject itself.
