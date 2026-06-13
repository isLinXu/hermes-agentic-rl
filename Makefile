.PHONY: help install install-dev test test-fast lint format typecheck typecheck-strict audit ci clean docker docker-test smoke

PYTHON ?= python
IMG ?= hermes-agentic-rl:0.6
STRICT_MYPY_TARGETS := \
	hermes_agentic_rl/collectors/replay_quality.py \
	hermes_agentic_rl/rewards/toolcall_reward.py \
	hermes_agentic_rl/rewards/next_turn_feedback.py

help:
	@echo "Common targets:"
	@echo "  make install       - pip install -e '.[rl,test]'"
	@echo "  make install-dev   - pip install -e '.[rl,dev]' + pre-commit install"
	@echo "  make test          - run full test suite (pytest)"
	@echo "  make test-fast     - run tests in parallel (needs pytest-xdist)"
	@echo "  make lint          - ruff check + ruff format --check"
	@echo "  make format        - ruff check --fix + ruff format"
	@echo "  make typecheck     - mypy (non-blocking in CI)"
	@echo "  make typecheck-strict - strict mypy for hardened core modules"
	@echo "  make audit         - pip-audit against requirements-lock.txt"
	@echo "  make ci            - lint + strict typecheck + test"
	@echo "  make smoke         - end-to-end smoke: echo GRPO on tiny backend"
	@echo "  make docker        - build Docker image ($(IMG))"
	@echo "  make docker-test   - run tests inside the Docker image"
	@echo "  make clean         - remove build artefacts and caches"

install:
	$(PYTHON) -m pip install -e '.[rl,test]'

install-dev:
	$(PYTHON) -m pip install -e '.[rl,dev]'
	pre-commit install

test:
	$(PYTHON) -m pytest tests -q

test-fast:
	$(PYTHON) -m pytest tests -q -n auto

lint:
	$(PYTHON) -m ruff check hermes_agentic_rl tests
	$(PYTHON) -m ruff format --check hermes_agentic_rl tests

format:
	$(PYTHON) -m ruff check hermes_agentic_rl tests --fix
	$(PYTHON) -m ruff format hermes_agentic_rl tests

typecheck:
	$(PYTHON) -m mypy hermes_agentic_rl || true

typecheck-strict:
	$(PYTHON) -m mypy --config-file=mypy-strict.ini $(STRICT_MYPY_TARGETS)

audit:
	$(PYTHON) -m pip_audit -r requirements-lock.txt

ci: lint typecheck-strict test

smoke:
	$(PYTHON) -m hermes_agentic_rl.cli.main train-rl \
		--config configs/echo_grpo_mvp.yaml \
		--output outputs/smoke_mvp

docker:
	docker build -t $(IMG) .

docker-test:
	docker run --rm $(IMG) pytest tests -q --maxfail=1

clean:
	rm -rf build dist *.egg-info hermes_agentic_rl.egg-info
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
