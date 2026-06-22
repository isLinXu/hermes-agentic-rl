# CPU-only Docker image for hermes-agentic-rl (multi-stage).
#
# Build:
#   docker build -t hermes-agentic-rl:0.11 .
# Run (smoke GRPO):
#   docker run --rm hermes-agentic-rl:0.11 \
#       python -m hermes_agentic_rl.cli.main train-rl \
#       --config configs/echo_grpo_mvp.yaml --output /tmp/out
# Run (tests):
#   docker run --rm hermes-agentic-rl:0.11 pytest tests -q
#
# For local development with live-mounted source, use docker-compose.yml.

# ---------------------------------------------------------------------------
# Stage 1 — install Python dependencies into a virtualenv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

COPY pyproject.toml README.md ./
COPY hermes_agentic_rl ./hermes_agentic_rl

RUN pip install -e '.[rl,test,config]'

# ---------------------------------------------------------------------------
# Stage 2 — minimal runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /opt/hermes-agentic-rl

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY pyproject.toml README.md ./
COPY hermes_agentic_rl ./hermes_agentic_rl
COPY tests ./tests
COPY configs ./configs
COPY scripts ./scripts

CMD ["python", "-m", "hermes_agentic_rl.cli.main", "--help"]
