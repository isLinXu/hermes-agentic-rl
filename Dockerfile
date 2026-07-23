# CPU-only Docker image for hermes-agentic-rl (multi-stage).
#
# Build:
#   docker build -t hermes-agentic-rl:0.12 .
# Run (smoke GRPO):
#   docker run --rm hermes-agentic-rl:0.12 \
#       python -m hermes_agentic_rl.cli.main train-rl \
#       --config configs/echo_grpo_mvp.yaml --output /tmp/out
# Run (tests):
#   docker run --rm hermes-agentic-rl:0.12 pytest tests -q
#
# For local development with live-mounted source, use docker-compose.yml.

LABEL org.opencontainers.image.title="hermes-agentic-rl"
LABEL org.opencontainers.image.version="0.12.0"
LABEL org.opencontainers.image.maintainer="gatilin"

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

RUN pip install -e '.[rl,test,config,data,metrics,hf]'

# ---------------------------------------------------------------------------
# Stage 2 — minimal runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="hermes-agentic-rl"
LABEL org.opencontainers.image.version="0.12.0"
LABEL org.opencontainers.image.maintainer="gatilin"

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

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hermes_agentic_rl; print('ok')" || exit 1

USER nobody

CMD ["python", "-m", "hermes_agentic_rl.cli.main", "--help"]
