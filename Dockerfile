# CPU-only Docker image for hermes-agentic-rl.
#
# Build:
#   docker build -t hermes-agentic-rl:0.6 .
# Run (smoke GRPO):
#   docker run --rm hermes-agentic-rl:0.6 \
#       python -m hermes_agentic_rl.cli.main train-rl \
#       --config configs/echo_grpo_mvp.yaml --output /tmp/out
# Run (tests):
#   docker run --rm hermes-agentic-rl:0.6 pytest tests -q
#
# Image target: <600MB compressed. Torch CPU wheel from pytorch.org.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/hermes-agentic-rl

# System deps: only what's needed for torch + tests. No CUDA, no build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in layers for better caching.
# 1) CPU torch first (largest, most stable).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

# 2) project metadata (only what's needed to resolve deps).
COPY pyproject.toml README.md ./

# 3) source tree.
COPY hermes_agentic_rl ./hermes_agentic_rl
COPY tests ./tests
COPY configs ./configs
COPY scripts ./scripts

# 4) install the package (editable mode for easy volume-mount overrides).
RUN pip install -e '.[test]'

# Default command: print the CLI help so the image is introspectable.
CMD ["python", "-m", "hermes_agentic_rl.cli.main", "--help"]
