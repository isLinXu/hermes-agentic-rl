#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "[1/3] Sync root environment with integration extra"
env -u PYTHONHOME -u PYTHONPATH uv sync --extra integration

echo "[2/3] Initialize top-level integration submodules"
git submodule update --init subprojects/hermes-agent subprojects/atropos subprojects/tinker-atropos

echo "[3/3] Install local integration subprojects into the uv environment"
env -u PYTHONHOME -u PYTHONPATH uv pip install \
  -e subprojects/hermes-agent

# Keep Atropos / Tinker-Atropos importable without forcing heavyweight runtime
# extras (for example `vllm`) during a local preflight/bootstrap flow.
env -u PYTHONHOME -u PYTHONPATH uv pip install --no-deps \
  -e subprojects/atropos \
  -e subprojects/tinker-atropos

echo
echo "Integration environment ready."
echo "Recommended checks:"
echo "  env -u PYTHONHOME -u PYTHONPATH ./.venv/bin/python -m hermes_agentic_rl.cli.main hermes-preflight"
echo "  env -u PYTHONHOME -u PYTHONPATH ./.venv/bin/python -m hermes_agentic_rl.cli.main atropos-preflight"
echo "  env -u PYTHONHOME -u PYTHONPATH ./.venv/bin/python scripts/check_real_hermes.py"
