#!/bin/bash
# Launch logprob-steering self-distillation on Qwen3-30B-A3B
#
# Starts 3 processes:
#   1. Atropos trajectory API server (port 8000)
#   2. Tinker trainer + FastAPI inference server (port 8001)
#   3. LogpSteeringEnv worker
#
# Usage: bash run_logp_steering.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/python"

echo "============================================"
echo " Logprob-Steering Self-Distillation"
echo " Model: Qwen/Qwen3-30B-A3B"
echo " Steering: emojification"
echo " Steps: 75"
echo "============================================"
echo ""

# 1. Atropos trajectory API (port 8000)
echo "[1/3] Starting Atropos API server on port 8000..."
$VENV -m uvicorn atroposlib.api.server:app --host 0.0.0.0 --port 8000 &
ATROPOS_PID=$!
echo "  PID: $ATROPOS_PID"
sleep 2

# 2. Tinker trainer + inference server (port 8001)
echo "[2/3] Starting Tinker trainer + inference server on port 8001..."
cd "$SCRIPT_DIR"
$VENV launch_training.py --config configs/logp_steering_qwen3_30b.yaml &
TRAINER_PID=$!
echo "  PID: $TRAINER_PID"
sleep 5

# 3. LogpSteeringEnv (from tinker_atropos/environments/)
echo "[3/3] Starting LogpSteeringEnv..."
$VENV -m tinker_atropos.environments.logp_steering serve &
ENV_PID=$!
echo "  PID: $ENV_PID"

echo ""
echo "All processes started:"
echo "  Atropos API:  PID $ATROPOS_PID (port 8000)"
echo "  Tinker:       PID $TRAINER_PID (port 8001)"
echo "  Environment:  PID $ENV_PID"
echo ""
echo "Press Ctrl+C to stop all."

# Trap cleanup
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $ENV_PID $TRAINER_PID $ATROPOS_PID 2>/dev/null
    wait
    echo "Done."
}
trap cleanup EXIT INT TERM

# Wait for any to exit
wait
