#!/bin/bash
# scripts/serve.sh — Launch KawaiiLLM inference server
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-output/kawaii_v1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
NUM_MEM_TOKENS="${NUM_MEM_TOKENS:-128}"

echo "Starting KawaiiLLM inference server..."
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Device:     ${DEVICE}"
echo "  Listen:     ${HOST}:${PORT}"

CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
DEVICE="${DEVICE}" \
NUM_MEM_TOKENS="${NUM_MEM_TOKENS}" \
python -m uvicorn src.inference.server:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1 \
  --log-level info
