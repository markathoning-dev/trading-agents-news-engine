#!/usr/bin/env bash
#
# serve.sh — Launch a vLLM inference server for the quantized market-news model.
#
# Usage:
#   ./scripts/serve.sh --model ./outputs/quantized --port 8000
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- defaults ----------
MODEL="./outputs/quantized"
PORT="8000"
HOST="0.0.0.0"
GPU_MEMORY_UTIL="0.90"
MAX_MODEL_LEN="8192"
DTYPE="auto"
TENSOR_PARALLEL_SIZE="1"

# ---------- parse flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";          shift 2 ;;
        --port)        PORT="$2";           shift 2 ;;
        --host)        HOST="$2";           shift 2 ;;
        --gpu-memory)  GPU_MEMORY_UTIL="$2"; shift 2 ;;
        --max-len)     MAX_MODEL_LEN="$2";   shift 2 ;;
        --dtype)       DTYPE="$2";          shift 2 ;;
        --tensor-parallel) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --)           shift; break ;;
        *)            break ;;
    esac
done

# ---------- environment ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

echo "=============================================="
echo "  vLLM Inference Server"
echo "  Model:       ${MODEL}"
echo "  Host:        ${HOST}:${PORT}"
echo "  GPU memory:  ${GPU_MEMORY_UTIL}"
echo "  Max tokens:  ${MAX_MODEL_LEN}"
echo "  Devices:     ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

cd "$REPO_DIR"

python -m market_news.inference.server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    "$@"