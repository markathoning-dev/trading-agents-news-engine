#!/usr/bin/env bash
#
# train.sh — Single-command QLoRA training for market-news Gemma models.
#
# Usage:
#   ./scripts/train.sh --model google/gemma-4-12b-it --dataset market_news/data/financial_news.jsonl
#
# Optional flags are passed through to the trainer trainer.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- defaults ----------
MODEL="google/gemma-4-12b-it"
DATASET=""
OUTPUT_DIR="./outputs/train"
QUANTIZE=""            # empty → no quant; "4bit" or "8bit"

# ---------- parse long flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --quantize)   QUANTIZE="$2";   shift 2 ;;
        --)           shift; break ;;
        *)            break ;;  # pass remaining to python
    esac
done

# ---------- validation ----------
if [[ -z "$DATASET" ]]; then
    echo "[ERROR] --dataset is required."
    echo "Usage: $0 --model <name> --dataset <path> [--output-dir <dir>] [--quantize 4bit|8bit]"
    exit 1
fi

# ---------- environment ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  QLoRA Training"
echo "  Model:       ${MODEL}"
echo "  Dataset:     ${DATASET}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Quantize:    ${QUANTIZE:-None}"
echo "  Device(s):   ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

cd "$REPO_DIR"

python -m market_news.training.trainer \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    ${QUANTIZE:+--quantize "$QUANTIZE"} \
    "$@"