#!/usr/bin/env bash
#
# quantize.sh — 4-bit / 8-bit quantization of a trained model checkpoint.
#
# Uses bitsandbytes (4-bit NF4/FP4) or GPTQ via auto-gptq.
#
# Usage:
#   ./scripts/quantize.sh --model-dir ./outputs/train/final --output-dir ./outputs/quantized --bits 4
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- defaults ----------
MODEL_DIR=""
OUTPUT_DIR="./outputs/quantized"
BITS="4"               # 4 or 8
METHOD="bitsandbytes"  # bitsandbytes | gptq
DTYPE="float16"
CALIBRATION_DATA=""

# ---------- parse flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)   MODEL_DIR="$2";   shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --bits)        BITS="$2";        shift 2 ;;
        --method)      METHOD="$2";      shift 2 ;;
        --dtype)       DTYPE="$2";       shift 2 ;;
        --calibration) CALIBRATION_DATA="$2"; shift 2 ;;
        --)           shift; break ;;
        *)            break ;;
    esac
done

# ---------- validation ----------
if [[ -z "$MODEL_DIR" ]]; then
    echo "[ERROR] --model-dir is required."
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[ERROR] Model directory does not exist: $MODEL_DIR"
    exit 1
fi

# ---------- environment ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Post-training Quantization"
echo "  Model dir:    ${MODEL_DIR}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Bits:         ${BITS}-bit"
echo "  Method:       ${METHOD}"
echo "  Dtype:        ${DTYPE}"
echo "=============================================="

cd "$REPO_DIR"

python -m market_news.training.quantize \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --bits "$BITS" \
    --method "$METHOD" \
    --dtype "$DTYPE" \
    ${CALIBRATION_DATA:+--calibration-data "$CALIBRATION_DATA"} \
    "$@"