#!/usr/bin/env bash
#
# evaluate.sh — Run the full evaluation suite for a trained/quantized model.
#
# Metrics:
#   - Perplexity (on held-out financial news corpus)
#   - Sentiment classification accuracy
#   - Coherence / faithfulness scoring
#
# Usage:
#   ./scripts/evaluate.sh --model ./outputs/quantized --data ./data/eval.jsonl
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------- defaults ----------
MODEL="./outputs/quantized"
DATA="./data/eval.jsonl"
BATCH_SIZE="4"
OUTPUT_DIR="./outputs/evaluation"
METRICS="perplexity,sentiment,coherence"

# ---------- parse flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2";    shift 2 ;;
        --data)     DATA="$2";     shift 2 ;;
        --batch)    BATCH_SIZE="$2"; shift 2 ;;
        --output)   OUTPUT_DIR="$2"; shift 2 ;;
        --metrics)  METRICS="$2";  shift 2 ;;
        --)        shift; break ;;
        *)         break ;;
    esac
done

# ---------- validation ----------
if [[ ! -d "$MODEL" ]] && [[ ! -f "$MODEL" ]]; then
    echo "[WARN] Model path does not exist (yet?): $MODEL"
fi
if [[ ! -f "$DATA" ]]; then
    echo "[ERROR] Evaluation data not found: $DATA"
    exit 1
fi

# ---------- environment ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Model Evaluation"
echo "  Model:       ${MODEL}"
echo "  Data:        ${DATA}"
echo "  Metrics:     ${METRICS}"
echo "  Output:      ${OUTPUT_DIR}"
echo "=============================================="

cd "$REPO_DIR"

python -m market_news.evaluation.evaluator \
    --model "$MODEL" \
    --data "$DATA" \
    --batch-size "$BATCH_SIZE" \
    --metrics "$METRICS" \
    --output-dir "$OUTPUT_DIR" \
    "$@"