#!/usr/bin/env bash
set -euo pipefail

# This script acts as the automated entry point for the submission pipeline.
# Usage: ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Output the causal summary to the same directory as the predictions
SUMMARY_PATH="$(dirname "$OUTPUT_PATH")/causal_summary.txt"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "============================================================"
echo "AIgnition 2026: Probabilistic Revenue Forecaster Pipeline"
echo "============================================================"
echo "  Data Directory: $DATA_DIR"
echo "  Model Bundle:   $MODEL_PATH"
echo "  Predictions:    $OUTPUT_PATH"
echo "  Causal Summary: $SUMMARY_PATH"
echo "------------------------------------------------------------"

# 1. Generate probabilistic revenue and ROAS predictions
# Note: predict.py internally handles loading and feature engineering
# the raw data before applying the model bundle.
echo ">> Running prediction pipeline..."
python src/predict.py \
    --data-dir "$DATA_DIR" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

# 2. Generate the rule-based (offline) causal business summary
echo ">> Generating causal business summary..."
python src/llm_summary.py \
    --predictions "$OUTPUT_PATH" \
    --model "$MODEL_PATH" \
    --output "$SUMMARY_PATH"

echo "------------------------------------------------------------"
echo "Pipeline completed successfully!"
echo "Outputs generated:"
echo "  - $OUTPUT_PATH"
echo "  - $SUMMARY_PATH"
echo "============================================================"
