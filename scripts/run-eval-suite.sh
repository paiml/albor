#!/bin/bash
# Run full model evaluation suite
#
# Orchestrates checkpoint validation, perplexity evaluation, and code
# completion benchmarks. Designed to be run after training completes.
#
# Usage:
#   ./scripts/run-eval-suite.sh checkpoints/albor-base-350m/
#   ./scripts/run-eval-suite.sh checkpoints/albor-base-50m/ --quick
#
# Prerequisites:
#   - Python venv with numpy, safetensors, pyarrow
#   - Pre-tokenized validation data in data/pretokenized-2048/val/
#   - Benchmark files in configs/eval/

set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir> [--quick]}"
QUICK="${2:-}"

PYTHON=".venv/bin/python"
EVAL_DIR="eval"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REPORT_DIR="${EVAL_DIR}/${TIMESTAMP}"

echo "==========================================================="
echo " Albor Model Evaluation Suite"
echo "==========================================================="
echo ""
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Report:     ${REPORT_DIR}"
echo "  Time:       $(date)"
echo ""

mkdir -p "${REPORT_DIR}"

# --- Step 1: Checkpoint Validation ---
echo "--- Step 1: Checkpoint Validation ---"
if ${PYTHON} scripts/eval-perplexity.py "${CHECKPOINT_DIR}" --validate-checkpoint 2>&1 | tee "${REPORT_DIR}/checkpoint-validation.txt"; then
    echo "  PASS: Checkpoint contains trained weights"
    CKPT_VALID=true
else
    echo "  FAIL: Checkpoint validation failed (ALB-038)"
    echo "  WARNING: Continuing evaluation but results will be meaningless"
    CKPT_VALID=false
fi
echo ""

# --- Step 2: Model Metadata ---
echo "--- Step 2: Model Metadata ---"
if [ -f "${CHECKPOINT_DIR}/config.json" ]; then
    cat "${CHECKPOINT_DIR}/config.json" | tee "${REPORT_DIR}/model-config.json"
else
    echo "  No config.json found"
fi

if [ -f "${CHECKPOINT_DIR}/final_model.json" ]; then
    echo ""
    echo "  Training metadata:"
    cat "${CHECKPOINT_DIR}/final_model.json" | tee "${REPORT_DIR}/training-metadata.json"
fi
echo ""

# --- Step 3: Checkpoint File Info ---
echo "--- Step 3: Checkpoint Files ---"
find "${CHECKPOINT_DIR}" -name "*.safetensors" -o -name "*.json" | sort | while read -r f; do
    SIZE=$(du -h "$f" | awk '{print $1}')
    SHA=$(sha256sum "$f" | cut -c1-16)
    echo "  ${f} (${SIZE}, sha256:${SHA}...)"
done | tee "${REPORT_DIR}/checkpoint-files.txt"
echo ""

# --- Step 4: Perplexity Evaluation ---
echo "--- Step 4: Perplexity Evaluation ---"
VAL_DATA="data/pretokenized-2048/val/val.parquet"
TRAIN_DATA="data/pretokenized-2048/train/train.parquet"

if [ -f "${VAL_DATA}" ]; then
    SEQ_LEN=2048
    MAX_SEQ=100
    if [ "${QUICK}" = "--quick" ]; then
        MAX_SEQ=10
        SEQ_LEN=128
    fi
    echo "  Evaluating on validation data (${MAX_SEQ} sequences, seq_len=${SEQ_LEN})..."
    ${PYTHON} scripts/eval-perplexity.py "${CHECKPOINT_DIR}" \
        --data "${VAL_DATA}" \
        --max-sequences "${MAX_SEQ}" \
        --seq-len "${SEQ_LEN}" \
        --threshold 30 2>&1 | tee "${REPORT_DIR}/perplexity-val.txt" || true
elif [ -f "${TRAIN_DATA}" ]; then
    echo "  No val data, evaluating on training data..."
    ${PYTHON} scripts/eval-perplexity.py "${CHECKPOINT_DIR}" \
        --data "${TRAIN_DATA}" \
        --max-sequences 50 \
        --seq-len 128 \
        --threshold 50 2>&1 | tee "${REPORT_DIR}/perplexity-train.txt" || true
else
    echo "  No evaluation data found. Skipping perplexity."
fi
echo ""

# --- Step 5: Code Completion Benchmarks ---
echo "--- Step 5: Code Completion Benchmarks ---"
echo "  (Validate canonical solutions — no model inference needed)"

for bench in configs/eval/*.jsonl; do
    NAME=$(basename "$bench" .jsonl)
    echo ""
    echo "  Benchmark: ${NAME}"
    ${PYTHON} scripts/eval-code.py "$bench" --validate-only 2>&1 | tee "${REPORT_DIR}/code-eval-${NAME}.txt" || true
done
echo ""

# --- Step 6: Summary Report ---
echo "==========================================================="
echo " Evaluation Summary"
echo "==========================================================="
echo ""
echo "  Checkpoint valid:  ${CKPT_VALID}"
echo "  Report saved to:   ${REPORT_DIR}/"
echo "  Files generated:   $(ls ${REPORT_DIR} | wc -l)"

if [ "${CKPT_VALID}" = "false" ]; then
    echo ""
    echo "  NOTE: Checkpoint failed validation (ALB-038)."
    echo "  All perplexity results are unreliable."
    echo "  See: https://github.com/paiml/albor/issues/36"
fi

echo ""
echo "  Report files:"
ls -la "${REPORT_DIR}/"
echo "==========================================================="
