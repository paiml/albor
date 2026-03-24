#!/usr/bin/env bash
# eval-humaneval.sh — Run HumanEval evaluation on a checkpoint
#
# Usage:
#   ./scripts/eval-humaneval.sh checkpoints/albor-base-350m-v15/model-best.apr
#   ./scripts/eval-humaneval.sh checkpoints/albor-base-350m-v15/model-step-25000.apr --device cuda
#
# Requires: tokenizer.json symlinked in checkpoint directory
set -euo pipefail

APR="${APR:-/mnt/nvme-raid0/targets/aprender/release/apr}"
CHECKPOINT="${1:?Usage: $0 <checkpoint.apr> [--device cpu|cuda]}"
DEVICE="${2:---device cpu}"
SAMPLES="${SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"

# Ensure tokenizer exists in checkpoint dir
CKPT_DIR=$(dirname "$CHECKPOINT")
if [ ! -f "$CKPT_DIR/tokenizer.json" ]; then
    echo "Linking tokenizer to $CKPT_DIR/"
    ln -sf "$(pwd)/models/albor-tokenizer-v2/tokenizer.json" "$CKPT_DIR/tokenizer.json"
fi

echo "=== HumanEval Evaluation ==="
echo "  Checkpoint: $CHECKPOINT"
echo "  Device: $DEVICE"
echo "  Samples: $SAMPLES"
echo "  Temperature: $TEMPERATURE"
echo ""

$APR eval "$CHECKPOINT" \
    --dataset custom --data data/humaneval.jsonl \
    --task humaneval \
    --samples "$SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    $DEVICE \
    --json 2>&1 | tee "eval/humaneval-$(basename "$CHECKPOINT" .apr)-$(date +%Y%m%d).json"
