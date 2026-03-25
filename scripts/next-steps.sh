#!/usr/bin/env bash
# next-steps.sh — What to run when the GPU is free
#
# Run this after v15 training finishes or is killed.
# It executes the distillation pipeline in order.
set -euo pipefail

APR="/mnt/nvme-raid0/targets/aprender/release/apr"

echo "=== Step 1: Distill v9 (ppl=129) with existing mixed-v3 data ==="
echo "  This takes ~30 min on RTX 4090"
echo "  Config: configs/train/distill-student-v3.yaml"
echo ""
echo "  $APR train apply --task pretrain --config configs/train/distill-student-v3.yaml"
echo ""

echo "=== Step 2: Evaluate distilled model ==="
echo "  ./scripts/eval-humaneval.sh checkpoints/albor-distill-v3/model-best.apr"
echo ""

echo "=== Step 3: If HumanEval > 0%, proceed to distill v15 ==="
echo "  $APR train apply --task pretrain --config configs/train/distill-student-v4.yaml"
echo "  (requires generating textbook data first: ./scripts/generate-textbook-data.sh)"
echo ""

echo "=== Step 4: If v15 converged better, distill from v15 ==="
echo "  Update distill-student-v4.yaml model.path to v15 model-best.apr"
echo ""

echo "=== Quick start (run this): ==="
echo "  $APR train apply --task pretrain --config configs/train/distill-student-v3.yaml 2>&1 | tee logs/distill-v3-retry.log"
