#!/bin/bash
# Evaluate v28 best checkpoint on HumanEval (164 problems).
# Contract: v28-humaneval-eval-v1.yaml (FALSIFY-V28-EVAL-001)
#
# Prerequisites:
#   - v28 training complete (or checkpoint exists)
#   - GPU free (training finished) OR use --device cpu (slow, ~24h)
#
# Usage:
#   bash scripts/eval-v28-humaneval.sh          # GPU (default, ~1.4h)
#   bash scripts/eval-v28-humaneval.sh cpu       # CPU (~24h)
#   bash scripts/eval-v28-humaneval.sh gx10      # Run on gx10 via SSH

set -euo pipefail

MODE="${1:-gpu}"
CHECKPOINT="checkpoints/albor-base-350m-v28/model-best.apr"
HUMANEVAL="data/humaneval.jsonl"
PORT=18080
RESULTS_DIR="results/v28-humaneval"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "=== HumanEval Evaluation (v28, $MODE) ==="

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Is v28 training complete?"
    exit 1
fi

# Verify HumanEval data
if [ ! -f "$HUMANEVAL" ]; then
    echo "ERROR: HumanEval data not found: $HUMANEVAL"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "Checkpoint: $CHECKPOINT ($(du -h $CHECKPOINT | cut -f1))"
echo "Problems: $(wc -l < $HUMANEVAL)"
echo "Results: $RESULTS_DIR/eval-$TIMESTAMP.json"
echo ""

case "$MODE" in
    gpu)
        echo "Starting inference server (GPU)..."
        bin/apr-train serve run "$CHECKPOINT" --port "$PORT" &
        SERVER_PID=$!
        sleep 10  # Wait for model to load

        echo "Running HumanEval eval..."
        python3 scripts/eval-code.py "$HUMANEVAL" \
            --api "http://localhost:$PORT" \
            --samples 1 \
            2>&1 | tee "$RESULTS_DIR/eval-$TIMESTAMP.log"

        kill $SERVER_PID 2>/dev/null || true
        ;;

    cpu)
        echo "Starting inference server (CPU, slow)..."
        bin/apr-train serve run "$CHECKPOINT" --port "$PORT" --device cpu &
        SERVER_PID=$!
        sleep 30  # CPU load is slower

        echo "Running HumanEval eval (ETA ~24h at ~10 tok/s)..."
        python3 scripts/eval-code.py "$HUMANEVAL" \
            --api "http://localhost:$PORT" \
            --samples 1 \
            2>&1 | tee "$RESULTS_DIR/eval-$TIMESTAMP.log"

        kill $SERVER_PID 2>/dev/null || true
        ;;

    gx10)
        echo "Copying checkpoint to gx10..."
        scp "$CHECKPOINT" gx10:~/src/albor/checkpoints/albor-base-350m-v28/model-best.apr

        echo "Running eval on gx10 (GPU)..."
        ssh gx10 "cd ~/src/albor && bash scripts/eval-v28-humaneval.sh gpu" \
            2>&1 | tee "$RESULTS_DIR/eval-$TIMESTAMP.log"

        echo "Copying results back..."
        scp "gx10:~/src/albor/results/v28-humaneval/eval-*.json" "$RESULTS_DIR/" 2>/dev/null || true
        ;;

    *)
        echo "Usage: $0 [gpu|cpu|gx10]"
        exit 1
        ;;
esac

echo ""
echo "=== Results ==="
if [ -f "$RESULTS_DIR/eval-$TIMESTAMP.log" ]; then
    grep -E "pass@|total|solved" "$RESULTS_DIR/eval-$TIMESTAMP.log" || echo "Check log for details"
fi
echo "Log: $RESULTS_DIR/eval-$TIMESTAMP.log"
