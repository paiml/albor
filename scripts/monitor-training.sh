#!/bin/bash
# Monitor active training run
# Usage: ./scripts/monitor-training.sh [checkpoint_dir]

CKPT_DIR="${1:-checkpoints/albor-base-350m}"
LOG="$CKPT_DIR/training.log"

echo "==========================================="
echo " Albor Training Monitor"
echo "==========================================="
echo ""

# Check for running training process
PROC=$(ps aux | grep "apr train" | grep -v grep)
if [ -n "$PROC" ]; then
    PID=$(echo "$PROC" | awk '{print $2}')
    CPU=$(echo "$PROC" | awk '{print $3}')
    MEM=$(echo "$PROC" | awk '{print $4}')
    TIME=$(echo "$PROC" | awk '{print $10}')
    echo "Process: PID $PID, CPU ${CPU}%, MEM ${MEM}%, elapsed $TIME"
else
    echo "Process: NOT RUNNING"
fi

echo ""

# GPU status
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>/dev/null | \
        awk -F', ' '{printf "  GPU: %s util, %s / %s VRAM, %s°C\n", $1, $2, $3, $4}'
else
    echo "  nvidia-smi not available"
fi

echo ""

# Training log
echo "--- Log ($LOG) ---"
if [ -f "$LOG" ]; then
    tail -15 "$LOG"
else
    echo "  No log file found"
fi

echo ""

# Checkpoints
echo "--- Checkpoints ---"
CKPTS=$(find "$CKPT_DIR" -name "*.safetensors" 2>/dev/null | sort)
if [ -n "$CKPTS" ]; then
    for f in $CKPTS; do
        SIZE=$(du -h "$f" | awk '{print $1}')
        echo "  $f ($SIZE)"
    done
else
    echo "  No checkpoints saved yet"
fi

echo ""

# Final model metadata
if [ -f "$CKPT_DIR/final_model.json" ]; then
    echo "--- Final Model ---"
    cat "$CKPT_DIR/final_model.json"
fi

echo ""
echo "==========================================="
