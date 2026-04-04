#!/bin/bash
# Resume teacher completions — runs directly on gx10 (no tunnel needed).
# Contract: teacher-completions-pipeline-v1.yaml (C-TEACHER-*)
# Issue: paiml/albor#117 (ALB-136)
#
# Prerequisites:
#   - gx10 running: realizar serve --model ~/models/Qwen3-8B-Q4_K_M.gguf --gpu --openai-api --port 8091
#   - SSH access to gx10
#   - albor repo synced on gx10 (or script + data copied)
#
# Usage:
#   bash scripts/resume-teacher-completions.sh              # Run on gx10 (default)
#   bash scripts/resume-teacher-completions.sh local 8091   # Run locally via tunnel
#   bash scripts/resume-teacher-completions.sh local 18091  # Run locally, tunnel already up

set -euo pipefail

MODE="${1:-gx10}"
PORT="${2:-8091}"
LIMIT="${3:-1000}"

PROMPTS="data/distill/prompts-filtered-50k.jsonl"
OUTPUT="data/distill/completions-1k.jsonl"

echo "=== Teacher Completions Pipeline (Resume) ==="
echo "Mode: $MODE, Port: $PORT, Limit: $LIMIT"

case "$MODE" in
    gx10)
        # Sync script + data to gx10, run there directly (avoids tunnel issues)
        echo "Syncing to gx10..."
        rsync -az scripts/generate_teacher_completions_api.py gx10:~/src/albor/scripts/
        rsync -az "$PROMPTS" "gx10:~/src/albor/$PROMPTS"
        rsync -az "$OUTPUT" "gx10:~/src/albor/$OUTPUT" 2>/dev/null || true

        # Validate server on gx10
        HEALTH=$(ssh gx10 "curl -s http://localhost:$PORT/health" 2>/dev/null)
        echo "Server: $HEALTH"

        # Validate completions actually work (not just health)
        echo "Testing completion endpoint..."
        TEST=$(ssh gx10 "curl -s -X POST http://localhost:$PORT/v1/completions \
            -H 'Content-Type: application/json' \
            -d '{\"model\":\"teacher\",\"prompt\":\"def add(a,b):\\n\",\"max_tokens\":10,\"temperature\":0.0}'" 2>/dev/null)
        if echo "$TEST" | grep -q "error"; then
            echo "ERROR: Completion endpoint broken: $TEST"
            echo "Try restarting realizar on gx10:"
            echo "  ssh gx10 'realizar serve --model ~/models/Qwen3-8B-Q4_K_M.gguf --gpu --openai-api --port $PORT'"
            exit 1
        fi
        echo "Completion endpoint OK"

        # Run on gx10 via SSH (nohup so it survives SSH disconnect)
        echo ""
        echo "Launching on gx10 (nohup)..."
        ssh gx10 "cd ~/src/albor && nohup python3 scripts/generate_teacher_completions_api.py \
            --server http://localhost:$PORT \
            --prompts $PROMPTS \
            --output $OUTPUT \
            --limit $LIMIT \
            --max-retries 5 \
            > logs/teacher-completions-resume.log 2>&1 &
            echo \"PID: \$!\"
            echo 'Monitor: ssh gx10 tail -f ~/src/albor/logs/teacher-completions-resume.log'
            echo 'Results: ssh gx10 wc -l ~/src/albor/$OUTPUT'"
        ;;

    local)
        # Run locally via SSH tunnel (fragile — use gx10 mode instead)
        LOCAL_PORT="$PORT"

        if ! curl -s "http://localhost:$LOCAL_PORT/health" >/dev/null 2>&1; then
            echo "Setting up SSH tunnel with keepalive..."
            ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
                -L "$LOCAL_PORT:localhost:8091" gx10 -N -f
            sleep 2
        fi

        HEALTH=$(curl -s "http://localhost:$LOCAL_PORT/health" 2>/dev/null)
        if [ -z "$HEALTH" ]; then
            echo "ERROR: Cannot reach server at localhost:$LOCAL_PORT"
            exit 1
        fi
        echo "Server: $HEALTH"

        existing=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
        echo "Existing: $existing, Remaining: ~$((LIMIT - existing))"

        exec python3 scripts/generate_teacher_completions_api.py \
            --server "http://localhost:$LOCAL_PORT" \
            --prompts "$PROMPTS" \
            --output "$OUTPUT" \
            --limit "$LIMIT" \
            --max-retries 5
        ;;

    *)
        echo "Usage: $0 [gx10|local] [port] [limit]"
        exit 1
        ;;
esac
