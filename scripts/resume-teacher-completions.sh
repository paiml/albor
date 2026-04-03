#!/bin/bash
# Resume teacher completions via SSH tunnel to gx10.
# Uses the resilient script with retry/resume/dedup (C-TEACHER-*).
#
# Prerequisites:
#   - gx10 running: realizar serve --model ~/models/Qwen3-8B-Q4_K_M.gguf --gpu --openai-api --port 8090
#   - SSH access to gx10
#
# Usage:
#   bash scripts/resume-teacher-completions.sh [--limit N]

set -euo pipefail

LIMIT="${1:-1000}"  # Default: 1K prompts (was original pilot size)
LOCAL_PORT=18090
REMOTE_PORT=8090

echo "=== Teacher Completions Pipeline (Resume) ==="
echo "Target: $LIMIT prompts"
echo "Output: data/distill/completions-1k.jsonl"

# Check if tunnel already exists
if curl -s "http://localhost:$LOCAL_PORT/health" >/dev/null 2>&1; then
    echo "SSH tunnel already active on port $LOCAL_PORT"
else
    echo "Setting up SSH tunnel to gx10:$REMOTE_PORT..."
    ssh -L "$LOCAL_PORT:localhost:$REMOTE_PORT" gx10 -N -f
    sleep 1
    if ! curl -s "http://localhost:$LOCAL_PORT/health" >/dev/null 2>&1; then
        echo "ERROR: SSH tunnel failed. Is gx10 reachable and realizar running?"
        exit 1
    fi
    echo "Tunnel active"
fi

# Show server health
echo "Server: $(curl -s http://localhost:$LOCAL_PORT/health)"

# Count existing completions
existing=$(wc -l < data/distill/completions-1k.jsonl 2>/dev/null || echo 0)
echo "Existing completions: $existing"
echo "Remaining: $((LIMIT - existing)) (approx)"
echo ""

# Resume
exec python3 scripts/generate_teacher_completions_api.py \
    --server "http://localhost:$LOCAL_PORT" \
    --prompts data/distill/prompts-filtered-50k.jsonl \
    --output data/distill/completions-1k.jsonl \
    --limit "$LIMIT" \
    --max-retries 5
