#!/usr/bin/env bash
# cleanup-gpu.sh — Kill stale processes holding GPU memory
#
# Run before launching training to ensure GPU is clean.
# Finds python/apr processes with /dev/nvidia0 open and kills them.
#
# Usage:
#   ./scripts/cleanup-gpu.sh          # dry-run (show what would be killed)
#   ./scripts/cleanup-gpu.sh --kill   # actually kill them
set -euo pipefail

DRY_RUN=true
[[ "${1:-}" == "--kill" ]] && DRY_RUN=false

echo "=== GPU Process Audit ==="
nvidia-smi 2>/dev/null | grep MiB | head -1

# Find processes holding /dev/nvidia0 (exclude Xorg, gnome-shell, firefox, chrome)
PIDS=$(fuser /dev/nvidia0 2>/dev/null | tr -s ' ' '\n' | sort -u)
STALE_PIDS=()

for pid in $PIDS; do
    CMD=$(ps -p "$pid" -o comm= 2>/dev/null || echo "")
    case "$CMD" in
        Xorg|gnome-shell|snapd-desktop-*|xdg-desktop-*|firefox*|chrome*|spotify*|ttop)
            # Desktop processes — leave alone
            ;;
        python*|apr|uv|cargo|rustc)
            STALE_PIDS+=("$pid")
            CMDLINE=$(ps -p "$pid" -o args= 2>/dev/null | head -c 80)
            RSS=$(ps -p "$pid" -o rss= 2>/dev/null | awk '{printf "%.0fMB", $1/1024}')
            echo "  STALE: PID=$pid CMD=$CMD RSS=$RSS"
            echo "    $CMDLINE"
            ;;
    esac
done

if [ ${#STALE_PIDS[@]} -eq 0 ]; then
    echo "GPU is clean — no stale processes."
    exit 0
fi

echo ""
echo "${#STALE_PIDS[@]} stale process(es) found."

if $DRY_RUN; then
    echo "Run with --kill to terminate them."
else
    for pid in "${STALE_PIDS[@]}"; do
        echo "  Killing PID $pid..."
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
    echo "After cleanup:"
    nvidia-smi 2>/dev/null | grep MiB | head -1
fi
