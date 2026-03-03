#!/usr/bin/env bash
# train-guard.sh — Crash-resilient training supervisor
#
# Production-grade process monitoring for GPU training. Detects silent
# crashes, captures structured crash reports, and restarts with backoff.
#
# Architecture (from FlashRecovery/ByteRobust/Meta patterns):
#   train-guard.sh  --supervises-->  apr train apply
#   (crash reports, GPU health,       (the actual training)
#    backoff, heartbeat monitoring)
#
# Usage:
#   bash scripts/train-guard.sh configs/train/pretrain-350m-v2.yaml
#   bash scripts/train-guard.sh configs/train/pretrain-350m-v2.yaml --max-restarts 3
#   bash scripts/train-guard.sh configs/train/pretrain-350m-v2.yaml --no-restart
#   bash scripts/train-guard.sh configs/train/pretrain-350m-v2.yaml --cuda-blocking
#
# Auto-diagnostic: if training crashes early with SIGABRT/SIGSEGV (the async
# CUDA error pattern), the guard automatically enables CUDA_LAUNCH_BLOCKING=1
# on the next restart to surface the exact failing kernel.
#
# Refs: ALB-064 (https://github.com/paiml/albor/issues/46)

set -euo pipefail

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

CONFIG="${1:?Usage: train-guard.sh <config.yaml> [--max-restarts N] [--no-restart]}"
APR="${APR_BIN:-/mnt/nvme-raid0/targets/aprender/release/apr}"
CRASH_DIR="crash-reports"
HEARTBEAT_INTERVAL=15        # Poll heartbeat every N seconds
HEARTBEAT_STALE_THRESHOLD=300  # Declare hang after N seconds of stale heartbeat
MAX_RESTARTS=5
NO_RESTART=false
BACKOFF_BASE=30              # Initial backoff in seconds
BACKOFF_CAP=600              # Max backoff in seconds
STABLE_RESET_SECS=3600      # Reset crash counter after 1 hour of stable training
CUDA_BLOCKING=false          # Set automatically on async CUDA crash diagnosis

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-restarts)   MAX_RESTARTS="$2"; shift 2 ;;
        --no-restart)     NO_RESTART=true; shift ;;
        --cuda-blocking)  CUDA_BLOCKING=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ═══════════════════════════════════════════════════════════
# CRASH CLASSIFICATION (Five Whys: what killed the process?)
# ═══════════════════════════════════════════════════════════

classify_crash() {
    local exit_code=$1

    case $exit_code in
        0)   echo "clean_exit" ;;         # Training completed normally
        1)   echo "restartable" ;;        # Generic error
        2)   echo "oom_cuda" ;;           # CUDA OOM — needs config change
        134) echo "restartable" ;;        # SIGABRT (assertion, driver abort)
        135) echo "fatal_bus" ;;          # SIGBUS (PCIe failure)
        136) echo "restartable" ;;        # SIGFPE
        137) echo "oom_system" ;;         # SIGKILL (OOM killer)
        139) echo "restartable" ;;        # SIGSEGV (buffer overflow)
        143) echo "graceful_stop" ;;      # SIGTERM (manual stop)
        *)   echo "restartable" ;;        # Default: try restart
    esac
}

signal_name() {
    local exit_code=$1
    if [[ $exit_code -gt 128 ]]; then
        local sig=$((exit_code - 128))
        case $sig in
            6)  echo "SIGABRT" ;;
            7)  echo "SIGBUS" ;;
            8)  echo "SIGFPE" ;;
            9)  echo "SIGKILL" ;;
            11) echo "SIGSEGV" ;;
            15) echo "SIGTERM" ;;
            *)  echo "SIG$sig" ;;
        esac
    else
        echo "EXIT_$exit_code"
    fi
}

# ═══════════════════════════════════════════════════════════
# GPU STATE CAPTURE
# ═══════════════════════════════════════════════════════════

capture_gpu_state() {
    local output_file="$1"
    {
        echo "=== nvidia-smi snapshot ==="
        nvidia-smi --query-gpu=timestamp,name,pci.bus_id,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,pstate --format=csv 2>/dev/null || echo "nvidia-smi FAILED"

        echo ""
        echo "=== GPU compute processes ==="
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "no compute processes"

        echo ""
        echo "=== Xid errors (last 20 lines) ==="
        sudo dmesg -T 2>/dev/null | grep -i 'NVRM\|Xid' | tail -20 || echo "no Xid errors found"

        echo ""
        echo "=== OOM killer (last 10 lines) ==="
        sudo dmesg -T 2>/dev/null | grep -i 'oom\|killed process' | tail -10 || echo "no OOM events"
    } > "$output_file" 2>&1
}

# ═══════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════

preflight() {
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  train-guard.sh — Crash-Resilient Training Monitor  ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""

    # Verify config exists
    if [[ ! -f "$CONFIG" ]]; then
        echo "FATAL: Config not found: $CONFIG"
        exit 1
    fi
    echo "  Config:  $CONFIG"

    # Verify apr binary
    if [[ ! -x "$APR" ]]; then
        echo "FATAL: apr binary not found: $APR"
        exit 1
    fi
    echo "  Binary:  $APR"

    # Kill stale GPU processes (CLAUDE.md rule: ALWAYS kill before training)
    local stale_pids
    stale_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [[ -n "$stale_pids" ]]; then
        echo "  WARNING: Killing stale GPU processes: $stale_pids"
        echo "$stale_pids" | xargs -r kill -9 2>/dev/null || true
        sleep 2
    else
        echo "  GPU:     No stale processes"
    fi

    # GPU health check
    local gpu_temp
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null)
    if [[ -z "$gpu_temp" ]]; then
        echo "FATAL: GPU not responsive (nvidia-smi failed)"
        exit 1
    fi
    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "  GPU:     ${gpu_temp}°C, ${gpu_mem} MiB used"

    # Check for fatal Xid errors
    local xid_fatal
    xid_fatal=$(sudo dmesg -T 2>/dev/null | grep -oP 'Xid.*?: \K\d+' | tail -5 | grep -E '^(48|79|95)$' || true)
    if [[ -n "$xid_fatal" ]]; then
        echo "FATAL: Xid errors detected: $xid_fatal (GPU needs reset or reboot)"
        exit 1
    fi

    # Create crash report directory
    mkdir -p "$CRASH_DIR"
    echo "  Crashes: $CRASH_DIR/"
    echo "  Restart: $( $NO_RESTART && echo "DISABLED" || echo "max $MAX_RESTARTS" )"
    echo "  CUDA:    $( $CUDA_BLOCKING && echo "BLOCKING (diagnostic mode)" || echo "async (normal)" )"
    echo ""
}

# ═══════════════════════════════════════════════════════════
# HEARTBEAT MONITORING
# ═══════════════════════════════════════════════════════════

read_heartbeat() {
    local checkpoint_dir
    checkpoint_dir=$(grep -oP 'output_dir:\s*"?\K[^"]+' "$CONFIG" | tr -d '"' || echo "")
    local state_file="${checkpoint_dir}/training_state.json"

    if [[ -f "$state_file" ]]; then
        local ts
        ts=$(python3 -c "import json; f=open('$state_file'); d=json.load(f); print(d.get('timestamp_ms', 0))" 2>/dev/null || echo "0")
        local step
        step=$(python3 -c "import json; f=open('$state_file'); d=json.load(f); print(d.get('step', -1))" 2>/dev/null || echo "-1")
        local loss
        loss=$(python3 -c "import json; f=open('$state_file'); d=json.load(f); print(d.get('loss', -1))" 2>/dev/null || echo "-1")
        echo "$ts $step $loss"
    else
        echo "0 -1 -1"
    fi
}

# ═══════════════════════════════════════════════════════════
# CRASH REPORT WRITER
# ═══════════════════════════════════════════════════════════

write_crash_report() {
    local run_id="$1"
    local crash_num="$2"
    local exit_code="$3"
    local classification="$4"
    local last_step="$5"
    local last_loss="$6"
    local action="$7"
    local backoff="$8"
    local elapsed="$9"

    local report_file="${CRASH_DIR}/crash-${run_id}-$(printf '%03d' "$crash_num").json"
    local gpu_file="${CRASH_DIR}/gpu-state-${run_id}-$(printf '%03d' "$crash_num").txt"

    # Capture GPU state
    capture_gpu_state "$gpu_file"

    # Write structured crash report
    cat > "$report_file" <<REPORT
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "run_id": "$run_id",
  "config": "$CONFIG",
  "crash_number": $crash_num,
  "exit_code": $exit_code,
  "signal": "$(signal_name "$exit_code")",
  "classification": "$classification",
  "last_step": $last_step,
  "last_loss": $last_loss,
  "elapsed_seconds": $elapsed,
  "action_taken": "$action",
  "backoff_seconds": $backoff,
  "gpu_state_file": "$gpu_file",
  "log_file": "/tmp/albor-train-${run_id}.log"
}
REPORT

    echo "  Crash report: $report_file"
}

# ═══════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════

main() {
    preflight

    local run_id
    run_id=$(date +%Y%m%d-%H%M%S)
    local crash_count=0
    local last_stable_time
    last_stable_time=$(date +%s)

    while true; do
        local log_file="/tmp/albor-train-${run_id}-attempt${crash_count}.log"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Starting training (attempt $((crash_count + 1))/$((MAX_RESTARTS + 1)))"
        echo "  Log: $log_file"
        echo "  Time: $(date)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Capture GPU state BEFORE training
        capture_gpu_state "${CRASH_DIR}/gpu-pre-${run_id}-attempt${crash_count}.txt"

        # Launch training in background
        local start_time
        start_time=$(date +%s)

        if $CUDA_BLOCKING; then
            echo "  Mode: DIAGNOSTIC (CUDA_LAUNCH_BLOCKING=1, RUST_BACKTRACE=1)"
            echo "  WARNING: ~50x slower than normal — use for crash diagnosis only"
            CUDA_LAUNCH_BLOCKING=1 RUST_BACKTRACE=1 \
                "$APR" train apply --task pretrain --config "$CONFIG" > "$log_file" 2>&1 &
        else
            "$APR" train apply --task pretrain --config "$CONFIG" > "$log_file" 2>&1 &
        fi
        local train_pid=$!
        echo "  PID: $train_pid"

        # Monitor loop
        local last_heartbeat_ts=0
        local last_heartbeat_check
        last_heartbeat_check=$(date +%s)

        while true; do
            sleep "$HEARTBEAT_INTERVAL"

            # Check if process is alive
            if ! kill -0 "$train_pid" 2>/dev/null; then
                # Process died — capture exit code
                wait "$train_pid" 2>/dev/null
                local exit_code=$?
                local elapsed=$(( $(date +%s) - start_time ))

                echo ""
                echo "  ╔══════════════════════════════════════════╗"
                echo "  ║  CRASH DETECTED                          ║"
                echo "  ╚══════════════════════════════════════════╝"
                echo "  Exit code:  $exit_code ($(signal_name "$exit_code"))"
                echo "  Elapsed:    ${elapsed}s"
                echo "  Log tail:"
                echo "  ---"
                tail -5 "$log_file" | sed 's/^/  | /'
                echo "  ---"

                # Read last known training state
                local heartbeat
                heartbeat=$(read_heartbeat)
                local last_step last_loss
                last_step=$(echo "$heartbeat" | awk '{print $2}')
                last_loss=$(echo "$heartbeat" | awk '{print $3}')
                echo "  Last step:  $last_step"
                echo "  Last loss:  $last_loss"

                # Classify crash
                local classification
                classification=$(classify_crash "$exit_code")
                echo "  Class:      $classification"

                # Check for clean exit
                if [[ "$classification" == "clean_exit" ]]; then
                    echo ""
                    echo "  Training completed successfully."
                    write_crash_report "$run_id" "$crash_count" "$exit_code" \
                        "$classification" "$last_step" "$last_loss" "none" 0 "$elapsed"
                    exit 0
                fi

                # Check for graceful stop (user sent SIGTERM)
                if [[ "$classification" == "graceful_stop" ]]; then
                    echo ""
                    echo "  Training stopped by user (SIGTERM)."
                    write_crash_report "$run_id" "$crash_count" "$exit_code" \
                        "$classification" "$last_step" "$last_loss" "none" 0 "$elapsed"
                    exit 0
                fi

                crash_count=$((crash_count + 1))

                # Calculate backoff
                local backoff=$((BACKOFF_BASE * (2 ** (crash_count - 1))))
                if [[ $backoff -gt $BACKOFF_CAP ]]; then
                    backoff=$BACKOFF_CAP
                fi

                # Detect async CUDA crash pattern: early death + signal crash + no CUDA_LAUNCH_BLOCKING
                # Five Whys root cause: kernel launches are async, errors are queued,
                # process dies at next sync point with SIGABRT/SIGSEGV and no stderr output.
                # Diagnosis: restart with CUDA_LAUNCH_BLOCKING=1 to get synchronous errors.
                if [[ "$last_step" -le 0 && "$elapsed" -lt 120 && ! $CUDA_BLOCKING ]] \
                   && [[ "$exit_code" -eq 134 || "$exit_code" -eq 139 || "$exit_code" -eq 136 ]]; then
                    echo ""
                    echo "  ╔══════════════════════════════════════════════════════════╗"
                    echo "  ║  ASYNC CUDA ERROR DETECTED                               ║"
                    echo "  ║  Pattern: early crash + signal death + no error output    ║"
                    echo "  ║  Action: enabling CUDA_LAUNCH_BLOCKING=1 for diagnosis    ║"
                    echo "  ╚══════════════════════════════════════════════════════════╝"
                    echo ""
                    echo "  NOTE: Training will be ~50x slower but will surface the exact"
                    echo "  failing CUDA kernel. Check the log for the error message."
                    CUDA_BLOCKING=true
                fi

                # Determine action
                local action="restart_with_backoff"

                if $NO_RESTART; then
                    action="halt_no_restart"
                elif [[ "$classification" == "fatal_bus" ]]; then
                    action="halt_fatal_hardware"
                elif [[ "$classification" == "oom_cuda" ]]; then
                    action="halt_oom_needs_config"
                elif [[ "$classification" == "oom_system" ]]; then
                    action="halt_oom_system"
                elif [[ $crash_count -gt $MAX_RESTARTS ]]; then
                    action="halt_max_restarts"
                fi

                # Write crash report
                write_crash_report "$run_id" "$crash_count" "$exit_code" \
                    "$classification" "$last_step" "$last_loss" "$action" "$backoff" "$elapsed"

                # Act on classification
                case "$action" in
                    restart_with_backoff)
                        echo "  Action:     Restart after ${backoff}s backoff (crash $crash_count/$MAX_RESTARTS)"
                        echo ""

                        # Pre-restart health check
                        local stale
                        stale=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
                        if [[ -n "$stale" ]]; then
                            echo "  Killing stale GPU processes: $stale"
                            echo "$stale" | xargs -r kill -9 2>/dev/null || true
                            sleep 2
                        fi

                        echo "  Waiting ${backoff}s..."
                        sleep "$backoff"

                        # Check GPU health before restart
                        if ! nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader >/dev/null 2>&1; then
                            echo "  FATAL: GPU not responsive after crash. Halting."
                            exit 1
                        fi

                        break  # Break inner loop to restart training
                        ;;
                    halt_*)
                        echo "  Action:     HALT — $action"
                        echo ""
                        echo "  Training guard halted. Review crash reports in $CRASH_DIR/"
                        exit 1
                        ;;
                esac
            fi

            # Process is alive — check heartbeat freshness
            local now
            now=$(date +%s)
            local heartbeat
            heartbeat=$(read_heartbeat)
            local hb_ts
            hb_ts=$(echo "$heartbeat" | awk '{print $1}')

            # Convert ms to seconds for comparison
            local hb_ts_secs=$((hb_ts / 1000))

            if [[ $hb_ts_secs -gt 0 && $hb_ts_secs == "$last_heartbeat_ts" ]]; then
                local stale_secs=$((now - last_heartbeat_check))
                if [[ $stale_secs -gt $HEARTBEAT_STALE_THRESHOLD ]]; then
                    echo ""
                    echo "  WARNING: Heartbeat stale for ${stale_secs}s (threshold: ${HEARTBEAT_STALE_THRESHOLD}s)"
                    echo "  Sending SIGTERM to PID $train_pid..."
                    kill -TERM "$train_pid" 2>/dev/null || true
                    sleep 10
                    if kill -0 "$train_pid" 2>/dev/null; then
                        echo "  Process did not exit, sending SIGKILL..."
                        kill -KILL "$train_pid" 2>/dev/null || true
                    fi
                    # Let the main loop pick up the exit code
                fi
            else
                last_heartbeat_ts=$hb_ts_secs
                last_heartbeat_check=$now
            fi

            # Reset crash counter after stable running
            local running_secs=$((now - start_time))
            if [[ $running_secs -gt $STABLE_RESET_SECS && $crash_count -gt 0 ]]; then
                echo "  [$(date +%H:%M:%S)] Stable for ${running_secs}s — resetting crash counter (was $crash_count)"
                crash_count=0
                last_stable_time=$now
            fi

            # Periodic status (every 5 minutes)
            if [[ $((running_secs % 300)) -lt $HEARTBEAT_INTERVAL ]]; then
                local hb_step hb_loss
                hb_step=$(echo "$heartbeat" | awk '{print $2}')
                hb_loss=$(echo "$heartbeat" | awk '{print $3}')
                local gpu_temp gpu_mem
                gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "?")
                gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "?")
                echo "  [$(date +%H:%M:%S)] step=$hb_step loss=$hb_loss gpu=${gpu_temp}°C/${gpu_mem}MiB elapsed=${running_secs}s"
            fi
        done
    done
}

# ═══════════════════════════════════════════════════════════
# SIGNAL HANDLING (graceful shutdown of guard + training)
# ═══════════════════════════════════════════════════════════

cleanup() {
    echo ""
    echo "  train-guard.sh received signal — shutting down..."
    # Forward signal to training process if it exists
    if [[ -n "${train_pid:-}" ]] && kill -0 "$train_pid" 2>/dev/null; then
        echo "  Forwarding SIGTERM to training PID $train_pid"
        kill -TERM "$train_pid" 2>/dev/null || true
    fi
    exit 143
}

trap cleanup SIGTERM SIGINT SIGHUP

main
