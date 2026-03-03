#!/usr/bin/env python3
"""Training watchdog: fault-tolerant supervisor for GPU training runs.

Implements three fault tolerance features from the albor survey:
  #12 - Automatic restart on crash (configurable max retries)
  #17 - Hang detection via JSONL log inactivity monitoring
  #20 - Post-crash diagnostic dump (GPU state, logs, exit info)

Works alongside train-guard.sh but adds Python-native log monitoring
with thread-based hang detection and structured diagnostic reports.

Usage:
    # Supervise a training run with auto-restart
    python scripts/training-watchdog.py \\
        --command "apr train apply --task pretrain --config configs/train/pretrain-350m.yaml" \\
        --max-retries 3 --hang-timeout 600

    # With explicit log file monitoring
    python scripts/training-watchdog.py \\
        --command "apr train apply --task pretrain --config configs/train/pretrain-350m.yaml" \\
        --log-file checkpoints/albor-base-350m/training_log.jsonl \\
        --diagnostic-dir /tmp/diagnostics/
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Utility: structured stderr logging
# ---------------------------------------------------------------------------

def _log(level: str, msg: str) -> None:
    """Write a timestamped status line to stderr."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", file=sys.stderr, flush=True)


def log_info(msg: str) -> None:
    _log("INFO", msg)


def log_warn(msg: str) -> None:
    _log("WARN", msg)


def log_error(msg: str) -> None:
    _log("ERROR", msg)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

_CLI_ARGS = [
    ("--command", dict(required=True, help="Training command to supervise (as a shell string)")),
    ("--max-retries", dict(type=int, default=3, help="Max restart attempts (default: 3)")),
    ("--hang-timeout", dict(type=int, default=600, help="Seconds of log inactivity before restart (default: 600)")),
    ("--log-file", dict(type=Path, default=None, help="Path to JSONL training log (auto-detected if omitted)")),
    ("--diagnostic-dir", dict(type=Path, default=Path("diagnostics"), help="Directory for crash dumps (default: diagnostics/)")),
    ("--cooldown", dict(type=int, default=30, help="Seconds to wait between restarts (default: 30)")),
]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the watchdog."""
    parser = argparse.ArgumentParser(
        description="Training watchdog: auto-restart, hang detection, crash diagnostics",
    )
    for name, kwargs in _CLI_ARGS:
        parser.add_argument(name, **kwargs)
    return parser


# ---------------------------------------------------------------------------
# Auto-detect JSONL log path from the training command
# ---------------------------------------------------------------------------

def _extract_config_path(command: str) -> Path | None:
    """Extract --config value from a training command string."""
    match = re.search(r"--config\s+(\S+)", command)
    if match:
        return Path(match.group(1))
    return None


def _read_output_dir(config_path: Path) -> Path | None:
    """Read output_dir from a YAML config without PyYAML dependency."""
    if not config_path.exists():
        return None
    for line in config_path.read_text().splitlines():
        match = re.match(r'\s*output_dir:\s*["\']?([^"\'#\s]+)', line)
        if match:
            return Path(match.group(1))
    return None


def auto_detect_log_file(command: str) -> Path | None:
    """Try to find the JSONL log from the training command's config."""
    config_path = _extract_config_path(command)
    if config_path is None:
        return None
    output_dir = _read_output_dir(config_path)
    if output_dir is None:
        return None
    candidate = output_dir / "training_log.jsonl"
    return candidate


# ---------------------------------------------------------------------------
# Signal name resolution
# ---------------------------------------------------------------------------

_SIGNAL_MAP = {
    signal.SIGABRT: "SIGABRT",
    signal.SIGFPE: "SIGFPE",
    signal.SIGKILL: "SIGKILL",
    signal.SIGSEGV: "SIGSEGV",
    signal.SIGTERM: "SIGTERM",
}


def signal_name_from_code(exit_code: int) -> str:
    """Return human-readable signal name for an exit code."""
    if exit_code >= 0:
        return f"EXIT_{exit_code}"
    sig_num = -exit_code
    name = _SIGNAL_MAP.get(sig_num)
    return name if name else f"SIG{sig_num}"


# ---------------------------------------------------------------------------
# GPU state capture
# ---------------------------------------------------------------------------

def capture_gpu_state() -> str:
    """Capture nvidia-smi output, returning it as a string."""
    sections = []
    for label, cmd in [
        ("GPU summary", ["nvidia-smi"]),
        ("Compute processes", [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv",
        ]),
    ]:
        sections.append(f"=== {label} ===")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
            sections.append(result.stdout or "(no output)")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            sections.append("(nvidia-smi unavailable)")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Read recent JSONL log entries
# ---------------------------------------------------------------------------

def read_recent_jsonl(log_path: Path | None, count: int = 20) -> list[str]:
    """Return the last `count` lines from a JSONL file."""
    if log_path is None or not log_path.exists():
        return []
    lines = log_path.read_text().splitlines()
    return lines[-count:]


# ---------------------------------------------------------------------------
# Diagnostic report writer (survey item #20)
# ---------------------------------------------------------------------------

def _build_report(
    attempt: int,
    exit_code: int,
    reason: str,
    stdout_tail: list[str],
    stderr_tail: list[str],
    log_path: Path | None,
) -> tuple[str, dict]:
    """Assemble the diagnostic report dict and timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "timestamp": ts,
        "attempt": attempt,
        "exit_code": exit_code,
        "signal": signal_name_from_code(exit_code),
        "reason": reason,
        "stdout_tail": stdout_tail[-50:],
        "stderr_tail": stderr_tail[-50:],
        "gpu_state": capture_gpu_state(),
        "recent_log_entries": read_recent_jsonl(log_path, count=20),
    }
    return ts, report


def write_diagnostic_report(
    diag_dir: Path,
    attempt: int,
    exit_code: int,
    reason: str,
    stdout_tail: list[str],
    stderr_tail: list[str],
    log_path: Path | None,
) -> Path:
    """Write a timestamped diagnostic report after a crash or hang."""
    diag_dir.mkdir(parents=True, exist_ok=True)
    ts, report = _build_report(
        attempt, exit_code, reason, stdout_tail, stderr_tail, log_path,
    )
    report_path = diag_dir / f"crash-{ts}-attempt{attempt}.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    log_info(f"Diagnostic report written to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Log file monitor thread (survey item #17)
# ---------------------------------------------------------------------------

class LogFileMonitor:
    """Watches a JSONL log file for new entries.

    Sets `self.stale` to True when no new entry has appeared within
    the configured timeout, signalling a hang.
    """

    def __init__(self, log_path: Path | None, timeout: int) -> None:
        self.log_path = log_path
        self.timeout = timeout
        self.last_activity = time.monotonic()
        self._last_size: int = 0
        self.stale = False
        self._stop = threading.Event()

    def reset(self) -> None:
        """Reset activity timestamp (call on restart)."""
        self.last_activity = time.monotonic()
        self._last_size = self._current_size()
        self.stale = False

    def _current_size(self) -> int:
        """Return current file size, 0 if missing."""
        if self.log_path is None:
            return 0
        try:
            return self.log_path.stat().st_size
        except OSError:
            return 0

    def _check_once(self) -> None:
        """Check log file for new content once."""
        current = self._current_size()
        if current > self._last_size:
            self._last_size = current
            self.last_activity = time.monotonic()
            self.stale = False
        elif self.log_path is not None:
            elapsed = time.monotonic() - self.last_activity
            if elapsed > self.timeout:
                self.stale = True

    def run(self) -> None:
        """Polling loop, intended to run in a daemon thread."""
        while not self._stop.is_set():
            self._check_once()
            self._stop.wait(timeout=5)

    def stop(self) -> None:
        """Signal the monitor thread to exit."""
        self._stop.set()


# ---------------------------------------------------------------------------
# Process lifecycle helpers
# ---------------------------------------------------------------------------

def start_training(command: str) -> subprocess.Popen:
    """Launch the training command as a subprocess."""
    global _child_proc
    log_info(f"Starting: {command}")
    proc = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _child_proc = proc
    return proc


def terminate_process(proc: subprocess.Popen) -> int:
    """Send SIGTERM, wait 10s, escalate to SIGKILL if needed."""
    log_warn(f"Sending SIGTERM to PID {proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log_warn(f"SIGTERM ignored, sending SIGKILL to PID {proc.pid}")
        proc.kill()
        proc.wait(timeout=5)
    return proc.returncode


def drain_output(proc: subprocess.Popen) -> tuple[list[str], list[str]]:
    """Read remaining stdout/stderr from a finished process."""
    stdout_bytes, stderr_bytes = proc.communicate(timeout=10)
    stdout_lines = (stdout_bytes or b"").decode("utf-8", errors="replace").splitlines()
    stderr_lines = (stderr_bytes or b"").decode("utf-8", errors="replace").splitlines()
    return stdout_lines, stderr_lines


# ---------------------------------------------------------------------------
# Main supervisor loop
# ---------------------------------------------------------------------------

def run_supervisor(args: argparse.Namespace) -> int:
    """Core supervisor loop: start, monitor, restart on failure.

    Returns 0 if training eventually succeeds, 1 if retries exhausted.
    """
    log_path = args.log_file or auto_detect_log_file(args.command)
    if log_path:
        log_info(f"Monitoring log file: {log_path}")
    else:
        log_warn("No log file detected; hang detection disabled")

    attempt = 0
    while attempt <= args.max_retries:
        exit_code = _run_one_attempt(args, attempt, log_path)
        if exit_code == 0:
            log_info("Training completed successfully (exit 0)")
            return 0
        attempt += 1
        if attempt <= args.max_retries:
            log_info(f"Cooling down {args.cooldown}s before retry")
            time.sleep(args.cooldown)

    log_error(f"Max retries ({args.max_retries}) exhausted")
    return 1


def _run_one_attempt(
    args: argparse.Namespace,
    attempt: int,
    log_path: Path | None,
) -> int:
    """Run a single training attempt with monitoring.

    Returns the process exit code (0 = success).
    """
    log_info(f"Attempt {attempt + 1}/{args.max_retries + 1}")
    proc = start_training(args.command)
    monitor = LogFileMonitor(log_path, args.hang_timeout)
    monitor.reset()

    monitor_thread = threading.Thread(target=monitor.run, daemon=True)
    monitor_thread.start()

    try:
        exit_code = _poll_process(proc, monitor)
    finally:
        monitor.stop()
        monitor_thread.join(timeout=2)

    return _handle_exit(proc, exit_code, attempt, args, log_path)


def _poll_process(
    proc: subprocess.Popen,
    monitor: LogFileMonitor,
) -> int | None:
    """Poll until process exits or hang is detected.

    Returns exit code (int) if process exited, or None if hang detected.
    """
    while True:
        retcode = proc.poll()
        if retcode is not None:
            return retcode
        if monitor.stale:
            log_warn("Hang detected: log file inactive")
            return None
        time.sleep(2)


def _resolve_exit(
    proc: subprocess.Popen,
    exit_code: int | None,
    hang_timeout: int,
) -> tuple[int, str]:
    """Resolve exit code and reason string from poll result."""
    if exit_code is None:
        actual = terminate_process(proc)
        return actual, f"hang (no log activity for {hang_timeout}s)"
    return exit_code, f"crash (exit code {exit_code})"


def _handle_exit(
    proc: subprocess.Popen,
    exit_code: int | None,
    attempt: int,
    args: argparse.Namespace,
    log_path: Path | None,
) -> int:
    """Handle process exit or hang, capture diagnostics."""
    actual_code, reason = _resolve_exit(proc, exit_code, args.hang_timeout)
    stdout_tail, stderr_tail = drain_output(proc)
    if actual_code == 0:
        return 0
    log_error(f"{reason}, exit_code={actual_code}")
    _log_stderr_tail(stderr_tail)
    write_diagnostic_report(
        args.diagnostic_dir, attempt, actual_code,
        reason, stdout_tail, stderr_tail, log_path,
    )
    return actual_code


def _log_stderr_tail(stderr_tail: list[str], count: int = 10) -> None:
    """Print last N lines of stderr to the watchdog's stderr."""
    if not stderr_tail:
        return
    log_info("Last stderr lines:")
    for line in stderr_tail[-count:]:
        print(f"  | {line}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Signal forwarding
# ---------------------------------------------------------------------------

_child_proc: subprocess.Popen | None = None


def _forward_signal(signum: int, _frame: object) -> None:
    """Forward received signal to the child training process."""
    if _child_proc and _child_proc.poll() is None:
        log_info(f"Forwarding signal {signum} to PID {_child_proc.pid}")
        _child_proc.send_signal(signum)


def install_signal_handlers() -> None:
    """Install handlers that forward SIGTERM/SIGINT to the child."""
    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args and run the supervisor loop."""
    parser = build_parser()
    args = parser.parse_args()

    install_signal_handlers()

    log_info("Training watchdog starting")
    log_info(f"  command:      {args.command}")
    log_info(f"  max_retries:  {args.max_retries}")
    log_info(f"  hang_timeout: {args.hang_timeout}s")
    log_info(f"  cooldown:     {args.cooldown}s")
    log_info(f"  diagnostic_dir: {args.diagnostic_dir}")

    exit_code = run_supervisor(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
