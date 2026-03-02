#!/usr/bin/env python3
"""Validate training convergence by analyzing loss from training log.

Parses entrenar training log output, extracts per-step loss values,
and validates FALSIFY-ALBOR-001: EMA(loss) is monotonically decreasing
after warmup period.

Usage:
    # Analyze completed training
    python scripts/validate-training-convergence.py \
        checkpoints/albor-base-350m/training.log

    # Monitor active training (reads current log state)
    python scripts/validate-training-convergence.py \
        checkpoints/albor-base-350m/training.log --watch

    # Custom parameters
    python scripts/validate-training-convergence.py training.log \
        --ema-window 100 --warmup-steps 50 --max-spike 2.0
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path


def parse_training_log(log_path):
    """Extract step/loss pairs from entrenar training log.

    Returns list of (step, loss) tuples.
    """
    losses = []
    step_pattern = re.compile(
        r"(?:step|batch|iteration)\s*[=:]\s*(\d+).*?loss\s*[=:]\s*([\d.]+)",
        re.IGNORECASE,
    )
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/\d+:\s+loss=([\d.]+)",
    )
    loss_only_pattern = re.compile(
        r"^\s*loss[=:\s]+([\d.]+)",
        re.IGNORECASE,
    )

    with open(log_path) as f:
        step = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = step_pattern.search(line)
            if m:
                losses.append((int(m.group(1)), float(m.group(2))))
                continue

            m = epoch_pattern.search(line)
            if m:
                losses.append((step, float(m.group(2))))
                step += 1
                continue

            m = loss_only_pattern.search(line)
            if m:
                losses.append((step, float(m.group(1))))
                step += 1

    return losses


def compute_ema(values, window):
    """Compute exponential moving average."""
    alpha = 2.0 / (window + 1)
    ema = []
    current = values[0] if values else 0.0
    for v in values:
        current = alpha * v + (1 - alpha) * current
        ema.append(current)
    return ema


def validate_convergence(losses, ema_window=100, warmup_steps=50, max_spike=2.0):
    """Validate FALSIFY-ALBOR-001: loss convergence.

    Checks:
    1. Loss generally decreases after warmup
    2. No catastrophic spikes (>max_spike above EMA)
    3. Final loss < initial loss
    """
    if len(losses) < 2:
        return {"status": "INSUFFICIENT_DATA", "message": "Need at least 2 data points"}

    steps = [s for s, _ in losses]
    values = [l for _, l in losses]

    ema = compute_ema(values, ema_window)

    results = {
        "total_steps": len(losses),
        "initial_loss": values[0],
        "final_loss": values[-1],
        "min_loss": min(values),
        "max_loss": max(values),
        "loss_reduction": values[0] - values[-1],
        "reduction_pct": (values[0] - values[-1]) / values[0] * 100,
    }

    # Check 1: overall decrease
    results["loss_decreased"] = values[-1] < values[0]

    # Check 2: EMA monotonicity after warmup
    post_warmup_ema = ema[warmup_steps:]
    violations = 0
    worst_violation = 0.0
    for i in range(1, len(post_warmup_ema)):
        if post_warmup_ema[i] > post_warmup_ema[i - 1]:
            violations += 1
            diff = post_warmup_ema[i] - post_warmup_ema[i - 1]
            worst_violation = max(worst_violation, diff)

    results["ema_violations"] = violations
    results["ema_violation_rate"] = violations / max(len(post_warmup_ema) - 1, 1)
    results["worst_ema_violation"] = worst_violation

    # Check 3: catastrophic spikes
    spikes = []
    for i, (step, loss) in enumerate(losses):
        if i < warmup_steps:
            continue
        if loss > ema[i] + max_spike:
            spikes.append((step, loss, ema[i]))

    results["spike_count"] = len(spikes)
    results["spikes"] = spikes[:5]  # first 5

    # Overall verdict
    passed = (
        results["loss_decreased"]
        and results["ema_violation_rate"] < 0.3
        and results["spike_count"] == 0
    )
    results["status"] = "PASS" if passed else "FAIL"

    return results


def print_report(results, losses):
    """Print convergence analysis report."""
    print("=" * 60)
    print("  FALSIFY-ALBOR-001: Training Convergence Validation")
    print("=" * 60)
    print()
    print(f"  Steps analyzed:     {results['total_steps']}")
    print(f"  Initial loss:       {results['initial_loss']:.4f}")
    print(f"  Final loss:         {results['final_loss']:.4f}")
    print(f"  Min loss:           {results['min_loss']:.4f}")
    print(f"  Max loss:           {results['max_loss']:.4f}")
    print(f"  Loss reduction:     {results['loss_reduction']:.4f} "
          f"({results['reduction_pct']:.1f}%)")
    print()

    if results.get("initial_loss", 0) > 0:
        initial_ppl = math.exp(min(results["initial_loss"], 20))
        final_ppl = math.exp(min(results["final_loss"], 20))
        print(f"  Initial perplexity: {initial_ppl:.2f}")
        print(f"  Final perplexity:   {final_ppl:.2f}")
        print()

    print(f"  Loss decreased:     {'YES' if results['loss_decreased'] else 'NO'}")
    print(f"  EMA violations:     {results['ema_violations']} "
          f"({results['ema_violation_rate']:.1%} of post-warmup steps)")
    print(f"  Catastrophic spikes: {results['spike_count']}")

    if results["spikes"]:
        print()
        print("  Spikes detected:")
        for step, loss, ema_val in results["spikes"]:
            print(f"    Step {step}: loss={loss:.4f} (EMA={ema_val:.4f}, "
                  f"delta={loss-ema_val:.4f})")

    print()
    status = results["status"]
    if status == "PASS":
        print(f"  Result: PASS — FALSIFY-ALBOR-001 CORROBORATED")
    elif status == "INSUFFICIENT_DATA":
        print(f"  Result: SKIP — {results.get('message', 'not enough data')}")
    else:
        print(f"  Result: FAIL — FALSIFY-ALBOR-001 REFUTED")

    print("=" * 60)

    # Print loss curve (ASCII)
    if len(losses) >= 5:
        print()
        print("  Loss Curve (ASCII):")
        print_ascii_chart(losses)


def _downsample(values, width):
    """Downsample values to fit width."""
    if len(values) <= width:
        return values
    step = len(values) / width
    return [values[int(i * step)] for i in range(width)]


def _chart_row_label(row, height, lo, hi):
    """Generate label for a chart row."""
    if row == 0:
        return f"{hi:7.2f} |"
    if row == height - 1:
        return f"{lo:7.2f} |"
    return "        |"


def print_ascii_chart(losses, width=50, height=12):
    """Print a simple ASCII loss curve."""
    values = [l for _, l in losses]
    if not values:
        return

    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1

    sampled = _downsample(values, width)

    for row in range(height):
        threshold = hi - (row / (height - 1)) * (hi - lo)
        label = _chart_row_label(row, height, lo, hi)
        bar = "".join("#" if v >= threshold else " " for v in sampled)
        print(f"  {label}{bar}")

    print("         +" + "-" * len(sampled))
    print(f"          0{' ' * (len(sampled) - 5)}steps")


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate training convergence (FALSIFY-ALBOR-001)"
    )
    parser.add_argument("log_file", type=Path, help="Path to training.log")
    parser.add_argument("--ema-window", type=int, default=100,
                        help="EMA window size for smoothing (default: 100)")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Steps to ignore for monotonicity check (default: 50)")
    parser.add_argument("--max-spike", type=float, default=2.0,
                        help="Max acceptable loss spike above EMA (default: 2.0)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode: re-read log every 30 seconds")
    return parser


def run_watch(args):
    """Run in watch mode, re-reading log periodically."""
    while True:
        losses = parse_training_log(args.log_file)
        if losses:
            print("\033[2J\033[H")  # clear screen
            results = validate_convergence(
                losses, args.ema_window, args.warmup_steps, args.max_spike
            )
            print_report(results, losses)
            print(f"\n  Last updated: {time.strftime('%H:%M:%S')}")
            print(f"  Watching {args.log_file} (Ctrl+C to stop)")
        else:
            print(f"No loss data found in {args.log_file}")
        time.sleep(30)


def main():
    args = build_parser().parse_args()

    if not args.log_file.exists():
        sys.exit(f"ERROR: Log file not found: {args.log_file}")

    if args.watch:
        run_watch(args)
        return

    losses = parse_training_log(args.log_file)
    if not losses:
        print(f"No loss data found in {args.log_file}")
        print("entrenar may not write per-step loss (see ALB-035)")
        sys.exit(1)

    results = validate_convergence(
        losses, args.ema_window, args.warmup_steps, args.max_spike
    )
    print_report(results, losses)
    sys.exit(0 if results["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
