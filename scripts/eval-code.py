#!/usr/bin/env python3
"""Evaluate code completion models on functional test benchmarks.

Runs code completion benchmarks (HumanEval format) by:
1. Loading problems from JSONL (prompt + test cases)
2. Generating completions via apr serve API or direct model inference
3. Executing generated code + tests in sandboxed subprocess
4. Reporting pass@k metrics

Benchmark format (JSONL, one per line):
{
    "task_id": "unique_name",
    "prompt": "def function_name(args):\n    \"\"\"Docstring.\"\"\"\n",
    "test": "assert function_name(input) == expected",
    "canonical_solution": "    return answer\n"  // optional, for validation
}

Usage:
    # Validate canonical solutions (no model needed)
    python scripts/eval-code.py configs/eval/python-intermediate.jsonl --validate-only

    # Evaluate via apr serve API (requires running server)
    python scripts/eval-code.py configs/eval/python-intermediate.jsonl \
        --api http://localhost:8080 --samples 10

    # Evaluate using canonical solutions (baseline — should be 100%)
    python scripts/eval-code.py configs/eval/python-intermediate.jsonl --use-canonical

Requirements: pip install requests (for API mode)

NOTE: Model inference via apr serve currently blocked by ALB-037 (weight loading bug).
Until fixed, use --validate-only or --use-canonical modes.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


def load_problems(path: Path) -> list[dict]:
    """Load problems from JSONL file."""
    problems = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                problem = json.loads(line)
                assert "task_id" in problem, "Missing task_id"
                assert "prompt" in problem, "Missing prompt"
                assert "test" in problem, "Missing test"
                problems.append(problem)
            except (json.JSONDecodeError, AssertionError) as e:
                print(f"WARNING: Line {line_num}: {e}")
    return problems


def execute_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute Python code in a subprocess, return (passed, output)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr[-500:] if result.stderr else "Unknown error"
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as e:
            return False, str(e)
        finally:
            Path(f.name).unlink(missing_ok=True)


def generate_completion_api(prompt: str, api_url: str, max_tokens: int = 256,
                            temperature: float = 0.2) -> str:
    """Generate completion via apr serve API."""
    try:
        import requests
    except ImportError:
        sys.exit("ERROR: requests required for API mode. Install: pip install requests")

    resp = requests.post(
        f"{api_url}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["\nclass ", "\ndef ", "\n#", "\nif __name__"],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["text"]


def evaluate_problem(problem: dict, completion: str) -> tuple[bool, str]:
    """Evaluate a single problem with a completion."""
    full_code = problem["prompt"] + completion + "\n" + problem["test"]
    return execute_code(full_code)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric.

    n: total samples, c: correct samples, k: k for pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c, n - c - k, -1)) / math.prod(range(n, n - k, -1))


def run_validation(problems: list[dict]) -> dict:
    """Validate canonical solutions — all should pass."""
    results = {"passed": 0, "failed": 0, "errors": []}

    for problem in problems:
        canonical = problem.get("canonical_solution", "")
        if not canonical:
            print(f"  SKIP {problem['task_id']} (no canonical solution)")
            continue

        passed, output = evaluate_problem(problem, canonical)
        if passed:
            results["passed"] += 1
            print(f"  PASS {problem['task_id']}")
        else:
            results["failed"] += 1
            results["errors"].append({
                "task_id": problem["task_id"],
                "error": output,
            })
            print(f"  FAIL {problem['task_id']}: {output[:100]}")

    return results


def run_eval_canonical(problems: list[dict]) -> dict:
    """Evaluate using canonical solutions (baseline, should be 100%)."""
    results = {"total": 0, "passed": 0, "per_problem": {}}

    for problem in problems:
        canonical = problem.get("canonical_solution", "")
        if not canonical:
            continue

        passed, output = evaluate_problem(problem, canonical)
        results["total"] += 1
        if passed:
            results["passed"] += 1
        results["per_problem"][problem["task_id"]] = {
            "passed": passed,
            "error": None if passed else output[:200],
        }

    return results


def run_eval_api(problems: list[dict], api_url: str, samples: int = 1,
                 max_tokens: int = 256, temperature: float = 0.2) -> dict:
    """Evaluate using apr serve API."""
    results = {"total": 0, "per_problem": {}}

    for problem in problems:
        task_id = problem["task_id"]
        correct = 0

        for s in range(samples):
            try:
                completion = generate_completion_api(
                    problem["prompt"], api_url, max_tokens, temperature
                )
                passed, output = evaluate_problem(problem, completion)
                if passed:
                    correct += 1
            except Exception as e:
                print(f"  ERROR {task_id} sample {s}: {e}")

        results["total"] += 1
        results["per_problem"][task_id] = {
            "samples": samples,
            "correct": correct,
            "pass@1": pass_at_k(samples, correct, 1) if samples >= 1 else 0,
        }

        status = f"{correct}/{samples}"
        print(f"  {task_id}: {status}")

    # Compute aggregate pass@k
    all_n = [v["samples"] for v in results["per_problem"].values()]
    all_c = [v["correct"] for v in results["per_problem"].values()]

    if all_n:
        results["pass@1"] = sum(
            pass_at_k(n, c, 1) for n, c in zip(all_n, all_c)
        ) / len(all_n)

    return results


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate code completion models")
    parser.add_argument("benchmark", type=Path,
                        help="JSONL benchmark file (HumanEval format)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate canonical solutions")
    parser.add_argument("--use-canonical", action="store_true",
                        help="Evaluate using canonical solutions (baseline)")
    parser.add_argument("--api", type=str,
                        help="apr serve API URL (e.g., http://localhost:8080)")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--json", action="store_true")
    return parser


def report_validation(results):
    """Print validation report and return exit code."""
    total = results["passed"] + results["failed"]
    print(f"\nValidation: {results['passed']}/{total} passed")
    for e in results.get("errors", []):
        print(f"    - {e['task_id']}: {e['error'][:80]}")
    return 1 if results["failed"] > 0 else 0


def main():
    args = build_parser().parse_args()

    print(f"Loading benchmark: {args.benchmark}")
    problems = load_problems(args.benchmark)
    print(f"  {len(problems)} problems loaded\n")

    if args.validate_only:
        print("Validating canonical solutions...")
        sys.exit(report_validation(run_validation(problems)))

    if args.use_canonical:
        print("Evaluating canonical solutions (baseline)...")
        results = run_eval_canonical(problems)
        pct = results["passed"] / max(results["total"], 1) * 100
        print(f"\nBaseline: {results['passed']}/{results['total']} ({pct:.1f}%)")
        sys.exit(0 if pct == 100 else 1)

    if args.api:
        print(f"Evaluating via API: {args.api}")
        results = run_eval_api(
            problems, args.api, args.samples, args.max_tokens, args.temperature
        )
        print(f"\nResults: {results['total']} problems, pass@1: {results.get('pass@1', 0):.1%}")
        return

    build_parser().error("Specify --validate-only, --use-canonical, or --api URL")


if __name__ == "__main__":
    main()
