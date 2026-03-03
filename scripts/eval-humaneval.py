#!/usr/bin/env python3
"""HumanEval pass@k evaluation for code generation models (R-020).

Evaluates code generation quality using the HumanEval benchmark (164 problems)
with the unbiased pass@k estimator from Chen et al. (2021). Generates completions
via realizar (the sovereign stack inference engine) and executes them in sandboxed
subprocesses with test assertions.

Usage:
    # Full HumanEval with realizar model
    python scripts/eval-humaneval.py \
        --model-dir checkpoints/albor-base-350m/ \
        --n 200 --k 1 10 100

    # Quick smoke test (5 embedded problems, mock generation)
    python scripts/eval-humaneval.py --mock --quick

    # Use local HumanEval JSONL
    python scripts/eval-humaneval.py \
        --model-dir checkpoints/albor-base-350m/ \
        --humaneval data/humaneval.jsonl --n 20 --k 1

    # JSON output for CI
    python scripts/eval-humaneval.py \
        --model-dir checkpoints/albor-base-350m/ \
        --quick --json

    # Parallel execution with custom timeout
    python scripts/eval-humaneval.py \
        --model-dir checkpoints/albor-base-350m/ \
        --workers 8 --timeout 10 --n 50

Requirements: pip install requests (for download only)

Refs: R-020 (HumanEval evaluation), openai/human-eval
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# URL for the canonical HumanEval dataset
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval"
    "/master/data/HumanEval.jsonl.gz"
)

# Modules forbidden in sandboxed execution
FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "signal",
    "socket", "http", "urllib", "requests", "ctypes",
    "importlib", "pathlib", "glob", "tempfile",
    "multiprocessing", "threading", "pickle",
})

# Stop sequences that indicate end of function body
STOP_SEQUENCES = ["\nclass ", "\ndef ", "\n#", "\nif __name__"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def download_humaneval(dest: Path) -> Path:
    """Download HumanEval JSONL.gz from GitHub and decompress."""
    import gzip
    print(f"Downloading HumanEval from {HUMANEVAL_URL} ...")
    gz_path = dest.parent / "HumanEval.jsonl.gz"
    urllib.request.urlretrieve(HUMANEVAL_URL, gz_path)
    with gzip.open(gz_path, "rt") as gz:
        dest.write_text(gz.read())
    gz_path.unlink()
    print(f"  Saved {dest}")
    return dest


def load_humaneval_jsonl(path: Path) -> list[dict]:
    """Load problems from a HumanEval-format JSONL file."""
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def resolve_humaneval_path(user_path: str | None) -> Path:
    """Find or download the HumanEval dataset."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"HumanEval file not found: {p}")
    default = Path("data/HumanEval.jsonl")
    if default.exists():
        return default
    default.parent.mkdir(parents=True, exist_ok=True)
    return download_humaneval(default)


def get_quick_problems() -> list[dict]:
    """Return 5 embedded problems for quick smoke testing."""
    return [
        {
            "task_id": "HumanEval/0",
            "prompt": (
                "from typing import List\n\n\n"
                "def has_close_elements("
                "numbers: List[float], threshold: float) -> bool:\n"
                '    """Check if in given list of numbers, are any two '
                "numbers closer to each other than\n"
                "    given threshold.\n"
                "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
                "    False\n"
                "    >>> has_close_elements("
                "[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
                "    True\n"
                '    """\n'
            ),
            "test": (
                "assert has_close_elements("
                "[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
                "assert has_close_elements("
                "[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False"
            ),
            "entry_point": "has_close_elements",
        },
        {
            "task_id": "HumanEval/2",
            "prompt": (
                "\n\ndef truncate_number(number: float) -> float:\n"
                '    """Given a positive floating point number, it can '
                "be decomposed into\n"
                "    and integer part (largest integer smaller than "
                "given number) and decimals\n"
                "    (leftover part always smaller than 1).\n\n"
                "    Return the decimal part of the number.\n"
                "    >>> truncate_number(3.5)\n"
                "    0.5\n"
                '    """\n'
            ),
            "test": (
                "assert truncate_number(3.5) == 0.5\n"
                "assert abs(truncate_number(1.33) - 0.33) < 1e-6\n"
                "assert abs(truncate_number(123.456) - 0.456) < 1e-6"
            ),
            "entry_point": "truncate_number",
        },
        {
            "task_id": "HumanEval/23",
            "prompt": (
                "\n\ndef strlen(string: str) -> int:\n"
                '    """Return length of given string\n'
                "    >>> strlen('')\n"
                "    0\n"
                "    >>> strlen('abc')\n"
                "    3\n"
                '    """\n'
            ),
            "test": (
                "assert strlen('') == 0\n"
                "assert strlen('x') == 1\n"
                "assert strlen('asdasnakj') == 9"
            ),
            "entry_point": "strlen",
        },
        {
            "task_id": "HumanEval/28",
            "prompt": (
                "from typing import List\n\n\n"
                "def concatenate(strings: List[str]) -> str:\n"
                '    """Concatenate list of strings into a single '
                "string\n"
                "    >>> concatenate([])\n"
                "    ''\n"
                "    >>> concatenate(['a', 'b', 'c'])\n"
                "    'abc'\n"
                '    """\n'
            ),
            "test": (
                "assert concatenate([]) == ''\n"
                "assert concatenate(['x', 'y', 'z']) == 'xyz'\n"
                "assert concatenate("
                "['x', 'y', 'z', 'w', 'k']) == 'xyzwk'"
            ),
            "entry_point": "concatenate",
        },
        {
            "task_id": "HumanEval/34",
            "prompt": (
                "\n\ndef unique(l: list):\n"
                '    """Return sorted unique elements in a list\n'
                "    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])\n"
                "    [0, 2, 3, 5, 9, 123]\n"
                '    """\n'
            ),
            "test": (
                "assert unique("
                "[5, 3, 5, 2, 3, 3, 9, 0, 123]"
                ") == [0, 2, 3, 5, 9, 123]"
            ),
            "entry_point": "unique",
        },
    ]


# ---------------------------------------------------------------------------
# Completion generation
# ---------------------------------------------------------------------------

def generate_with_realizar(
    model_dir: str, prompt: str, max_tokens: int, temperature: float
) -> str:
    """Generate a single completion via realizar subprocess."""
    cmd = [
        "realizar", "run", model_dir, prompt,
        "--raw",
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--format", "text",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return ""
    return truncate_at_stop(result.stdout)


def truncate_at_stop(text: str) -> str:
    """Truncate completion at the first stop sequence."""
    earliest = len(text)
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


def generate_mock_completion(prompt: str) -> str:
    """Generate a random completion for testing without a model."""
    bodies = [
        "    pass\n",
        "    return None\n",
        "    return []\n",
        "    return 0\n",
        "    return ''\n",
        "    return True\n",
        "    return False\n",
    ]
    return random.choice(bodies)


def generate_completions(
    prompt: str,
    n: int,
    model_dir: str | None,
    max_tokens: int,
    temperature: float,
    mock: bool,
) -> list[str]:
    """Generate n completions for a single prompt."""
    completions = []
    for _ in range(n):
        if mock:
            c = generate_mock_completion(prompt)
        else:
            c = generate_with_realizar(
                model_dir, prompt, max_tokens, temperature
            )
        completions.append(c)
    return completions


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

def build_import_guard() -> str:
    """Build Python code that blocks dangerous imports."""
    blocked = ", ".join(f'"{m}"' for m in sorted(FORBIDDEN_MODULES))
    return (
        "import builtins as _builtins\n"
        f"_BLOCKED = frozenset([{blocked}])\n"
        "_orig_import = _builtins.__import__\n"
        "def _safe_import(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in _BLOCKED:\n"
        f"        raise ImportError(f'Import blocked: {{name}}')\n"
        "    return _orig_import(name, *args, **kwargs)\n"
        "_builtins.__import__ = _safe_import\n\n"
    )


def build_execution_code(prompt: str, completion: str, test: str) -> str:
    """Assemble the full program: guard + prompt + completion + tests."""
    guard = build_import_guard()
    return guard + prompt + completion + "\n" + test + "\n"


def set_resource_limits():
    """Set CPU time and memory limits for sandboxed execution."""
    try:
        import resource
        # 10 seconds CPU time
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        # 256 MB memory
        mem = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ImportError, ValueError):
        pass  # Windows or unsupported


def execute_completion(
    prompt: str, completion: str, test: str, timeout: int
) -> tuple[bool, str]:
    """Execute a single completion in a sandboxed subprocess."""
    code = build_execution_code(prompt, completion, test)
    fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        return _run_sandboxed(tmp_path, timeout)
    finally:
        _cleanup_tempfile(tmp_path)


def _run_sandboxed(
    script_path: str, timeout: int
) -> tuple[bool, str]:
    """Run a script file in a restricted subprocess."""
    env = _make_restricted_env()
    try:
        result = subprocess.run(
            [sys.executable, "-u", script_path],
            capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        if result.returncode == 0:
            return True, "OK"
        return False, _truncate_error(result.stderr)
    except subprocess.TimeoutExpired:
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        return False, f"Execution error: {e}"


def _make_restricted_env() -> dict:
    """Create a restricted environment for subprocess execution."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONSTARTUP", None)
    return env


def _truncate_error(stderr: str) -> str:
    """Truncate stderr to last 300 chars for reporting."""
    if len(stderr) > 300:
        return "..." + stderr[-300:]
    return stderr


def _cleanup_tempfile(path: str):
    """Remove a temporary file, ignoring errors."""
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

def execute_problem_completions(
    problem: dict, completions: list[str], timeout: int, workers: int
) -> list[tuple[bool, str]]:
    """Execute all completions for a problem in parallel."""
    prompt = problem["prompt"]
    test = problem["test"]
    if workers <= 1:
        return _execute_serial(prompt, completions, test, timeout)
    return _execute_parallel(
        prompt, completions, test, timeout, workers
    )


def _execute_serial(
    prompt: str, completions: list[str], test: str, timeout: int
) -> list[tuple[bool, str]]:
    """Execute completions sequentially."""
    results = []
    for c in completions:
        results.append(execute_completion(prompt, c, test, timeout))
    return results


def _execute_parallel(
    prompt: str,
    completions: list[str],
    test: str,
    timeout: int,
    workers: int,
) -> list[tuple[bool, str]]:
    """Execute completions in parallel using ProcessPoolExecutor."""
    results = [None] * len(completions)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, c in enumerate(completions):
            f = pool.submit(
                execute_completion, prompt, c, test, timeout
            )
            futures[f] = i
        for f in as_completed(futures):
            idx = futures[f]
            try:
                results[idx] = f.result()
            except Exception as e:
                results[idx] = (False, f"Worker error: {e}")
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    pass@k = 1 - C(n-c, k) / C(n, k)
    where n = total samples, c = correct samples, k = k value.
    Uses log-space computation for numerical stability.
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    return 1.0 - _comb_ratio(n, c, k)


def _comb_ratio(n: int, c: int, k: int) -> float:
    """Compute C(n-c, k) / C(n, k) in log-space."""
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)
    return math.exp(log_ratio)


def compute_aggregate_pass_at_k(
    problem_results: list[dict], k_values: list[int]
) -> dict[int, float]:
    """Compute aggregate pass@k across all problems."""
    metrics = {}
    for k in k_values:
        scores = []
        for pr in problem_results:
            if pr["n"] >= k:
                scores.append(pass_at_k(pr["n"], pr["c"], k))
        if scores:
            metrics[k] = sum(scores) / len(scores)
        else:
            metrics[k] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate_single_problem(
    problem: dict,
    n_samples: int,
    model_dir: str | None,
    max_tokens: int,
    temperature: float,
    timeout: int,
    workers: int,
    mock: bool,
) -> dict:
    """Evaluate a single HumanEval problem, return results dict."""
    task_id = problem["task_id"]
    completions = generate_completions(
        problem["prompt"], n_samples, model_dir,
        max_tokens, temperature, mock,
    )
    exec_results = execute_problem_completions(
        problem, completions, timeout, workers
    )
    return _summarize_problem_results(task_id, exec_results)


def _summarize_problem_results(
    task_id: str, exec_results: list[tuple[bool, str]]
) -> dict:
    """Summarize execution results for a single problem."""
    n = len(exec_results)
    c = sum(1 for passed, _ in exec_results if passed)
    errors = sum(
        1 for _, msg in exec_results if "error" in msg.lower()
    )
    return {
        "task_id": task_id,
        "n": n,
        "c": c,
        "errors": errors,
        "pass_rate": c / n if n > 0 else 0.0,
    }


def run_evaluation(problems: list[dict], args) -> dict:
    """Run full evaluation across all problems."""
    t_start = time.time()
    problem_results = []
    total_samples = 0
    total_errors = 0

    for i, problem in enumerate(problems):
        pr = evaluate_single_problem(
            problem, args.n, args.model_dir, args.max_tokens,
            args.temperature, args.timeout, args.workers, args.mock,
        )
        problem_results.append(pr)
        total_samples += pr["n"]
        total_errors += pr["errors"]
        _print_problem_progress(i, len(problems), pr)

    elapsed = time.time() - t_start
    return _build_eval_report(
        problem_results, args.k, elapsed,
        total_samples, total_errors,
    )


def _print_problem_progress(idx: int, total: int, pr: dict):
    """Print progress line for a single problem."""
    pct = pr["pass_rate"] * 100
    print(
        f"  [{idx + 1}/{total}] {pr['task_id']}: "
        f"{pr['c']}/{pr['n']} passed ({pct:.1f}%)"
    )


def _build_eval_report(
    problem_results: list[dict],
    k_values: list[int],
    elapsed: float,
    total_samples: int,
    total_errors: int,
) -> dict:
    """Build the final evaluation report dict."""
    pass_at_k_metrics = compute_aggregate_pass_at_k(
        problem_results, k_values
    )
    return {
        "total_problems": len(problem_results),
        "total_samples": total_samples,
        "total_errors": total_errors,
        "elapsed_seconds": round(elapsed, 2),
        "pass_at_k": {
            str(k): round(v, 6) for k, v in pass_at_k_metrics.items()
        },
        "per_problem": problem_results,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_text_report(report: dict, k_values: list[int]):
    """Print human-readable evaluation report."""
    print(f"\n{'=' * 64}")
    print("  HumanEval Evaluation Results (R-020)")
    print(f"{'=' * 64}")
    _print_summary_section(report)
    _print_pass_at_k_section(report, k_values)
    _print_per_problem_section(report)
    print(f"{'=' * 64}")


def _print_summary_section(report: dict):
    """Print the summary section of the text report."""
    print(f"  Problems:          {report['total_problems']}")
    print(f"  Total samples:     {report['total_samples']}")
    print(f"  Execution errors:  {report['total_errors']}")
    elapsed = report['elapsed_seconds']
    print(f"  Wall time:         {elapsed:.1f}s")
    if elapsed > 0 and report['total_samples'] > 0:
        rate = report['total_samples'] / elapsed
        print(f"  Throughput:        {rate:.1f} samples/s")


def _print_pass_at_k_section(report: dict, k_values: list[int]):
    """Print the pass@k metrics section."""
    print(f"\n  {'Pass@k Metrics':^40}")
    print(f"  {'-' * 40}")
    for k in k_values:
        key = str(k)
        if key in report["pass_at_k"]:
            val = report["pass_at_k"][key]
            print(f"  pass@{k:<6} {val:.4f} ({val * 100:.2f}%)")


def _print_per_problem_section(report: dict):
    """Print per-problem results (top 10 worst)."""
    problems = sorted(
        report["per_problem"], key=lambda p: p["pass_rate"]
    )
    worst = problems[:10]
    if not worst:
        return
    print(f"\n  {'Hardest Problems (bottom 10)':^40}")
    print(f"  {'-' * 40}")
    for pr in worst:
        pct = pr["pass_rate"] * 100
        print(f"  {pr['task_id']:<20} {pct:6.1f}%")


def print_json_report(report: dict):
    """Print machine-readable JSON report."""
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="HumanEval pass@k evaluation (R-020)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_model_args(parser)
    _add_dataset_args(parser)
    _add_generation_args(parser)
    _add_execution_args(parser)
    _add_output_args(parser)
    return parser


def _add_model_args(parser: argparse.ArgumentParser):
    """Add model-related arguments."""
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to model checkpoint directory (required unless --mock)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock completions (random stubs) for testing",
    )


def _add_dataset_args(parser: argparse.ArgumentParser):
    """Add dataset-related arguments."""
    parser.add_argument(
        "--humaneval", type=str, default=None,
        help="Path to HumanEval JSONL (downloads if not found)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use 5 embedded problems for quick testing",
    )


def _add_generation_args(parser: argparse.ArgumentParser):
    """Add generation-related arguments."""
    parser.add_argument(
        "--n", type=int, default=200,
        help="Number of samples per problem (default: 200)",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 10, 100],
        help="Values of k for pass@k (default: 1 10 100)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per completion (default: 512)",
    )


def _add_execution_args(parser: argparse.ArgumentParser):
    """Add execution-related arguments."""
    parser.add_argument(
        "--timeout", type=int, default=5,
        help="Per-execution timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel execution workers (default: 4)",
    )


def _add_output_args(parser: argparse.ArgumentParser):
    """Add output-related arguments."""
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output results as JSON",
    )


def validate_args(args) -> bool:
    """Validate CLI argument combinations."""
    if not args.mock and not args.model_dir:
        print(
            "ERROR: --model-dir is required unless --mock is set",
            file=sys.stderr,
        )
        return False
    for k in args.k:
        if k > args.n:
            print(
                f"WARNING: k={k} > n={args.n}, "
                f"pass@{k} will be skipped",
                file=sys.stderr,
            )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_problems(args) -> list[dict]:
    """Load problems based on CLI flags."""
    if args.quick:
        problems = get_quick_problems()
        print(f"Quick mode: {len(problems)} embedded problems")
        return problems
    path = resolve_humaneval_path(args.humaneval)
    problems = load_humaneval_jsonl(path)
    print(f"Loaded {len(problems)} problems from {path}")
    return problems


def main():
    """Entry point for HumanEval evaluation."""
    parser = build_parser()
    args = parser.parse_args()

    if not validate_args(args):
        sys.exit(1)

    problems = load_problems(args)
    if not problems:
        print("ERROR: No problems loaded", file=sys.stderr)
        sys.exit(1)

    _print_config(args, len(problems))
    report = run_evaluation(problems, args)

    if args.json_output:
        print_json_report(report)
    else:
        print_text_report(report, args.k)


def _print_config(args, n_problems: int):
    """Print the evaluation configuration."""
    mode = "mock" if args.mock else args.model_dir
    print(f"\nEvaluation config:")
    print(f"  Model:       {mode}")
    print(f"  Problems:    {n_problems}")
    print(f"  Samples/p:   {args.n}")
    print(f"  k values:    {args.k}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Timeout:     {args.timeout}s")
    print(f"  Workers:     {args.workers}")
    print()


if __name__ == "__main__":
    main()
