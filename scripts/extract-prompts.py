#!/usr/bin/env python3
"""Extract Python function/class prompts from codeparrot-clean for distillation.

Reads raw codeparrot Parquet shards, extracts function signatures with optional
docstrings as prompts for teacher model completion.

Output: JSONL with {"prompt": "def ...", "source": "shard-NNNN:row", "kind": "function|class"}

Usage:
    python scripts/extract-prompts.py \
        --input /mnt/nvme-raid0/albor-data/codeparrot-clean/ \
        --output data/distill/prompts.jsonl \
        --max-prompts 100000 \
        --min-body-lines 3

Prompt extraction strategy (spec 4.5):
    - Function headers: `def name(args):` + docstring (if present)
    - Class headers: `class Name(bases):` + docstring + __init__ signature
    - Method headers: indented `def name(self, ...):`
    - Skip: single-line functions, test functions, __dunder__ except __init__
"""

import argparse
import ast
import json
import os
import random
import sys
import textwrap
from pathlib import Path

import pyarrow.parquet as pq


def extract_functions_from_source(source: str) -> list[dict]:
    """Extract function/class headers from Python source code.

    Returns list of {"prompt": str, "kind": str, "original_body_lines": int}
    """
    prompts = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return prompts

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prompt = _extract_function_prompt(node, source)
            if prompt:
                prompts.append(prompt)
        elif isinstance(node, ast.ClassDef):
            prompt = _extract_class_prompt(node, source)
            if prompt:
                prompts.append(prompt)

    return prompts


def _extract_function_prompt(
    node: ast.FunctionDef | ast.AsyncFunctionDef, source: str
) -> dict | None:
    """Extract a function header as a prompt."""
    name = node.name

    # Skip test functions, dunders (except __init__), and very short functions
    if name.startswith("test_"):
        return None
    if name.startswith("__") and name.endswith("__") and name != "__init__":
        return None

    # Count body lines (excluding docstring)
    body = node.body
    body_start = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body_start = 1  # skip docstring

    body_lines = sum(
        (getattr(stmt, "end_lineno", stmt.lineno) - stmt.lineno + 1)
        for stmt in body[body_start:]
    )

    if body_lines < 2:
        return None

    # Build the prompt: signature + docstring
    lines = source.splitlines()
    # Get the def line
    def_line_start = node.lineno - 1  # 0-indexed
    # Find where the body starts
    if body:
        body_first_line = body[0].lineno - 1
    else:
        return None

    # Include everything from def line to end of docstring (or just def line)
    has_docstring = body_start == 1
    if (
        has_docstring
        and hasattr(body[0], "end_lineno")
        and body[0].end_lineno is not None
    ):
        # Include docstring
        prompt_end = body[0].end_lineno  # 1-indexed, inclusive
        prompt_lines = lines[def_line_start:prompt_end]
    else:
        # Just the signature (may span multiple lines for long arg lists)
        prompt_end = body_first_line
        prompt_lines = lines[def_line_start:prompt_end]

    if not prompt_lines:
        return None

    prompt_text = "\n".join(prompt_lines) + "\n"

    # Dedent if it's a method (indented)
    if prompt_text and prompt_text[0] == " ":
        prompt_text = textwrap.dedent(prompt_text)

    # Skip very short prompts (no context for teacher)
    if len(prompt_text.strip().splitlines()) < 2 and not has_docstring:
        return None

    return {
        "prompt": prompt_text,
        "kind": "function",
        "original_body_lines": body_lines,
        "has_docstring": has_docstring,
    }


def _extract_class_prompt(node: ast.ClassDef, source: str) -> dict | None:
    """Extract a class header as a prompt (class line + docstring + __init__ sig)."""
    name = node.name

    # Skip very small classes
    body = node.body
    if len(body) < 2:
        return None

    lines = source.splitlines()
    class_line_start = node.lineno - 1

    # Find docstring end
    has_docstring = (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )

    if has_docstring and hasattr(body[0], "end_lineno") and body[0].end_lineno:
        prompt_end = body[0].end_lineno
    else:
        # Just the class line
        if body:
            prompt_end = body[0].lineno - 1
        else:
            return None

    # Find __init__ signature if present
    init_end = prompt_end
    for stmt in body:
        if (
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name == "__init__"
        ):
            # Include __init__ signature + its docstring
            init_body = stmt.body
            if (
                init_body
                and isinstance(init_body[0], ast.Expr)
                and isinstance(init_body[0].value, ast.Constant)
                and isinstance(init_body[0].value.value, str)
                and hasattr(init_body[0], "end_lineno")
                and init_body[0].end_lineno
            ):
                init_end = init_body[0].end_lineno
            elif init_body:
                init_end = init_body[0].lineno - 1
            else:
                init_end = stmt.end_lineno or stmt.lineno
            break

    prompt_end = max(prompt_end, init_end)
    prompt_lines = lines[class_line_start:prompt_end]

    if not prompt_lines:
        return None

    prompt_text = "\n".join(prompt_lines) + "\n"
    prompt_text = textwrap.dedent(prompt_text)

    total_body_lines = (
        (node.end_lineno or node.lineno) - node.lineno - (prompt_end - class_line_start)
    )

    if total_body_lines < 3:
        return None

    return {
        "prompt": prompt_text,
        "kind": "class",
        "original_body_lines": total_body_lines,
    }


def process_shard(
    shard_path: Path,
    max_prompts_per_shard: int | None = None,
) -> list[dict]:
    """Process a single Parquet shard and extract prompts."""
    table = pq.read_table(str(shard_path))
    shard_name = shard_path.stem
    prompts = []

    for row_idx in range(len(table)):
        source = str(table["text"][row_idx])
        extracted = extract_functions_from_source(source)

        for item in extracted:
            item["source"] = f"{shard_name}:{row_idx}"
            prompts.append(item)

            if max_prompts_per_shard and len(prompts) >= max_prompts_per_shard:
                return prompts

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Extract Python function/class prompts for distillation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/nvme-raid0/albor-data/codeparrot-clean/"),
        help="Directory containing raw codeparrot Parquet shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/distill/prompts.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=100_000,
        help="Maximum number of prompts to extract",
    )
    parser.add_argument(
        "--min-body-lines",
        type=int,
        default=3,
        help="Minimum function body lines (filters trivial functions)",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Process only first N shards (for testing)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle output prompts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    # Find shard files
    shards = sorted(args.input.glob("shard-*.parquet"))
    if not shards:
        print(f"No shard files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.max_shards:
        shards = shards[: args.max_shards]

    print(f"Processing {len(shards)} shards from {args.input}")
    print(f"Target: {args.max_prompts} prompts, min body lines: {args.min_body_lines}")

    all_prompts = []
    for i, shard in enumerate(shards):
        remaining = args.max_prompts - len(all_prompts)
        if remaining <= 0:
            break

        prompts = process_shard(shard, max_prompts_per_shard=remaining)

        # Filter by min body lines
        prompts = [p for p in prompts if p["original_body_lines"] >= args.min_body_lines]

        all_prompts.extend(prompts)
        print(
            f"  [{i+1}/{len(shards)}] {shard.name}: {len(prompts)} prompts "
            f"(total: {len(all_prompts)})"
        )

    # Truncate to max
    if len(all_prompts) > args.max_prompts:
        all_prompts = all_prompts[: args.max_prompts]

    # Shuffle
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(all_prompts)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for prompt in all_prompts:
            # Remove internal field before writing
            body_lines = prompt.pop("original_body_lines", 0)
            json.dump(prompt, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nWrote {len(all_prompts)} prompts to {args.output}")

    # Stats
    kinds = {}
    for p in all_prompts:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    for kind, count in sorted(kinds.items()):
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
