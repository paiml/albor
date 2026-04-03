#!/usr/bin/env python3
"""Extract distillation prompts from filtered codeparrot corpus.

Parses Python files, extracts function/class definitions with docstrings,
creates prompts for teacher model completion. The teacher (Qwen3-Coder)
generates the function body; the student trains on prompt+completion.

Prompt types:
  - function: "def func_name(args):\n    \"\"\"docstring\"\"\"\n"
  - class: "class ClassName:\n    \"\"\"docstring\"\"\"\n"
  - method: "    def method(self, args):\n        \"\"\"docstring\"\"\"\n"

Output: JSONL with {prompt, kind, source, signature, docstring}

Usage:
    python3 scripts/extract_distill_prompts.py \
        --input data/filtered/train/ \
        --output data/distill/prompts-filtered.jsonl \
        --limit 50000
"""

import argparse
import ast
import json
import os
import sys
import textwrap
import glob

import pyarrow.parquet as pq


def extract_prompts_from_code(code: str, source: str) -> list[dict]:
    """Extract function/class prompts with docstrings from Python code."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return []

    lines = code.split('\n')
    prompts = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        # Must have a docstring
        if not (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            continue

        docstring = node.body[0].value.value
        if len(docstring.strip()) < 10:
            continue

        # Extract the signature line(s) + docstring as prompt
        start_line = node.lineno - 1  # 0-indexed
        # Find the end of the docstring
        doc_end_line = node.body[0].end_lineno  # 1-indexed

        if start_line < 0 or doc_end_line is None or doc_end_line > len(lines):
            continue

        # Prompt = everything from def/class line through end of docstring
        prompt_lines = lines[start_line:doc_end_line]
        prompt = '\n'.join(prompt_lines) + '\n'

        # Skip very short or very long prompts
        if len(prompt) < 30 or len(prompt) > 2000:
            continue

        # Determine kind
        if isinstance(node, ast.ClassDef):
            kind = "class"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check if it's a method (inside a class)
            kind = "function"
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.iter_child_nodes(parent):
                        if child is node:
                            kind = "method"
                            break
        else:
            kind = "function"

        # Extract just the signature (first line)
        sig_line = lines[start_line].strip()

        # Skip dunder methods and trivial functions
        name = node.name
        if name.startswith('__') and name.endswith('__') and name != '__init__':
            continue

        prompts.append({
            "prompt": prompt,
            "kind": kind,
            "source": source,
            "signature": sig_line,
            "docstring": docstring.strip()[:200],
        })

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Extract distillation prompts")
    parser.add_argument("--input", default="data/filtered/train/")
    parser.add_argument("--output", default="data/distill/prompts-filtered.jsonl")
    parser.add_argument("--limit", type=int, default=50000, help="Max prompts to extract")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not files:
        print(f"ERROR: No parquet files in {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_files = 0
    total_prompts = 0
    kind_counts = {"function": 0, "method": 0, "class": 0}

    with open(args.output, 'w') as out:
        for fi, f in enumerate(files):
            table = pq.read_table(f, columns=["content"])
            for row_idx, text_col in enumerate(table["content"]):
                code = text_col.as_py()
                if not code:
                    continue
                total_files += 1

                source = f"{os.path.basename(f)}:{row_idx}"
                prompts = extract_prompts_from_code(code, source)

                for p in prompts:
                    out.write(json.dumps(p) + '\n')
                    total_prompts += 1
                    kind_counts[p["kind"]] += 1

                    if total_prompts >= args.limit:
                        break

                if total_prompts >= args.limit:
                    break

            print(f"  [{fi+1}/{len(files)}] {os.path.basename(f)}: "
                  f"{total_prompts:,} prompts from {total_files:,} files",
                  flush=True)

            if total_prompts >= args.limit:
                break

    print(f"\nDone: {total_prompts:,} prompts from {total_files:,} files")
    print(f"  function: {kind_counts['function']:,}")
    print(f"  method: {kind_counts['method']:,}")
    print(f"  class: {kind_counts['class']:,}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
