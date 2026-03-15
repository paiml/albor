#!/usr/bin/env python3
"""Generate synthetic Python code via teacher model for distillation.

Reads prompts JSONL (from extract-prompts.py), calls realizar for each
prompt to generate completions, and writes synthetic training data.

Two modes:
  --mode server: Use HTTP API (requires `realizar serve --gpu`)
  --mode subprocess: Call `realizar run --gpu --format json` per prompt
                     (slower due to 5s model load per request, but works
                     with Q4K GPU path which serve doesn't support yet)

Output: JSONL with {"text": "prompt + completion", "prompt": "...",
                     "completion": "...", "tokens_generated": N, "source": "..."}

Usage:
    # Subprocess mode (Q4K GPU, works now):
    python scripts/generate-synthetic.py \
        --prompts data/distill/prompts.jsonl \
        --output data/distill/synthetic.jsonl \
        --mode subprocess \
        --model /mnt/nvme-raid0/models/qwen3-coder-30b-q4k.apr \
        --max-tokens 256 --temperature 0.8 --max-samples 50

    # Server mode (when serve supports GPU Q4K):
    python scripts/generate-synthetic.py \
        --prompts data/distill/prompts.jsonl \
        --output data/distill/synthetic.jsonl \
        --mode server --server http://localhost:8080

Spec 4.5: Temperature 0.8, top-p 0.95, max 512 tokens, 1-3 samples per prompt.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def generate_via_subprocess(
    realizar_bin: str,
    model_path: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.8,
) -> dict | None:
    """Call `realizar run` as subprocess for Q4K GPU inference."""
    cmd = [
        realizar_bin,
        "run",
        model_path,
        prompt,
        "--gpu",
        "--raw",
        "--format", "json",
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            # Filter out verbose loading lines, show only real errors
            stderr_lines = [
                l for l in result.stderr.strip().splitlines()
                if not l.startswith("[") and not l.startswith("Backend:")
                and not l.startswith("Loading") and not l.startswith("Model:")
                and not l.startswith("MoE:") and not l.startswith("Uploaded")
                and not l.startswith("[GpuProfile]") and not l.startswith("Prompt")
                and not l.startswith("Temperature")
            ]
            if stderr_lines:
                print(f"  Error: {stderr_lines[-1]}", file=sys.stderr)
            return None

        # Parse JSON output
        output = result.stdout.strip()
        if not output:
            return None

        data = json.loads(output)
        return {
            "text": data.get("generated_text", ""),
            "tokens_generated": data.get("tokens_generated", 0),
            "tok_per_sec": data.get("tokens_per_second", 0),
        }
    except subprocess.TimeoutExpired:
        print("  Timeout", file=sys.stderr)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Parse error: {e}", file=sys.stderr)
        return None


def generate_via_server(
    server: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.8,
) -> dict | None:
    """Send a prompt to the realizar HTTP server."""
    import requests

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            f"{server}/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "text": data.get("text", ""),
            "tokens_generated": data.get("num_generated", data.get("tokens_generated", 0)),
        }
    except Exception as e:
        print(f"  Request failed: {e}", file=sys.stderr)
        return None


def filter_completion(text: str, prompt: str) -> str:
    """Extract the completion part from full generated text.

    The realizar output includes the prompt + completion concatenated.
    We need to strip the prompt and clean up the completion.
    """
    # Strip prompt prefix
    if text.startswith(prompt):
        completion = text[len(prompt):]
    else:
        completion = text

    # Remove trailing whitespace-only lines
    lines = completion.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def check_server(server: str) -> bool:
    """Check if the realizar server is running."""
    import requests
    try:
        resp = requests.get(f"{server}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Python code via teacher model"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Input prompts JSONL from extract-prompts.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/distill/synthetic.jsonl"),
        help="Output synthetic data JSONL",
    )
    parser.add_argument(
        "--mode",
        choices=["server", "subprocess"],
        default="subprocess",
        help="Inference mode: server (HTTP API) or subprocess (realizar run per prompt)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8080",
        help="Realizar server URL (for server mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.apr",
        help="Model path (for subprocess mode)",
    )
    parser.add_argument(
        "--realizar-bin",
        type=str,
        default="/mnt/nvme-raid0/targets/realizar/release/realizar",
        help="Path to realizar binary",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per completion",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of prompts to process",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Number of completions per prompt (1-3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already-processed sources)",
    )
    parser.add_argument(
        "--min-completion-tokens",
        type=int,
        default=10,
        help="Skip completions shorter than this",
    )
    args = parser.parse_args()

    # Validate mode
    if args.mode == "server":
        if not check_server(args.server):
            print(f"Server not reachable at {args.server}", file=sys.stderr)
            print("Start it with: realizar serve --model <path> --gpu", file=sys.stderr)
            sys.exit(1)
    elif args.mode == "subprocess":
        bin_path = Path(args.realizar_bin)
        if not bin_path.exists():
            print(f"Realizar binary not found: {args.realizar_bin}", file=sys.stderr)
            sys.exit(1)
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Model not found: {args.model}", file=sys.stderr)
            sys.exit(1)

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))

    if args.max_samples:
        prompts = prompts[: args.max_samples]

    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    print(f"Mode: {args.mode}")
    if args.mode == "subprocess":
        print(f"Model: {args.model}")
        print(f"Binary: {args.realizar_bin}")
    else:
        print(f"Server: {args.server}")
    print(f"Max tokens: {args.max_tokens}, temperature: {args.temperature}")
    print(f"Samples per prompt: {args.samples_per_prompt}")

    # Load already-processed sources for resume
    processed_sources = set()
    if args.resume and args.output.exists():
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                processed_sources.add(d.get("source", ""))
        print(f"Resuming: {len(processed_sources)} already processed")

    # Generate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    total_tokens = 0
    generated = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    with open(args.output, mode) as out:
        for i, prompt_data in enumerate(prompts):
            source = prompt_data.get("source", f"prompt-{i}")

            if source in processed_sources:
                skipped += 1
                continue

            prompt = prompt_data["prompt"]

            for sample_idx in range(args.samples_per_prompt):
                if args.mode == "subprocess":
                    result = generate_via_subprocess(
                        args.realizar_bin,
                        args.model,
                        prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                else:
                    result = generate_via_server(
                        args.server,
                        prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )

                if result is None:
                    errors += 1
                    continue

                full_text = result.get("text", "")
                tokens = result.get("tokens_generated", 0)

                if tokens < args.min_completion_tokens:
                    skipped += 1
                    continue

                completion = filter_completion(full_text, prompt)
                text = prompt + completion

                record = {
                    "text": text,
                    "prompt": prompt,
                    "completion": completion,
                    "tokens_generated": tokens,
                    "source": source,
                    "kind": prompt_data.get("kind", "unknown"),
                    "sample_idx": sample_idx,
                }

                json.dump(record, out, ensure_ascii=False)
                out.write("\n")
                out.flush()

                total_tokens += tokens
                generated += 1

            # Progress
            elapsed = time.time() - start_time
            if elapsed > 0:
                tok_per_sec = total_tokens / elapsed
            else:
                tok_per_sec = 0

            if (i + 1) % 5 == 0 or i == len(prompts) - 1:
                print(
                    f"  [{i+1}/{len(prompts)}] generated={generated} "
                    f"tokens={total_tokens} ({tok_per_sec:.1f} tok/s) "
                    f"errors={errors} skipped={skipped}"
                )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Generated: {generated} completions, {total_tokens} tokens")
    print(f"Errors: {errors}, Skipped: {skipped}")
    print(f"Output: {args.output}")
    if elapsed > 0:
        print(f"Throughput: {total_tokens / elapsed:.1f} tok/s")
        if args.mode == "subprocess":
            print(
                f"  (includes ~5s model load per prompt — "
                f"effective decode: ~13 tok/s)"
            )


if __name__ == "__main__":
    main()
