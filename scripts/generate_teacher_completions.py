#!/usr/bin/env python3
"""Generate teacher completions for distillation.

Reads prompts JSONL, sends each to the teacher model via `apr run`,
collects completions. Output: JSONL with {prompt, completion, kind, source}.

Usage (on gx10 with Qwen2.5-Coder-32B):
    scp scripts/generate_teacher_completions.py gx10:/tmp/
    scp data/distill/prompts-filtered-50k.jsonl gx10:/tmp/
    ssh gx10 'python3 /tmp/generate_teacher_completions.py \
        --model ~/src/apr-leaderboard/checkpoints/qwen2.5-coder-32b-instruct-q4km.apr \
        --prompts /tmp/prompts-filtered-50k.jsonl \
        --output /tmp/completions-50k.jsonl \
        --max-tokens 512 --limit 10000'
    scp gx10:/tmp/completions-50k.jsonl data/distill/
"""

import argparse
import json
import os
import subprocess
import sys
import time


def generate_completion(apr_path: str, model: str, prompt: str,
                        max_tokens: int) -> str | None:
    """Run apr to generate a single completion."""
    try:
        result = subprocess.run(
            [apr_path, "run", model, "--prompt", prompt,
             "--max-tokens", str(max_tokens)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None
        # apr run outputs the completion after the prompt
        output = result.stdout
        # Strip ANSI codes and header
        lines = output.split('\n')
        # Find "Output:" marker
        in_output = False
        completion_lines = []
        for line in lines:
            if 'Output:' in line:
                in_output = True
                # Get text after "Output:" on same line
                after = line.split('Output:')[-1].strip()
                if after:
                    completion_lines.append(after)
                continue
            if in_output:
                completion_lines.append(line)
        completion = '\n'.join(completion_lines).strip()
        return completion if completion else None
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  [WARN] Generation failed: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate teacher completions")
    parser.add_argument("--model", required=True, help="Path to teacher .apr model")
    parser.add_argument("--prompts", required=True, help="Prompts JSONL file")
    parser.add_argument("--output", required=True, help="Output completions JSONL")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Max prompts (0=all)")
    parser.add_argument("--apr", default="apr", help="Path to apr binary")
    args = parser.parse_args()

    # Verify model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Load prompts
    with open(args.prompts) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        prompts = prompts[:args.limit]
    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Generate completions
    completed = 0
    failed = 0
    t0 = time.time()

    with open(args.output, 'w') as out:
        for i, p in enumerate(prompts):
            completion = generate_completion(
                args.apr, args.model, p["prompt"], args.max_tokens
            )

            if completion and len(completion) > 10:
                record = {
                    "prompt": p["prompt"],
                    "completion": completion,
                    "text": p["prompt"] + completion,
                    "kind": p["kind"],
                    "source": p["source"],
                }
                out.write(json.dumps(record) + '\n')
                completed += 1
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(prompts)}] {completed} completed, "
                      f"{failed} failed, {rate:.1f} prompts/s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Completed: {completed}/{len(prompts)} ({completed/len(prompts)*100:.1f}%)")
    print(f"  Failed: {failed}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
