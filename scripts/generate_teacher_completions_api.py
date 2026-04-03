#!/usr/bin/env python3
"""Generate teacher completions via API server (realizar serve).

Resilient pipeline with retry, resume, and quality filtering.
Contract: teacher-completions-pipeline-v1.yaml (C-TEACHER-*)

Prerequisites:
    # On gx10 (or localhost):
    realizar serve --model ~/models/Qwen3-8B-Q4_K_M.gguf --gpu --openai-api --port 8090

Usage:
    python3 scripts/generate_teacher_completions_api.py \
        --server http://gx10:8090 \
        --prompts data/distill/prompts-filtered-50k.jsonl \
        --output data/distill/completions-50k.jsonl \
        --limit 10000

Resume:
    # Same command — automatically skips completed prompts (C-TEACHER-RESUME-001)
    python3 scripts/generate_teacher_completions_api.py \
        --server http://gx10:8090 \
        --prompts data/distill/prompts-filtered-50k.jsonl \
        --output data/distill/completions-50k.jsonl
"""

import argparse
import atexit
import hashlib
import json
import os
import signal
import sys
import time
import urllib.request
import urllib.error


def generate_completion(server: str, prompt: str, max_tokens: int,
                        max_retries: int = 3) -> str | None:
    """Call OpenAI-compatible completions API with retry (C-TEACHER-RETRY-001)."""
    url = f"{server}/v1/completions"
    payload = json.dumps({
        "model": "teacher",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": ["\nclass ", "\ndef ", "\n\n\n"],
    }).encode()

    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("text", "")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                ConnectionResetError, OSError) as e:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            print(f"  [WARN] API error (attempt {attempt+1}/{max_retries}): {e}",
                  flush=True)
            if attempt < max_retries - 1:
                print(f"  [WARN] Retrying in {wait}s...", flush=True)
                time.sleep(wait)
    return None


def load_completed_prompts(output_path: str) -> set[str]:
    """Load set of prompt hashes from existing output (C-TEACHER-RESUME-001)."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path) as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    h = hashlib.sha256(record["prompt"].encode()).hexdigest()[:16]
                    completed.add(h)
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def prompt_hash(prompt: str) -> str:
    """Short hash for dedup (C-TEACHER-DEDUP-001)."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8090")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Health check
    try:
        with urllib.request.urlopen(f"{args.server}/health", timeout=5) as r:
            health = json.loads(r.read())
            print(f"Server: {health.get('status')} ({health.get('compute_mode', '?')})",
                  flush=True)
    except Exception as e:
        print(f"ERROR: Server not reachable at {args.server}: {e}")
        sys.exit(1)

    # Load prompts
    with open(args.prompts) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        prompts = prompts[:args.limit]
    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Resume: load completed prompts (C-TEACHER-RESUME-001)
    completed_hashes = load_completed_prompts(args.output)
    if completed_hashes:
        print(f"Resuming: {len(completed_hashes)} already completed, "
              f"{len(prompts) - len(completed_hashes)} remaining", flush=True)

    prev_completed = len(completed_hashes)
    completed = prev_completed
    failed = 0
    skipped = 0
    t0 = time.time()

    # Summary on exit (C-TEACHER-SUMMARY-001)
    def print_summary():
        elapsed = time.time() - t0
        total = completed + failed
        print(f"\nSummary: {completed} completed, {failed} failed, "
              f"{skipped} skipped (resume), {elapsed/3600:.1f}h elapsed",
              flush=True)
        print(f"  Output: {args.output}", flush=True)

    atexit.register(print_summary)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Generate completions (append mode for resume safety)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'a') as out:
        for i, p in enumerate(prompts):
            h = prompt_hash(p["prompt"])

            # Skip completed prompts (C-TEACHER-DEDUP-001)
            if h in completed_hashes:
                skipped += 1
                continue

            completion = generate_completion(
                args.server, p["prompt"], args.max_tokens, args.max_retries)

            # Quality filter (C-TEACHER-QUALITY-001)
            if completion and len(completion.strip()) > 10:
                record = {
                    "prompt": p["prompt"],
                    "completion": completion,
                    "text": p["prompt"] + completion,
                    "kind": p["kind"],
                    "source": p["source"],
                }
                out.write(json.dumps(record) + '\n')
                out.flush()  # C-TEACHER-PROGRESS-001
                completed += 1
                completed_hashes.add(h)
            else:
                failed += 1

            # Progress logging (C-TEACHER-LOG-001)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                new_this_run = (completed - prev_completed) + failed
                if new_this_run > 0 and elapsed > 0:
                    rate = new_this_run / elapsed * 3600
                    remaining = len(prompts) - i - 1
                    eta = remaining / (new_this_run / elapsed)
                    print(f"  [{i+1}/{len(prompts)}] {completed} ok, {failed} fail, "
                          f"{skipped} skip, {rate:.0f}/h, ETA {eta/3600:.1f}h",
                          flush=True)


if __name__ == "__main__":
    main()
