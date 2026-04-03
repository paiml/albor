#!/usr/bin/env python3
"""Generate teacher completions via API server (apr serve).

Much faster than subprocess-per-prompt: model stays loaded, HTTP reuse.

Prerequisites:
    # On gx10:
    apr serve run ~/src/apr-leaderboard/checkpoints/qwen2.5-coder-32b-instruct-q4km.apr --port 8090

Usage:
    python3 scripts/generate_teacher_completions_api.py \
        --server http://gx10:8090 \
        --prompts data/distill/prompts-filtered-50k.jsonl \
        --output data/distill/completions-50k.jsonl \
        --limit 10000
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error


def generate_completion(server: str, prompt: str, max_tokens: int) -> str | None:
    """Call OpenAI-compatible completions API."""
    url = f"{server}/v1/completions"
    payload = json.dumps({
        "model": "teacher",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": ["\nclass ", "\ndef ", "\n\n\n"],
    }).encode()

    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("text", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  [WARN] API error: {e}", flush=True)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8090")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
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

    with open(args.prompts) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        prompts = prompts[:args.limit]
    print(f"Loaded {len(prompts)} prompts", flush=True)

    completed = 0
    failed = 0
    t0 = time.time()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as out:
        for i, p in enumerate(prompts):
            completion = generate_completion(args.server, p["prompt"], args.max_tokens)

            if completion and len(completion.strip()) > 10:
                record = {
                    "prompt": p["prompt"],
                    "completion": completion,
                    "text": p["prompt"] + completion,
                    "kind": p["kind"],
                    "source": p["source"],
                }
                out.write(json.dumps(record) + '\n')
                out.flush()
                completed += 1
            else:
                failed += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 3600
                eta = (len(prompts) - i - 1) / max(1, (i + 1) / elapsed)
                print(f"  [{i+1}/{len(prompts)}] {completed} ok, {failed} fail, "
                      f"{rate:.0f}/h, ETA {eta/3600:.1f}h", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/3600:.1f}h")
    print(f"  Completed: {completed}/{len(prompts)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
