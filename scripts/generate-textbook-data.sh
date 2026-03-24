#!/usr/bin/env bash
# generate-textbook-data.sh — Generate synthetic textbook data from Qwen3-Coder-30B
#
# Uses the teacher model to generate high-quality Python completions from
# textbook-style prompts. Follows phi-1 playbook: textbook data >> raw code.
#
# Requires: GPU free (not running training), Qwen3-Coder-30B Q4K model
#
# Usage:
#   ./scripts/generate-textbook-data.sh                    # all 118 prompts
#   ./scripts/generate-textbook-data.sh --limit 10         # first 10 only (test)
#   SAMPLES=5 ./scripts/generate-textbook-data.sh          # 5 completions per prompt
set -euo pipefail

APR="${APR:-/mnt/nvme-raid0/targets/aprender/release/apr}"
TEACHER="/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.apr"
PROMPTS="data/distill/textbook-prompts-v2.jsonl"
OUTPUT="data/distill/textbook-completions-v1.jsonl"
SAMPLES="${SAMPLES:-3}"
LIMIT="${1:---limit 0}"  # 0 = all

if [ ! -f "$TEACHER" ]; then
    echo "ERROR: Teacher model not found at $TEACHER"
    echo "Download: apr model pull Qwen/Qwen3-Coder-30B-A3B-Instruct"
    exit 1
fi

echo "=== Textbook Data Generation ==="
echo "  Teacher: $TEACHER"
echo "  Prompts: $PROMPTS ($(wc -l < "$PROMPTS") total)"
echo "  Samples per prompt: $SAMPLES"
echo "  Output: $OUTPUT"
echo ""

# Generate completions
python3 -c "
import json, subprocess, sys

prompts_file = '$PROMPTS'
output_file = '$OUTPUT'
teacher = '$TEACHER'
samples = $SAMPLES
limit = int('${LIMIT##--limit }'.strip() or '0')

with open(prompts_file) as f:
    prompts = [json.loads(l) for l in f]

if limit > 0:
    prompts = prompts[:limit]

print(f'Generating {len(prompts)} × {samples} = {len(prompts)*samples} completions...')

results = []
for i, prompt_data in enumerate(prompts):
    prompt = prompt_data['prompt']
    category = prompt_data.get('category', 'unknown')

    for s in range(samples):
        # Use apr run for inference
        cmd = [
            '$APR', 'run', teacher,
            '--prompt', prompt,
            '--max-tokens', '512',
            '--temperature', '0.7',
            '--device', 'cuda',
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            completion = result.stdout.strip()
            if completion:
                entry = {
                    'prompt': prompt,
                    'completion': completion,
                    'category': category,
                    'kind': 'textbook',
                    'sample': s,
                    'tokens': len(completion.split()),  # rough estimate
                }
                results.append(entry)
        except Exception as e:
            print(f'  [{i}/{len(prompts)}] Sample {s} failed: {e}', file=sys.stderr)

    if (i + 1) % 10 == 0:
        print(f'  [{i+1}/{len(prompts)}] {len(results)} completions generated')

with open(output_file, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

print(f'Done: {len(results)} completions written to {output_file}')
"
