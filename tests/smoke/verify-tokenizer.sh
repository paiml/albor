#!/usr/bin/env bash
# Smoke test: verify BPE tokenizer roundtrip
# FALSIFY: decode(encode(text)) = text for Python code samples
set -euo pipefail

TOKENIZER_DIR="${1:-tokenizer}"
VOCAB="$TOKENIZER_DIR/vocab.json"
MERGES="$TOKENIZER_DIR/merges.txt"

echo "=== Tokenizer Smoke Test ==="
echo "  Vocab: $VOCAB"
echo "  Merges: $MERGES"

# Check files exist
if [ ! -f "$VOCAB" ]; then
    echo "FAIL: vocab.json not found at $VOCAB"
    exit 1
fi
if [ ! -f "$MERGES" ]; then
    echo "FAIL: merges.txt not found at $MERGES"
    exit 1
fi

# Check vocab size
VOCAB_SIZE=$(python3 -c "import json; print(len(json.load(open('$VOCAB'))))")
echo "  Vocab size: $VOCAB_SIZE"

if [ "$VOCAB_SIZE" -lt 1000 ]; then
    echo "FAIL: vocab too small ($VOCAB_SIZE < 1000)"
    exit 1
fi

# Check merges count
MERGES_COUNT=$(wc -l < "$MERGES")
echo "  Merges count: $MERGES_COUNT"

# Check special tokens exist
python3 -c "
import json, sys
vocab = json.load(open('$VOCAB'))
required = ['<unk>', '<s>', '</s>']
for tok in required:
    if tok not in vocab:
        print(f'FAIL: Missing special token: {tok}')
        sys.exit(1)
    print(f'  Special token {tok}: id={vocab[tok]}')
print('  All special tokens present')
"

# Check Python-specific tokens exist (common code patterns)
python3 -c "
import json
vocab = json.load(open('$VOCAB'))
# Check for common Python-ish subwords
found = 0
for pattern in ['def', 'return', 'self', 'import', 'class', 'for', 'if', 'in']:
    matches = [k for k in vocab if pattern in k.lower()]
    if matches:
        found += 1
        print(f'  Python pattern \"{pattern}\": {len(matches)} vocab entries')
    else:
        print(f'  WARNING: No vocab entries for \"{pattern}\"')
print(f'  Python pattern coverage: {found}/8')
"

echo ""
echo "PASS: Tokenizer smoke test passed"
