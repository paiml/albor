#!/usr/bin/env python3
"""Train ByteLevel BPE tokenizer for Albor (v2).

Produces a whitespace-preserving tokenizer suitable for Python code models.
Output: models/albor-tokenizer-v2/tokenizer.json

Usage:
    python3 scripts/train-tokenizer-v2.py --corpus data/staging/corpus-raw.txt
    python3 scripts/train-tokenizer-v2.py --corpus data/staging/corpus-raw.txt --vocab-size 32768
"""
import argparse
import json
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, AddedToken


def main():
    parser = argparse.ArgumentParser(description="Train ByteLevel BPE tokenizer")
    parser.add_argument("--corpus", required=True, help="Path to training corpus (text file)")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Target vocabulary size")
    parser.add_argument("--output", default="models/albor-tokenizer-v2", help="Output directory")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency")
    args = parser.parse_args()

    special_tokens = [
        "<unk>", "<s>", "</s>", "<pad>",
        "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
    ]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=args.min_frequency,
        show_progress=True,
    )

    print(f"Training ByteLevel BPE tokenizer (vocab={args.vocab_size})...")
    tokenizer.train([args.corpus], trainer)
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    for st in special_tokens:
        tokenizer.add_special_tokens([AddedToken(st, special=True)])

    # Roundtrip validation
    test_cases = [
        "def hello():\n    return 42",
        "import os\nimport sys",
        "class Foo:\n    def __init__(self):\n        self.x = 1",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
        "    indented_line = True",
        "# comment\n\ndef func():\n    pass",
    ]

    print("\nRoundtrip validation:")
    all_pass = True
    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        if decoded != text:
            print(f"  FAIL: {repr(text[:60])}")
            print(f"     -> {repr(decoded[:60])}")
            all_pass = False
        else:
            print(f"  PASS: {repr(text[:60])} -> {len(encoded.ids)} tokens")

    if not all_pass:
        print("\nWARNING: Some roundtrip tests failed!")
        return 1

    # Convert merges from arrays to strings for entrenar/aprender compatibility
    os.makedirs(args.output, exist_ok=True)
    tokenizer.save(os.path.join(args.output, "tokenizer.json"))

    # Post-process: convert merges format
    path = os.path.join(args.output, "tokenizer.json")
    with open(path) as f:
        data = json.load(f)
    if data["model"]["merges"] and isinstance(data["model"]["merges"][0], list):
        data["model"]["merges"] = [f"{m[0]} {m[1]}" for m in data["model"]["merges"]]
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print("Converted merges to string format (entrenar compatibility)")

    size = os.path.getsize(path)
    print(f"\nSaved: {path} ({size:,} bytes)")
    print("All roundtrip tests: PASS")
    return 0


if __name__ == "__main__":
    exit(main())
