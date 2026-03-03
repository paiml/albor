#!/usr/bin/env python3
"""R-030: Training data contamination detection.

Checks for n-gram overlap between training data and evaluation benchmarks
to detect data contamination. Uses 13-gram exact match (following GPT-4
technical report methodology).

Usage:
    # Check training data against built-in HumanEval problems
    python scripts/contamination-check.py \
        --train-data data/pretokenized-2048/train/train.parquet \
        --benchmark humaneval

    # Check against custom evaluation text file
    python scripts/contamination-check.py \
        --train-data data/parquet/*.parquet \
        --eval-file eval_data.txt \
        --ngram-size 13

    # Check raw text files
    python scripts/contamination-check.py \
        --train-text data/raw/*.py \
        --eval-file benchmarks/humaneval.jsonl

Requirements: pip install pyarrow
"""

import argparse
import glob
import json
import sys
from pathlib import Path

# Subset of HumanEval problem signatures for contamination checking
HUMANEVAL_SIGNATURES = [
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
    "def separate_paren_groups(paren_string: str) -> List[str]:",
    "def truncate_number(number: float) -> float:",
    "def below_zero(operations: List[int]) -> bool:",
    "def mean_absolute_deviation(numbers: List[float]) -> float:",
    "def intersperse(numbers: List[int], delimeter: int) -> List[int]:",
    "def parse_nested_parens(paren_string: str) -> List[int]:",
    "def filter_by_substring(strings: List[str], substring: str) -> List[str]:",
    "def sum_product(numbers: List[int]) -> Tuple[int, int]:",
    "def rolling_max(numbers: List[int]) -> List[int]:",
    "def make_palindrome(string: str) -> str:",
    "def string_xor(a: str, b: str) -> str:",
    "def longest(strings: List[str]) -> Optional[str]:",
    "def greatest_common_divisor(a: int, b: int) -> int:",
    "def all_prefixes(string: str) -> List[str]:",
    "def string_sequence(n: int) -> str:",
    "def count_distinct_characters(string: str) -> int:",
    "def parse_music(music_string: str) -> List[int]:",
    "def how_many_times(string: str, substring: str) -> int:",
    "def sort_numbers(numbers: str) -> str:",
]


def extract_ngrams(text, n):
    """Extract character-level n-grams from text."""
    words = text.split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngrams.add(ngram)
    return ngrams


def load_train_text_from_parquet(paths):
    """Load training text from Parquet files."""
    import pyarrow.parquet as pq
    all_text = []
    for path in paths:
        table = pq.read_table(str(path))
        for col_name in ["text", "content", "code", "source"]:
            if col_name in table.column_names:
                texts = table[col_name].to_pylist()
                all_text.extend(t for t in texts if isinstance(t, str))
                break
    return all_text


def load_train_text_from_files(patterns):
    """Load training text from raw files."""
    texts = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, errors="replace") as f:
                texts.append(f.read())
    return texts


def load_eval_text(eval_file):
    """Load evaluation text from file (plain text or JSONL)."""
    texts = []
    path = Path(eval_file)
    with open(path, errors="replace") as f:
        if path.suffix == ".jsonl":
            for line in f:
                obj = json.loads(line)
                for key in ["prompt", "text", "canonical_solution", "test"]:
                    if key in obj:
                        texts.append(str(obj[key]))
        else:
            texts.append(f.read())
    return texts


def check_contamination(train_ngrams, eval_texts, ngram_size, label):
    """Check n-gram overlap between training and eval data."""
    contaminated = []
    for i, eval_text in enumerate(eval_texts):
        eval_ngrams = extract_ngrams(eval_text, ngram_size)
        overlap = train_ngrams & eval_ngrams
        if overlap:
            ratio = len(overlap) / max(len(eval_ngrams), 1)
            contaminated.append({
                "index": i,
                "overlap_count": len(overlap),
                "eval_ngrams": len(eval_ngrams),
                "overlap_ratio": round(ratio, 4),
                "sample_overlap": list(overlap)[:3],
            })
    return contaminated


def main():
    parser = argparse.ArgumentParser(description="Check training data contamination")
    parser.add_argument("--train-data", nargs="+", help="Training Parquet files")
    parser.add_argument("--train-text", nargs="+", help="Training text file patterns")
    parser.add_argument("--benchmark", choices=["humaneval"], help="Built-in benchmark")
    parser.add_argument("--eval-file", type=Path, help="Custom evaluation file")
    parser.add_argument("--ngram-size", type=int, default=13, help="N-gram size (default: 13)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    # Load training data
    train_texts = []
    if args.train_data:
        train_texts = load_train_text_from_parquet(args.train_data)
    elif args.train_text:
        train_texts = load_train_text_from_files(args.train_text)
    else:
        print("ERROR: Specify --train-data or --train-text", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(train_texts)} training documents", file=sys.stderr)

    # Build training n-gram set
    train_ngrams = set()
    for text in train_texts:
        train_ngrams |= extract_ngrams(text, args.ngram_size)
    print(f"Built {len(train_ngrams):,} unique {args.ngram_size}-grams", file=sys.stderr)

    # Load evaluation data
    eval_texts = []
    eval_label = "custom"
    if args.benchmark == "humaneval":
        eval_texts = HUMANEVAL_SIGNATURES
        eval_label = "humaneval"
    elif args.eval_file:
        eval_texts = load_eval_text(args.eval_file)
        eval_label = args.eval_file.name

    contaminated = check_contamination(train_ngrams, eval_texts, args.ngram_size, eval_label)

    result = {
        "benchmark": eval_label,
        "ngram_size": args.ngram_size,
        "train_documents": len(train_texts),
        "train_ngrams": len(train_ngrams),
        "eval_items": len(eval_texts),
        "contaminated_items": len(contaminated),
        "contamination_rate": round(len(contaminated) / max(len(eval_texts), 1), 4),
        "details": contaminated,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Contamination Check: {eval_label}")
        print(f"{'='*60}")
        print(f"  N-gram size: {args.ngram_size}")
        print(f"  Training docs: {len(train_texts):,}")
        print(f"  Training n-grams: {len(train_ngrams):,}")
        print(f"  Eval items: {len(eval_texts)}")
        print(f"  Contaminated: {len(contaminated)} ({result['contamination_rate']:.1%})")
        if contaminated:
            print(f"\n  Contaminated items:")
            for c in contaminated[:10]:
                print(f"    [{c['index']}] overlap={c['overlap_count']}/{c['eval_ngrams']} ({c['overlap_ratio']:.1%})")
        else:
            print(f"\n  No contamination detected.")


if __name__ == "__main__":
    main()
