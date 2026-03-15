#!/usr/bin/env python3
"""Mix synthetic distillation data with codeparrot for student training.

Pipeline:
  1. Read synthetic-5m.jsonl (teacher completions)
  2. Tokenize with albor tokenizer
  3. Pack into seq_len sequences (concatenate + chunk)
  4. Sample codeparrot shards for 90/10 mix
  5. Write mixed pretokenized parquet

Usage:
    python scripts/mix-distill-data.py \
        --synthetic data/distill/synthetic-5m.jsonl \
        --codeparrot data/pretokenized-1024-v3/train/ \
        --output data/distill/mixed-v3/ \
        --tokenizer models/albor-tokenizer-v2/tokenizer.json \
        --seq-len 1024 \
        --synthetic-ratio 0.10
"""
import argparse
import glob
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def tokenize_synthetic(jsonl_path: str, tokenizer: Tokenizer, seq_len: int) -> list[list[int]]:
    """Tokenize synthetic completions and pack into fixed-length sequences."""
    all_tokens: list[int] = []
    count = 0
    skipped = 0

    with open(jsonl_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                text = d.get("completion", "")
                if not text or len(text) < 20:  # skip trivially short
                    skipped += 1
                    continue
                enc = tokenizer.encode(text)
                all_tokens.extend(enc.ids)
                count += 1
            except (json.JSONDecodeError, KeyError):
                skipped += 1

    print(f"Synthetic: {count} completions, {skipped} skipped")
    print(f"  Total tokens: {len(all_tokens):,}")

    # Pack into seq_len sequences
    num_seqs = len(all_tokens) // seq_len
    sequences = []
    for i in range(num_seqs):
        start = i * seq_len
        sequences.append(all_tokens[start:start + seq_len])

    print(f"  Packed into {num_seqs} sequences of length {seq_len}")
    print(f"  Dropped {len(all_tokens) - num_seqs * seq_len} remainder tokens")
    return sequences


def sample_codeparrot(codeparrot_dir: str, n_seqs: int, seed: int = 42) -> list[list[int]]:
    """Sample n_seqs sequences from codeparrot pretokenized shards."""
    files = sorted(glob.glob(os.path.join(codeparrot_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {codeparrot_dir}")

    # Count total sequences across shards
    shard_sizes = []
    total = 0
    for f in files:
        meta = pq.read_metadata(f)
        shard_sizes.append(meta.num_rows)
        total += meta.num_rows

    print(f"Codeparrot: {len(files)} shards, {total:,} total sequences")

    if n_seqs >= total:
        print(f"  Requested {n_seqs:,} >= total {total:,}, using all")
        all_seqs = []
        for f in files:
            t = pq.read_table(f, columns=["input_ids"])
            for row in t["input_ids"]:
                all_seqs.append(row.as_py())
        return all_seqs

    # Reservoir sampling across shards
    rng = random.Random(seed)
    # Determine how many to sample from each shard (proportional)
    indices = rng.sample(range(total), n_seqs)
    indices.sort()

    # Map global indices to (shard, local_index) pairs
    sampled = []
    cum = 0
    idx_ptr = 0
    for shard_idx, f in enumerate(files):
        shard_end = cum + shard_sizes[shard_idx]
        # Collect indices for this shard
        local_indices = []
        while idx_ptr < len(indices) and indices[idx_ptr] < shard_end:
            local_indices.append(indices[idx_ptr] - cum)
            idx_ptr += 1

        if local_indices:
            t = pq.read_table(f, columns=["input_ids"])
            col = t["input_ids"]
            for li in local_indices:
                sampled.append(col[li].as_py())
            print(f"  {os.path.basename(f)}: sampled {len(local_indices)}")

        cum = shard_end

    print(f"  Sampled {len(sampled):,} codeparrot sequences")
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Mix synthetic + codeparrot for distillation")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic-5m.jsonl")
    parser.add_argument("--codeparrot", required=True, help="Path to codeparrot pretokenized dir")
    parser.add_argument("--output", required=True, help="Output dir for mixed parquet")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--synthetic-ratio", type=float, default=0.10, help="Fraction of synthetic data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    # Step 1: Tokenize synthetic data
    synthetic_seqs = tokenize_synthetic(args.synthetic, tokenizer, args.seq_len)
    n_synthetic = len(synthetic_seqs)

    if n_synthetic == 0:
        print("ERROR: No synthetic sequences produced")
        return

    # Step 2: Calculate codeparrot budget for desired ratio
    # synthetic_ratio = n_synthetic / (n_synthetic + n_codeparrot)
    # => n_codeparrot = n_synthetic * (1 - ratio) / ratio
    n_codeparrot = int(n_synthetic * (1 - args.synthetic_ratio) / args.synthetic_ratio)
    print(f"\nMix target: {args.synthetic_ratio*100:.0f}% synthetic, {(1-args.synthetic_ratio)*100:.0f}% codeparrot")
    print(f"  Synthetic: {n_synthetic:,} sequences")
    print(f"  Codeparrot: {n_codeparrot:,} sequences")
    print(f"  Total: {n_synthetic + n_codeparrot:,} sequences")
    print(f"  Total tokens: {(n_synthetic + n_codeparrot) * args.seq_len:,}")

    # Step 3: Sample codeparrot
    codeparrot_seqs = sample_codeparrot(args.codeparrot, n_codeparrot, args.seed)

    # Step 4: Interleave and shuffle
    all_seqs = synthetic_seqs + codeparrot_seqs
    rng = random.Random(args.seed)
    rng.shuffle(all_seqs)
    print(f"\nShuffled {len(all_seqs):,} total sequences")

    # Step 5: Write as parquet
    os.makedirs(os.path.join(args.output, "train"), exist_ok=True)
    table = pa.table({"input_ids": all_seqs})

    # Write in shards of ~250K sequences (matches codeparrot shard size)
    shard_size = 250_000
    n_shards = max(1, (len(all_seqs) + shard_size - 1) // shard_size)

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(all_seqs))
        shard_table = table.slice(start, end - start)
        out_path = os.path.join(args.output, "train", f"shard-{shard_idx:04d}.parquet")
        pq.write_table(shard_table, out_path, compression="zstd")
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Written: {out_path} ({size_mb:.1f} MB, {end - start:,} sequences)")

    total_mb = sum(
        os.path.getsize(os.path.join(args.output, "train", f))
        for f in os.listdir(os.path.join(args.output, "train"))
        if f.endswith(".parquet")
    ) / (1024 * 1024)
    print(f"\nDone: {len(all_seqs):,} sequences, {len(all_seqs) * args.seq_len:,} tokens, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
