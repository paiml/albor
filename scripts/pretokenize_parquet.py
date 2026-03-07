#!/usr/bin/env python3
"""Pre-tokenize a Parquet dataset into fixed-length sequences for LM training.

Usage:
    python scripts/pretokenize_parquet.py \
        --input /mnt/nvme-raid0/data/code-search-net-python/data/ \
        --output /mnt/nvme-raid0/data/pretokenized-csn-python-2048/ \
        --tokenizer models/albor-tokenizer-v2/tokenizer.json \
        --seq-len 2048 \
        --column code
"""
import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def pretokenize(input_dir: str, output_dir: str, tokenizer_path: str,
                seq_len: int, column: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {input_dir}")

    print(f"Found {len(files)} parquet files")
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")
    print(f"Sequence length: {seq_len}")

    # Tokenize all text and concatenate into one long token stream
    all_tokens: list[int] = []
    total_rows = 0
    for f in files:
        table = pq.read_table(f, columns=[column])
        for text in table[column]:
            txt = text.as_py()
            if txt:
                enc = tokenizer.encode(txt)
                all_tokens.extend(enc.ids)
        total_rows += len(table)
        print(f"  {os.path.basename(f)}: {len(table)} rows, "
              f"running total: {len(all_tokens)} tokens")

    print(f"\nTotal: {total_rows} rows, {len(all_tokens)} tokens")

    # Chunk into seq_len sequences (drop remainder)
    num_seqs = len(all_tokens) // seq_len
    print(f"Creating {num_seqs} sequences of length {seq_len}")
    print(f"Dropping {len(all_tokens) - num_seqs * seq_len} remainder tokens")

    sequences = []
    for i in range(num_seqs):
        start = i * seq_len
        sequences.append(all_tokens[start:start + seq_len])

    # Write as Parquet
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    table = pa.table({"input_ids": sequences})
    out_path = os.path.join(output_dir, "train", "train.parquet")
    pq.write_table(table, out_path, compression="zstd")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWritten: {out_path} ({size_mb:.1f} MB, {num_seqs} sequences)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize Parquet for LM training")
    parser.add_argument("--input", required=True, help="Input dir with .parquet files")
    parser.add_argument("--output", required=True, help="Output dir for pretokenized data")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--column", default="code", help="Text column name")
    args = parser.parse_args()
    pretokenize(args.input, args.output, args.tokenizer, args.seq_len, args.column)
