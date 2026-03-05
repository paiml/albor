#!/usr/bin/env python3
"""Pre-tokenize Parquet data for Albor training.

Reads text data from Parquet, tokenizes with a HuggingFace tokenizer,
chunks into fixed-length sequences, and saves as Parquet with input_ids column.

Supports single files, directories of shards, and glob patterns.
For large datasets, processes shards incrementally and writes output shards
to avoid holding all data in memory.

Usage:
    python3 scripts/pretokenize.py \
        --input data/tokenized/train/mixed.parquet \
        --tokenizer models/albor-tokenizer-v2/tokenizer.json \
        --seq-len 2048 \
        --output data/pretokenized-2048/train/train.parquet

    python3 scripts/pretokenize.py \
        --input /mnt/nvme-raid0/albor-data/codeparrot-clean/ \
        --tokenizer models/albor-tokenizer-v2/tokenizer.json \
        --seq-len 1024 \
        --output data/pretokenized-1024-v3/train/ \
        --text-column text --shard-output
"""
import argparse
import glob
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def get_input_files(input_path):
    """Resolve input path to list of Parquet files."""
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    elif "*" in input_path:
        files = sorted(glob.glob(input_path))
    else:
        files = [input_path]
    return files


def tokenize_and_chunk(texts, tokenizer, seq_len, leftover=None):
    """Tokenize texts and chunk into fixed-length sequences.

    Returns (chunks, leftover_ids) where leftover_ids are tokens that
    didn't fill a complete chunk (carried to next shard).
    """
    token_buf = list(leftover) if leftover else []
    chunks = []

    for text in texts:
        if text is None:
            continue
        encoded = tokenizer.encode(text)
        token_buf.extend(encoded.ids)

        while len(token_buf) >= seq_len:
            chunks.append(token_buf[:seq_len])
            token_buf = token_buf[seq_len:]

    return chunks, token_buf


def write_chunks(chunks, output_path):
    """Write chunks to a Parquet file."""
    input_ids_type = pa.list_(pa.uint32())
    arrow_ids = pa.array(chunks, type=input_ids_type)
    out_table = pa.table({"input_ids": arrow_ids})
    pq.write_table(out_table, output_path)
    return os.path.getsize(output_path)


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize Parquet data")
    parser.add_argument("--input", required=True, help="Input Parquet file, directory, or glob")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length for chunking")
    parser.add_argument("--output", required=True, help="Output Parquet file or directory (with --shard-output)")
    parser.add_argument("--text-column", default="text", help="Name of text column")
    parser.add_argument("--shard-output", action="store_true",
                        help="Write output shards (one per input shard) instead of single file")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    input_files = get_input_files(args.input)
    print(f"Found {len(input_files)} input file(s)")

    if not input_files:
        print("ERROR: No input files found")
        return

    os.makedirs(args.output if args.shard_output else os.path.dirname(args.output), exist_ok=True)

    start = time.time()
    total_docs = 0
    total_tokens = 0
    total_chunks = 0
    leftover = None
    all_chunks = []

    for file_idx, filepath in enumerate(input_files):
        table = pq.read_table(filepath)
        texts = table.column(args.text_column).to_pylist()
        total_docs += len(texts)

        chunks, leftover = tokenize_and_chunk(texts, tokenizer, args.seq_len, leftover)
        n_tokens = len(chunks) * args.seq_len
        total_tokens += n_tokens
        total_chunks += len(chunks)

        if args.shard_output and chunks:
            out_path = os.path.join(args.output, f"shard-{file_idx:04d}.parquet")
            size = write_chunks(chunks, out_path)
            elapsed = time.time() - start
            print(f"  [{file_idx+1}/{len(input_files)}] {os.path.basename(filepath)}: "
                  f"{len(texts):,} docs → {len(chunks):,} seqs ({n_tokens:,} tokens) "
                  f"[{size/1e6:.0f} MB] ({elapsed:.0f}s)")
        else:
            all_chunks.extend(chunks)
            if (file_idx + 1) % 5 == 0 or file_idx == len(input_files) - 1:
                elapsed = time.time() - start
                print(f"  [{file_idx+1}/{len(input_files)}] {total_docs:,} docs, "
                      f"{total_chunks:,} seqs, {total_tokens:,} tokens ({elapsed:.0f}s)")

    # Write single output file (non-shard mode)
    if not args.shard_output:
        if all_chunks:
            size = write_chunks(all_chunks, args.output)
            print(f"Saved: {args.output} ({size:,} bytes)")
        else:
            print("WARNING: No chunks produced")

    elapsed = time.time() - start
    print(f"\nDone: {total_docs:,} docs → {total_chunks:,} sequences of {args.seq_len} tokens")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Leftover tokens (discarded): {len(leftover) if leftover else 0}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
