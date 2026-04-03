#!/usr/bin/env python3
"""Streaming pretokenization — constant memory, sharded output.

Tokenizes text Parquet files into fixed-length token sequences,
writing output shards incrementally instead of buffering all tokens.

Usage:
    python3 scripts/pretokenize_streaming.py \
        --input data/filtered/train/ \
        --output data/pretokenized-1024-v4/train/ \
        --tokenizer models/albor-tokenizer-v2/tokenizer.json \
        --seq-len 1024
"""
import argparse
import glob
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def pretokenize_streaming(input_dir, output_dir, tokenizer_path, seq_len,
                          column, shard_size=50000):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Input: {len(files)} parquet files from {input_dir}")
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")
    print(f"Sequence length: {seq_len}, shard size: {shard_size}")

    token_buffer = []  # rolling buffer of leftover tokens
    sequences = []     # current shard's sequences
    shard_idx = 0
    total_seqs = 0
    total_tokens = 0
    total_rows = 0
    t0 = time.time()

    for fi, f in enumerate(files):
        table = pq.read_table(f, columns=[column])
        for text in table[column]:
            txt = text.as_py()
            if not txt:
                continue
            enc = tokenizer.encode(txt)
            token_buffer.extend(enc.ids)
            total_tokens += len(enc.ids)

            # Extract complete sequences from buffer
            while len(token_buffer) >= seq_len:
                sequences.append(token_buffer[:seq_len])
                token_buffer = token_buffer[seq_len:]

                # Write shard when full
                if len(sequences) >= shard_size:
                    shard_path = os.path.join(output_dir, f"shard-{shard_idx:04d}.parquet")
                    out_table = pa.table({"input_ids": sequences})
                    pq.write_table(out_table, shard_path, compression="zstd")
                    total_seqs += len(sequences)
                    print(f"  shard-{shard_idx:04d}: {len(sequences)} seqs → {shard_path}",
                          flush=True)
                    shard_idx += 1
                    sequences = []

        total_rows += len(table)
        elapsed = time.time() - t0
        print(f"  [{fi+1}/{len(files)}] {os.path.basename(f)}: "
              f"{len(table)} rows, {total_tokens:,} tokens, "
              f"{total_seqs + len(sequences):,} seqs, {elapsed:.0f}s",
              flush=True)

    # Write remaining sequences
    if sequences:
        shard_path = os.path.join(output_dir, f"shard-{shard_idx:04d}.parquet")
        out_table = pa.table({"input_ids": sequences})
        pq.write_table(out_table, shard_path, compression="zstd")
        total_seqs += len(sequences)
        print(f"  shard-{shard_idx:04d}: {len(sequences)} seqs → {shard_path}",
              flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Input: {total_rows:,} files, {total_tokens:,} tokens")
    print(f"  Output: {total_seqs:,} sequences × {seq_len} = "
          f"{total_seqs * seq_len / 1e9:.2f}B tokens")
    print(f"  Dropped: {len(token_buffer)} remainder tokens")
    print(f"  Shards: {shard_idx + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--column", default="content")
    parser.add_argument("--shard-size", type=int, default=50000)
    args = parser.parse_args()
    pretokenize_streaming(args.input, args.output, args.tokenizer,
                          args.seq_len, args.column, args.shard_size)
