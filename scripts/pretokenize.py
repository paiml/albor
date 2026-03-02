#!/usr/bin/env python3
"""Pre-tokenize Parquet data for Albor training.

Reads text data from Parquet, tokenizes with a HuggingFace tokenizer,
chunks into fixed-length sequences, and saves as Parquet with input_ids column.

Usage:
    python3 scripts/pretokenize.py \
        --input data/tokenized/train/mixed.parquet \
        --tokenizer models/albor-tokenizer-v2/tokenizer.json \
        --seq-len 2048 \
        --output data/pretokenized-2048/train/train.parquet

    python3 scripts/pretokenize.py \
        --input data/tokenized/val/val.parquet \
        --tokenizer models/albor-tokenizer-v2/tokenizer.json \
        --seq-len 2048 \
        --output data/pretokenized-2048/val/val.parquet
"""
import argparse
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize Parquet data")
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length for chunking")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--text-column", default="text", help="Name of text column")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")

    table = pq.read_table(args.input)
    texts = table.column(args.text_column).to_pylist()
    print(f"Loaded {len(texts)} rows from {args.input}")

    # Tokenize
    start = time.time()
    all_ids = []
    for i, text in enumerate(texts):
        if text is None:
            continue
        encoded = tokenizer.encode(text)
        all_ids.append(encoded.ids)
        if (i + 1) % 5000 == 0:
            print(f"  Tokenized {i+1}/{len(texts)} ({time.time()-start:.1f}s)")

    elapsed = time.time() - start
    total_tokens = sum(len(ids) for ids in all_ids)
    print(f"Tokenized {len(all_ids)} docs in {elapsed:.1f}s ({total_tokens:,} tokens)")

    # Chunk into fixed-length sequences
    chunks = []
    for ids in all_ids:
        for i in range(0, len(ids) - args.seq_len + 1, args.seq_len):
            chunks.append(ids[i:i + args.seq_len])

    print(f"Chunked to {len(chunks)} sequences of {args.seq_len} tokens")
    print(f"Total tokens: {len(chunks) * args.seq_len:,}")

    # Save
    input_ids_type = pa.list_(pa.uint32())
    arrow_ids = pa.array(chunks, type=input_ids_type)
    out_table = pa.table({"input_ids": arrow_ids})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pq.write_table(out_table, args.output)
    size = os.path.getsize(args.output)
    print(f"Saved: {args.output} ({size:,} bytes)")


if __name__ == "__main__":
    main()
