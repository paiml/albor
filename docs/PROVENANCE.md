# Data Provenance Manifest

Generated: 2026-03-01
Tool: alimentar 0.2.6, mix seed=42

## Source Corpora

| Source | Git Repo | Files | Rows | SHA-256 |
|--------|----------|-------|------|---------|
| depyler | /home/noah/src/depyler | 1,843 | 1,843 | `a5f5da16...` |
| hf-ground-truth | /home/noah/src/hf-ground-truth-corpus | 11,928 | 11,493 | `d0e40937...` |
| jax-ground-truth | /home/noah/src/jax-ground-truth-corpus | 2,697 | 2,637 | `1fbd6484...` |
| vllm-ground-truth | /home/noah/src/vllm-ground-truth-corpus | 1,118 | 1,100 | `066c39f6...` |

## Training Mix

| Split | File | Rows | Size | Weights | SHA-256 |
|-------|------|------|------|---------|---------|
| train | data/tokenized/train/mixed.parquet | 17,070 | 201MB | depyler:0.4 hf:0.3 jax:0.15 vllm:0.15 | `bdfe8742...` |
| val | data/tokenized/val/val.parquet | 500 | 7MB | equal 0.25 each | `6be03768...` |
| test | data/tokenized/test/test.parquet | 200 | 2.4MB | equal 0.25 each | `14388092...` |
| train-fim | data/tokenized/train/mixed-fim.parquet | 17,070 | 192MB | FIM@50% PSM of train | `da258ae4...` |

## Mix Parameters

- Seed: 42 (train), 123 (val), 456 (test)
- Depyler upsampled from 1,843 to 6,829 (3.7x) — Tier 1 priority
- HF downsampled from 11,493 to 5,121
- FIM: PSM format, 50% rate, seed 42
- Schema: `{text: Utf8, source: Utf8, file: Utf8}`

## BPE Tokenizer

| File | Vocab Size | Lines Trained | Training Time | SHA-256 |
|------|-----------|---------------|---------------|---------|
| tokenizer/vocab.json | 32,768 | 100,000 | 2022.5s | `aca6fa72...` |
| tokenizer/merges.txt | 32,518 merges | — | — | `7cf4dede...` |

- Algorithm: BPE (whitespace-split, `</w>` suffix)
- Tool: `apr tokenize apply` (aprender)
- Corpus: 100K lines from mixed training data (real newlines, one code line per line)
- Special tokens: `<unk>` (0), `<s>` (1), `</s>` (2), `<pad>` (3)
- Python pattern coverage: 8/8 (def, return, self, import, class, for, if, in)
- Limitation: normalizes whitespace (see ALB-036)

## ByteLevel BPE Tokenizer (v2)

| File | Vocab Size | Training Data | SHA-256 |
|------|-----------|---------------|---------|
| models/albor-tokenizer-v2/tokenizer.json | 32,768 | 5.6M lines (corpus-raw.txt) | `d999cc9e...` |

- Algorithm: ByteLevel BPE (preserves whitespace, newlines, indentation)
- Tool: Python `tokenizers` library (HuggingFace)
- Pre-tokenizer: ByteLevel (add_prefix_space=False)
- Special tokens: `<unk>`, `<s>`, `</s>`, `<pad>`, `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`
- Roundtrip: 6/6 PASS (preserves Python indentation)

## Pre-Tokenized Data

| File | Sequences | Seq Length | Total Tokens | SHA-256 |
|------|-----------|-----------|--------------|---------|
| data/pretokenized-2048/train/train.parquet | 22,079 | 2048 | 45,217,792 | `4f54e422...` |
| data/pretokenized-2048/val/val.parquet | 814 | 2048 | 1,667,072 | `c9c1d093...` |

- Tokenizer: ByteLevel BPE v2
- Chunking: non-overlapping 2048-token segments (remainder dropped)
- Column: `input_ids` (List<u32>)
