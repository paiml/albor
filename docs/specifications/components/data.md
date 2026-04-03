# Data Pipeline

**Parent**: [albor-llm-spec.md](../albor-llm-spec.md) §5

---

## 1. Available Datasets

### 1.1 codeparrot-clean (Primary Pre-training)

| Property | Value |
|----------|-------|
| Source | HuggingFace `codeparrot/codeparrot-clean` (Python subset) |
| Raw tokens | 5.3B |
| Sequences | 5.16M |
| Seq length | 1024 (pre-tokenized) |
| Shards | 20 parquet files (shard-0000 to shard-0019) |
| Size on disk | 7.5 GB (pretokenized parquet) |
| Quality | **Raw** — unfiltered GitHub scrape |
| Path | `data/pretokenized-1024-v3/train/` |
| Tokenizer | albor-tokenizer-v2 (ByteLevel BPE, 32K vocab) |

### 1.2 Train/Val Split

| Split | Source | Sequences | Notes |
|-------|--------|-----------|-------|
| Train | shards 0000-0018 | ~4.9M | 19 shards |
| Val | shard 0019 (subsampled) | 1,000 | Seed=42, same distribution |
| Held-out | shard 0019 (remainder) | ~270K | Available for test |

**Critical**: Val data MUST come from same source as train data. Previous
val set was from a different dataset → false val_ppl regression (see ALB v6
post-mortem). Fixed by subsampling from shard-0019.

### 1.3 CodeSearchNet Python (Secondary)

| Property | Value |
|----------|-------|
| Tokens | 133M |
| Sequences | 65K |
| Seq length | 2048 |
| Quality | Medium (curated GitHub, with docstrings) |
| Path | `/mnt/nvme-raid0/data/pretokenized-csn-python-2048/` |
| Status | Ready, not yet used in training |

### 1.4 Filtered codeparrot-clean (Quality-Filtered Pre-training)

| Property | Value |
|----------|-------|
| Source | codeparrot-clean, filtered by `scripts/filter_codeparrot.py` |
| Input files | 2.95M (streamed from HuggingFace) |
| Passed filter | 850K files (28.7%) |
| Raw tokens | 2.04B |
| Sequences | 1,988,843 |
| Seq length | 1024 (pre-tokenized) |
| Shards | 40 parquet files |
| Path | `data/pretokenized-1024-v4/train/` |
| Status | **Ready for v29** |

**Filters applied** (AST-based, no ML classifier):
1. `ast.parse()` — must be valid Python
2. Docstring density — rejects files with no documentation
3. Import diversity — rejects single-import utility scripts
4. Generated code detection — rejects auto-generated files

**v29 config**: `configs/train/pretrain-350m-v29.yaml` (15,530 steps, ~2.4 days).

### 1.5 Merged Dataset

| Property | Value |
|----------|-------|
| Tokens | 180M |
| Sequences | 88K |
| Seq length | 2048 |
| Path | `/mnt/nvme-raid0/data/pretokenized-merged-2048/` |
| Status | Ready, not yet used |

---

## 2. Data Quality Strategy

### 2.1 The phi-1 Insight

A 350M model trained on 7B curated tokens beats one trained on 577B raw tokens.
The quality pipeline is the single highest-leverage intervention.

phi-1's data stack:
1. **6B tokens "CodeTextbooks"**: GPT-4-trained classifier scores GitHub code
   for "textbook quality". Top ~17% of 35B raw tokens kept.
2. **1B tokens synthetic textbooks**: GPT-3.5 generates Python teaching material
   with explanations interleaved with code.
3. **180M tokens CodeExercises**: Function stubs + solutions in HumanEval style.

### 2.2 Our Adaptation

| Step | Data | Tokens | Method | Status |
|------|------|--------|--------|--------|
| 1. Filter | codeparrot-clean | 5.3B → **2.04B** | AST + heuristic filter | **DONE** |
| 2. Synthetic completions | Teacher output | 10-128M | Execution-verified | Pilot (330/1K) |
| 3. Synthetic textbooks | Teacher output | 50-200M | Multi-step explanations | TODO |
| 4. CodeExercises | Teacher output | 5-20M | HumanEval-style stubs | TODO |

### 2.3 Quality Filtering (COMPLETE)

**Implemented**: `scripts/filter_codeparrot.py` — deterministic AST-based filter.

Simpler than the originally planned ML classifier, but effective:
- 2.95M files processed (streamed from HuggingFace)
- 850K passed (28.7%) → 2.04B tokens
- Filters: valid Python AST, docstring density, import diversity, no generated code
- Pass rate matches the ~20-30% expected yield from §2.1

**ML classifier (planned, not yet needed)**: The AST-based filter achieved the
target yield. An ML classifier (random forest on code features, Qwen3-Coder
annotated) remains available as a second filtering pass if v29 results suggest
further curation is needed.

---

## 3. Tokenization Pipeline

### 3.1 alimentar Integration

```bash
# Tokenize raw dataset
apr data plan --source hf:codeparrot/codeparrot-clean \
              --tokenizer models/albor-tokenizer-v2/tokenizer.json \
              --seq-len 1024 \
              --output data/pretokenized-1024-v3/

apr data apply
```

### 3.2 Token Format

Pre-tokenized parquet files with schema:
```
input_ids: list<int32>  # Token IDs, length = seq_len
```

Each row is one pre-packed sequence of exactly `seq_len` tokens. Documents
separated by `<|endoftext|>` (token ID 0). Padding with `<|padding|>` (ID 1)
only at sequence end.

### 3.3 FIM Transform (Planned)

Fill-in-the-Middle transformation for code completion. Not yet implemented
in alimentar. Will require:
1. Three new special tokens: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`
2. Random split point within each document
3. 50% FIM rate (half of sequences get FIM transform)

---

## 4. Data Loading

### 4.1 Current Architecture

entrenar loads all parquet shards into memory at startup via
`ParquetDataLoader`. For 19 shards of codeparrot-clean:
- 5.16M sequences × ~4KB each ≈ 20 GB in RAM
- With OS overhead → 39 GB swap usage
- Progressive throughput degradation: 8.4K → 6.5K tok/s

### 4.2 Planned: Streaming Data Loader

Replace all-in-RAM loading with streaming:
1. Load shard metadata (file paths, row counts) at startup
2. Memory-map one shard at a time
3. Yield batches from current shard
4. Advance to next shard when current exhausted
5. Shuffle within shard (sufficient for pre-training)

Expected improvement: RSS from 25 GB to <2 GB, throughput recovery to 8K+ tok/s.

### 4.3 Data Sampling

Current: sequential iteration through shards. No cross-shard shuffling.
This is acceptable for pre-training (each shard is already shuffled internally)
but not ideal. Full shuffling requires either:
- Streaming shuffle buffer (e.g., 10K sequences)
- Pre-shuffle across shards (alimentar transform)

---

## 5. Synthetic Data Generation

### 5.1 Prompt Engineering

For teacher completion generation, prompts are structured as:

**Style 1: Function completion**
```python
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number.

    Args:
        n: The position in the Fibonacci sequence (0-indexed)

    Returns:
        The nth Fibonacci number
    """
```

**Style 2: Class implementation**
```python
class LRUCache:
    """Least Recently Used cache with O(1) get and put operations."""

    def __init__(self, capacity: int):
```

**Style 3: Algorithm from description**
```python
# Implement merge sort that works on a list of integers.
# The function should sort the list in-place and return None.
def merge_sort(arr: list[int]) -> None:
```

### 5.2 Execution Verification

Every teacher completion is executed in a sandbox:
1. Parse completion to check syntax (`ast.parse`)
2. Execute with timeout (10s) and memory limit (256 MB)
3. Run any embedded assertions or test cases
4. Check for runtime errors (TypeError, ValueError, etc.)
5. Optionally: run mypy for type correctness

Only completions that pass ALL checks are kept for SFT.

### 5.3 Deduplication

AST-based deduplication to avoid training on near-identical completions:
1. Parse each completion to AST
2. Normalize: remove comments, standardize variable names
3. Hash normalized AST
4. Remove duplicates (keep first occurrence)

---

## 6. Data Paths Reference

| Path | Contents | Tokens |
|------|----------|--------|
| `data/pretokenized-1024-v3/train/` | codeparrot-clean raw (19 shards) | 5.0B |
| `data/pretokenized-1024-v3/val/` | codeparrot-clean holdout (1K seqs) | 1M |
| `data/pretokenized-1024-v4/train/` | **codeparrot-clean filtered (40 shards)** | **2.04B** |
| `data/filtered/train/` | Filtered parquet (pre-tokenization) | — |
| `/mnt/nvme-raid0/data/pretokenized-csn-python-2048/` | CodeSearchNet Python | 133M |
| `/mnt/nvme-raid0/data/pretokenized-merged-2048/` | codeparrot + CSN merged | 180M |
| `models/albor-tokenizer-v2/tokenizer.json` | ByteLevel BPE tokenizer | — |
