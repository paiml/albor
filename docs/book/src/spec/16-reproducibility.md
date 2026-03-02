# 16. Reproducibility Protocol

Every artifact in the albor pipeline is reproducible from source. This chapter
documents the exact commands, seeds, and checksums needed to reproduce the
full training pipeline from raw code corpora to trained model.

## 16.1 Artifact Tracking

| Artifact | How Recorded |
|----------|----------|
| Random seed | 42 (global), per-component seeds derived |
| Data versions | HuggingFace dataset commit SHAs + local repo git SHAs |
| Data provenance | `docs/PROVENANCE.md` (source path, git SHA, file count, token count per source) |
| Data checksums | SHA-256 of every Parquet shard (recorded in PROVENANCE.md) |
| Tokenizer v1 | `models/albor-tokenizer/` (vocab.json + merges.txt + tokenizer.json) |
| Tokenizer v2 | `models/albor-tokenizer-v2/tokenizer.json` (ByteLevel BPE) |
| Training config | YAML checked into git (`configs/train/*.yaml`) |
| Checkpoint hashes | SHA-256 of model.safetensors |
| Software versions | `apr --version`, `alimentar --version`, `pv --version` |
| Hardware | nvidia-smi + free -h captured in training logs |
| Training logs | `checkpoints/*/training.log` + `final_model.json` |
| Eval results | `configs/eval/*.jsonl` (benchmarks) + eval scripts |

## 16.2 Full Reproduction Commands

### Step 1: Corpus Preparation

```bash
# Import local Python codebases
alimentar import local /path/to/depyler -o data/raw/depyler.parquet
alimentar import local /path/to/hf-ground-truth-corpus -o data/raw/hf.parquet
alimentar import local /path/to/jax-ground-truth-corpus -o data/raw/jax.parquet
alimentar import local /path/to/vllm-ground-truth-corpus -o data/raw/vllm.parquet

# Mix training split (weighted sampling)
alimentar mix \
    data/raw/depyler.parquet:0.4 \
    data/raw/hf.parquet:0.3 \
    data/raw/jax.parquet:0.15 \
    data/raw/vllm.parquet:0.15 \
    -o data/tokenized/train/mixed.parquet \
    --seed 42

# Mix validation split (equal weights)
alimentar mix \
    data/raw/depyler.parquet:0.25 \
    data/raw/hf.parquet:0.25 \
    data/raw/jax.parquet:0.25 \
    data/raw/vllm.parquet:0.25 \
    -o data/tokenized/val/val.parquet \
    -n 500 --seed 123
```

### Step 2: Tokenizer Training

```bash
# v1 tokenizer (whitespace-split BPE — has ALB-036 limitation)
apr tokenize apply \
    --data data/staging/corpus-raw.txt \
    --vocab-size 32768 \
    --algorithm bpe \
    -o models/albor-tokenizer/ \
    --max-lines 100000

# v2 tokenizer (ByteLevel BPE — preserves whitespace)
python scripts/train-tokenizer-v2.py \
    --corpus data/staging/corpus-raw.txt \
    --vocab-size 32768 \
    --output models/albor-tokenizer-v2/
```

### Step 3: Pre-Tokenization

```bash
# Pre-tokenize full training data (v2 tokenizer, 2048-token chunks)
python scripts/pretokenize.py \
    --input data/tokenized/train/mixed.parquet \
    --tokenizer models/albor-tokenizer-v2/tokenizer.json \
    --seq-len 2048 \
    --output data/pretokenized-2048/train/train.parquet

# Pre-tokenize validation data
python scripts/pretokenize.py \
    --input data/tokenized/val/val.parquet \
    --tokenizer models/albor-tokenizer-v2/tokenizer.json \
    --seq-len 2048 \
    --output data/pretokenized-2048/val/val.parquet
```

### Step 4: Model Training

```bash
# 50M pipeline validation (< 2 minutes)
make train-50m
# Equivalent to:
# apr train apply --task pretrain --config configs/train/pretrain-50m.yaml

# 350M base model (~20 hours on RTX 4090)
make train-350m
# Equivalent to:
# apr train apply --task pretrain --config configs/train/pretrain-350m.yaml
```

### Step 5: Checkpoint Conversion (for evaluation)

```bash
# Convert entrenar 1D-flat SafeTensors to realizar 2D format
python scripts/convert-checkpoint.py checkpoints/albor-base-350m/ \
    --config configs/train/pretrain-350m.yaml
```

### Step 6: Evaluation

```bash
# Validate all benchmarks (no model needed)
make eval-validate

# Perplexity evaluation (needs trained model)
make eval-perplexity-350m

# Monitor active training
make training-status
```

## 16.3 Key SHA-256 Checksums

See `docs/PROVENANCE.md` for complete checksums. Key artifacts:

| Artifact | SHA-256 (first 8 hex) |
|----------|----------------------|
| Training data (mixed.parquet) | `bdfe8742` |
| Val data (val.parquet) | `6be03768` |
| v1 tokenizer (vocab.json) | `aca6fa72` |
| v2 tokenizer (tokenizer.json) | `d999cc9e` |
| Pre-tokenized train (2048) | `4f54e422` |
| Pre-tokenized val (2048) | `c9c1d093` |

## 16.4 Verification

```bash
# Verify data checksums
sha256sum data/tokenized/train/mixed.parquet
sha256sum data/pretokenized-2048/train/train.parquet
sha256sum models/albor-tokenizer-v2/tokenizer.json

# Verify training config reproducibility
apr train plan --task pretrain --config configs/train/pretrain-350m.yaml

# Verify contract integrity
pv validate contracts/*.yaml
pv coverage contracts
pv audit contracts/*.yaml
```
