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

**v1 pipeline** (Tier 1 only, 17K rows):

```bash
# Import Tier 1 ground truth corpora
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
```

**v2 pipeline** (Tier 1 10x + 8 Tier 2 repos, 45K rows → 68K sequences):

```bash
# Convert Tier 2 source repos to Parquet (alimentar can't read source dirs)
for repo in pytorch hf-repos mlflow vllm-full tgi algo-corpus cuda-python llms-with-hf; do
    python3 scripts/source-to-parquet.py ~/src/$repo $repo data/parquet/tier2/$repo.parquet
done

# Mix Tier 1 (10x upsampled) + Tier 2 (1x)
alimentar mix \
    data/parquet/depyler/shard_0000.parquet:10.0 \
    data/parquet/hf-ground-truth/shard_0000.parquet:10.0 \
    data/parquet/jax/shard_0000.parquet:10.0 \
    data/parquet/vllm/shard_0000.parquet:10.0 \
    data/parquet/tier2/pytorch.parquet:1.0 \
    data/parquet/tier2/hf-repos.parquet:1.0 \
    data/parquet/tier2/mlflow.parquet:1.0 \
    data/parquet/tier2/vllm-full.parquet:1.0 \
    data/parquet/tier2/tgi.parquet:1.0 \
    data/parquet/tier2/algo-corpus.parquet:1.0 \
    data/parquet/tier2/cuda-python.parquet:1.0 \
    data/parquet/tier2/llms-with-hf.parquet:1.0 \
    -o data/staging/mixed-expanded.parquet --seed 42

# Apply FIM (50% PSM)
alimentar fim data/staging/mixed-expanded.parquet \
    -o data/staging/mixed-expanded-fim.parquet --rate 0.5 --format psm --seed 42
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

**v3 pipeline** (codeparrot-clean, 5.3B tokens, 19 shards at seq_len=1024):

```bash
# Download + pretokenize codeparrot-clean (2M Python files)
python scripts/pretokenize-codeparrot.py \
    --tokenizer models/albor-tokenizer-v2/tokenizer.json \
    --seq-len 1024 \
    --output data/pretokenized-1024-v3/train/ \
    --val-output data/pretokenized-1024-v3/val/val.parquet \
    --val-size 1000 \
    --num-shards 19
```

### Step 4: Model Training

```bash
# 50M pipeline validation (< 2 minutes)
/mnt/nvme-raid0/targets/aprender/release/apr train apply \
    --task pretrain --config configs/train/pretrain-50m-quick.yaml

# 350M v13 — current production run (~7 days on RTX 4090)
# All ALB-079–119 fixes applied, GPU optimizer state checkpointed (ALB-118)
/mnt/nvme-raid0/targets/aprender/release/apr train apply \
    --task pretrain --config configs/train/pretrain-350m-v13.yaml
# v13: cosine LR over 155K steps, ga=8, batch=4, seq=1024, 32K tokens/step
# Data: codeparrot-clean 5.0B tokens (19 shards, streaming loader)
# RoPE forward+backward (ALB-119), GPU optimizer state saved (ALB-118)

# Build apr-cli with local patches
cd ~/src/aprender && CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/aprender \
    cargo build --release -p apr-cli
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

### v3 Data (codeparrot-clean, 5.3B tokens, production)

| Artifact | SHA-256 |
|----------|---------|
| v2 tokenizer (`tokenizer.json`) | `d999cc9e283d7934a726a863b196ccf91c143b1d25a87fcda4fb984ab469e403` |
| Val (`val.parquet`) | `95db0c72a108507460f2dcd6b04b387fc4a6beb647016d4759d684932cdfc7e4` |
| `shard-0000.parquet` | `d49c7d9e7c9a524de1eafd5cfc02b736d737ff6c5ae918480ad005ab2b914037` |
| `shard-0001.parquet` | `b39dc2d001efc1c4cc95de291f6380988a303d2cf2c9fbc5a24cc01210399eb6` |
| `shard-0002.parquet` | `384934cd3448d76cc5b282344de5345ede0b95329973d1e9d890ac9bab3c9bee` |
| `shard-0003.parquet` | `e542b6291b7c432c32a359e640a798ac1102709262040bd4ea62b2a282e23572` |
| `shard-0004.parquet` | `0d139da5bcde3fff8573e1bad40c6b0a5ae34ef1f006c304f2deb30aa1e225e1` |
| `shard-0005.parquet` | `7c229c697281e66678bdf108dcfb584e79387e8c986cb528bf956ccbd96e65a2` |
| `shard-0006.parquet` | `a02ed61cb66358f6e05332f81b3df6d64ebeb117d7645e8f74fa0f59e40c5aed` |
| `shard-0007.parquet` | `c159e5f83a45ee8cb116a7568e04fb5c860d93866a25ee49431065e5be0966eb` |
| `shard-0008.parquet` | `8b3b6388724e72cc6c2c721501a62da360c99b8eb5b5b74e60c198b50d5683a2` |
| `shard-0009.parquet` | `85cd172b060171eb99c8cb6651cfd24bea6d59a3bc33d57921aa4fa290ef0087` |
| `shard-0010.parquet` | `f41fe614cb7bb7a516e656e088b57e82fcdcbddacd478b6fd4f98a2d52f8bdd5` |
| `shard-0011.parquet` | `c2be1e6b8230f3105ec6caffd949f86447e4cf52dd1af19085a7ad74f7e5a4ab` |
| `shard-0012.parquet` | `6b3392880fc5e4f18be2f52de1f05268302aaf2c8010964fc9bf7125bb4afa71` |
| `shard-0013.parquet` | `2d2b5ea6062c2a318b52bec1c646740b77bc10b217d71a0f30b81ae3f335ccf1` |
| `shard-0014.parquet` | `4c501eed6392808faba747f32f2ecd5458118133a4f96611fc44137ebe071ff1` |
| `shard-0015.parquet` | `236c111cf7572b11bf4615aeeb798100db3041f57d0e528a02e27a38def81a55` |
| `shard-0016.parquet` | `e13a54c04b074f17787f9c95a267f07612ce09a79258ea37350f42cc5dbf24ea` |
| `shard-0017.parquet` | `ae35c3231d94a2076ad21c275638a9a8663d74e48657d2052b33db08d046e213` |
| `shard-0018.parquet` | `6ad5fa5616b33478e1f3f9229fd89016a46b5c7a8b3e2293cfdca76d21d98dbb` |

### Legacy Data

| Artifact | SHA-256 (first 8 hex) |
|----------|----------------------|
| Training data (mixed.parquet) | `bdfe8742` |
| Val data (val.parquet) | `6be03768` |
| v1 tokenizer (vocab.json) | `aca6fa72` |
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
