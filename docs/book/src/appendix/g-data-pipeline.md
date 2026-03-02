# Appendix G: Data Pipeline

> Documents the Phase 1 data ingestion, tokenization, and augmentation pipeline.

## Source Corpora

| Source | Repository | Files | Rows | Parquet Size |
|--------|-----------|-------|------|-------------|
| depyler | depyler examples + TDD book | 1,843 | 1,843 | 6MB |
| hf-ground-truth | HuggingFace ground truth corpus | 11,928 | 11,493 | 197MB |
| jax-ground-truth | JAX ground truth corpus | 2,697 | 2,637 | 50MB |
| vllm-ground-truth | vLLM ground truth corpus | 1,118 | 1,100 | 18MB |

All sources are Python code, collected via `alimentar import local`.

## Training Mix

Weighted sampling with Tier 1 (depyler) upsampled:

```
alimentar mix \
  depyler.parquet:0.4 \
  hf.parquet:0.3 \
  jax.parquet:0.15 \
  vllm.parquet:0.15 \
  --output mixed.parquet \
  --seed 42
```

Result: **17,070 rows** (depyler upsampled 3.7x from 1,843 to ~6,829).

## Data Splits

| Split | Rows | Size | Seed | Weights |
|-------|------|------|------|---------|
| train | 17,070 | 201MB | 42 | depyler:0.4 hf:0.3 jax:0.15 vllm:0.15 |
| val | 500 | 7MB | 123 | equal 0.25 each |
| test | 200 | 2.4MB | 456 | equal 0.25 each |

## FIM Augmentation

Fill-in-the-Middle transforms applied via `alimentar fim`:

```
alimentar fim mixed.parquet \
  --output mixed-fim.parquet \
  --column text \
  --rate 0.5 \
  --format psm \
  --seed 42
```

- Format: PSM (Prefix-Suffix-Middle)
- Rate: 50% of rows receive FIM transform
- Sentinel tokens: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`

## BPE Tokenizer

Trained via `apr tokenize apply`:

```
apr tokenize apply \
  --data corpus-raw.txt \
  --vocab-size 32768 \
  --algorithm bpe \
  --max-lines 100000 \
  -o tokenizer/
```

Results:
- Final vocab size: 32,768
- Merges: 32,518
- Training time: 2022.5s (~33.7 min)
- Training data: 100K lines of Python code
- Special tokens: `<unk>`, `<s>`, `</s>`, `<pad>`
- Python pattern coverage: 8/8 (`def`, `return`, `self`, `import`, `class`, `for`, `if`, `in`)
- Output: `tokenizer/vocab.json` + `tokenizer/merges.txt`

## HuggingFace tokenizer.json Conversion

Entrenar requires HuggingFace `tokenizer.json` format, but `apr tokenize apply`
produces raw `vocab.json` + `merges.txt`. A Python conversion step bridges the gap
([ALB-033](https://github.com/paiml/albor/issues/31)):

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
bpe = models.BPE(vocab=vocab, merges=merges, end_of_word_suffix='</w>')
tokenizer = Tokenizer(bpe)
tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=' ', behavior='removed')
tokenizer.decoder = decoders.BPEDecoder(suffix='</w>')
tokenizer.save('models/albor-tokenizer/tokenizer.json')
```

Key details:
- Merges must be string format (`"i n"`) not array format (`["i", "n"]`)
- Pre-tokenizer matches aprender's `split_whitespace()` behavior
- `</w>` end-of-word suffix matches aprender's BPE encoding
- Regular vocab: 32,768 tokens (IDs 0-32767)
- FIM special tokens: 3 additional (IDs 32768-32770)

## Parquet Schema

All data files use a consistent schema:

```
{
  text: Utf8,    -- Python source code
  source: Utf8,  -- Corpus name (depyler, hf, jax, vllm)
  file: Utf8     -- Original file path
}
```

## Provenance

SHA-256 hashes for all data artifacts are recorded in `docs/PROVENANCE.md`.
Each split uses a different random seed for reproducibility.

## Tools Used

- `alimentar import local` — JSONL to Parquet conversion
- `alimentar mix` — weighted sampling with upsampling
- `alimentar fim` — Fill-in-the-Middle augmentation
- `apr tokenize plan/apply` — BPE vocabulary training
- `entrenar` (parquet feature) — Parquet-to-LMBatch bridge for training
