# 5. Training Data

### 5.1 Data Philosophy
- All datasets either locally owned (MIT/Apache 2.0) or publicly available with permissive licenses
- **Local-first**: Sovereign ground truth corpora are our highest-quality data — curated,
  tested, type-annotated, and owned. They are upsampled to punch above their token weight.
- Exact download URLs, versions, and SHA-256 hashes recorded for all external data
- Preprocessing pipeline is deterministic (fixed seed, recorded transforms)
- Quality validated by `alimentar quality check`

### 5.2 Data Mix (Target: ~10B tokens)

Following the phi-1 playbook: maximum concentration on Python. phi-1 proved that
a small model (1.3B) with focused data and distillation can hit 50% HumanEval —
outperforming models 10x its size trained on diluted multi-language corpora.

**Key insight from phi-1**: Data quality matters more than quantity at small
param counts. A 350M model trained on 1B tokens of textbook-quality code can
outperform a 350M model trained on 100B tokens of raw GitHub scrapes. We have
~71K curated Python files locally — this is our unfair advantage.

| Source | Tokens (est.) | Weight | License | Rationale |
|--------|--------------|--------|---------|-----------|
| StarCoder Python subset (HF) | ~4B | 40% | Apache 2.0 | Bulk Python code diversity; aligns with Qwen3-Coder teacher |
| **Local ground truth corpora** (upsampled 10x) | ~50-100M raw → ~500M-1B effective | **10%** | MIT | Highest-quality anchor — see §5.2.1 |
| **Local ML framework code** | ~200-400M | **10%** | MIT / Apache 2.0 | ML/AI Python patterns — see §5.2.2 |
| FineWeb-Edu (subset) | ~2B | 20% | ODC-BY | Educational web text for docstring understanding |
| Python textbooks + tutorials (HF) | ~1B | 10% | Apache 2.0 / CC | "Textbooks Are All You Need" — public educational code |
| Python docs + PEPs + Stack Overflow | ~1B | 10% | CC BY-SA | API knowledge, idiomatic patterns |

Total: ~10B tokens. Chinchilla-optimal for 350M params is ~7B; we slightly
overtrain for benchmark performance (common practice in SmolLM, Phi-1.5).

**Python concentration**: 80% of training data is Python or Python-adjacent
(code, textbooks, docs). The remaining 20% (FineWeb-Edu) provides general
language understanding for docstrings, comments, and natural language prompts.

#### 5.2.1 Local Ground Truth Corpora (Tier 1 — Upsampled)

These are our "textbook-quality" data — the phi-1 equivalent. Every file has
been curated, tested to 98%+ coverage, and validated by CI. They are upsampled
10x during training because their per-token teaching signal is 10-100x higher
than raw GitHub code.

| Corpus | Path | Files | Lines (est.) | Quality Signal |
|--------|------|-------|-------------|----------------|
| depyler examples + tdd-book | `../depyler/examples/`, `../depyler/tdd-book/` | 1,845 | ~219K | Type-annotated, transpiler-validated, 27 stdlib modules, property-tested |
| hf-ground-truth-corpus | `../hf-ground-truth-corpus/` | 11,928 | ~500K+ | 98%+ test coverage, zero lint violations, production HF recipes |
| jax-ground-truth-corpus | `../jax-ground-truth-corpus/` | 2,697 | ~200K+ | 100% test coverage, full type checking, numerical computing |
| vllm-ground-truth-corpus | `../vllm-ground-truth-corpus/` | 1,118 | ~100K+ | Production inference optimization code |
| **Total** | | **17,588** | **~1M+** | **All MIT licensed, all CI-validated** |

**Why upsampling works**: phi-1's "textbook" data was <10% of total tokens but
had outsized impact on HumanEval. Our ground truth corpora share the same
properties: clean types, complete docstrings, tested correctness, educational
structure. The model sees these examples multiple times, reinforcing correct
patterns over noisy GitHub code.

**depyler corpus is uniquely valuable**: Every Python function in the depyler
corpus was validated by a transpiler — it has clear types, clean control flow,
and provably correct semantics. The tdd-book covers 27 stdlib modules
(json, datetime, collections, itertools, os, pathlib, re, etc.) with
property-based tests. This teaches the model Python's standard library idioms
at a depth no scraped dataset matches.

#### 5.2.2 Local ML Framework Code (Tier 2)

Large, high-quality Python codebases from our local repos. Not upsampled —
used at natural frequency for pattern diversity.

| Corpus | Path | Files | Notes |
|--------|------|-------|-------|
| huggingface-fine-tuning | `../huggingface-fine-tuning/` | 12,274 | Fine-tuning recipes and examples |
| llms-with-huggingface | `../llms-with-huggingface/` | 13,869 | LLM integration patterns |
| HF-Hub-Ecosystem | `../HF-Hub-Ecosystem/` | 16,978 | Comprehensive HF Hub code |
| pytorch | `../pytorch/` | 4,217 | ML framework fundamentals |
| vllm | `../vllm/` | 2,400 | Inference serving |
| databricks-data-engineering | `../databricks-data-engineering/` | 3,038 | Data engineering patterns |
| algorithm-competition-corpus | `../algorithm-competition-corpus/` | 201 | Algorithms + data structures |
| coursera-stats | `../coursera-stats/` | 430 | Statistical modeling |
| cuda-python | `../cuda-python/` | 161 | GPU computing |
| **Total** | | **53,568** | **All MIT / Apache 2.0** |

#### 5.2.3 Pre-Built Local Datasets

| File | Path | Format | Size |
|------|------|--------|------|
| hf_gtc_corpus.parquet | `../hf-ground-truth-corpus/hf_gtc_corpus.parquet` | Parquet | 2 MB |
| corpus_manifest_v1.json | `../depyler/corpus_manifest_v1.json` | JSON | Tier metadata |
| corpus_tiers.json | `../depyler/corpus_tiers.json` | JSON | Complexity metrics |

#### 5.2.4 Data Sourcing Summary

```
Local owned data (~71K files, ~1-2M lines):
├── Tier 1: Ground truth corpora (17,588 files) → upsampled 10x
├── Tier 2: ML framework code   (53,568 files) → natural frequency
└── Pre-built: Parquet + JSON manifests

External data (HuggingFace, ~8B tokens):
├── StarCoder Python subset     (~4B tokens)   → bulk diversity
├── FineWeb-Edu                 (~2B tokens)   → general language
├── Python textbooks/tutorials  (~1B tokens)   → educational code
└── Python docs + PEPs + SO     (~1B tokens)   → API knowledge
```

**Sovereign data advantage**: 20% of training tokens come from data we own,
curate, and can improve. Unlike scraped web data, we know the provenance,
license, and quality of every file. If benchmarks reveal weaknesses in specific
Python patterns, we can add targeted examples to our ground truth corpora and
retrain — a feedback loop no public-dataset-only approach can match.

### 5.3 Fill-in-the-Middle (FIM) Training

Code completion requires fill-in-the-middle capability, not just left-to-right
generation. During training, a fraction of code sequences are transformed using
the PSM (Prefix-Suffix-Middle) format:

```
<fim_prefix>def fibonacci(n):<fim_suffix>    return fib_sequence<fim_middle>
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FIM rate | 50% of code sequences | SantaCoder/StarCoder standard |
| FIM format | PSM (Prefix-Suffix-Middle) | Most common, best tooling support |
| Special tokens | `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>` | Added to BPE vocabulary |
| Context split | Random split point per sequence | Uniform distribution over valid positions |

**Gap ALB-018**: Verify `entrenar` / `alimentar` support FIM data transforms.
If not, implement PSM/SPM sequence transformation in the data pipeline.

This is critical — without FIM, the model is a text generator, not a code
completion engine.

### 5.4 Data Pipeline

```bash
# ── Step 1: Ingest local ground truth corpora (Tier 1 — highest quality) ──
alimentar import local ../depyler/examples/ ../depyler/tdd-book/tests/ \
  --lang python --output ./data/local/depyler.parquet
alimentar import local ../hf-ground-truth-corpus/ \
  --lang python --output ./data/local/hf-gtc.parquet
alimentar import local ../jax-ground-truth-corpus/ \
  --lang python --output ./data/local/jax-gtc.parquet
alimentar import local ../vllm-ground-truth-corpus/ \
  --lang python --output ./data/local/vllm-gtc.parquet

# ── Step 2: Ingest local ML framework code (Tier 2) ──
alimentar import local \
  ../huggingface-fine-tuning/ ../llms-with-huggingface/ ../HF-Hub-Ecosystem/ \
  ../pytorch/ ../vllm/ ../databricks-data-engineering/ \
  ../algorithm-competition-corpus/ ../coursera-stats/ ../cuda-python/ \
  --lang python --output ./data/local/ml-frameworks.parquet

# ── Step 3: Download external data (on intel — 300GB RAM) ──
alimentar import hf bigcode/starcoderdata --lang python --output ./data/starcoder-python/
alimentar import hf HuggingFaceFW/fineweb-edu --output ./data/fineweb-edu/

# ── Step 4: Quality validation ──
alimentar quality check ./data/local/ --profile ml-training
alimentar quality check ./data/starcoder-python/ --profile ml-training
alimentar quality check ./data/fineweb-edu/ --profile ml-training

# ── Step 5: Filter, dedup, shard ──
alimentar filter ./data/starcoder-python/ --lang python --min-tokens 32 --max-tokens 8192 \
  --dedup --output ./data/processed/starcoder-python.parquet
alimentar convert ./data/fineweb-edu/ ./data/processed/fineweb-edu.parquet

# ── Step 6: Build training mix with upsampling ──
alimentar mix \
  --input ./data/processed/starcoder-python.parquet --weight 0.40 \
  --input ./data/local/depyler.parquet --weight 0.025 --upsample 10 \
  --input ./data/local/hf-gtc.parquet --weight 0.025 --upsample 10 \
  --input ./data/local/jax-gtc.parquet --weight 0.025 --upsample 10 \
  --input ./data/local/vllm-gtc.parquet --weight 0.025 --upsample 10 \
  --input ./data/local/ml-frameworks.parquet --weight 0.10 \
  --input ./data/processed/fineweb-edu.parquet --weight 0.20 \
  --input ./data/processed/textbooks.parquet --weight 0.10 \
  --input ./data/processed/python-docs.parquet --weight 0.10 \
  --output ./data/mixed/ \
  --seed 42 --shuffle

# ── Step 7: Record provenance ──
alimentar provenance ./data/mixed/ --output ./data/provenance.json
```

**Gap ALB-019**: Verify `alimentar import local` supports recursive Python file
ingestion from local directories. If not, implement local filesystem source.

**Gap ALB-020**: Verify `alimentar mix` supports weighted multi-source mixing
with upsampling. If not, implement mix command with upsample parameter.

### 5.5 Tokenizer

**Existing capability**: `aprender::text::tokenize::BpeTokenizer` with full
`train()` / `encode()` / `decode()` support. `entrenar::tokenizer::BPETokenizer`
provides the training-pipeline integration.

```bash
# Plan: validate inputs, estimate vocab training time
apr tokenize plan \
  --input ./data/processed/*.parquet \
  --vocab-size 32768 \
  --algorithm bpe \
  --output ./models/albor-tokenizer/

# Apply: train the tokenizer
apr tokenize apply \
  --input ./data/processed/*.parquet \
  --vocab-size 32768 \
  --algorithm bpe \
  --output ./models/albor-tokenizer/ \
  --seed 42
```

**Gap ALB-001**: Verify `apr tokenize plan/apply` exists as a CLI subcommand.
If not, wire `aprender::text::tokenize::BpeTokenizer::train()` into apr with
the plan/apply contract (see §1.5.2).
