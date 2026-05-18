<p align="center">
  <img src="assets/hero.svg" alt="Albor — Sovereign Python Code Completion" width="100%"/>
</p>

<p align="center">
  <em>Albor</em> (Spanish: "dawn") — A sovereign Python code completion model trained from first principles using only the Sovereign AI stack.
</p>

<p align="center">
  <a href="https://paiml.github.io/albor/">Specification Book</a> &middot;
  <a href="docs/book/src/spec/06-training.md">Training Log</a> &middot;
  <a href="docs/book/src/spec/11-gaps.md">Gap Register</a>
</p>

---

> ### 🎉 Albor SHIPPED (2026-05-18) — via the aprender monorepo
>
> Active development of Albor moved into the [`paiml/aprender`](https://github.com/paiml/aprender) monorepo after the APR-MONO consolidation. The shipped artifact is **[`paiml/albor-370m-v1`](https://huggingface.co/paiml/albor-370m-v1)** on HuggingFace Hub — a 494M-parameter Qwen2-architecture model trained from `Qwen/Qwen2.5-Coder-0.5B-Instruct` init + fine-tuned on bigcode/the-stack-dedup + codeparrot/codeparrot-clean (Python permissive subset), val_loss=4.6227. Apache-2.0. Three usage paths verified: `apr run` (Rust), HF Transformers (`AutoModelForCausalLM.from_pretrained`), and llama.cpp (GGUF Q4_K).
>
> The ship is framed as a **§88 stack-existence-proof** per [SPEC-SHIP-MODEL-2 §84](https://github.com/paiml/aprender/blob/main/docs/specifications/aprender-train/ship-model-2-spec.md) — demonstrating that the pure-Rust pipeline runs end-to-end on real data, not that the resulting model is competitive at code completion. The compute-bounded target (val_loss ≤ 4.7 within 48 GPU-hours) was met; the strict target (val_loss ≤ 2.2) is deferred to the distillation epic.
>
> The publish pipeline is now codified in [**SPEC-HF-PUBLISH-001**](https://github.com/paiml/aprender/blob/main/docs/specifications/aprender-train/model-hf-publish-pipeline-spec.md) — 12-file minimum, YAML schema, NDJSON-commit rule, LFS-batch flow, 13-tier crates.io cascade, three-path verification. Future Albor iterations follow that spec.
>
> Post-ship audit with Popperian falsification + Sorscher 2022 ([arXiv:2206.14486](https://arxiv.org/abs/2206.14486)) literature support: [`audits/albor-370.md`](https://github.com/paiml/aprender/blob/main/docs/specifications/audits/albor-370.md).
>
> **This repo (`paiml/albor`) remains the historical reference for the standalone v0.1.0–v0.2.x lineage and the v1–v28 training history.** Last active commit: 2026-04-05. The 54/54 ALB-* contracts and the v28/v29 data-quality work (ALB-134) remain valid context for the monorepo's continued iteration.

---

## What Is Albor?

A **350M-parameter decoder-only transformer** for Python code completion, trained entirely in Rust with zero Python dependencies. Every operation — data loading, tokenization, training, evaluation, checkpointing — uses the Sovereign AI stack. No PyTorch, no pip, no conda.

**The project has two goals:**
1. Produce a **usable Python code completion model** that runs anywhere Rust compiles
2. Identify and fix every gap in the [Sovereign AI stack](https://github.com/paiml) that blocks end-to-end LLM development — **129+ gaps found, 53 contracts validated**

## Current Status

**v28 STOPPED** — peaked at step 6K, diverged by step 11K. Next: v29 on filtered data.

| Metric | Value |
|--------|-------|
| Training run | v28 fresh — **STOPPED** at step 11K (diverging on raw data) |
| Best val_ppl | **38.53** at step 6K (best checkpoint saved) |
| Final val_ppl | 75.65 at step 11K (regression — raw data ceiling) |
| Prior best (v28 orig) | **5.88** at step 3.5K (killed by `cargo-killer`) |
| Config | lr=7.35e-5, GA=32 (131K tok/step), wd=0.012 |
| Throughput | **11,000 tok/s, 36.3% MFU** (ALB-078 fused gradient clipping) |
| Hardware | RTX 4090 (24 GB), single GPU |
| Next | **v29** on quality-filtered data (2.04B tokens, `data/pretokenized-1024-v4/`) |

### Training History: v1 to v28

28 training runs across three phases:

**Phase 1 — Correctness (v1-v22):** Fixed 7 critical backward pass bugs via
systematic Five Whys analysis and PyTorch canary comparison:

| Root Cause | Impact | Gap |
|-----------|--------|-----|
| Missing residual skip gradients | **val_ppl 726 → 9.44** (6.3x) | ALB-126/127 |
| Gradient clipping using ‖g‖⁴ | 3% improvement | ALB-124 |
| No causal attention mask | Minimal | ALB-125 |
| Wrong weight init (sinusoidal) | Step-0 parity | entrenar#309 |
| Sequential shard overfitting | val_ppl regression at step 3K | entrenar#315 |

**Phase 2 — Hyperparameters (v22-v27):** Systematic HPO via `apr train halving`:

| Run | LR | Batch (tok/step) | Best val_ppl | Key Finding |
|-----|-----|-------------------|-------------|-------------|
| v22 | 3e-4 | 32K | 9.44 (memorized) | Overfitting without shuffle |
| v25 | 3e-4 | 32K | 69 → 314 (spike) | LR too high for small batch |
| v26 | 1e-4 | 32K | 15 → 162 → 41 | Still oscillating |
| v27 | 7.35e-5 | 131K | 9.39 → 82 (diverged) | HPO-validated, but cosine schedule broken |

**Phase 3 — Schedule fix (v28):** ALB-129 — `max_steps` calibrated to actual
training horizon. One config change, best result ever:

| Step | v28 val_ppl | v27 val_ppl |
|------|-----------|-----------|
| 2K | 9.65 | 9.39 (peak) |
| 2.5K | 6.87 | 12.57 (diverging) |
| 3K | 6.16 | 16.07 |
| 3.5K | **5.88** | 32.02 |
| 5K | 6.70 | 55.86 |

v27's cosine schedule used `max_steps=155K` for a 38K-step run — LR never
decayed (99% of peak at step 10K). v28 sets `max_steps=38349` so cosine
completes its full decay. The model pushes through the divergence zone and
keeps improving.

### Pipeline to HumanEval

| Phase | What | Status | ETA |
|-------|------|--------|-----|
| **v28** | Full epoch, unfiltered codeparrot (5B tokens, 38K steps) | **STOPPED** step 11K (diverged) | Best: step 6K |
| **v29** | Filtered codeparrot (2B clean tokens, 15.5K steps) | **READY** | ~2.4 days |
| **Distill** | Best checkpoint + Qwen3-Coder-30B teacher | **CONFIG READY** | after v29 |
| **HumanEval** | Target: 5-15% pass@1 | pending | after distill |

### Data Filtering

codeparrot-clean (raw GitHub Python) filtered to high-quality subset:

| Metric | Unfiltered (v28) | Filtered (v29) |
|--------|-----------------|----------------|
| Files | 2.9M | 850K (29% pass rate) |
| Tokens | 5.08B | 2.04B |
| Syntax valid | ~87% | **100%** |
| Has docstrings | ~40% | **100%** (≥1 per 50 lines) |
| Has imports | ~60% | **100%** (≥2 unique) |
| No generated code | ~80% | **100%** |

Hypothesis: 2B clean tokens > 5B noisy tokens for code generation quality,
following phi-1's finding that data quality dominates model scale.

### Research Context

**Why this is hard** ([Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)):
phi-1-small (350M) achieved 45% HumanEval — but with 7B tokens of synthetic
textbook-quality data generated by GPT-3.5. Albor trains on codeparrot-clean
(raw GitHub Python). Data quality is the primary ceiling, not model architecture.
v29 addresses this with quality filtering; distillation adds synthetic data.

**Scaling position** ([Chinchilla](https://arxiv.org/abs/2203.15556)):
Our 5.08B tokens on 350M params (14.5:1) is below Chinchilla-optimal (20:1 = 7B tokens).
The [Step Law](https://arxiv.org/abs/2503.04715) formula gives lr=3.2e-3 and
batch=183K tokens for this model size — our HPO-validated lr=7.35e-5 is
conservative because of the interleaved per-block optimizer architecture.

**HPO methodology** ([μTransfer](https://arxiv.org/abs/2203.03466),
[Hyperband](https://arxiv.org/abs/1603.06560)):
Hyperparameters are tuned on a 50M proxy model via `apr train sweep` +
`apr train halving`, then transferred to 350M via μTransfer width scaling.
Contract: C-HPO-001 (`contracts/hyperparameter-tuning-v1.yaml`).

## Architecture

```
LLaMA-style decoder-only transformer
├── 24 layers, 1024 hidden dim, 16 attention heads, 4 KV heads (GQA)
├── SwiGLU FFN (4096 intermediate), RoPE, RMSNorm (pre-norm)
├── 32,768 vocab (ByteLevel BPE v2), 1024 context (GPU-resident)
├── ~370M parameters, GPU-resident AdamW on RTX 4090 (~14.8 GB VRAM)
└── Cosine LR schedule (7.35e-5 peak, 38K steps, 93-step warmup, GA=32)
```

## Automatic Hyperparameter Tuning (C-HPO-001)

```bash
# Step 1: Generate 16 random sweep configs from 50M proxy base
apr train sweep --config configs/train/pretrain-50m-sweep.yaml \
    --strategy random --num-configs 16 --seed 42 \
    --output-dir sweeps/50m-hpo/

# Step 2: Run successive halving (kills worst half each round)
apr train halving --sweep-dir sweeps/50m-hpo/ \
    --rounds 3 --steps-per-round 500 \
    --source-width 512 --target-width 1024 \
    --output sweeps/50m-hpo/results.json

# Step 3: Winner's LR is μTransfer-scaled for 350M
cat sweeps/50m-hpo/results.json | jq '.winner'
```

References:
[μTransfer](https://arxiv.org/abs/2203.03466),
[Hyperband](https://arxiv.org/abs/1603.06560),
[MiniCPM](https://arxiv.org/abs/2404.06395),
[Step Law](https://arxiv.org/abs/2503.04715)

## Sovereign AI Stack

| Component | Role | Gaps Fixed |
|-----------|------|-----------|
| [entrenar](https://github.com/paiml/entrenar) | Training engine (CUDA kernels, optimizer) | 45+ |
| [trueno](https://github.com/paiml/trueno) | GPU tensor ops (RoPE, RMSNorm, PTX) | 15+ |
| [aprender](https://github.com/paiml/aprender) (`apr`) | CLI (train, sweep, halving, eval) | 12+ |
| [realizar](https://github.com/paiml/realizar) | Inference (Qwen3 MoE, Q4K) | 5+ |
| [alimentar](https://github.com/paiml/alimentar) | Data pipeline (Parquet, FIM) | 5+ |
| [provable-contracts](https://github.com/paiml/provable-contracts) (`pv`) | Verification (53 contracts) | — |
| [renacer](https://github.com/paiml/renacer) | Tracing (BrickTracer, spans) | 3+ |

## Provable Contracts

Every training invariant is encoded as a YAML contract verified by `pv validate`:

| Contract | What It Proves |
|----------|---------------|
| C-RESIDUAL-001 | Residual skip bypasses RMSNorm backward |
| C-CLIP-001 | Gradient clipping uses L2 norm, not squared |
| C-BACKPARITY-001 | Backward pass matches PyTorch canary |
| C-HPO-001 | μTransfer scaling, successive halving protocol |
| C-INTERLEAVED-001 | LR bound for interleaved optimizer |
| C-TIEDOPT-001 | Tied weight optimizer single-step invariant |

## Reproduce

```bash
# Build the CLI (requires sibling repos: entrenar, trueno, realizar, renacer)
cd ~/src/aprender && CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/aprender \
    cargo build --release -p apr-cli

# IMPORTANT: Pin the binary — other projects sharing CARGO_TARGET_DIR will
# overwrite it with feature-incompatible builds (no train subcommand)
mkdir -p bin && cp /mnt/nvme-raid0/targets/aprender/release/apr bin/apr-train

# Quick regression test (50M model, 5 steps, < 2 min)
bin/apr-train train apply --task pretrain --config configs/train/pretrain-50m-quick.yaml

# Full 350M training on unfiltered data (RTX 4090, ~5 days)
bin/apr-train train apply --task pretrain --config configs/train/pretrain-350m-v28.yaml

# Full 350M training on filtered data (RTX 4090, ~2.4 days)
bin/apr-train train apply --task pretrain --config configs/train/pretrain-350m-v29.yaml

# Evaluate
bin/apr-train eval --task humaneval --data data/humaneval.jsonl \
    --device cpu checkpoints/albor-base-350m-v28/model-best.apr

# Filter codeparrot-clean (CPU, ~50 min)
python3 scripts/filter_codeparrot.py --output data/filtered/

# Pretokenize filtered data (CPU, ~30 min)
python3 scripts/pretokenize_streaming.py \
    --input data/filtered/train/ \
    --output data/pretokenized-1024-v4/train/ \
    --tokenizer models/albor-tokenizer-v2/tokenizer.json
```

## License

Apache-2.0
