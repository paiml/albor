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

## What Is Albor?

A **350M-parameter decoder-only transformer** for Python code completion, trained entirely in Rust with zero Python dependencies. Every operation — data loading, tokenization, training, evaluation, checkpointing — uses the Sovereign AI stack. No PyTorch, no pip, no conda.

**The project has two goals:**
1. Produce a **usable Python code completion model** that runs anywhere Rust compiles
2. Identify and fix every gap in the [Sovereign AI stack](https://github.com/paiml) that blocks end-to-end LLM development — **97 gaps found and fixed so far**

## Current Status

**v15 training RUNNING** (March 22, 2026) — Phase change confirmed at step 3K.

| Metric | Value |
|--------|-------|
| Training run | v15 (15th attempt, seed=123) |
| Step | 5,000 / 155,000 (3.2%) |
| Best val_ppl | **333** (step 5K) |
| Throughput | 8,500 tok/s, 24.6% MFU |
| Hardware | RTX 4090 (24 GB), single GPU |
| Data | codeparrot-clean, 5.08B tokens (73% Chinchilla) |
| ETA | ~5 days from launch |

**v15 is the strongest early convergence of any run** — 21% ahead of the previous best (v13) at step 5K. Phase change occurred at step 3K (v13 at 4K, v9 at 4.75K).

### What We've Built

Through 15 training runs and 97 bug fixes, the project has:

- **Trained a 350M transformer on a single GPU entirely in Rust** — no Python anywhere in the stack
- **Achieved 8.5K tok/s at 24.6% MFU** on an RTX 4090 with hand-written PTX kernels + cuBLAS
- **Discovered and fixed 97 infrastructure gaps** across 6 upstream repos, including:
  - Silent memory corruption in CUDA backward kernels (ALB-041, 043, 059)
  - Missing RoPE backward pass (ALB-119) — model trained without position gradients
  - GPU optimizer state not checkpointed (ALB-118) — resume destroyed weights
  - Data loader position not checkpointed (ALB-120) — resume caused data overlap
  - Activation gradient overflow at GPU-CPU boundary (ALB-044) — NaN in embeddings
  - Stream synchronization race conditions (ALB-065) — stale GPU data on D2H transfer
- **Reached val_ppl=129** (v9) on 490M tokens — v15 is on track to beat this with 10x more data
- **Built 39 provable contracts** verified by `pv` (provable-contracts)
- **108/108 batuta falsification tests PASS** (Toyota Standard grade)

### What Needs to Happen

| Milestone | When | Success Criteria |
|-----------|------|------------------|
| v15 surpasses v9 (val_ppl < 129) | Step 10-15K (~2 days) | Model learns sentence-level patterns |
| v15 converges (val_ppl < 50) | Step 155K (~5 days) | Model captures syntactic structure |
| HumanEval pass@1 > 8% | After convergence | Model generates valid Python |
| Distillation from Qwen3-Coder-30B | After base model | Synthetic data pipeline ready |
| HumanEval pass@1 > 15% | After distillation | Beat CodeGen-350M-mono (12.8%) |
| Big Code Leaderboard submission | After distillation | First sub-1B model on the board |

### When We Declare Success

**Minimum viable (Phase 3):** Base model converges to val_ppl < 100 and achieves HumanEval pass@1 > 8%. This proves the sovereign stack can train a working code model.

**Good (Phase 5):** Distilled model hits HumanEval pass@1 > 15%, beating CodeGen-350M-mono. This proves distillation from a 30B MoE teacher works through the sovereign stack.

**Full success (Phase 8):** All 6 model variants benchmarked, Q4 model under 100MB runs at <50ms/token on CPU, submitted to Big Code Leaderboard as the first sub-1B entry.

## Architecture

```
LLaMA-style decoder-only transformer
├── 24 layers, 1024 hidden dim, 16 attention heads, 4 KV heads (GQA)
├── SwiGLU FFN (4096 intermediate), RoPE, RMSNorm (pre-norm)
├── 32,768 vocab (ByteLevel BPE v2), 1024 context (GPU-resident)
├── ~370M parameters, GPU-resident AdamW on RTX 4090 (~13 GB VRAM)
└── Cosine LR schedule (3e-4 peak, 155K steps, 2K warmup)
```

## Training History

15 training runs, each revealing and fixing infrastructure bugs:

| Run | Steps | Best val_ppl | Outcome | Key Fix |
|-----|-------|-------------|---------|---------|
| v2 | 1K | 1,008 | Crashed | ALB-073: PTX instruction bug |
| v3 | 28K | 1,018 | Plateau | ALB-079: no cosine LR decay |
| v5 | — | — | Failed | ALB-092: gradient accumulation bug |
| v8 | 5K | — | Killed | ALB-106: trained without RoPE |
| **v9** | **15K** | **129** | **Stopped** | **Best genuine result** (490M tokens) |
| v13 | 62K | 239 (inflated) | Stopped | ALB-120: data position not checkpointed |
| v14 | 20K | 782 | Killed | Degenerate init (seed=42) |
| **v15** | **5K+** | **333** | **Running** | **Phase change at step 3K** |

## Sovereign AI Stack

| Component | Role | Gaps Fixed |
|-----------|------|-----------|
| [entrenar](https://github.com/paiml/entrenar) | Training engine | 40+ (CUDA kernels, optimizer, checkpoint) |
| [trueno](https://github.com/paiml/trueno) | GPU tensor ops | 15+ (RoPE, RMSNorm, cuBLAS, PTX) |
| [aprender](https://github.com/paiml/aprender) (`apr`) | CLI | 10+ (eval, train, checkpoint) |
| [realizar](https://github.com/paiml/realizar) | Inference | 5+ (Qwen3 MoE, Q4K) |
| [alimentar](https://github.com/paiml/alimentar) | Data pipeline | 5+ (Parquet, FIM) |
| [provable-contracts](https://github.com/paiml/provable-contracts) | Verification | 39 contracts |

## Reproduce

```bash
# Build the CLI
cd ~/src/aprender && cargo build --release -p apr-cli

# Train from scratch (RTX 4090, ~5 days)
apr train apply --task pretrain --config configs/train/pretrain-350m-v15.yaml

# Evaluate
apr eval --task humaneval --model checkpoints/albor-base-350m-v15/model-best.apr
```

## License

Apache-2.0
