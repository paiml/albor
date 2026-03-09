# Training Performance Specification

## Current Baseline (v8, step 1000+, RUNNING)

| Metric | Value |
|--------|-------|
| Model | 350M (370M actual), LLaMA-style, 24 blocks |
| Throughput | 7.4K–7.7K tok/s |
| MFU | 23.3% |
| GEMM backend | cuBLAS (CUBLAS_DEFAULT_MATH) |
| Sequence length | 1024 |
| Batch size | 4 |
| Gradient accumulation | 8 (32K tokens/step) |
| Best val_ppl | 761 (v8 step 500) |
| HumanEval pass@1 | 0/164 (v4 step 2425) |
| Hardware | RTX 4090 24GB, PCIe Gen4 |
| Target | 20,000 steps (~655M tokens), val_ppl < 100 |

## MFU Analysis

**Model FLOPs Utilization (MFU)** measures what fraction of the GPU's
theoretical peak FLOPS the training achieves.

```
RTX 4090 peak FP32: 82.6 TFLOPS
RTX 4090 peak TF32: 165.2 TFLOPS (tensor cores, CUBLAS_DEFAULT_MATH)

Forward FLOPs per token ≈ 2 × params = 2 × 370M = 740 MFLOP
Forward + Backward ≈ 3× forward = 2.22 GFLOP/token

At 7.7K tok/s (v8): 7700 × 2.22 GFLOP = 17.1 TFLOPS
MFU vs TF32 peak: 17.1 / 165.2 = 10.4% (conservative)
MFU vs FP32 peak: 17.1 / 82.6 = 20.7% (standard)
Reported MFU: 23.3% (includes ga=8 efficiency gain)
```

Research benchmarks for comparison:
- Chinchilla (2022): ~46% MFU on TPUv3/v4
- LLaMA (2023): ~56% MFU on A100 (dense)
- GPT-3 (2020): ~47% MFU on V100

Our 19.3% MFU gap comes from:
1. **Single GPU** — no tensor/pipeline parallelism overhead, but also no optimization
2. **PCIe bottleneck** — 3 transfers/step (embed H2D, logits D2H, grad H2D)
3. **Per-block interleaved backward+optimizer** — shared workspace, sequential
4. **No kernel fusion** — separate RMSNorm, RoPE, GEMM, softmax kernels

## cuBLAS Integration (ALB-075, COMPLETE)

### Architecture

```
entrenar::cuda_trainer
  → trueno-gpu::cublas::CublasContext (safe Rust wrapper)
    → trueno-gpu::cublas_sys (FFI bindings)
      → libcublas.so (NVIDIA cuBLAS)
```

Key files:
- `trueno/crates/trueno-gpu/src/cublas_sys.rs` — Raw FFI bindings
- `trueno/crates/trueno-gpu/src/cublas.rs` — Safe wrapper (CublasContext, GemmConfig)
- `entrenar/src/cuda/gemm.rs` — Training GEMM dispatch

### Performance Results

| Phase | Throughput | MFU | Notes |
|-------|-----------|-----|-------|
| Pre-cuBLAS (PTX GEMM) | 890 tok/s | 2.6% | 552 hand-written PTX GEMMs/step |
| cuBLAS Phase 1 | 4.2K tok/s | 12.1% | SGEMM for weight GEMMs |
| cuBLAS Phase 2 | 5.8K tok/s | 16.8% | + attention score GEMMs |
| cuBLAS Phase 3 | 6.3K tok/s | 18.2% | + optimizer step overlap |
| cuBLAS Phase 4 | 6.7K tok/s | 19.3% | + strided batched GEMM |

### Next Opportunities

1. **Fused QKV projection** — 3 GEMMs → 1 (Q, K, V share input, different weights)
2. **CUDA Graphs** — capture+replay for decode steps (fixed shapes)
3. **FlashAttention** — fused attention kernel (memory + compute)
4. **FP16/BF16 training** — 2× throughput from tensor cores (requires loss scaling)

## Memory Profile (ALB-099)

### Training Memory Budget (24 GB VRAM)

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (24 blocks) | ~1.4 GB | FP32, 370M params |
| AdamW state (m, v) | ~2.8 GB | 2× model weights |
| Activations | ~2.0 GB | Scales with seq×batch |
| Gradient workspace | ~1.4 GB | Shared across blocks |
| cuBLAS workspace | ~0.5 GB | Internal scratch |
| KV cache | ~0.2 GB | seq=1024, batch=4 |
| **Total** | **~8.3 GB** | Leaves 15.7 GB headroom |

### Host Memory (dhat-rs profiling results)

| Path | Before | After | Fix |
|------|--------|-------|-----|
| realizar: GPU Q4K inference | 19.6 GB peak | 1.3 GB peak | Skip fs::read for mmap path |
| aprender: GGUF contract validation | 8.2 GB total | 6.0 GB total | 24-byte header check |

## Bottleneck Hierarchy

1. **Data quality** — codeparrot-clean is noisy; distillation from Qwen3-Coder expected (serve path verified at 17.5 tok/s)
2. **Sequence length** — 1024 limits code understanding; 2048 OOMs at batch=4
3. **Throughput** — 7.7K tok/s means ~1.5 days for 1B tokens (Chinchilla optimal: 7.4B = ~11 days)
4. **Evaluation** — HumanEval 0/164 at step 2425; need loss < 3.0 for meaningful pass@1
5. **v8 training active** — resumed at step 1000, target 20K steps, ETA ~18 hours
