# ADR-001: GPU-Resident Training Architecture

**Date**: 2026-02-28
**Status**: Accepted
**Context**: Need to train 350M model on RTX 4090 (24GB VRAM)

## Decision

Use CudaTransformerTrainer with per-block interleaved backward+optimizer.
All 24 transformer blocks remain GPU-resident; only 3 PCIe transfers per step.

## Alternatives Considered

1. **CPU fallback training** — Too slow (~100x slower than CUDA)
2. **Gradient checkpointing** — Not yet implemented in entrenar
3. **Model parallelism** — Single GPU only

## Negative Results

- **seq_len=2048 OOM**: Logits tensor [batch, 2048, 32768] exceeds VRAM headroom.
  Reduced to seq_len=1024. Filed as architectural limitation.
- **batch_size=8 OOM**: Even with seq_len=512, large batches overflow with 24 blocks.
  Reduced to batch_size=4 with grad_accum=128.
- **ALB-043 backward_ffn overflow**: SwiGLU intermediate buffer was 4x undersized.
  Silent memory corruption manifested as wrong gradients, not crashes.

## Consequences

- Training throughput: ~377 tok/s on RTX 4090
- Must checkpoint frequently (save_interval=25) due to long optimizer steps
- Future: gradient checkpointing would allow seq_len=2048
