# 3. Model Architecture

### 3.1 Architecture: LLaMA-Style Decoder-Only Transformer

entrenar's transformer is a pre-norm LLaMA-style architecture with RMSNorm,
SwiGLU FFN, Grouped-Query Attention, and RoPE. This is hardcoded in the
`Transformer` struct — we configure it via YAML, we don't build it from scratch.

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Parameters | ~350M | Fits in 4090 VRAM with optimizer state in fp16 |
| Layers | 24 | GPT-2 Medium proven at this depth |
| Hidden dim (d_model) | 1024 | Standard for this param count |
| Attention heads | 16 | d_head = 64, well-studied |
| KV heads | 4 | GQA with 4:1 ratio (memory efficient) |
| FFN dim (intermediate) | 4096 | ~4x hidden dim (SwiGLU gate + up + down) |
| Vocab size | 32,768 | BPE trained on corpus (power of 2 for GPU efficiency) |
| Context length | 2048 | Sufficient for benchmarks, manageable VRAM |
| Position encoding | RoPE | Built into entrenar's `MultiHeadAttention` |
| Attention | GQA | Built into entrenar, fewer KV heads than Q heads |
| Normalization | RMSNorm | Built into entrenar, pre-norm (before attn + FFN) |
| FFN activation | SwiGLU | Built into entrenar (gate_proj, up_proj, down_proj) |
| Dropout | 0.0 | Modern practice for pre-training (regularize via data) |

### 3.2 Progressive Model Sizing

To validate the pipeline quickly, we train progressively larger models.
Each gets its own YAML config file (see §6.2 for full config format).

| Model | Config | Params | Layers | Hidden | Heads | Purpose |
|-------|--------|--------|--------|--------|-------|---------|
| albor-50M | `pretrain-50m.yaml` | ~50M | 12 | 512 | 8 | Pipeline validation (hours) |
| albor-125M | `pretrain-125m.yaml` | ~125M | 16 | 768 | 12 | Intermediate, first benchmarks (1-2 days) |
| albor-350M | `pretrain-350m.yaml` | ~350M | 24 | 1024 | 16 | Final base model (3-7 days) |

The 50M model proves the entire stack works end-to-end before committing
days of GPU time to the 350M run.

### 3.3 VRAM Budget (fp16 mixed precision, RTX 4090)

| Component | Size |
|-----------|------|
| Model weights (fp16) | ~700 MB |
| Adam optimizer states (fp32 m, v) | ~2.8 GB |
| Gradients (fp16) | ~700 MB |
| Activations (grad checkpoint, batch=8, seq=2048) | ~8-12 GB |
| **Total estimated** | **~13-16 GB** |
| **4090 headroom** | **~8-11 GB** |

Fits comfortably. Batch size tunable up to ~16 with gradient checkpointing.
