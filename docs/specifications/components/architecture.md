# Model Architecture

**Parent**: [albor-llm-spec.md](../albor-llm-spec.md) §2

---

## 1. Overview

Albor is a **LLaMA-style decoder-only transformer** with 350M parameters
(~370M actual including embeddings). It uses Grouped Query Attention (GQA),
SwiGLU FFN, RMSNorm, and Rotary Position Embeddings (RoPE).

The architecture is nearly identical to phi-1-small (350M, h=1024, 20 layers,
45% HumanEval) — Albor has 4 more layers, which gives slightly more capacity
at minimal VRAM cost.

---

## 2. Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size (d) | 1024 | |
| num_hidden_layers | 24 | phi-1: 20 |
| num_attention_heads (Q) | 16 | head_dim = 64 |
| num_kv_heads (KV) | 4 | GQA ratio = 4:1 |
| intermediate_size | 4096 | SwiGLU: 2× gate projection |
| vocab_size | 32768 | ByteLevel BPE |
| max_position_embeddings | 1024 | RoPE, theta=10000 |
| rms_norm_eps | 1e-5 | |
| rope_theta | 10000 | Standard LLaMA value |

### 2.1 Parameter Count Breakdown

| Component | Parameters | Formula |
|-----------|-----------|---------|
| Token embedding | 33.6M | vocab × d = 32768 × 1024 |
| Per-layer attention | 1.31M | d×d + 2×(d×d/4) + d×d = 3.5d² with GQA |
| Per-layer FFN | 8.39M | 3 × d × intermediate = 3 × 1024 × 4096 (SwiGLU) |
| Per-layer norms (×2) | 2K | 2 × d = 2 × 1024 |
| 24 layers total | 233M | 24 × (1.31M + 8.39M + 2K) |
| Final norm | 1K | d |
| LM head | 33.6M | d × vocab (weight-tied with embedding) |
| **Total unique** | ~267M | Excluding tied LM head |
| **Total with LM head** | ~370M | If untied (current config) |

### 2.2 Weight Tying

Current config: LM head weights are **NOT** tied with token embeddings.
This adds 33.6M parameters. phi-1 uses untied heads. We follow suit.

If VRAM becomes a concern, tying saves 33.6M params (130 MB at f32).

---

## 3. Attention

### 3.1 Grouped Query Attention (GQA)

16 query heads share 4 KV heads (ratio 4:1). Each query group of 4 heads
attends to the same key and value projections.

```
Q: [batch, seq, 16, 64] → 16 heads × 64 dim
K: [batch, seq, 4, 64]  →  4 KV heads × 64 dim
V: [batch, seq, 4, 64]  →  4 KV heads × 64 dim
```

KV cache at inference: 4 heads instead of 16 → 4× smaller cache.

### 3.2 Rotary Position Embeddings (RoPE)

Applied to Q and K before attention. theta=10000, max_position=1024.

```
cos/sin tables: [max_seq, head_dim/2] precomputed
Apply: rotate pairs of dimensions by position-dependent angle
```

### 3.3 Causal Mask

Lower-triangular attention mask. No sliding window, no sparse attention.
Full causal attention up to 1024 tokens.

---

## 4. Feed-Forward Network (SwiGLU)

```
FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down

W_gate: [d, intermediate] = [1024, 4096]
W_up:   [d, intermediate] = [1024, 4096]
W_down: [intermediate, d] = [4096, 1024]
```

Three weight matrices per layer (vs two for standard FFN). The gating
mechanism (SiLU activation on W_gate) provides smoother optimization
landscape than ReLU.

---

## 5. Normalization

**RMSNorm** (not LayerNorm). Applied:
1. Before attention (pre-norm): `attn_input = RMSNorm(x)`
2. Before FFN (pre-norm): `ffn_input = RMSNorm(attn_out + x)`
3. After final layer: `logits_input = RMSNorm(last_hidden)`

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma

gamma: [d] learnable scale (no bias)
eps: 1e-5
```

Pre-norm architecture (norm before sublayer, not after) — standard for
modern transformers. More stable training, especially at scale.

### 5.1 ALB-092 Fix: Gradient Flow Through Norms

v3-v5 had a critical bug: `BatchedRmsNormBackwardKernel` declared
`grad_gamma_ptr` but never wrote to it. Norm gamma weights received zero
gradient — only weight decay updated them. Fixed in trueno PR #178.

50 norm parameters (24 attn + 24 ffn + 2 final) now receive proper gradients.

---

## 6. Residual Connections

Standard pre-norm residual:
```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

No residual scaling, no layer scaling. With proper initialization and
gradient clipping, this is stable for 24 layers.

---

## 7. Initialization

All weights initialized from N(0, 0.02). No special initialization for
residual connections (no 1/√N scaling). This is the standard LLaMA approach.

AdamW optimizer handles the large initial gradients via adaptive learning rate.
Combined with warmup (500 steps) and gradient clipping (max_norm=1.0), training
is stable from step 1.

---

## 8. Tokenizer

**ByteLevel BPE**, 32K vocabulary.

| Property | Value |
|----------|-------|
| Type | ByteLevel BPE (HuggingFace tokenizers) |
| Vocab size | 32768 |
| Path | `models/albor-tokenizer-v2/tokenizer.json` |
| Special tokens | `<|endoftext|>` (ID 0), `<|padding|>` (ID 1) |
| Byte fallback | Yes (covers all UTF-8) |

Trained on codeparrot-clean Python corpus. Optimized for Python syntax:
common keywords (`def`, `class`, `import`, `return`), indentation patterns,
and standard library names are single tokens.

---

## 9. Sequence Format

### 9.1 Pre-training

Standard causal language modeling on concatenated Python files:
```
<|endoftext|> [code file 1] <|endoftext|> [code file 2] <|endoftext|> ...
```

Packed sequences to `max_position_embeddings` (1024 tokens). No padding
within sequences — only between the last `<|endoftext|>` and sequence end.

### 9.2 Fill-in-the-Middle (FIM) — Future

FIM transform for code completion (not yet implemented in alimentar):
```
<|fim_prefix|> [before cursor] <|fim_suffix|> [after cursor] <|fim_middle|> [completion]
```

Requires 3 additional special tokens. Will be added post-distillation.

---

## 10. Comparison to Reference Models

| Property | Albor | phi-1-small | CodeGen-350M |
|----------|-------|-------------|--------------|
| Params | 370M | 350M | 350M |
| Layers | 24 | 20 | 20 |
| Hidden | 1024 | 1024 | 1024 |
| Heads | 16 | 16 | 16 |
| KV heads | 4 (GQA) | 16 (MHA) | 16 (MHA) |
| FFN | SwiGLU 4096 | GELU 4096 | GELU 4096 |
| Norm | RMSNorm | LayerNorm | LayerNorm |
| Vocab | 32K | 51K | 51K |
| Seq len | 1024 | 2048 | 2048 |
| HumanEval | 0% (undertrained) | 45% | 12% |

Albor has modern architectural improvements (GQA, RMSNorm, SwiGLU) over both
references. The performance gap is entirely about data quality and training
volume — not architecture.
