# Fused Kernel Optimizations Contract

**Contract**: `contracts/fused-kernels-v1.yaml`
**Version**: 1.0.0
**Status**: NEW (ALB-075 Phase 4+)
**Depends on**: cublas-gemm-v1, training-gpu-kernel-v1, training-step-budget-v1
**Source**: [unslothai/unsloth](https://github.com/unslothai/unsloth) analysis

## Equations

### fused_cross_entropy

```
For each row r in logits [B*S, V]:
  logsumexp_r = log(sum(exp(logit[r, i])))
  loss_r = logsumexp_r - logit[r, label_r]
  grad_r[i] = exp(logit[r, i] - logsumexp_r) - delta(i, label_r)
```

Single kernel pass. FP32 accumulation. Softmax tensor never materialized.
Backward grad overwrites logits buffer in-place (zero extra allocation).

### rmsnorm_activation_reuse

```
Forward: save ONLY inv_var [B*S] (not normed — recompute in backward)
Backward: normed = X_cached * inv_var_saved (bit-exact recompute)
Memory savings: 24 layers * B * S * H * 4 bytes = 384 MB
```

### swiglu_inplace_backward

```
d_up = grad_output * silu(gate)          → written to up buffer
d_gate = grad_output * up * silu'(gate)  → written to gate buffer
```

gate and up consumed before overwrite. Peak workspace reduced by 128 MB.

### rope_head_grouping

```
Load sin/cos once per group (G=4 heads)
Apply to all heads in group with single memory load
Q: 4 groups of 4, K: 1 group of 4
```

Bit-exact with per-head RoPE. ~10% attention speedup from L2 cache reuse.

### fused_tiled_attention

```
For tile_q, tile_k in tiled [0, S):
  scores_tile = Q[tile_q] @ K[tile_k]^T / sqrt(d_k)
  Online softmax (Milakov & Gimelshein 2018):
    m_new = max(m_old, max(scores_tile))
    l_new = l_old * exp(m_old - m_new) + sum(exp(scores_tile - m_new))
  O += exp(scores_tile - m_new) @ V[tile_k]
O = O / l_new
```

Full [S, S] attention matrix never materialized. Memory: O(B*H*S*d_k)
instead of O(B*H*S*S). Saves 256 MB per layer.

### chunked_cross_entropy (deferred)

For vocab > 65K: split logsumexp into 65K chunks. Mathematically exact
(logsumexp is associative). Current vocab=32K: single chunk, no overhead.

## Proof Obligations (10)

| ID | Type | Property |
|----|------|----------|
| 1 | equivalence | Fused CE matches separate CE (< 1e-5) |
| 2 | invariant | Fused CE never allocates softmax tensor |
| 3 | equivalence | RMS norm recompute is bit-exact |
| 4 | bound | Activation memory reduced by >= 300 MB |
| 5 | equivalence | SwiGLU in-place backward correct (< 1e-5) |
| 6 | equivalence | RoPE grouped matches individual (bitwise) |
| 7 | equivalence | Fused attention matches separate (< 1e-3) |
| 8 | bound | Fused attention memory < separate / 4 |
| 9 | invariant | Training stability preserved (loss finite) |
| 10 | invariant | Gradient flow preserved (all params) |

## Falsification Tests (10)

| ID | Rule | Prediction |
|----|------|------------|
| FALSIFY-FUSED-001 | Fused CE matches separate | `max_abs_diff(loss) < 1e-5` 50 steps |
| FALSIFY-FUSED-002 | RMS norm recompute exact | Bitwise match all 24 layers |
| FALSIFY-FUSED-003 | SwiGLU in-place correct | `max_abs_diff(d_gate, d_up) < 1e-5` |
| FALSIFY-FUSED-004 | RoPE grouped matches | Bit-exact 16 Q + 4 K heads |
| FALSIFY-FUSED-005 | Fused attention matches | `max_abs_diff < 1e-3` (FP32) |
| FALSIFY-FUSED-006 | Memory savings >= 300 MB | Activation peak reduction measured |
| FALSIFY-FUSED-007 | No full softmax alloc | Peak CE memory < `B*S*V*4` |
| FALSIFY-FUSED-008 | Grad checkpoint exact | Bitwise gradient match |
| FALSIFY-FUSED-009 | Fused attn backward OK | All params get grads, loss within 1% |
| FALSIFY-FUSED-010 | No instability | 100 steps, loss finite, gnorm < 100 |

## Priority Matrix

| # | Optimization | Gain | Memory | Phase |
|---|-------------|------|--------|-------|
| 1 | Fused CE loss | 20-40ms/step | -512 MB bandwidth | 4 |
| 2 | RMS norm reuse | 0 compute | **-384 MB** | 4 |
| 3 | SwiGLU in-place | 10-20ms/step | -128 MB peak | 4 |
| 4 | RoPE grouping | 5-10ms/step | 0 | 4 |
| 5 | Fused attention | 15% attn speedup | **-256 MB/layer** | 5 |
| 6 | Chunked CE | future | 0 | Deferred |
| 7 | Grad checkpoint | -2x backward | **-66% activations** | 7 |

## QA Gate

**F-FUSED-001**: All 10 falsification tests must pass. If combined run shows
instability, bisect fusions individually to identify the culprit.
