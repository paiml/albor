# 4. Distillation Teacher: Qwen3.5-35B-A3B

### 4.1 Teacher Model Profile

| Property | Value |
|----------|-------|
| Model | [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Parameters | 35B total, 3B active per token (MoE) |
| Architecture | Hybrid: 30 Gated DeltaNet + 10 full GQA layers, MoE FFN (256 experts, top-8 + 1 shared) |
| Hidden dim | 2048, head_dim=256, 16 Q heads, 2 KV heads |
| Layers | 40 (pattern: 3 linear + 1 full attention, repeating) |
| Expert FFN | SwiGLU, intermediate_size=512 per expert |
| Context | 262K tokens (extensible to ~1M via YaRN) |
| License | Apache 2.0 |
| Specialization | Code generation, agentic reasoning |

### 4.2 Why This Teacher

- **Apache 2.0**: Legally clean for distillation, no license contamination
- **35B knowledge at 3B cost**: MoE activates only 8+1 experts per token.
  Inference FLOP budget matches a dense 1.8B model, but the 256 experts
  collectively encode 35B parameters of knowledge. Soft targets are far
  richer than a dense 3B teacher.
- **Fits on a single 4090**: At Q4 quantization, weights occupy ~17.5 GB.
  With activations and KV cache (only 10 full-attention layers need KV
  cache), total VRAM is ~18.3 GB — leaving 5.7 GB headroom on 24 GB.
- **Coding focus**: Distilled student inherits strong code capabilities,
  making it competitive on HumanEval/MBPP — benchmarks where tiny models
  normally fail.
- **realizar already supports most of the architecture**: Gated DeltaNet
  linear attention (GH-278), SwiGLU FFN, GQA, hybrid `layer_types` config,
  and MoE routing (CapacityFactorRouter, PowerOfTwoChoicesRouter) all exist.
  The missing pieces are expert weight loading and dispatch integration.
- **Novel architecture** (DeltaNet + MoE): Exercising realizar's model
  loading on a non-standard architecture is exactly the kind of gap-finding
  that validates the stack.

### 4.2.1 VRAM Budget (Q4, batch=1, seq=2048)

| Component | Size | Notes |
|-----------|------|-------|
| Weights (Q4) | 17.5 GB | 35B params × 0.5 bytes/param |
| KV cache (10 layers) | 0.08 GB | Only full-attention layers (every 4th) |
| Activations (40 layers) | 0.67 GB | hidden=2048, single-token inference |
| Router logits | 0.08 GB | 2048 × 256 experts × f32 |
| **Total** | **18.3 GB** | **5.7 GB headroom on RTX 4090** |

### 4.2.2 Realizar MoE Readiness Assessment

| Component | Status | Location |
|-----------|--------|----------|
| MoE routing (2 strategies) | Exists | `src/moe/mod.rs` |
| Gated DeltaNet linear attention | Exists (GH-278) | `src/gpu/scheduler/types.rs` |
| SwiGLU FFN | Exists | `src/gpu/scheduler/forward_block.rs` |
| GQA attention | Exists | `src/gpu/scheduler/forward_block.rs` |
| Hybrid `layer_types` config | Exists | `types.rs` `is_linear_layer()` |
| Safetensors loading | Exists | `src/safetensors/` |
| **Expert weight struct** | **Missing** | Add `MoeExpertWeights` to `BlockWeights` |
| **Router gate loading** | **Missing** | Load `mlp.gate.weight` [256, 2048] |
| **Expert dispatch** | **Missing** | softmax → top-8 → SwiGLU × 8 → weighted sum |
| **Shared expert** | **Missing** | Always-on SwiGLU, separate gate/up/down |
| **Fused gate_up_proj** | **Missing** | Unfuse [256, 1024, 2048] tensor |

**Estimated new code**: ~300-400 lines in realizar for full MoE inference.

### 4.3 Distillation Architecture

**Primary path**: GPU-resident teacher inference on lambda (RTX 4090). The
35B model at Q4 fits in 18.3 GB VRAM — teacher inference and logit caching
run on the same machine as student training.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  lambda (RTX 4090, 24 GB)                                              │
│                                                                         │
│  Phase 1: Pre-compute teacher logits (GPU, ~18.3 GB)                   │
│  ┌──────────────────────────┐     Parquet shards      ┌──────────────┐ │
│  │ Qwen3.5-35B-A3B (Q4)    │ ──────────────────────► │ teacher_logits│ │
│  │ realizar MoE inference   │    top-k=128 logits     │ ~50-100 GB   │ │
│  │ 18.3 GB VRAM             │                          └──────────────┘ │
│  └──────────────────────────┘                                           │
│                                                                         │
│  Phase 2: Train student (GPU, ~5 GB)                                   │
│  ┌──────────────────────────┐     ┌─────────────────────────────────┐  │
│  │ Student: albor-350M      │ ◄── │ Pre-computed logits + train data │  │
│  │ KD loss + CE loss        │     │ (loaded from disk at GPU speed)  │  │
│  │ entrenar distill         │     └─────────────────────────────────┘  │
│  └──────────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Fallback path**: If GPU VRAM is tight (teacher + student simultaneously),
pre-compute logits on CPU. Intel box (300 GB RAM) can run the 35B model at
Q4 (~18 GB RAM) or Q8 (~35 GB) with ~5-15 tok/s throughput.

### 4.4 Pre-Computed Logits Strategy

Teacher and student do NOT run simultaneously. We **pre-compute teacher
logits offline**, then train the student from cached logits at full GPU speed:

1. Lambda runs Qwen3.5-35B-A3B inference (Q4, GPU) on all training data
2. Teacher top-k logits (k=128) saved as sharded Parquet via `alimentar`
3. Student training loads pre-computed logits from disk — no teacher in VRAM
4. Sequential phases = no VRAM contention

```bash
# Step 0: Plan — check teacher fits, estimate logit disk usage
apr distill plan configs/train/distill.yaml

# Step 1: Pre-compute teacher logits on lambda GPU (Q4, ~18.3 GB)
apr distill apply configs/train/distill.yaml --stage precompute

# Step 2: Train student on lambda using pre-computed logits (~5 GB)
apr distill apply configs/train/distill.yaml --stage train --seed 42
```

**Estimated teacher throughput (Qwen3.5-35B-A3B)**:

| Device | Quantization | VRAM/RAM | Throughput | 500M tokens |
|--------|-------------|----------|------------|-------------|
| RTX 4090 (GPU) | Q4 | 18.3 GB | ~50-100 tok/s | ~1.5-3 days |
| Xeon 48T (CPU) | Q4 | ~18 GB | ~5-15 tok/s | ~10-30 days |
| Xeon 48T (CPU) | Q8 | ~35 GB | ~3-8 tok/s | ~18-48 days |

### 4.5 Distillation Data Budget

| Approach | Teacher Tokens | Time (est.) | Quality |
|----------|---------------|-------------|---------|
| Full corpus (10B tokens) | 10B | ~30-60 days | Best |
| Representative subset (2B) | 2B | ~6-12 days | Good — focus on diverse/hard examples |
| Curated hard examples (500M) | 500M | ~2-3 days | Targeted — highest knowledge density |

**Recommended**: Start with the local ground truth corpora (~50-100M raw tokens)
plus curated hard examples from StarCoder Python (~400M tokens) for ~500M total.
The ground truth corpora should be distilled **first** — they are our highest
quality data and benefit most from teacher knowledge. Scale to 2B with broader
StarCoder data if benchmarks justify the compute. Python-only focus means all
teacher compute goes toward the language we care about.

### 4.6 Fallback Teacher: Qwen2.5-Coder-3B

If ALB-010 (MoE inference in realizar) proves harder than estimated, we fall
back to **Qwen2.5-Coder-3B** as a dense teacher:

| Property | Value |
|----------|-------|
| Model | [Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) |
| Parameters | 3B (dense) |
| Architecture | Qwen2 (standard transformer — already supported by realizar) |
| Compression ratio | 8.6x (3B → 350M) — within recommended 5-20x range |
| CPU inference | ~12 GB RAM, ~2 tok/s on 48 cores |
| License | Apache 2.0 |

**Why this is the fallback, not the primary**:
- Dense 3B has ~10x less knowledge capacity than 35B MoE
- Weaker code capabilities → lower distillation quality ceiling
- Soft targets less informative for the student

**Why it's still viable**:
- Already supported by realizar's Qwen2 architecture loader (no MoE/DeltaNet)
- `apr distill --stage precompute` verified working with 3B teacher (2026-03-03)
- CPU precompute feasible on lambda box (~12 GB RAM)
- 8.6x compression ratio is in the sweet spot for KD

**Config**: `configs/train/distill-qwen3b.yaml` — teacher: Qwen2.5-Coder-3B,
student: albor-base-350m, temperature=4.0, alpha=0.5, LoRA rank 16.

### 4.7 ALB-010 Implementation Status: MoE Inference in Realizar

**Status: MERGED** — Steps 1-5b merged to main ([PR #133](https://github.com/paiml/realizar/pull/133), squash-merged).

**Step 1: Expert weight types + loading** — DONE
- `MoeExpertWeights` struct in `gpu/scheduler/types.rs` (45 files updated)
- Fields: `gate_weight`, `expert_gate_up`, `expert_down`, `shared_{gate,up,down}`
- `GpuModelConfig` extended with `num_experts`, `num_experts_per_tok`, `expert_intermediate_size`

**Step 2: Router forward** — DONE (`moe_dispatch.rs`)
- `moe_route()`: softmax (max-subtracted) → top-k selection → renormalize
- 3 contract-derived tests pass: stability, uniform routing, order preservation

**Step 3: Expert dispatch** — DONE (`moe_dispatch.rs`)
- `expert_swiglu()`: per-expert `down(SiLU(gate(x)) * up(x))`
- `moe_forward_token()`: routes to k experts + shared expert, weighted sum
- 2 contract-derived tests pass: shared expert always active, uniform routing averages

**Step 4: Integration into forward pass** — DONE
- All 5 forward block variants integrated: `forward_block_refcell`, `forward_block_single`,
  `forward_block_incremental`, `forward_block_incremental_optimized`, `forward_block_idx`
- MoE path activates when `block.moe_experts.is_some()`
- Multi-token `forward_block_idx` loops per token (MoE routes independently per token)
- 15,053 total tests pass (0 failures)

**Remaining: Safetensors weight loading**
- Map HuggingFace tensor names (`model.layers.{N}.mlp.experts.*`) to `MoeExpertWeights`
- Fuse individual expert gate/up projections into `expert_gate_up` tensor
- Blocked on: model download (Qwen3.5-35B-A3B, ~70 GB)

### 4.8 Provable Contracts for MoE Inference

Two design-by-contract YAMLs written and validated (`pv validate` PASS) before
implementation begins, per engineering discipline Rule #6:

**`contracts/moe-router-v1.yaml`** (Router forward):
- 4 equations: router_logits, softmax_normalization, topk_selection, weight_renormalization
- 6 invariants: softmax_valid, topk_ordered, renorm_sum_one, expert_count, index_bounds, deterministic
- 5 falsification tests: softmax stability with large logits, top-8 correctness, renorm ordering, zero gate weight, shape mismatch rejection
- 1 Kani harness (stub_float strategy for symbolic f32)

**`contracts/moe-expert-dispatch-v1.yaml`** (Expert dispatch):
- 5 equations: expert_swiglu, routed_output, shared_expert, moe_output, fused_gate_up_unfuse
- 6 invariants: expert_output_shape, weighted_sum_preserves_shape, shared_expert_always_active, expert_independence, unfuse_covers_all, numerical_stability
- 7 falsification tests: single-expert routing, uniform routing, unfuse round-trip, shared expert unconditional, bounds check, finite outputs, dense FFN equivalence
- 2 Kani harnesses (bounded_int strategy)

**Performance characteristics** (from `docs/specifications/training-performance.md` §6.19):
- 28 GEMMs per token per MoE layer (vs 3 for dense FFN)
- Expert GEMMs are tiny ([2048, 512]) — memory-bandwidth bound at batch=1
- Router overhead negligible vs expert computation
- Estimated teacher throughput: 50-100 tok/s on RTX 4090 at Q4

### 4.9 Qwen3.5-35B-A3B Tensor Name Mapping

Architecture class: `Qwen3_5MoeForConditionalGeneration` (model_type: `qwen3_5_moe`).
All layer tensors use `model.language_model.layers.{L}` prefix (multimodal wrapper).

**MoE Expert Tensors** (packed per-layer, not per-expert):

| Tensor Name | Shape | Description |
|------------|-------|-------------|
| `...layers.{L}.mlp.gate.weight` | [256, 2048] | Router: nn.Parameter (not nn.Linear) |
| `...layers.{L}.mlp.experts.gate_up_proj` | [256, 1024, 2048] | All 256 experts' fused gate+up |
| `...layers.{L}.mlp.experts.down_proj` | [256, 2048, 512] | All 256 experts' down projection |
| `...layers.{L}.mlp.shared_expert.gate_proj.weight` | [512, 2048] | Shared expert gate (SwiGLU) |
| `...layers.{L}.mlp.shared_expert.up_proj.weight` | [512, 2048] | Shared expert up |
| `...layers.{L}.mlp.shared_expert.down_proj.weight` | [2048, 512] | Shared expert down |
| `...layers.{L}.mlp.shared_expert_gate.weight` | [1, 2048] | Sigmoid gate scaling shared expert |

**Key architectural detail**: The shared expert output is scaled by
`sigmoid(shared_expert_gate(x))` before adding to the routed expert sum.
This was discovered from the HuggingFace source (`Qwen3_5MoeSparseMoeBlock`)
and added to `MoeExpertWeights.shared_expert_gate_weight` in realizar.

**Expert weights are packed**: Unlike per-expert indexing (`experts.{E}.gate_proj`),
the main model stores all 256 experts in bulk tensors (`experts.gate_up_proj`).
The MTP (multi-token prediction) head uses per-expert indexing. Realizar handles
the packed format directly in `MoeExpertWeights.expert_gate_up`.
