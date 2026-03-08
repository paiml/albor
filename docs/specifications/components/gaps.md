# Gap Register

**Parent**: [albor-llm-spec.md](../albor-llm-spec.md) §10

---

## 1. Summary

**92 gaps filed. 52 fixed.** The gap register tracks every missing capability,
bug, or integration issue discovered during end-to-end LLM development with the
sovereign stack.

The model is the proof; the stack improvements are the lasting value.

---

## 2. Critical Path Gaps (Distillation Blockers)

| ID | Gap | Status | Component | Notes |
|----|-----|--------|-----------|-------|
| ALB-010 | `qwen3_moe` teacher loading (re-scoped) | Steps 4-8 | realizar | **BLOCKER**: Qwen3-Coder-30B-A3B |
| ALB-089 | GPU-accelerated inference for eval | DOGFOODING | realizar | Needed for fast teacher generation |

### 2.1 ALB-010: Qwen3.5-35B-A3B MoE Support

**The single most important gap.** Without this, we cannot use our primary
teacher model.

**Re-scoped**: Teacher changed from Qwen3.5-35B-A3B (`qwen3_5_moe`) to
Qwen3-Coder-30B-A3B-Instruct (`qwen3_moe`). Falsification revealed the
original teacher has no FIM support and isn't code-specialized.

| Step | Description | Status |
|------|-------------|--------|
| 1-3 | MoE routing, dispatch, forward (architecture-agnostic) | MERGED (PR #133) |
| 4* | Config parsing (adapt `qwen3_5_moe` → `qwen3_moe`) | NEEDS UPDATE |
| 5* | Tests (re-validate for 128 experts, no shared) | NEEDS UPDATE |
| 6 | Download model + APR import + tensor→slot mapping | **TODO** |
| 7 | Q4K quantization via `apr quantize` | TODO |
| 8 | End-to-end generation dogfood | BLOCKED on 6-7 |

**Architecture differences** (`qwen3_moe` vs `qwen3_5_moe`):
- 128 experts (not 256), no shared expert
- Standard GQA (no Gated DeltaNet, no linear attention layers)
- `model.layers.{L}` tensor prefix (no `language_model` nesting)
- 48 layers (not 40), h=2048 (not 2048), vocab 151936 (not 248320)

**Simpler**: No DeltaNet attention, no shared expert routing. Core MoE
dispatch from Steps 1-3 transfers directly.

**Fallback**: Qwen2.5-Coder-3B (dense, 84.1% HumanEval, already supported).

---

## 3. Fixed Gaps (Selected — Training-Critical)

### 3.1 ALB-092: Norm Grads + Accum Init + Resume (FIXED)

Three bugs found together:
1. **RMSNorm backward**: `grad_gamma_ptr` declared but never written. 50 norm
   weights got zero gradient (only weight decay). Fix: atomicAdd in Pass 2
   (trueno PR #178).
2. **Gradient accumulator uninitialized**: `GpuGradientAccumulator::new()` used
   `GpuBuffer::new()` (cuMemAlloc = uninitialized VRAM). Fix: `accum.zero_all()`
   after construction (entrenar PR #257).
3. **Step counter reset on resume**: CUDA trainer didn't restore step on
   checkpoint load → LR schedule + AdamW bias correction reset to step 0.
   Fix: `set_initial_step()` (entrenar PR #257).

### 3.2 ALB-079: Cosine LR Not Consumed (FIXED)

`lr_scheduler: "cosine"` parsed from YAML but never used in training loop.
Training ran with constant LR. Fix: wire cosine decay into CUDA trainer.

### 3.3 ALB-080: Batch Size Too Small (FIXED)

4K tokens/step (batch=4, ga=1) vs 131K+ recommended for 350M params.
Gradient noise too high for stable convergence. Fix: ga=8 → 32K tokens/step.

### 3.4 ALB-077: cuBLAS Tensor Core NaN (FIXED)

`CUBLAS_GEMM_DEFAULT_TENSOR_OP` produces NaN for transposed GEMMs at high
magnitude. Fix: use `CUBLAS_DEFAULT_MATH` (SIMD, no tensor cores). Tensor cores
available via TF32 for compatible operations (trueno-gpu@0.4.26).

### 3.5 ALB-075: cuBLAS Integration (COMPLETE)

Hand-written PTX GEMMs replaced with cuBLAS FFI. 4 phases completed:
1. FFI bindings (`cublas_sys.rs`)
2. Safe Rust wrapper (`cublas.rs`)
3. Forward + backward GEMMs
4. Strided batched attention GEMMs

Result: 6.7K tok/s steady state, 19.3% MFU.

### 3.6 ALB-065: Stream Sync Race (FIXED)

`cuMemcpyDtoH` without stream sync → read stale GPU data → garbage clip scale
→ NaN → SIGABRT. Fix: explicit `stream.synchronize()` before every D2H.
Stable with `CUDA_LAUNCH_BLOCKING=1` but crashed without it.

### 3.7 ALB-044: Activation Grad Overflow (FIXED)

24-layer backward amplified activation gradients to ~1e35. CPU AdamW's second
moment overflowed f32 → 1298 NaN in 33.5M embedding table. Fix: clip activation
gradient norm before CPU optimizer.

### 3.8 ALB-043: FFN Buffer Overflow (FIXED)

`gate_out` buffer (intermediate_size=4096) reused for `grad_hidden`
(hidden_size=1024). 4× overflow into adjacent GPU allocations. Silent memory
corruption → wrong gradients. Fix: dedicated buffer per purpose.

### 3.9 ALB-038: Missing Norm Backward (FIXED)

`RMSNorm::forward_batched()` had no backward op. Blocked ALL gradient flow
through the network. Training appeared to work (loss decreased via embedding
gradients only) but produced garbage weights. Fix: implement batched backward.

---

## 4. Infrastructure Gaps

| ID | Gap | Status | Notes |
|----|-----|--------|-------|
| ALB-086 | SafeTensors 2D shapes | FIXED | entrenar PR #255 |
| ALB-087 | Auto eval scheduling | FIXED | entrenar PR #254 |
| ALB-088 | Multi-sample pass@k | FIXED | aprender PR #432 |
| ALB-091 | GPU-resident grad accum | FIXED | entrenar (CudaGradWorkspace) |

---

## 5. Open Gaps (Non-Blocking)

| ID | Gap | Priority | Notes |
|----|-----|----------|-------|
| ALB-002 | Model merging | Low | Post-distillation |
| ALB-003 | LoRA finetuning | Low | Post-distillation |
| ALB-004 | Structured pruning | Low | Post-distillation |
| ALB-005 | 4-bit quantization | Medium | Needed for M6 |
| ALB-008 | WASM inference | Low | Demo deployment |
| ALB-024 | FIM tokenization | Medium | Post-distillation |
| ALB-025 | Streaming data loader | Medium | Performance (39 GB swap fix) |
| ALB-026 | Cross-shard shuffle | Low | Data quality |
| ALB-031 | Checkpoint resume fix | High | `with_model()` broken |
| ALB-036 | Multi-GPU DDP | Low | Only have 1 GPU |
| ALB-042 | Mixed precision (fp16/bf16) | Medium | 2× throughput potential |

---

## 6. Gap Lifecycle

```
Discovery → File GitHub issue → Implement upstream → Wire into apr
    → Dogfood in albor → pv/pmat verify → Close gap
```

Every gap gets:
1. **GitHub issue** with reproduction steps
2. **Contract** in `contracts/` if safety-critical
3. **Fix** in the upstream repo (entrenar, trueno, realizar, etc.)
4. **Verification** via dogfood training run
5. **Bug pattern** added to `bug-patterns.md` for prevention

---

## 7. Bug Pattern Cross-Reference

Common failure patterns discovered during dogfooding. Full catalog in
project memory (`bug-patterns.md`). Key patterns:

| Pattern | Gaps | Prevention |
|---------|------|------------|
| Buffer size mismatch | ALB-041, -043 | Algebraic verification at kernel boundary |
| Missing backward op | ALB-038 | Verify gradient flow after 1 step |
| Hyperparameter defaults | ALB-044 | Explicit constructor, never default_params() |
| Stream sync race | ALB-065 | stream.synchronize() before every D2H |
| Uninitialized GPU buffers | ALB-059, -092 | from_host() or zero_all() |
| Config parsed but unused | ALB-068, -079 | Grep for field usage in actual loop |
| PTX arg order | ALB-059, -069 | Verify against constructor signature |
