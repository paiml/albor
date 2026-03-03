# 12. Provable Quality & Design by Contract

Every computational kernel used in Albor must have a provable-contracts YAML
specification with Popperian falsification tests, property-based probar tests,
and Kani bounded model checking harnesses. This is not optional — it is a
first-class deliverable alongside the model.

## 12.1 Verification Ladder

Four levels of assurance, from cheapest to most rigorous:

```
Level 4: Kani bounded model check    ─── PROOF (exhaustive for inputs ≤ N)
Level 3: probar property tests       ─── HIGH CONFIDENCE (10,000+ random inputs)
Level 2: Falsification tests         ─── TARGETED (specific edge cases)
Level 1: Type system                 ─── BY CONSTRUCTION (Rust compiler)
Level 0: Code review                 ─── HUMAN (necessary but insufficient)
```

**Requirement**: Every kernel reaches at least Level 3. Critical kernels
(softmax, attention, cross-entropy, KD loss) reach Level 4.

## 12.2 Contract Registry for Albor

Albor requires contracts for every kernel in the training + post-training pipeline.
Many already exist in provable-contracts; new ones must be written.

### Existing Contracts (bind to aprender implementations)

| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| `softmax-kernel-v1.yaml` | softmax | 6 (normalization, positivity, monotonicity, SIMD parity, translation invariance, bound) | Exists, 289 bindings |
| `rmsnorm-kernel-v1.yaml` | RMSNorm | 5 (finiteness, scale invariance, SIMD parity, idempotency) | Exists |
| `attention-kernel-v1.yaml` | scaled dot-product attention | Multiple (causal mask, score bounds, gradient flow) | Exists |
| `rope-kernel-v1.yaml` | Rotary Position Embedding | Multiple (rotation invariant, frequency spectrum) | Exists |
| `gelu-kernel-v1.yaml` | GELU activation | Bound, monotonicity, SIMD parity | Exists |
| `matmul-kernel-v1.yaml` | matrix multiplication | Associativity, SIMD parity, bound | Exists |
| `cross-entropy-kernel-v1.yaml` | cross-entropy loss | Non-negativity, gradient correctness | Exists |
| `adamw-kernel-v1.yaml` | AdamW optimizer | Bias correction, weight decay decoupling | Exists |
| `gqa-kernel-v1.yaml` | Grouped Query Attention | Equivalence to MHA when groups=heads | Exists |
| `swiglu-kernel-v1.yaml` | SwiGLU FFN | Gating invariants | Exists |

### New Contracts Required for Albor (ALB-013 through ALB-017)

| Contract (NEW) | Key Equations | Key Obligations | Priority |
|----------------|---------------|-----------------|----------|
| `knowledge-distillation-kernel-v1.yaml` | KD_loss = α·KL(σ(z_t/T) ∥ σ(z_s/T))·T² + (1-α)·CE(y, z_s) | KL non-negativity, temperature scaling invariant, gradient correctness, α interpolation bound | Critical |
| `bpe-tokenizer-kernel-v1.yaml` | BPE merge rules, byte-pair encoding | Roundtrip invariant: decode(encode(x)) = x, vocab coverage, merge ordering | High |
| `model-merging-kernel-v1.yaml` | SLERP: interp(θ, w₁, w₂) on unit sphere; TIES: trim + elect + disjoint merge | SLERP interpolation bound (‖result‖ ≈ 1), TIES sparsity guarantee | Medium |
| `pruning-kernel-v1.yaml` | WANDA: score = |w| · ‖x‖₂; magnitude: score = |w| | Sparsity invariant (exactly k% weights zeroed), score ordering preserved | Medium |
| `gradient-accumulation-kernel-v1.yaml` | G_accum = (1/N)·Σ g_i ≈ g_full | Numerical equivalence within tolerance, loss scaling correctness | High |
| `training-config-kernel-v1.yaml` | steps_per_epoch, total_achievable_steps, LR warmup coverage, Chinchilla tokens | Epoch sufficiency for max_steps, warmup completion, peak LR reached, data sufficiency | Critical |

## 12.3 Contract Workflow for Each Kernel

```bash
# 1. Write or validate YAML contract
pv validate contracts/knowledge-distillation-kernel-v1.yaml

# 2. Generate trait stubs + failing tests
pv scaffold contracts/knowledge-distillation-kernel-v1.yaml

# 3. Generate property-based tests (wired to actual aprender code)
pv probar contracts/knowledge-distillation-kernel-v1.yaml \
  --binding contracts/aprender/binding.yaml

# 4. Generate Kani bounded model checking harnesses
pv kani contracts/knowledge-distillation-kernel-v1.yaml

# 5. Run falsification sweep
pv audit contracts/knowledge-distillation-kernel-v1.yaml \
  --binding contracts/aprender/binding.yaml

# 6. Verify full contract status
pv status contracts/knowledge-distillation-kernel-v1.yaml
```

## 12.4 Falsification Tests: Albor-Specific

Every claim in this specification must be falsifiable. Below are the concrete
falsification tests for Albor's key properties.

### Training Correctness

```yaml
# FALSIFY-ALBOR-001: Loss decreases monotonically (smoothed)
- id: FALSIFY-ALBOR-001
  rule: "Training convergence"
  prediction: "EMA(loss, window=100) is monotonically decreasing after warmup"
  test: "Load training log, compute EMA, assert no sustained increase >5% over 500 steps"
  if_fails: "Learning rate too high, data corruption, or gradient computation bug"

# FALSIFY-ALBOR-002: Gradient norms are bounded
- id: FALSIFY-ALBOR-002
  rule: "Training stability"
  prediction: "Global gradient norm < 10.0 after clipping for all steps"
  test: "Parse training log, assert max gradient norm across all steps"
  if_fails: "Gradient clipping not applied, loss spike, or NaN propagation"

# FALSIFY-ALBOR-003: Checkpoint determinism
- id: FALSIFY-ALBOR-003
  rule: "Reproducibility"
  prediction: "Two runs with seed=42 produce identical checkpoints at step 1000"
  test: "Train twice, BLAKE3 hash both checkpoints, assert equality"
  if_fails: "Non-deterministic operation (async GPU, HashMap ordering, etc.)"
```

### Distillation Correctness

```yaml
# FALSIFY-ALBOR-004: KL divergence is non-negative
- id: FALSIFY-ALBOR-004
  rule: "KD loss validity"
  prediction: "KL(teacher || student) >= 0 for all batches"
  test: "proptest with 10000 random logit pairs, assert KL >= -1e-7"
  if_fails: "Log-domain computation error or softmax numerical instability"

# FALSIFY-ALBOR-005: Distillation improves over base
- id: FALSIFY-ALBOR-005
  rule: "Distillation value"
  prediction: "albor-distill avg benchmark > albor-base avg benchmark"
  test: "Run full eval suite on both, paired t-test with p < 0.05"
  if_fails: "Teacher logits corrupted, temperature too high/low, or alpha miscalibrated"

# FALSIFY-ALBOR-006: Teacher logit integrity
- id: FALSIFY-ALBOR-006
  rule: "Data pipeline integrity"
  prediction: "Pre-computed teacher logits match live teacher inference within 1e-4"
  test: "Sample 100 batches, run live teacher inference, compare against stored logits"
  if_fails: "Serialization precision loss, wrong batch ordering, or teacher model mismatch"
```

### Post-Training Invariants

```yaml
# FALSIFY-ALBOR-007: Merge interpolation bound
- id: FALSIFY-ALBOR-007
  rule: "SLERP correctness"
  prediction: "‖SLERP(w1, w2, t)‖ ≈ ‖w1‖ for t ∈ [0,1] (unit sphere)"
  test: "proptest with 10000 random weight pairs and t values"
  if_fails: "SLERP implementation uses LERP instead, or normalization missing"

# FALSIFY-ALBOR-008: Pruning sparsity guarantee
- id: FALSIFY-ALBOR-008
  rule: "WANDA correctness"
  prediction: "Exactly 50% of weights are zero after prune --sparsity 0.5"
  test: "Count zero weights, assert within ±0.1% of target sparsity"
  if_fails: "Pruning threshold computation error or layer exclusion bug"

# FALSIFY-ALBOR-009: Quantization round-trip
- id: FALSIFY-ALBOR-009
  rule: "Q4 fidelity"
  prediction: "Perplexity(Q4 model) < 1.05 × Perplexity(fp16 model)"
  test: "Evaluate both on held-out set, assert ratio < 1.05"
  if_fails: "Quantization calibration data insufficient or block size wrong"
```

## 12.5 Brick Profiling Architecture

Training a 350M model on a single 4090 is a systems engineering problem, not
a scaling problem. Every watt of GPU silicon must be accounted for. The
architecture achieves this by treating each component as a **brick** — a
self-contained unit with measurable inputs, outputs, and a provable contract.

### 12.5.1 Three Granularities of Profiling

**Per-kernel.** Every CUDA kernel (`gemm_forward`, `silu_backward`,
`rms_norm_forward`, `batched_transpose_forward`, etc.) is individually
measurable via `compute-sanitizer`, `nsys`, or `nvprof`. When a kernel
misbehaves, the brick boundary isolates the failure to a single function with
known input/output shapes. The contract for each kernel specifies buffer size
invariants that can be checked statically.

**Per-block.** `CudaTransformerBlock` encapsulates one transformer layer's
forward, backward, and optimizer step as a single GPU-resident unit. Diagnostic
sampling after backward (downloading 1K elements from each gradient buffer)
immediately distinguishes "math is wrong" (NaN in gradients) from "math is
right but magnitudes are wrong" (gradient explosion). The brick boundary
separates kernel correctness from training dynamics.

**Per-transfer.** The 3-transfer-per-step contract (`C-GPUTRAIN-002`) fixes
the PCIe budget:

```
Transfer 1 (H2D): embedding hidden states   ~S×H×4 bytes
Transfer 2 (D2H): logits for cross-entropy  ~S×V×4 bytes
Transfer 3 (H2D): grad_logits to GPU        ~S×V×4 bytes
```

Any deviation from 3 transfers is a bug, not a tuning knob. For 350M at
seq=2048: total ~544 MB/step, overhead ~17 ms on PCIe 4.0 x16 — under 5%
of compute time.

### 12.5.2 Chain of Thought: How Brick Boundaries Diagnose Bugs

When a training run fails, the brick architecture converts "something is
broken" into a structured diagnosis:

1. **Which granularity?** Check per-transfer (D2D size mismatch?), per-block
   (which layer's backward fails?), per-kernel (which GEMM overflows?).
2. **Local or global?** If one block fails and others succeed, the bug is in
   that block's kernels. If all blocks succeed but loss diverges, the bug is in
   training dynamics (LR, grad clipping, optimizer config).
3. **Static or dynamic?** Buffer overflow is a static invariant violation
   (detectable by algebraic dimension checking). Gradient explosion is a
   dynamic stability issue (detectable by runtime sampling).

### 12.5.3 Five Whys: From Symptom to Root Cause

The brick architecture enforces a disciplined root-cause chain. Concrete
example from dogfooding:

| Why | Finding | Brick boundary |
|-----|---------|----------------|
| **Why does 350M training produce NaN at step 2?** | Gradients reach 1e35, AdamW produces NaN weights | Per-block sampling: `grad_gate max=3.28e35` |
| **Why are gradients 1e35?** | 24-layer backward amplifies without clipping | Per-transfer: config has `grad_clip: 1.0` but CUDA path ignores it |
| **Why no gradient clipping in CUDA path?** | `CudaTransformerTrainer` copied from finetuning (pre-trained weights, small grads) | Brick mismatch: finetuning brick assumed well-conditioned weights |
| **Why wasn't this caught by the GPU training contract?** | Contract validates kernel correctness + transfer count, not training stability | Contract gap: no `C-TRAINSTABLE-001` obligation |
| **Why doesn't the contract cover stability?** | Contracts target kernel-level (local) correctness, not loop-level (global) dynamics | **Action**: add training-stability contract bridging kernel and loop levels |

This same pattern resolved four bugs during ALB-040 dogfooding:

| Bug | Profiling diagnosis | Contract that prevents recurrence |
|-----|--------------------|------------------------------------|
| ALB-043: `silu_backward` writes `[S,I]` into `[S,H]` buffer (4x overflow) | `compute-sanitizer` pinpoints illegal address in `silu_backward` | Buffer size invariant: output must be `[S, intermediate_size]` |
| ALB-041: D2D copy size mismatch in `backward_attention` | Error logged at exact block index; `gate_out` used as `grad_hidden` temp | D2D invariant: `src.len() == dst.len()` for `copy_from_buffer_async` |
| `backward_attention`: transpose `attn_scores [H,S,S]` into `attn_kv_temp2 [H,S,hd]` | Algebraic trace: `16×512×512 = 4.2M` into 524K buffer = 8x overflow | Transpose output buffer invariant: `output.len() >= batch × rows × cols` |
| `gpu_forward`: D2D copy fails when `seq_len < max_seq_len` | All forwards return None; traced to `PAR-023` size mismatch | Forward buffer invariant: input/output buffers at `max_seq_len` size |
| ALB-044: Unclipped activation gradient (~1e35) overflows CPU AdamW | Per-boundary sampling: embed weights have 1298 NaN after optimizer step | C-EMBED-GRAD-001: clip activation gradient at GPU→CPU boundary |
| ALB-044: CPU AdamW beta2=0.999 vs YAML beta2=0.95 (50x amplification) | Traced bias correction: v_hat = v/0.001 with beta2=0.999 vs v/0.05 with 0.95 | C-HYPERPARAMS-001: all optimizer fields must match YAML config |
| ALB-059: GEMM backward constructor args n/k swapped — output stride 64× too large | Per-kernel: v_w_k[block0] corrupted during `gemm_backward_a(LM head)`. Pointer analysis: 3 contiguous 256KB allocs. Stride 32768 writes rows into m_w_k/v_w_k. | C-GEMMARGS-001: kernel constructor args must match documented parameter order |
| ALB-059: Uninitialized optimizer m/v buffers (cuMemAlloc returns garbage) | Per-block: v_w_k nonzero before any backward op (not from overflow). `GpuBuffer::new()` ≠ zero-init. | C-GPUINIT-001: all optimizer state buffers must be zero-initialized |
| ALB-065: Missing stream.synchronize() before D2H gradient transfers | Per-transfer: cuMemcpyDtoH reads stale GPU buffers. Process stable with CUDA_LAUNCH_BLOCKING=1, crashes within 15s without it. Five Whys: trueno uses CU_STREAM_NON_BLOCKING; cuMemcpyDtoH doesn't sync with non-blocking streams. | C-STREAMSYNC-001: stream.synchronize() before every D2H transfer reading kernel output |

### 12.5.4 How Bricks and Contracts Interlock

The gap register (§11) is the feedback loop between profiling and contracts:

```
Brick profiling finds anomaly
  → File gap (ALB-0XX)
    → Write or update contract obligation
      → Fix upstream brick
        → Verify contract passes (`pv audit`)
          → Dogfood in albor pipeline
            → Close gap
```

Profiling finds bugs that contracts miss (runtime-only issues like gradient
explosion). Contracts prevent bugs that profiling misses (the 50M model's 2x
buffer overflow "worked" through undefined behavior — only a static size
invariant would have caught it). Together they form a ratchet: every bug found
by profiling becomes a permanent contract obligation that prevents recurrence.

## 12.6 Verification DAG (Albor End-to-End)

Like the Qwen 3.5 verification DAG in provable-contracts, Albor composes
sub-contracts into a full model verification:

```
softmax ← attention ← gqa
                        ↑
rmsnorm ──────────────── albor-forward ← training-loop
                        ↑                      ↑
gelu ← swiglu ──────────┘                     │
                                               │
rope ──────────────────── albor-forward        │
                                               │
matmul ← gqa                                   │
                                               │
cross-entropy ─────────── training-loss ────────┘
                              ↑
adamw ─────────── optimizer-step ──────── training-loop
                                               │
gradient-accumulation ─────────────────────────┘
                                               │
training-config ─── config-validation ─────────┘
                                               │
knowledge-distillation ── distill-loss ── distill-loop
                              ↑
bpe-tokenizer ─── data-pipeline ─── training-loop

model-merging ─── post-training ─── albor-merged
pruning ────────── post-training ─── albor-pruned
```

Each node in this DAG is a contract. `pv graph contracts/ --format mermaid`
renders the full dependency graph. A change to any sub-contract triggers
re-verification of all dependents.

## 12.7 Training Stability Contracts

The kernel-level contracts in §12.2 verify *local* correctness — each kernel
produces the right output for its input. They do NOT verify *global* training
stability — that the training loop converges without NaN, that hyperparameters
propagate correctly, or that gradients flow to all parameters.

ALB-038, ALB-041, ALB-043, and ALB-044 all passed kernel-level contracts
while producing training failures. These contracts bridge the gap between
kernel correctness and training correctness.

### C-TRAINSTABLE-001: Training Stability

All weights and loss must remain finite for the entire training run.

```yaml
obligations:
  - "loss.is_finite() for all steps"
  - "weight[i].is_finite() for all i, all steps"
  - "grad[i].is_finite() for all i after clipping, all steps"
falsification: |
  FALSIFY-STABLE-001: Train 100 steps on random init.
  Assert loss.is_finite() at every step.
  Assert no NaN in any model weight after every optimizer step.
```

### C-EMBED-GRAD-001: Activation Gradient Clipping at GPU-CPU Boundary

When GPU backward produces activation gradients that flow to a CPU optimizer,
those gradients must be clipped to `max_grad_norm` before the CPU processes
them.

**Status: VERIFIED** — 350M CUDA test (50 steps) produces zero NaN in embedding
weights. Fix in `entrenar@86eec38`.

```yaml
motivation: |
  Per-block gradient clipping in CudaGradWorkspace only clips WEIGHT gradients.
  The ACTIVATION gradient in grad_buf_a/b flows unclipped to the CPU embedding
  optimizer. For 24-layer random init, this gradient reaches ~1e35 — overflowing
  the CPU AdamW second moment buffer.
obligation: |
  Before scatter-adding activation gradients into CPU embedding weight gradient:
    grad_norm = L2_norm(activation_grad)
    if grad_norm > max_grad_norm:
        activation_grad *= max_grad_norm / grad_norm
falsification: |
  FALSIFY-EMBEDGRAD-001: Train 350M model (24 layers) for 5 steps.
  Assert embedding weights contain zero NaN values after each optimizer step.
```

### C-HYPERPARAMS-001: Optimizer Hyperparameter Propagation

Every optimizer hyperparameter in the YAML config must reach the actual
optimizer constructor. No implicit defaults.

**Status: VERIFIED** — 350M CUDA test uses explicit `AdamW::new()` with
YAML config values (beta2=0.95, wd=0.1). Fix in `entrenar@86eec38`.

```yaml
obligation: |
  For every optimizer in the training loop (GPU AdamW, CPU AdamW, LM head AdamW):
    assert optimizer.lr == config.lr (adjusted for warmup)
    assert optimizer.beta1 == config.beta1
    assert optimizer.beta2 == config.beta2
    assert optimizer.weight_decay == config.weight_decay
    assert optimizer.epsilon == 1e-8 (or config.epsilon if specified)
falsification: |
  FALSIFY-HYPERPARAMS-001: Construct CudaTransformerTrainer with non-default
  YAML config (beta2=0.95, wd=0.1). Verify CPU embed_optimizer.beta2 == 0.95
  and embed_optimizer.weight_decay == 0.1 (not 0.999 and 0.01).
anti_pattern: |
  NEVER: AdamW::default_params(lr)  — hides beta2, wd, epsilon
  ALWAYS: AdamW::new(lr, beta1, beta2, epsilon, wd)  — explicit from config
```

### C-BUFSIZE-001: CUDA Kernel Buffer Size Invariants

Every GPU buffer passed to a CUDA kernel must have algebraically verifiable
size matching the kernel's expected dimensions.

```yaml
obligation: |
  For gemm_forward(A, B, C, M, K, N):
    assert A.len() >= M * K
    assert B.len() >= K * N
    assert C.len() >= M * N
  For silu_backward(input, grad_output, output):
    assert output.len() >= input.len()
  For rms_norm_backward(input, weight, grad_output, grad_input, grad_weight, S, H):
    assert grad_input.len() >= S * H
    assert grad_weight.len() >= H
falsification: |
  FALSIFY-BUFSIZE-001: Run compute-sanitizer on 10-step 50M training.
  Assert zero illegal address errors.
anti_pattern: |
  NEVER: Reuse a buffer sized for hidden_size as temp for intermediate_size
  ALWAYS: Use dedicated buffers or verify size >= required before kernel call
```

### C-GEMMARGS-001: GEMM Kernel Constructor Argument Ordering

Every GEMM kernel constructor call must pass arguments in the exact order
documented by the kernel's API. Compile-time stride constants baked into PTX
are determined by constructor args — wrong order produces wrong strides, not
wrong results at the kernel boundary (bounds check passes but data lands in
wrong memory).

**Status: VERIFIED** — 350M CUDA test (50 steps) produces correct backward
gradients. Fix in `entrenar@846ae0c`.

```yaml
motivation: |
  GemmBackwardAKernel::tiled_unrolled(m, n, k, tile_size) bakes self.n and
  self.k as immediate PTX constants for row/col strides. When called as
  tiled_unrolled(m, k, n, tile) with k and n swapped, the output stride
  becomes vocab_size (32768) instead of hidden_size (512) — writing output
  rows 64× too far apart and overflowing into adjacent GPU allocations.
obligation: |
  For every kernel constructor call:
    assert arg_order matches constructor signature exactly
  Specifically for GEMM backward:
    GemmBackwardAKernel::tiled_unrolled(m, n, k, tile)  # NOT (m, k, n, tile)
    GemmBackwardBKernel::tiled_unrolled(m, n, k, tile)  # NOT (m, k, n, tile)
falsification: |
  FALSIFY-GEMMARGS-001: Train 350M model for 5 steps. Download v_w_k[block0]
  after backward. Assert zero corruption (all values ≥ 0 after optimizer init,
  no values from adjacent buffers).
anti_pattern: |
  NEVER: Guess argument order from variable names (m/n/k are ambiguous)
  ALWAYS: Check constructor signature in trueno-gpu kernel source
```

### C-GPUINIT-001: GPU Buffer Zero Initialization

All optimizer state buffers (m and v for AdamW) must be zero-initialized.
`GpuBuffer::new()` uses `cuMemAlloc` which returns uninitialized VRAM —
the contents are whatever was previously in that memory region.

**Status: VERIFIED** — All 34 optimizer buffers (18 per-block + 12 LoRA + 4 LM head/norm)
zero-initialized via `GpuBuffer::from_host(&ctx, &vec![0.0f32; n])`. Fix in `entrenar@846ae0c`.

```yaml
obligation: |
  For every GpuBuffer used as optimizer state (m, v):
    assert buffer is zero-initialized after allocation
    Use GpuBuffer::from_host(&ctx, &vec![0.0f32; n])
    NOT GpuBuffer::new(&ctx, n)  -- returns uninitialized VRAM
falsification: |
  FALSIFY-GPUINIT-001: Allocate optimizer state, download immediately.
  Assert all values == 0.0.
```

### C-GRADFLOW-001: Gradient Flow Verification

Every trainable parameter must receive a non-zero gradient after one
forward+backward step on a non-trivial batch.

```yaml
obligation: |
  After one forward+backward step on a batch with non-constant inputs:
    for param in model.trainable_parameters():
      assert param.grad().abs().max() > 0
falsification: |
  FALSIFY-GRADFLOW-001: Train 1 step on 50M model with random init.
  Verify all 110 parameter tensors have max(|grad|) > 0.
anti_pattern: |
  NEVER: Create tensors with requires_grad=false in the forward path
  NEVER: Use ops that don't register backward (e.g., manual array copies)
  ALWAYS: Verify gradient flow when adding new layers or ops
```

### C-TRAINCFG-001: Training Configuration Algebraic Consistency

Every training configuration must be algebraically validated BEFORE GPU time is
consumed. The epoch/step/data/LR relationship must be provably sufficient.

**Status: OPEN** — ALB-060. The 350M training ran only 43/5000 steps because
`epochs: 1` exhausted data before `max_steps`. Contract written, config fixed
(`epochs: 117`), awaiting re-training verification.

```yaml
motivation: |
  ALB-060: pretrain-350m.yaml had epochs=1 with 22K sequences and grad_accum=128.
  steps_per_epoch = floor(22079 / 4 / 128) = 43. max_steps=5000 unreachable.
  warmup_steps=2000 never completed. LR peaked at 6.45e-6 (target 3e-4).
  Loss flat at ~10.39 for all 43 steps. Checkpoint contains untrained weights.
  Total wasted: ~12 seconds GPU + debugging time. Contract prevents recurrence.
equations:
  - "steps_per_epoch = floor(num_sequences / batch_size / grad_accum)"
  - "total_achievable_steps = num_epochs × steps_per_epoch"
  - "total_achievable_steps >= max_steps  (HARD REQUIREMENT)"
  - "warmup_steps < total_achievable_steps  (warmup must complete)"
  - "warmup_fraction = warmup_steps / actual_total_steps <= 0.10"
  - "min_epochs = ceil(max_steps / steps_per_epoch)"
  - "total_tokens = actual_steps × batch_size × grad_accum × seq_len"
obligations:
  - "Epoch count sufficient: num_epochs >= ceil(max_steps / steps_per_epoch)"
  - "Warmup completes: warmup_steps < actual_total_steps"
  - "Peak LR reached: exists step t where lr(t) = lr_peak"
  - "Training tokens sufficient: total_tokens >= 10 × num_params"
falsification: |
  FALSIFY-CFG-001: Compute steps_per_epoch for pretrain-350m.yaml.
  With 22079 seqs, batch=4, accum=128: steps_per_epoch=43.
  Assert 1 × 43 < 5000 (proves epochs=1 is insufficient).
  FALSIFY-CFG-002: Assert warmup_steps (2000) > total_steps (43)
  (proves warmup never completes with epochs=1).
```

Full contract: `contracts/training-config-kernel-v1.yaml` — 7 equations,
8 proof obligations, 5 falsification tests, 2 Kani harnesses.

### C-STREAMSYNC-001: Stream Synchronization Before D2H Transfers

Every `cuMemcpyDtoH` (or `copy_to_host_at()`) call that reads data written by
GPU kernels on a non-default stream MUST be preceded by `stream.synchronize()`.

```yaml
motivation: |
  ALB-065: gradient clipping downloaded 9 GPU buffers via cuMemcpyDtoH
  without stream synchronization. trueno CudaStream uses CU_STREAM_NON_BLOCKING;
  cuMemcpyDtoH only synchronizes with the default stream. Backward kernels
  hadn't finished → garbage clip scale → NaN → silent SIGABRT (process death
  with no error output). Training was stable with CUDA_LAUNCH_BLOCKING=1 but
  crashed within 15 seconds without it.
obligation: |
  stream.synchronize() MUST precede every cuMemcpyDtoH that reads kernel output.
  No exceptions. The sync ensures all prior kernel launches have completed.
falsification: |
  FALSIFY-GPU-008: Run 350M training for 50+ steps WITHOUT CUDA_LAUNCH_BLOCKING=1.
  Verify process stays alive, loss is finite, no CUDA errors in dmesg/Xid log.
anti_pattern: |
  NEVER: call copy_to_host_at() after kernel launches without stream.synchronize()
  NEVER: rely on cuMemcpyDtoH to synchronize non-blocking streams (it doesn't)
  DIAGNOSTIC: if training crashes without CUDA_LAUNCH_BLOCKING=1 but works with it,
  this is the FIRST contract to check
```

Full contract: `contracts/training-gpu-kernel-v1.yaml` — stream_synchronization
equation + proof obligation.

### 12.7.1 Observability Discipline

**All training observability MUST use the renacer tracing infrastructure.**

entrenar integrates renacer in `src/run.rs` (span lifecycle: `create_span`,
`emit_metric_event`, `end_span`). The `src/monitor/drift.rs` module provides
anomaly detection (`DriftStatus`, `AnomalySeverity`) that can automatically
flag NaN, gradient explosion, and loss divergence.

```yaml
obligation: |
  NEVER: eprintln!(), println!(), dbg!() for training diagnostics
  ALWAYS: tracing::debug!(), tracing::warn!() with structured fields
  ALWAYS: emit_metric_event() for training metrics (loss, grad_norm, lr)
motivation: |
  Ad-hoc eprintln! creates cleanup debt, is invisible to tracing infra,
  loses brick profiling boundary isolation, and cannot be filtered at runtime.
  renacer BrickTracer provides structured, filterable, permanent observability.
```

---
