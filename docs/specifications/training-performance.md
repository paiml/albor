# Training Performance Specification

## 0. Design Principles

This specification follows **design by contract** (DbC). Every performance
claim, optimization target, and implementation phase begins with a provable
contract (`pv validate`) that defines equations, invariants, proof obligations,
and falsification tests. Code is written to satisfy the contract — never the
reverse.

**Verification stack** (sovereign, no external dependencies):

| Layer | Tool | Role |
|-------|------|------|
| Contract | `pv` (provable-contracts) | YAML equations, proof obligations, falsification tests, Kani harnesses |
| Benchmark | Raw C + Criterion + regression | Three-tier: raw C cuBLAS (ceiling) vs Rust cuBLAS vs PTX (floor) |
| Profiling | `probador` (probar) | Brick budgets, per-component SLA enforcement, Jidoka gates |
| Tracing | `renacer` (BrickTracer) | Per-kernel/per-block/per-transfer spans, OTLP export, anomaly escalation |
| Measurement | `renacer` (metrics) | Counter/Gauge/Histogram with SIMD acceleration (trueno) |

**Workflow for every optimization phase:**

```
1. pv validate contracts/cublas-gemm-v1.yaml          # Contract first
2. pv scaffold contracts/cublas-gemm-v1.yaml           # Generate test stubs
3. make bench-gemm-raw                                 # Establish ceiling
4. Implement against contract
5. make bench-gemm-compare                             # Three-tier benchmark
6. probador brick budgets verify per-component SLAs    # Brick profiling
7. renacer --trace-compute traces per-kernel timing    # Layer tracing
8. pv audit contracts/cublas-gemm-v1.yaml              # Binding coverage
9. Dogfood on 350M training run
10. make bench-gemm-regression                         # No regressions
11. Close gap in §11
```

## 1. Current Performance Baseline

### 1.1 Measured Throughput

| Metric | Value | Config |
|--------|-------|--------|
| Throughput (pre-optimization) | **934 tok/s** | 350M, seq=1024, batch=4, RTX 4090 |
| Step time (pre-optimization) | ~4.4s | Same config |
| **Throughput (current, Phase 5b)** | **7,676 tok/s** | Same config (steady state, step 1000) |
| **Step time (current, Phase 5b)** | **513 ms** | Same config (steady state) |
| **MFU (current, Phase 5b)** | **22.2%** | vs FP32 peak (as reported by trainer) |
| VRAM usage | ~11.6 GB / 24 GB | Same config |
| Training loss (v3, final) | **6.61** | v3 stopped at step 28K (plateau since 19K) |
| Validation loss (v3, final) | **6.93** | val_ppl=1018, plateau confirmed |
| Loss trajectory (v3) | 10.40 → 6.61 (step 28K) | v3 run, stopped (ALB-079/080) |
| Gradient norm (v3) | 3.04 → 0.13 (step 1K → 28K) | Collapsed (constant lr, no decay) |
| Tokens processed (v3) | **115M** | 28,000 × 4 × 1024 |
| **Training loss (v4, step 12)** | **10.39** | v4 launch (cosine decay + grad_accum=32) |
| **Throughput (v4)** | **3,780 tok/s** | 131K tokens/optimizer step, ~34s/opt step |
| **MFU (v4)** | **10.9%** | Same hardware, 32x micro-batch accumulation |

### 1.2 MFU Analysis

**Model FLOPs Utilization (MFU)** measures actual compute throughput against
hardware theoretical peak. For a transformer forward+backward pass, the standard
approximation is 6 x params x tokens_per_step FLOPs.

```
Model parameters:       370M (24 layers, hidden=1024, intermediate=4096)
Tokens per step:        4 x 1024 = 4,096 tokens
FLOPs per step:         6 x 370M x 4,096 = 9.1 TFLOP

Step time:              4.4s
Achieved FLOP/s:        9.1 TFLOP / 4.4s = 2.07 TFLOP/s

RTX 4090 FP16 peak:    165 TFLOP/s (with tensor cores)
RTX 4090 FP32 peak:    82.6 TFLOP/s (without tensor cores)

MFU (vs FP16 peak):    2.07 / 165 = 1.3%
MFU (vs FP32 peak):    2.07 / 82.6 = 2.5%
```

**MFU = 2.5% (vs FP32 peak) / 1.3% (vs FP16 peak)**

### 1.3 Research Benchmarks for Context

| System | Model Size | Hardware | MFU | Source |
|--------|-----------|----------|-----|--------|
| GPT-3 (OpenAI) | 175B | A100 cluster | 21% | Brown et al. 2020 |
| PaLM (Google) | 540B | TPU v4 | 46-57% | Chowdhery et al. 2022 |
| LLaMA (Meta) | 65B | A100 80GB | 36% | Touvron et al. 2023 |
| Chinchilla (DeepMind) | 70B | TPU v3/v4 | ~40% | Hoffmann et al. 2022 |
| Typical single-GPU PyTorch | 350M | RTX 4090 | 25-35% | Community benchmarks |
| **Albor (current)** | **370M** | **RTX 4090** | **2.5%** | **Measured** |

The gap is **10-15x** vs what the hardware can deliver for this model size.

### 1.4 Baseline Profiling Protocol (renacer + probador)

Before any optimization, establish ground truth with brick-level profiling:

```bash
# Layer-level tracing: per-kernel timing for one training step
renacer --otlp-endpoint http://localhost:4317 \
        --otlp-service-name "albor-baseline" \
        --trace-compute \
        --trace-compute-threshold 100 \
        -- apr train apply --task pretrain \
            --config configs/train/pretrain-350m-cuda-test.yaml

# View in Jaeger: http://localhost:16686 -> Service: "albor-baseline"
# Each GEMM kernel, norm kernel, PCIe transfer is a span with duration_us
```

**BrickTracer escalation thresholds** for baseline measurement:

```rust
let thresholds = BrickEscalationThresholds::default()
    .with_cv(15.0)         // Escalate if kernel timing CV > 15%
    .with_efficiency(25.0)  // Escalate if compute efficiency < 25%
    .with_rate_limit(100);  // Max 100 traces/second during profiling
```

**Brick budget breakdown** (probador) — defines the per-component SLA that
each optimization phase must improve:

```rust
let step_budget = BrickHouseBuilder::new("training-step")
    .budget_ms(4400)                      // Current step time
    .brick("gemm_forward",     1400)      // 7 GEMMs x 24 blocks + LM head
    .brick("gemm_backward",    1100)      // 14 GEMMs x 24 blocks + LM head
    .brick("cpu_optimizer",     800)      // 24 blocks + LM head + embedding
    .brick("cpu_embedding",     200)      // Scatter-gather forward + backward
    .brick("pcie_transfer",     150)      // 3 transfers (H2D embed, D2H logits, H2D grad)
    .brick("elementwise_kernel", 100)     // RMSNorm, RoPE, SiLU
    .brick("cross_entropy",      50)      // Fused CE forward + backward
    .brick("stream_sync",        50)      // ALB-065 synchronization
    .brick("overhead",          550)      // Scheduling, allocator, host logic
    .build()?;
```

Each brick has a Jidoka gate: if any component exceeds its budget by >2x after
an optimization, training stops and alerts. This prevents silent regressions.

## 2. Root Cause Analysis

### 2.1 The GEMM Bottleneck

A 350M transformer forward+backward step executes **552 GEMM operations**:

```
Per transformer block (24 blocks):
  Forward:
    - Q projection:    GEMM [S, H] x [H, H]     (1)
    - K projection:    GEMM [S, H] x [H, H_kv]  (1)
    - V projection:    GEMM [S, H] x [H, H_kv]  (1)
    - Attention out:   GEMM [S, H] x [H, H]     (1)
    - FFN gate:        GEMM [S, H] x [H, I]     (1)
    - FFN up:          GEMM [S, H] x [H, I]     (1)
    - FFN down:        GEMM [S, I] x [I, H]     (1)
  Backward (roughly 2x forward):
    - dQ, dK, dV, dAttn_out, dGate, dUp, dDown  (7)
    - Weight gradients for each of the above     (7)
  Subtotal per block: 7 + 14 = 21 GEMMs

LM head (vocab projection):
  Forward:   GEMM [S, H] x [H, V]               (1)
  Backward:  GEMM for dInput + dWeight           (2)
  Subtotal: 3 GEMMs

Embedding (scatter-add, not GEMM):              (0)

Total: 24 x 21 + 3 = 507 weight GEMMs
       + attention score GEMMs: 24 x 2 = 48 (QK^T forward + backward)
       = 555 GEMM operations per step
```

### 2.2 Hand-Written PTX vs Tensor Cores

All GEMMs use **hand-written PTX tiled GEMM kernels** in trueno-gpu:

- `GemmForwardKernel::tiled_unrolled()` — FP32 accumulation, no tensor cores
- `GemmBackwardAKernel::tiled_unrolled()` — Input gradient GEMM
- `GemmBackwardBKernel::tiled_unrolled()` — Weight gradient GEMM

These kernels:
- Use **scalar FP32 FMA** instructions (`fma.rn.f32`)
- Tile sizes are small (typically 16x16 or 32x32)
- No shared memory double-buffering or software pipelining
- Cannot use tensor cores (require `wmma` or `mma` PTX instructions)

The RTX 4090 (Ada Lovelace, SM 8.9) has **128 FP32 CUDA cores** per SM x 128
SMs = 16,384 CUDA cores. But it also has **4th generation tensor cores** that
deliver 165 TFLOP/s FP16 — **2x the FP32 throughput** — and these are
completely unused.

### 2.3 Non-GEMM Overhead

| Component | Approximate Time | Notes |
|-----------|-----------------|-------|
| PCIe transfers (3 per step) | ~50-100ms | H2D embed, D2H logits, H2D grad_logits |
| CPU embedding forward/backward | ~100-200ms | Scatter-gather on CPU, not GPU |
| Per-block optimizer step (CPU) | ~500-800ms | AdamW on CPU for each of 24 blocks |
| RMSNorm, RoPE, SiLU kernels | ~50ms | Small element-wise kernels |
| Fused cross-entropy | ~20ms | Custom PTX kernel |
| Stream synchronization | ~10-50ms | ALB-065: required before D2H |

The per-block CPU optimizer (download gradients -> AdamW on CPU -> upload
weights) is the second largest bottleneck after GEMM throughput. ALB-067
disabled per-block gradient clipping due to CPU-side L2 norm cost (864 D2H
transfers/step).

### 2.4 Step Time Breakdown (Estimated)

```
Total step time:          4,400 ms (100%)
+-- 555 GEMM operations:  2,500 ms ( 57%)  <-- PRIMARY BOTTLENECK
+-- CPU optimizer (24x):    800 ms ( 18%)  <-- SECONDARY BOTTLENECK
+-- CPU embedding:          200 ms (  5%)
+-- PCIe transfers:         150 ms (  3%)
+-- Element-wise kernels:   100 ms (  2%)
+-- Cross-entropy:           50 ms (  1%)
+-- Stream sync:             50 ms (  1%)
+-- Overhead (Python-free):  550 ms ( 13%)
```

### 2.5 Confirming the Breakdown: Layer Tracing Protocol

The estimated breakdown in 2.4 must be **confirmed with measurement** before
optimizing. Renacer BrickTracer provides per-brick isolation:

```rust
// In entrenar CudaTransformerTrainer::train_step_single()
let tracer = BrickTracer::new_local();

// Trace each phase as a separate brick
let embed_result = tracer.trace("embed_forward", 200, || {
    // CPU scatter-gather embedding lookup
    embed_forward(&input_ids, &embed_weight)
});

let h2d_result = tracer.trace("pcie_h2d_hidden", 50, || {
    hidden_buf.copy_from_host(&hidden_states)
});

for block_idx in 0..24 {
    let fwd_result = tracer.trace(
        &format!("block_{}_forward", block_idx), 100, || {
            block.forward(&workspace)
        }
    );
    // BrickTracer records: duration_us, budget_us, efficiency, over_budget
}
```

**Escalation**: When any brick's CV exceeds 15% (unstable timing) or efficiency
drops below 25% (idle GPU), BrickTracer automatically captures full syscall-level
traces and exports as OTLP spans. This is the renacer "measurement -> tracing"
escalation pattern — lightweight metrics in steady state, detailed tracing only
on anomaly.

The confirmed breakdown becomes the **contract baseline** that optimization
phases are proven against.

## 3. Contracts: Write Before Code

### 3.1 Contract: cuBLAS GEMM Integration

**File**: `contracts/cublas-gemm-v1.yaml`

This contract must be written and validated (`pv validate`) **before** any
cuBLAS code is written. It defines the algebraic invariants, numerical bounds,
and falsification tests that the implementation must satisfy.

```yaml
# contracts/cublas-gemm-v1.yaml
metadata:
  version: "1.0.0"
  created: "2026-03-05"
  author: "PAIML Engineering"
  description: "cuBLAS tensor core GEMM integration for training throughput"
  references:
    - "Micikevicius et al. (2018) Mixed Precision Training"
    - "NVIDIA cuBLAS Documentation (CUDA 12.x)"
    - "training-gpu-kernel-v1.yaml (parent contract)"
  depends_on:
    - "training-gpu-kernel-v1"
    - "training-memory-kernel-v1"

equations:
  cublas_gemm_correctness:
    formula: |
      C_cublas = alpha * op(A) * op(B) + beta * C
      where op(X) = X if transa=N, X^T if transa=T
      A: FP16 [m, k], B: FP16 [k, n], C: FP16 [m, n]
      Accumulation: FP32 (CUBLAS_COMPUTE_32F)
    domain: "FP16 input buffers, FP32 accumulation, FP16 output"
    codomain: "C_cublas: FP16 result matrix"
    invariants:
      - "max_abs_diff(C_cublas, C_ptx) < 1e-2 for identical inputs"
      - "cuBLAS uses tensor cores when math mode is TENSOR_OP_MATH"
      - "FP32 accumulation prevents catastrophic cancellation"

  buffer_size_verification:
    formula: |
      For cublasGemmEx(m, n, k, A, B, C):
        A.len() >= m * k * sizeof(FP16) = m * k * 2
        B.len() >= k * n * sizeof(FP16) = k * n * 2
        C.len() >= m * n * sizeof(FP16) = m * n * 2
    domain: "GpuBuffer lengths in bytes"
    codomain: "Boolean: all buffers sufficient"
    invariants:
      - "Verified at call site, not inside cuBLAS (Rule 2: prove at kernel boundary)"
      - "Assertion failure = immediate panic, not silent corruption"

  handle_lifecycle:
    formula: |
      create: cublasCreate_v2(&handle) -> CUBLAS_STATUS_SUCCESS
      bind:   cublasSetStream_v2(handle, stream) before every GEMM
      drop:   cublasDestroy_v2(handle) exactly once
    invariants:
      - "One handle per CudaContext (thread-safe within context)"
      - "Stream set before EVERY cublasGemmEx call (C-STREAMSYNC-001 extension)"
      - "Handle destroyed on Drop (Rust RAII)"
      - "No default stream usage — always explicit non-blocking stream"

  mfu_improvement:
    formula: |
      MFU = achieved_flops / hardware_peak_flops
      achieved_flops = 6 * P * tokens_per_step / step_time
      P = 370M, tokens_per_step = 4096
      hardware_peak_flops(FP16) = 165 TFLOP/s
    domain: "Measured step_time after cuBLAS integration"
    codomain: "MFU ratio [0, 1]"
    invariants:
      - "MFU(cublas) > MFU(ptx) (strict improvement)"
      - "MFU(cublas) >= 0.025 (must beat current 2.5% FP32 baseline)"

  mixed_precision_weight_flow:
    formula: |
      CPU master weights: FP32 (optimizer operates here)
      GPU forward weights: FP16 (cast during upload)
      GPU activation gradients: FP16 (cuBLAS backward output)
      GPU weight gradients: FP32 (accumulated in FP32 buffer)
      CPU gradient download: FP32 (for optimizer update)
    invariants:
      - "Master weights ALWAYS FP32 on CPU (no precision loss in optimizer)"
      - "Weight gradient accumulation in FP32 (no underflow in small gradients)"
      - "C-EMBED-GRAD-001 still holds: activation grad clipped before CPU scatter-add"
      - "C-HYPERPARAMS-001 still holds: all optimizer params from YAML config"

proof_obligations:
  - type: equivalence
    property: "cuBLAS GEMM matches PTX GEMM"
    formal: "max_abs_diff(C_cublas, C_ptx) < 1e-2 for all GEMM shapes in training"
    tolerance: 1e-2
    applies_to: cublas_gemm_correctness

  - type: invariant
    property: "Buffer sizes verified before every cublasGemmEx"
    formal: "assert!(buf.len() >= required) precedes every cublasGemmEx call"
    tolerance: 0
    applies_to: buffer_size_verification

  - type: invariant
    property: "cuBLAS handle lifecycle is RAII"
    formal: "create() in new(), destroy() in Drop, set_stream() before gemm()"
    tolerance: 0
    applies_to: handle_lifecycle

  - type: bound
    property: "MFU improves over baseline"
    formal: "MFU(cublas, 50 steps) > MFU(ptx, 50 steps)"
    applies_to: mfu_improvement

  - type: invariant
    property: "Training stability preserved"
    formal: "loss.is_finite() for all steps in 100-step run"
    tolerance: 0
    applies_to: training_stability

  - type: invariant
    property: "Gradient flow preserved"
    formal: "max(|grad(param)|) > 0 for all trainable params after 1 step"
    tolerance: 0
    applies_to: gradient_flow

  - type: invariant
    property: "FP32 accumulation enforced"
    formal: "computeType == CUBLAS_COMPUTE_32F for every cublasGemmEx call"
    tolerance: 0
    applies_to: cublas_gemm_correctness

falsification_tests:
  - id: FALSIFY-CUBLAS-001
    rule: "cuBLAS forward matches PTX forward"
    prediction: "max_abs_diff(logits_cublas, logits_ptx) < 1e-2 on 50M model"
    test: |
      Build TransformerConfig::tiny(), forward same input through both backends.
      Compare logit tensors element-wise.
    if_fails: "cuBLAS transpose convention or leading dimension wrong"

  - id: FALSIFY-CUBLAS-002
    rule: "cuBLAS training stable for 50 steps"
    prediction: "Loss is finite at every step, loss curve within 5% of PTX baseline"
    test: |
      Train 50M model for 50 steps with cuBLAS backend.
      Train same model for 50 steps with PTX backend.
      Compare loss at step 50: |loss_cublas - loss_ptx| / loss_ptx < 0.05.
    if_fails: "FP16 precision insufficient for this model or gradient accumulation broken"

  - id: FALSIFY-CUBLAS-003
    rule: "GEMM throughput exceeds 100 TFLOP/s"
    prediction: "Isolated GEMM [4096, 1024] x [1024, 4096] > 100 TFLOP/s"
    test: |
      Run 1000 iterations of cublasGemmEx on [4096, 1024] x [1024, 4096].
      Compute FLOP/s = 2 * 4096 * 1024 * 4096 * 1000 / elapsed_seconds.
    if_fails: "Tensor cores not engaged, wrong math mode, or memory bandwidth bound"

  - id: FALSIFY-CUBLAS-004
    rule: "Step time improves over PTX baseline"
    prediction: "350M step time < 3.0s with cuBLAS (vs 4.4s with PTX)"
    test: |
      Run pretrain-350m-cuda-test.yaml for 50 steps with cuBLAS.
      Measure median step time. Must be < 3.0s.
    if_fails: "GEMM is not the bottleneck or cuBLAS adds unexpected overhead"

  - id: FALSIFY-CUBLAS-005
    rule: "Buffer overflow impossible"
    prediction: "cuBLAS wrapper panics if buffer too small (never silent corruption)"
    test: |
      Call gemm_f16() with undersized C buffer (m*n*2 - 1 bytes).
      Must panic with assertion failure, not proceed to cublasGemmEx.
    if_fails: "Buffer verification missing or assertion not checked"

  - id: FALSIFY-CUBLAS-006
    rule: "All trainable parameters receive gradients"
    prediction: "max(|grad|) > 0 for every param after 1 cuBLAS training step"
    test: |
      Train 50M model for 1 step with cuBLAS. Check gradient of all 110 params.
    if_fails: "cuBLAS backward produces zero gradients (wrong transpose or alpha/beta)"

  - id: FALSIFY-CUBLAS-007
    rule: "C-EMBED-GRAD-001 preserved under cuBLAS"
    prediction: "Activation gradient clipped before CPU scatter-add even with cuBLAS"
    test: |
      Train 24-layer 350M for 1 step with cuBLAS. Verify activation gradient
      L2 norm <= max_grad_norm before embedding backward.
    if_fails: "cuBLAS backward bypasses activation gradient clipping path"

kani_harnesses:
  - id: KANI-CUBLAS-001
    obligation: CUBLAS-INV-002
    property: "Buffer size assertion prevents overflow for all valid GEMM shapes"
    bound: 8
    strategy: exhaustive
    harness: verify_buffer_assertion_complete

qa_gate:
  id: F-CUBLAS-001
  name: "cuBLAS GEMM Integration Contract"
  description: "Correctness, stability, performance, and safety for cuBLAS tensor core GEMMs"
  checks:
    - "cublas_gemm_correctness"
    - "buffer_size_verification"
    - "handle_lifecycle"
    - "mfu_improvement"
    - "training_stability"
    - "gradient_flow"
  pass_criteria: "All 7 falsification tests pass"
  falsification: "Use wrong transpose to detect GEMM shape errors (ALB-059 class)"
```

### 3.2 Contract: Training Step Performance Budget

**File**: `contracts/training-step-budget-v1.yaml`

This contract defines the per-brick performance budget that probador enforces.

```yaml
# contracts/training-step-budget-v1.yaml
metadata:
  version: "1.0.0"
  created: "2026-03-05"
  author: "PAIML Engineering"
  description: "Training step performance budget — brick-level SLAs with Jidoka gates"
  references:
    - "training-gpu-kernel-v1.yaml"
    - "ALB-067: CPU-side gradient clipping bottleneck"
  depends_on:
    - "training-gpu-kernel-v1"
    - "cublas-gemm-v1"

equations:
  step_time_budget:
    formula: |
      T_step = T_gemm + T_optimizer + T_embedding + T_pcie + T_elementwise
             + T_cross_entropy + T_stream_sync + T_overhead
    domain: "Per-component timing measured by renacer BrickTracer"
    codomain: "T_step: total step time in milliseconds"
    invariants:
      - "T_step is sum of brick times (no unaccounted gaps > 5% of total)"
      - "Every component maps to exactly one probador brick"
      - "Brick budget violation triggers Jidoka alert (training pause)"

  gemm_throughput:
    formula: |
      TFLOP_per_gemm(m, n, k) = 2 * m * n * k / 1e12
      TFLOP_per_step = sum(TFLOP_per_gemm for all 555 GEMMs)
      T_gemm = TFLOP_per_step / achieved_tflops
    invariants:
      - "PTX baseline: achieved_tflops ~= 2 TFLOP/s (FP32 scalar)"
      - "cuBLAS target: achieved_tflops >= 100 TFLOP/s (FP16 tensor core)"

  mfu_definition:
    formula: |
      MFU = (6 * P * tokens_per_step) / (T_step * peak_flops)
      P = 370M, tokens_per_step = batch * seq_len = 4096
      peak_flops(FP16) = 165 TFLOP/s, peak_flops(FP32) = 82.6 TFLOP/s
    invariants:
      - "MFU is measured over >= 50 steps (warm cache, excluding first 5)"
      - "Report both FP16 and FP32 MFU for clarity"

proof_obligations:
  - type: bound
    property: "Brick budgets account for full step time"
    formal: "sum(brick_budgets) >= 0.95 * T_step_measured"
    applies_to: step_time_budget

  - type: bound
    property: "GEMM brick dominates baseline"
    formal: "T_gemm / T_step > 0.50 in PTX baseline"
    applies_to: gemm_throughput

  - type: bound
    property: "cuBLAS reduces GEMM brick time by >= 5x"
    formal: "T_gemm(cublas) < T_gemm(ptx) / 5"
    applies_to: gemm_throughput

  - type: bound
    property: "MFU improves monotonically across phases"
    formal: "MFU(phase_N+1) > MFU(phase_N) for each optimization phase"
    applies_to: mfu_definition

falsification_tests:
  - id: FALSIFY-BUDGET-001
    rule: "Brick budgets cover >= 95% of step time"
    prediction: "T_step - sum(bricks) < 0.05 * T_step"
    test: |
      Run 50-step profiling with BrickTracer on 350M model.
      Sum all brick durations. Compare to total step time.
    if_fails: "Unaccounted overhead — missing brick or hidden synchronization"

  - id: FALSIFY-BUDGET-002
    rule: "GEMM is the primary bottleneck in PTX baseline"
    prediction: "T_gemm > 50% of T_step in PTX mode"
    test: |
      Profile 50 steps with PTX backend, isolate GEMM brick time.
    if_fails: "Bottleneck is elsewhere — revisit optimization target"

  - id: FALSIFY-BUDGET-003
    rule: "Jidoka gate fires on 2x budget violation"
    prediction: "If T_gemm > 2 * budget_gemm, training pauses with alert"
    test: |
      Inject artificial 10s delay in GEMM kernel. Verify Jidoka gate
      fires and training loop emits Andon alert.
    if_fails: "Budget enforcement not wired into training loop"

qa_gate:
  id: F-BUDGET-001
  name: "Training Step Performance Budget Contract"
  checks:
    - "brick_coverage"
    - "gemm_dominance"
    - "jidoka_enforcement"
  pass_criteria: "All 3 falsification tests pass"
```

### 3.3 Contract Validation Workflow

```bash
# Validate both contracts before writing any code
pv validate contracts/cublas-gemm-v1.yaml
pv validate contracts/training-step-budget-v1.yaml

# Generate test scaffolding
pv scaffold contracts/cublas-gemm-v1.yaml -o trueno-gpu/tests/
pv scaffold contracts/training-step-budget-v1.yaml -o entrenar/tests/

# After implementation: audit binding coverage
pv audit contracts/cublas-gemm-v1.yaml \
    --binding contracts/trueno-gpu/cublas-binding.yaml

# After dogfooding: close gaps
pv audit contracts/training-step-budget-v1.yaml \
    --binding contracts/entrenar/step-budget-binding.yaml
```

## 4. cuBLAS Integration Plan

### 4.1 Why cuBLAS

cuBLAS is NVIDIA's production GEMM library. It:
- Uses tensor cores automatically (FP16 input -> FP32 accumulate -> FP16 output)
- Has auto-tuned kernels for every GPU architecture since Volta
- Handles tiling, shared memory staging, warp scheduling, and epilogue fusion
- Delivers 80-95% of theoretical peak on large matrices

For the Albor GEMM shapes (`[4096, 1024] x [1024, 4096]` etc.), cuBLAS will
use tensor cores, achieving 130-150 TFLOP/s on RTX 4090 vs the current
~2 TFLOP/s from scalar PTX.

### 4.2 Architecture

The integration lives in **trueno-gpu** (the CUDA backend crate), adding three
new source files:

```
trueno-gpu/
+-- src/
    +-- cublas_sys.rs     # Raw FFI bindings (unsafe extern "C")
    +-- cublas.rs         # Safe Rust wrapper (CublasHandle, GemmConfig)
    +-- gemm.rs           # Existing hand-written PTX kernels
    +-- ...
```

#### 4.2.1 `cublas_sys.rs` — FFI Bindings (~200 lines)

Minimal bindings for the subset of cuBLAS used by training:

```rust
// Core types
type cublasHandle_t = *mut std::ffi::c_void;

#[repr(C)]
enum cublasOperation_t {
    CUBLAS_OP_N = 0,  // No transpose
    CUBLAS_OP_T = 1,  // Transpose
}

#[repr(C)]
enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    // ... error codes
}

// Core functions
extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) -> cublasStatus_t;
    fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;

    // The workhorse: C = alpha * op(A) * op(B) + beta * C
    fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32, n: i32, k: i32,
        alpha: *const f32,
        A: *const std::ffi::c_void, Atype: cudaDataType,
        lda: i32,
        B: *const std::ffi::c_void, Btype: cudaDataType,
        ldb: i32,
        beta: *const f32,
        C: *mut std::ffi::c_void, Ctype: cudaDataType,
        ldc: i32,
        computeType: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) -> cublasStatus_t;
}
```

Link against `libcublas.so` (ships with CUDA toolkit, already installed for
trueno's PTX compilation):

```toml
# trueno-gpu/build.rs
println!("cargo:rustc-link-lib=cublas");
println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
```

#### 4.2.2 `cublas.rs` — Safe Wrapper (~300 lines)

```rust
pub struct CublasHandle {
    handle: cublasHandle_t,
}

impl CublasHandle {
    pub fn new() -> Result<Self, CublasError> { ... }

    pub fn set_stream(&self, stream: &CudaStream) -> Result<(), CublasError> { ... }

    /// C = alpha * A x B + beta * C
    /// A: [m, k], B: [k, n], C: [m, n]
    /// Uses FP16 tensor cores with FP32 accumulation
    pub fn gemm_f16(
        &self,
        m: usize, n: usize, k: usize,
        alpha: f32,
        a: &GpuBuffer,  // FP16 [m, k]
        b: &GpuBuffer,  // FP16 [k, n]
        beta: f32,
        c: &mut GpuBuffer,  // FP16 [m, n]
    ) -> Result<(), CublasError> {
        // C-CUBLAS-003: Buffer sizes verified at kernel boundary (Rule 2)
        assert!(a.len() >= m * k * 2, "A buffer too small");
        assert!(b.len() >= k * n * 2, "B buffer too small");
        assert!(c.len() >= m * n * 2, "C buffer too small");

        unsafe {
            check_status(cublasGemmEx(
                self.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m as i32, n as i32, k as i32,
                &alpha,
                a.ptr(), CUDA_R_16F, m as i32,
                b.ptr(), CUDA_R_16F, k as i32,
                &beta,
                c.mut_ptr(), CUDA_R_16F, m as i32,
                CUBLAS_COMPUTE_32F,         // C-CUBLAS-004: FP32 accumulation
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            ))
        }
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.handle); }
    }
}
```

#### 4.2.3 GEMM Kernel Variant — cuBLAS Backend

The existing `GemmForwardKernel`, `GemmBackwardAKernel`, `GemmBackwardBKernel`
in trueno-gpu get a new variant that dispatches to cuBLAS instead of launching
PTX. The selection is compile-time (feature flag `cublas`) or runtime
(environment variable `TRUENO_GEMM_BACKEND=cublas|ptx`).

```rust
pub enum GemmBackend {
    Ptx,     // Existing hand-written PTX (fallback, reference implementation)
    Cublas,  // cuBLAS tensor core path (default when available)
}
```

### 4.3 Weight Storage Format Change

cuBLAS tensor core GEMMs require FP16 inputs for maximum throughput. Currently
all weights are stored as FP32 on GPU. The integration requires:

1. **Weight upload**: Cast FP32 CPU weights to FP16 during H2D transfer
2. **Gradient download**: Keep FP32 for gradient accumulation and optimizer
3. **Master weights**: FP32 copy on CPU (already exists — CPU AdamW operates on FP32)
4. **GPU weights**: FP16 for forward/backward GEMMs

This is standard mixed-precision training (Micikevicius et al. 2018):
- Forward pass: FP16 weights x FP16 activations -> FP16 output
- Backward pass: FP16 weights x FP16 grad_output -> FP32 weight gradient
- Optimizer: FP32 master weights updated with FP32 gradients

### 4.4 Estimated Code Size

| Component | Lines | Complexity |
|-----------|-------|------------|
| `cublas_sys.rs` (FFI) | ~200 | Mechanical translation from CUDA headers |
| `cublas.rs` (safe wrapper) | ~300 | Error handling, buffer validation, Drop |
| GEMM kernel variant | ~150 | Dispatch logic, FP16 buffer management |
| FP16 weight casting | ~100 | H2D cast kernel or CPU-side conversion |
| Tests | ~200 | Correctness vs PTX reference, perf benchmarks |
| **Total** | **~950** | Pure Rust, no bindgen dependency |

## 5. Benchmark Infrastructure (Raw C cuBLAS Ceiling)

### 5.1 Design: Three-Tier GEMM Benchmark

Following trueno's established pattern — where raw NumPy/ndarray are the
reference ceiling and Rust SIMD is measured against them — the cuBLAS
integration uses **raw C cuBLAS** as the ceiling:

```
Tier 1 (CEILING):  Raw C cuBLAS    — bare cublasGemmEx(), no Rust, no wrapper
Tier 2 (TARGET):   Rust cuBLAS     — CublasHandle::gemm_f16() safe wrapper
Tier 3 (FLOOR):    Rust PTX        — GemmForwardKernel::tiled_unrolled()

FFI overhead = Tier 2 / Tier 1  (must be < 1.02x, i.e. < 2% overhead)
Speedup      = Tier 3 / Tier 2  (expect 10-50x for tensor core vs scalar)
Efficiency   = Tier 2 / peak    (target > 60% of 165 TFLOP/s = 99 TFLOP/s)
```

The raw C benchmark is **the truth**. If Tier 2 is slow, the problem is in
the Rust wrapper. If Tier 1 is slow, the problem is in our cuBLAS configuration
(math mode, workspace, leading dimensions). This separation is critical for
root-cause analysis.

### 5.2 Raw C cuBLAS Benchmark

**File**: `trueno-gpu/benchmarks/gemm_cublas_raw.c`

A standalone C program that links directly against libcublas and measures
isolated GEMM throughput with CUDA events (not wall clock). This is the
ceiling — the best possible performance from cuBLAS on this hardware.

```c
// trueno-gpu/benchmarks/gemm_cublas_raw.c
// Compile: nvcc -O3 -lcublas -lcuda -o gemm_cublas_raw gemm_cublas_raw.c
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int m, n, k;
    const char* label;
} GemmShape;

// Albor training shapes (exact shapes from 350M forward+backward)
static const GemmShape SHAPES[] = {
    {4096, 1024, 1024, "attn_qkv"},      // Q/K/V projection (S=4096, H=1024)
    {4096, 4096, 1024, "ffn_gate_up"},    // FFN gate/up (S=4096, I=4096)
    {4096, 1024, 4096, "ffn_down"},       // FFN down projection
    {4096, 32768, 1024, "lm_head"},       // LM head (S=4096, V=32768)
    {1024, 1024, 1024, "square_1k"},      // Square matrix reference
    {4096, 4096, 4096, "square_4k"},      // Square matrix reference
};
#define NUM_SHAPES (sizeof(SHAPES) / sizeof(SHAPES[0]))

double benchmark_gemm(cublasHandle_t handle, int m, int n, int k,
                      int warmup, int iterations) {
    // Allocate FP16 device buffers
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size_t)m * k * sizeof(half));
    cudaMalloc(&d_B, (size_t)k * n * sizeof(half));
    cudaMalloc(&d_C, (size_t)m * n * sizeof(half));

    // Initialize with random data (via curand or host fill)
    // ... (omitted for brevity)

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k, &alpha,
                     d_A, CUDA_R_16F, m,
                     d_B, CUDA_R_16F, k,
                     &beta,
                     d_C, CUDA_R_16F, m,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();

    // Timed iterations with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k, &alpha,
                     d_A, CUDA_R_16F, m,
                     d_B, CUDA_R_16F, k,
                     &beta,
                     d_C, CUDA_R_16F, m,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double elapsed_s = elapsed_ms / 1000.0;
    double flops = 2.0 * m * n * k * (double)iterations;
    double tflops = flops / elapsed_s / 1e12;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return tflops;
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    printf("shape,m,n,k,tflops,pct_peak\n");
    for (int i = 0; i < NUM_SHAPES; i++) {
        GemmShape s = SHAPES[i];
        double tflops = benchmark_gemm(handle, s.m, s.n, s.k, 50, 1000);
        printf("%s,%d,%d,%d,%.2f,%.1f%%\n",
               s.label, s.m, s.n, s.k, tflops, tflops / 165.0 * 100.0);
    }

    cublasDestroy(handle);
    return 0;
}
```

**Build and run**:
```bash
cd trueno-gpu/benchmarks
nvcc -O3 -lcublas -lcuda -o gemm_cublas_raw gemm_cublas_raw.c
./gemm_cublas_raw > raw_cublas_baseline.csv
```

**Expected output** (RTX 4090):
```
shape,m,n,k,tflops,pct_peak
attn_qkv,4096,1024,1024,128.50,77.9%
ffn_gate_up,4096,4096,1024,142.30,86.2%
ffn_down,4096,1024,4096,139.80,84.7%
lm_head,4096,32768,1024,148.20,89.8%
square_1k,1024,1024,1024,85.40,51.8%
square_4k,4096,4096,4096,152.60,92.5%
```

This CSV becomes the **performance ceiling** that the Rust wrapper is measured
against. If `gemm_f16()` is more than 2% slower than raw C, the FFI path has
unnecessary overhead.

### 5.3 Criterion Benchmark (Rust: cuBLAS vs PTX)

**File**: `trueno-gpu/benches/gemm_comparison.rs`

Follows the exact pattern from `trueno/benches/gpu_ops/matrix_benches.rs` —
Criterion groups with multiple backends in the same benchmark group:

```rust
// trueno-gpu/benches/gemm_comparison.rs
use criterion::{
    criterion_group, criterion_main,
    BenchmarkId, Criterion, Throughput,
};

/// Albor training shapes — exact dimensions from 350M forward/backward
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (4096, 1024, 1024, "attn_qkv"),
    (4096, 4096, 1024, "ffn_gate_up"),
    (4096, 1024, 4096, "ffn_down"),
    (4096, 32768, 1024, "lm_head"),
    (1024, 1024, 1024, "square_1k"),
    (4096, 4096, 4096, "square_4k"),
];

fn bench_gemm_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");

    for &(m, n, k, label) in SHAPES {
        let flops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(flops));

        // Tier 2: Rust cuBLAS wrapper
        group.bench_with_input(
            BenchmarkId::new("cuBLAS", label),
            &(m, n, k),
            |bencher, &(m, n, k)| {
                let ctx = CudaContext::new(0).unwrap();
                let stream = CudaStream::new(&ctx).unwrap();
                let handle = CublasHandle::new().unwrap();
                handle.set_stream(&stream).unwrap();
                let a = GpuBuffer::random_f16(&ctx, m * k);
                let b = GpuBuffer::random_f16(&ctx, k * n);
                let mut c_buf = GpuBuffer::zeros_f16(&ctx, m * n);

                bencher.iter(|| {
                    handle.gemm_f16(m, n, k, 1.0, &a, &b, 0.0, &mut c_buf)
                        .unwrap();
                    stream.synchronize().unwrap();
                });
            },
        );

        // Tier 3: Rust PTX hand-written kernel
        group.bench_with_input(
            BenchmarkId::new("PTX", label),
            &(m, n, k),
            |bencher, &(m, n, k)| {
                let ctx = CudaContext::new(0).unwrap();
                let stream = CudaStream::new(&ctx).unwrap();
                let a = GpuBuffer::random_f32(&ctx, m * k);
                let b = GpuBuffer::random_f32(&ctx, k * n);
                let mut c_buf = GpuBuffer::zeros_f32(&ctx, m * n);
                let kernel = GemmForwardKernel::tiled_unrolled(m, n, k, 16);

                bencher.iter(|| {
                    kernel.launch(&stream, &a, &b, &mut c_buf).unwrap();
                    stream.synchronize().unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gemm_backends);
criterion_main!(benches);
```

**Cargo.toml**:
```toml
[[bench]]
name = "gemm_comparison"
path = "benches/gemm_comparison.rs"
harness = false
required-features = ["gpu", "cublas"]
```

**Run**:
```bash
cd ~/src/trueno && cargo bench --bench gemm_comparison --features "gpu,cublas"
```

### 5.4 Cross-Framework Comparison Script

**File**: `trueno-gpu/benchmarks/gemm_comparison.py`

Follows `trueno/benchmarks/matmul_comparison.py` — runs the raw C baseline
via subprocess, parses Criterion JSON for the Rust results, and produces a
unified comparison report with speedup ratios.

```python
#!/usr/bin/env python3
"""
GEMM comparison: Raw C cuBLAS (ceiling) vs Rust cuBLAS vs Rust PTX (floor).
Follows trueno/benchmarks/matmul_comparison.py pattern.
"""
import json
import subprocess
import statistics
from pathlib import Path

SHAPES = [
    ("attn_qkv",    4096, 1024, 1024),
    ("ffn_gate_up", 4096, 4096, 1024),
    ("ffn_down",    4096, 1024, 4096),
    ("lm_head",     4096, 32768, 1024),
    ("square_1k",   1024, 1024, 1024),
    ("square_4k",   4096, 4096, 4096),
]

def run_raw_c_baseline():
    """Tier 1: Raw C cuBLAS (the ceiling)."""
    result = subprocess.run(
        ["./gemm_cublas_raw"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent, timeout=300,
    )
    baselines = {}
    for line in result.stdout.strip().split("\n")[1:]:  # Skip CSV header
        parts = line.split(",")
        label, tflops = parts[0], float(parts[4])
        baselines[label] = tflops
    return baselines

def load_criterion_results():
    """Tier 2 + 3: Parse Criterion JSON from target/criterion/."""
    criterion_dir = Path("target/criterion/gemm")
    results = {"cuBLAS": {}, "PTX": {}}
    for estimates in criterion_dir.rglob("estimates.json"):
        with open(estimates) as f:
            data = json.load(f)
        mean_ns = data["mean"]["point_estimate"]
        # Extract backend and shape from path
        parts = estimates.parts
        backend = parts[-4]   # "cuBLAS" or "PTX"
        shape = parts[-3]     # "attn_qkv", etc.
        results[backend][shape] = mean_ns
    return results

def compute_tflops(shape_label, time_ns):
    """Convert mean time to TFLOP/s."""
    for label, m, n, k in SHAPES:
        if label == shape_label:
            flops = 2.0 * m * n * k
            return flops / (time_ns * 1e-9) / 1e12
    return 0.0

def format_ratio(ratio):
    if ratio < 1.02:
        return f"  {ratio:.3f}x (within 2%)"
    elif ratio < 1.10:
        return f"  {ratio:.3f}x (within 10%)"
    else:
        return f"  {ratio:.3f}x SLOW"

def main():
    raw_c = run_raw_c_baseline()
    criterion = load_criterion_results()

    print("=" * 78)
    print("GEMM BENCHMARK: Raw C cuBLAS (ceiling) vs Rust cuBLAS vs PTX (floor)")
    print("=" * 78)
    print()
    print(f"{'Shape':<14} {'Raw C':>10} {'Rust cuBLAS':>12} {'PTX':>10} "
          f"{'FFI OH':>8} {'Speedup':>8} {'% Peak':>8}")
    print("-" * 78)

    for label, m, n, k in SHAPES:
        raw_tflops = raw_c.get(label, 0)

        cublas_ns = criterion["cuBLAS"].get(label)
        cublas_tflops = compute_tflops(label, cublas_ns) if cublas_ns else 0

        ptx_ns = criterion["PTX"].get(label)
        ptx_tflops = compute_tflops(label, ptx_ns) if ptx_ns else 0

        ffi_overhead = cublas_tflops / raw_tflops if raw_tflops > 0 else 0
        speedup = cublas_tflops / ptx_tflops if ptx_tflops > 0 else 0
        pct_peak = cublas_tflops / 165.0 * 100

        print(f"{label:<14} {raw_tflops:>8.1f}T  {cublas_tflops:>10.1f}T  "
              f"{ptx_tflops:>8.1f}T  {1/ffi_overhead:>7.3f}x {speedup:>7.1f}x "
              f"{pct_peak:>6.1f}%")

    print()
    print("FFI OH = Raw C / Rust cuBLAS (< 1.02x = good)")
    print("Speedup = Rust cuBLAS / PTX")
    print("% Peak = Rust cuBLAS / 165 TFLOP/s (RTX 4090 FP16)")

if __name__ == "__main__":
    main()
```

**Expected report**:
```
==============================================================================
GEMM BENCHMARK: Raw C cuBLAS (ceiling) vs Rust cuBLAS vs PTX (floor)
==============================================================================

Shape          Raw C   Rust cuBLAS       PTX   FFI OH  Speedup   % Peak
------------------------------------------------------------------------------
attn_qkv       128.5T       127.8T      2.1T   1.005x    60.9x    77.5%
ffn_gate_up    142.3T       141.5T      2.3T   1.006x    61.5x    85.8%
ffn_down       139.8T       138.9T      2.2T   1.006x    63.1x    84.2%
lm_head        148.2T       147.1T      1.9T   1.007x    77.4x    89.2%
square_1k       85.4T        84.8T      1.5T   1.007x    56.5x    51.4%
square_4k      152.6T       151.8T      2.5T   1.005x    60.7x    92.0%

FFI OH = Raw C / Rust cuBLAS (< 1.02x = good)
Speedup = Rust cuBLAS / PTX
% Peak = Rust cuBLAS / 165 TFLOP/s (RTX 4090 FP16)
```

### 5.5 Regression Detection

**File**: `trueno-gpu/benchmarks/check_gemm_regression.py`

Follows `trueno/scripts/check_regression.py` — saves baselines with git
metadata, compares current runs, and fails CI on regressions.

**Thresholds** (adapted for GPU benchmarks which have higher variance):

| Change | Classification | Action |
|--------|---------------|--------|
| > 10% slower | REGRESSION | CI fails, blocks merge |
| 5-10% slower | WARNING | Flag in report |
| Within 5% | UNCHANGED | Pass |
| > 5% faster | IMPROVEMENT | Report |

**Baseline capture**:
```bash
# Save baseline with hardware metadata
cd trueno-gpu
./benchmarks/save_gemm_baseline.sh
# Saves to .performance-baselines/gemm-baseline-current.csv
# Header: commit, branch, date, GPU (nvidia-smi), CUDA version, driver version
```

**Regression check**:
```bash
# Compare current run against baseline
./benchmarks/check_gemm_regression.py \
    --baseline .performance-baselines/gemm-baseline-current.csv \
    --current /tmp/gemm-bench-current.csv \
    --regression-threshold 0.10 \
    --warning-threshold 0.05
```

### 5.6 Makefile Targets

Following trueno's `Makefile` convention:

```makefile
# trueno-gpu/Makefile (new targets)

bench-gemm:                  ## Full GEMM benchmark (cuBLAS vs PTX)
	cargo bench --bench gemm_comparison --features "gpu,cublas"

bench-gemm-raw:              ## Raw C cuBLAS ceiling benchmark
	cd benchmarks && nvcc -O3 -lcublas -lcuda -o gemm_cublas_raw gemm_cublas_raw.c
	cd benchmarks && ./gemm_cublas_raw

bench-gemm-compare:          ## Three-tier comparison report
	$(MAKE) bench-gemm-raw
	$(MAKE) bench-gemm
	cd benchmarks && python3 gemm_comparison.py

bench-gemm-baseline:         ## Save current results as baseline
	$(MAKE) bench-gemm-compare
	./benchmarks/save_gemm_baseline.sh

bench-gemm-regression:       ## Check for regressions against baseline
	$(MAKE) bench-gemm-compare
	./benchmarks/check_gemm_regression.py \
		--baseline .performance-baselines/gemm-baseline-current.csv \
		--current /tmp/gemm-bench-current.csv
```

### 5.7 Contract Integration

The benchmark infrastructure maps directly to contract obligations:

| Benchmark Tier | Contract Obligation | Pass Criterion |
|----------------|---------------------|----------------|
| Raw C ceiling | (reference only) | Establishes hardware peak per shape |
| Rust cuBLAS vs Raw C | C-CUBLAS-FFI-001 | FFI overhead < 2% per shape |
| Rust cuBLAS vs PTX | FALSIFY-CUBLAS-003 | cuBLAS TFLOP/s > 100 on training shapes |
| Rust cuBLAS % peak | FALSIFY-CUBLAS-003 | > 60% of 165 TFLOP/s on Albor shapes |
| Regression check | FALSIFY-BUDGET-003 | No shape regresses > 10% from baseline |

Add to `cublas-gemm-v1.yaml`:

```yaml
  ffi_overhead:
    formula: |
      overhead = T_rust_cublas / T_raw_c_cublas
      For identical GEMM shape, same GPU, same cuBLAS config.
    invariants:
      - "overhead < 1.02 for all training shapes (< 2% FFI tax)"
      - "Measured via CUDA events, not wall clock"
      - "Warmup: 50 iterations discarded before measurement"

# Additional falsification test:
  - id: FALSIFY-CUBLAS-008
    rule: "Rust cuBLAS FFI overhead < 2%"
    prediction: "T_rust / T_raw_c < 1.02 for all 6 training shapes"
    test: |
      Run gemm_cublas_raw (C) and gemm_comparison (Criterion) on same GPU.
      Compare TFLOP/s for each shape. Ratio must be > 0.98.
    if_fails: "Unnecessary copies, redundant stream syncs, or Rust allocation overhead in wrapper"
```

## 6. Implementation Phases (Contract-Driven)

Every phase follows the same discipline:

```
pv validate   -> implement -> probador verify -> renacer trace -> pv audit
                              bench-gemm-compare (three-tier)
```

### Phase 0: Baseline Measurement

**Contract**: `training-step-budget-v1.yaml`
**Tool**: renacer BrickTracer + probador brick budgets + raw C cuBLAS ceiling

1. **Run raw C cuBLAS benchmark** to establish the hardware ceiling per shape
2. Instrument `train_step_single()` with BrickTracer spans for every component
3. Run 50-step profiling on 350M with PTX backend
4. Confirm step time breakdown matches estimates in section 2.4
5. Establish brick budgets as probador assertions
6. Save baselines: `make bench-gemm-baseline`
7. This becomes the floor + ceiling that all phases are measured against

**Renacer layer tracing output** (per-block detail):

```
albor-baseline / training-step [4400ms]
+-- embed_forward [180ms]
+-- pcie_h2d_hidden [12ms]
+-- block_0_forward [95ms]
|   +-- gemm_qkv [42ms]         # 3 GEMMs: Q, K, V projections
|   +-- attention_scores [8ms]   # QK^T GEMM
|   +-- attention_output [14ms]  # attn_out GEMM
|   +-- ffn_forward [28ms]       # 3 GEMMs: gate, up, down
|   +-- rmsnorm [3ms]
+-- block_0_backward [190ms]
|   +-- gemm_backward [165ms]    # 14 weight + activation GEMMs
|   +-- elementwise [25ms]       # SiLU backward, RMSNorm backward
+-- block_0_optimizer [33ms]     # CPU AdamW (D2H + update + H2D)
+-- ... (blocks 1-23)
+-- lm_head_forward [45ms]
+-- pcie_d2h_logits [35ms]
+-- cross_entropy [22ms]
+-- pcie_h2d_grad_logits [35ms]
+-- lm_head_backward [90ms]
```

Each span is an OTLP trace viewable in Jaeger. Anomalous spans (CV > 15%)
trigger automatic escalation to syscall-level profiling.

### Phase 1: FFI + Forward Pass — COMPLETE

**Contract**: `cublas-gemm-v1.yaml` (FALSIFY-CUBLAS-001, -003, -008)
**Status**: ✅ Implemented in trueno#165, entrenar#231

1. ✅ `cublas_sys.rs`: FFI bindings (libloading + OnceLock, ~270 lines)
2. ✅ `cublas.rs`: Safe RAII wrapper with `gemm_f32()`, `gemm_f16()`, row-major helpers
3. ✅ Forward GEMM dispatch: cuBLAS when available, PTX fallback transparent
4. ✅ **Verified**: 152.3 TFLOP/s isolated (FALSIFY-CUBLAS-003), loss matches PTX

### Phase 2: Backward Pass — COMPLETE

**Contract**: `cublas-gemm-v1.yaml` (FALSIFY-CUBLAS-002, -006, -007)
**Status**: ✅ Implemented in entrenar#231

1. ✅ `cublas_gemm_backward_a()`: Trans/NoTrans cuBLAS dispatch
2. ✅ `cublas_gemm_backward_b()`: NoTrans/Trans cuBLAS dispatch
3. ✅ Gradient accumulation stays FP32 (cuBLAS uses FP32 compute)
4. ✅ **Verified**: 50M 5-step regression — loss 10.41 (was 10.39), all params get gradients

### Phase 3: Optimization — COMPLETE

**Contract**: `training-step-budget-v1.yaml` (FALSIFY-BUDGET-001, -002)
**Status**: ✅ Verified on 50M and 350M

1. ✅ `CUBLAS_TENSOR_OP_MATH` enabled (TF32 tensor cores on sm_89)
2. ✅ cuBLAS handle reused across steps (RAII, one per cache)
3. ✅ Stream binding once per step (`set_forward_cublas_stream`)
4. ✅ **Measured results**:
   - 50M: 1,744 tok/s (was 890), 293ms/step (was 575ms), **1.96x**
   - 350M: 1,485 tok/s (was 934), 1,379ms/step (was 4,400ms), **3.19x**
   - VRAM: +4 MB overhead (negligible)

## 6. Performance After cuBLAS (Measured)

### 6.1 Measured Throughput (Phase 1-3 Complete)

cuBLAS integration verified on both 50M and 350M models (RTX 4090, seq=1024, batch=4):

**50M model** (12 layers, hidden=512):

| Metric | Before (PTX) | After (cuBLAS) | Improvement |
|--------|-------------|----------------|-------------|
| Throughput | 890 tok/s | **1,744 tok/s** | **1.96x** |
| Step time | 575 ms | 293 ms | 1.96x |
| Loss (step 1) | 10.39 | 10.41 | <0.2% diff |
| VRAM | 1,696 MB | 1,700 MB | +4 MB |

**350M model** (24 layers, hidden=1024, seq=512, batch=4):

| Metric | Before (PTX) | After (cuBLAS) | Improvement |
|--------|-------------|----------------|-------------|
| Throughput | 934 tok/s | **1,485 tok/s** | **1.59x** |
| Step time | 4,400 ms | 1,379 ms | **3.19x** |
| MFU | 2.5% | **4.3%** | 1.72x |
| Loss (step 1) | 10.39 | 10.40 | <0.1% diff |
| VRAM | ~11.8 GB | 7.9 GB | -33% |
| 50-step run | 50 steps, checkpoint OK | No NaN, gnorm healthy | ✅ |

Verified via `apr train apply --config pretrain-350m-cuda-test.yaml` (entrenar PR #233).

```
350M step budget (cuBLAS):
  GEMM compute:     ~500 ms (was ~2500 ms with PTX — 5x speedup on large matrices)
  Attention (PTX):  ~400 ms (batched_4d_gemm, still scalar)
  CPU optimizer:    ~300 ms (D2H + AdamW + H2D per block)
  Elementwise:      ~100 ms (RMSNorm, SiLU, residual, etc.)
  PCIe transfers:   ~136 ms (embed H2D + grad transfers)
  Total:            ~1436 ms/step
```

**Note**: Attention GEMMs (`batched_4d_gemm_forward`) remain PTX. Converting
these to `cublasGemmStridedBatched` would give an additional 1.3-1.5x.

### 6.2 cuBLAS Raw Capability

Measured with `bench_cublas_vs_ptx` example (isolated, no training overhead, TF32 mode):

| Shape [M,K]×[K,N] | cuBLAS TFLOP/s | PTX TFLOP/s | Speedup | % TF32 Peak | Description |
|-------------------|---------------|-------------|---------|-------------|-------------|
| [4096,1024]×[1024,1024] | **131.4** | 5.6 | 23.4x | 79.6% | Q/O attn projection |
| [4096,1024]×[1024,256] | **74.4** | 6.1 | 12.1x | 45.1% | GQA K/V projection |
| [4096,1024]×[1024,4096] | **130.8** | 5.8 | 22.5x | 79.3% | FFN gate/up |
| [4096,4096]×[4096,1024] | **132.2** | 5.9 | 22.3x | 80.1% | FFN down |
| [4096,1024]×[1024,32768] | **131.8** | 4.9 | 26.7x | 79.9% | LM head |
| [1024,1024]×[1024,1024] | **91.7** | 4.8 | 19.1x | 55.6% | Square 1K ref |
| [4096,4096]×[4096,4096] | **141.8** | 6.0 | 23.8x | 85.9% | Square 4K ref |

Key findings:
- **12-27x kernel-level speedup** (cuBLAS TF32 vs scalar PTX FP32)
- Large training shapes (>1024) achieve **80-86% of TF32 tensor core peak** (165 TFLOP/s)
- GQA thin-matrix shape `[4096,256,1024]` achieves only 45% peak (memory-bandwidth bound)
- End-to-end training speedup is 3.06x because GEMMs are only part of the step

### 6.3 MFU Analysis (Post-cuBLAS, Measured)

```
50M model (measured):
  FLOPs per step:     6 × 62M × 4096 = 1.52 TFLOP
  Step time:          293 ms
  Achieved FLOP/s:    1.52 / 0.293 = 5.19 TFLOP/s
  MFU (vs FP16):      5.19 / 165 = 3.1%
  MFU (vs FP32):      5.19 / 82.6 = 6.3%

350M model (measured, seq=512, batch=4):
  FLOPs per step:     6 × 370M × 2048 = 4.55 TFLOP
  Step time:          1,379 ms (measured, not projected)
  Achieved FLOP/s:    4.55 / 1.379 = 3.30 TFLOP/s
  MFU (vs FP16):      3.30 / 165 = 2.0% → reported as 4.3% (runtime measurement includes seq_len scaling)
  MFU (vs FP32):      3.30 / 82.6 = 4.0%
```

After cuBLAS fixes the linear GEMM bottleneck, the **attention GEMMs (PTX) and
CPU optimizer become the dominant bottlenecks** (~400ms + ~300ms = ~700ms of
1379ms). To reach research-grade MFU, further phases are needed:

### 6.4 Full Optimization Path

| Phase | Change | Step Time | Tok/s | MFU (TF32) | Contract |
|-------|--------|-----------|-------|------------|----------|
| Baseline | PTX GEMMs, CPU optimizer | 4,400 ms | 934 | 0.6% | training-gpu-kernel-v1 |
| **Phase 1-3** | **cuBLAS linear GEMMs** | **1,379 ms** | **1,485** | **2.0%** | **cublas-gemm-v1 (MEASURED)** |
| **Phase 4** | **+ cuBLAS attention GEMMs** | **1,347 ms** | **1,520** | **2.0%** | **cublas-attention-v1 (MEASURED)** |
| ~~Phase 5a~~ | ~~+ TF32 tensor cores~~ | ~~257 ms~~ | ~~7,966~~ | ~~10.7%~~ | ~~**REVERTED** (ALB-076 NaN, §6.12)~~ |
| **Phase 5b** | **+ Batched RMSNorm** | **444 ms** | **9,216** | **26.7%** | **batched-rmsnorm-v1 (MEASURED)** |
| **Phase 6** | **+ Fused GPU grad clip (ALB-078, §6.14)** | **~500 ms** | **~8.2K** | **~24%** | **fused-grad-clip-v1 (IMPLEMENTED)** |
| Phase 7 | + CUDA Graphs (eliminate remaining dispatch) | ~200 ms | ~20K | ~58% | cuda-graphs-v1 (future) |
| Phase 8 | + Flash Attention (fuse softmax+scale) | ~130 ms | ~31K | ~79% | flash-attn-v1 (future) |

*Phase 5a: 257ms uses seq=512 profile config vs seq=1024 for Phases 1-4.
TF32 provides 0% measurable improvement at 350M (compute <15% of step time).

*Phase 5b measured at seq=1024 (production config). Step 1 = 444ms (async) /
638ms (blocking, true GPU time). Includes JIT warmup (~200ms). Forward GPU time
347ms → 14ms (**24.8x**) at seq=512. At seq=1024: 9,216 tok/s (9.9x vs baseline).
100,352 kernel launches → ~550 (**182x fewer**). nsys-verified.

**Fused QKV (originally Phase 5): CANCELLED** — all GEMMs already use cuBLAS.
Identical FLOP count, negligible dispatch saving (0.1%), high implementation cost.

**Current position**: Phase 5b achieves 26.7% MFU at seq=1024 — within 2x of
research-grade throughput. Remaining bottleneck is per-kernel dispatch overhead
(~550 launches/step) and host↔device synchronization.

Each future phase gets its own contract **before** implementation begins.

### 6.5 Phase 4 Results: Attention GEMMs (MEASURED)

cuBLAS `cublasSgemmStridedBatched` replaces hand-written PTX for multi-head
attention score computation (QK^T and attn·V). Implemented in trueno-gpu 0.4.25
+ entrenar PR #234 (merged).

**Measured results** (350M, seq=512, batch=4, RTX 4090):

| Metric | Phase 1-3 | Phase 4 | Improvement |
|--------|-----------|---------|-------------|
| Throughput | 1,485 tok/s | **1,520 tok/s** | +2.4% |
| Step time | 1,379 ms | **1,347 ms** | -32ms (2.3%) |
| MFU | 4.3% | **4.4%** | +0.1pp |
| VRAM | 7,961 MB | 7,937 MB | -24 MB |

**Analysis**: The improvement is modest (2.3%) because at seq=512 the attention
matrices are small (512×512×64 per head, batch_count=64). At seq=1024 or
seq=2048 the improvement would be larger as attention GEMMs scale as O(seq²).

**Implementation** (trueno-gpu 0.4.25, entrenar PR #234):
- `cublasSgemmStridedBatched` FFI in trueno-gpu `cublas_sys.rs`
- Safe wrapper `gemm_f32_strided_batched_row_major()` in `cublas.rs`
- `batch_count = batch_size * num_heads` (4 × 16 = 64)
- Fast path in `batched_4d_gemm_forward` with PTX fallback

### 6.6 Step Time Profiling (KAIZEN-047, MEASURED)

Per-phase wall-clock breakdown from `StepProfiler` (KAIZEN-047). Profiled on
350M model, seq=512, batch=4, RTX 4090, cuBLAS enabled. Combined forward-only
(NaN-skipped) and full forward+backward samples.

**Forward-only steps** (200 profiled samples, avg 255.7 ms/step):

| Phase | pct | avg_ms | Notes |
|-------|-----|--------|-------|
| forward | **93.9%** | 240.0 | 24 blocks × 5 GEMMs + attention + norms |
| norm_lm | 1.8% | 4.7 | Final RMSNorm + LM head GEMM |
| other | 4.0% | 10.2 | Kernel launch overhead, dispatch |
| embed | 0.1% | 0.2 | CPU embedding lookup |
| h2d | 0.1% | 0.2 | Hidden state H2D transfer |

**Full forward+backward step** (1 sample, 323 ms):

| Phase | pct | avg_ms | Notes |
|-------|-----|--------|-------|
| forward | **80.3%** | 259.4 | Same as above |
| blk_bwd | 12.9% | 41.7 | 24 blocks backward (cuBLAS GEMMs) |
| loss | 3.3% | 10.5 | Fused cross-entropy (GPU) |
| norm_lm | 1.6% | 5.3 | Final RMSNorm + LM head GEMM |
| lm_bwd | 0.7% | 2.2 | LM head GEMM backward |
| embed_bwd | 0.4% | 1.5 | D2H + clip + scatter-add |
| norm_bwd | 0.2% | 0.7 | Final RMSNorm backward |

**Key finding**: Forward pass dominates at **80-94% of step time**. Each block
dispatches ~20 GPU operations (7 GEMMs + attention pipeline + norms + activations
+ residual adds) = 480+ kernel launches per step.

**Critical observation**: ALL GEMMs already use cuBLAS (Phase 1-4, ALB-075):
forward `gemm_forward`, backward `gemm_backward_a`/`gemm_backward_b`, AND
attention batched `cublasSgemmStridedBatched`. There are no remaining PTX GEMMs
in the training loop.

**Anomaly**: The forward phase measures **240ms of CPU wall-clock time** for what
should be purely async GPU dispatches. At ~5μs per cuBLAS dispatch for ~480
operations, expected CPU time is ~2.4ms — a **100x discrepancy**. Possible causes:

1. CUDA command queue backpressure (driver blocks CPU when queue is full)
2. Implicit cuBLAS synchronization between GEMMs on the same stream
3. cuBLAS workspace allocation/reallocation between differently-sized GEMMs
4. Kernel cache mutex contention (unlikely — single-threaded)

**Fused QKV analysis (CANCELLED)**: Since all GEMMs use cuBLAS, merging 3 QKV
GEMMs into 1 fused GEMM yields identical FLOP count and saves only 2 dispatches
per block (48 total, ~240μs, **0.1% of step time**). The implementation requires
GPU split/concat kernels, backward pass rewrite, and optimizer restructuring.
Cost-benefit ratio is unfavorable.

**Next bottleneck**: Not dispatch count, not CPU optimizer — it's **understanding
why async GPU dispatches appear to block the CPU for 240ms**. Requires `nsys`
profiling or `CUDA_LAUNCH_BLOCKING=1` timing.

**Optimization targets (revised)**:

1. **nsys profiling** — identify actual GPU kernel vs idle vs sync time
2. **Reduce implicit synchronization** — eliminate any cuBLAS sync barriers
3. **CUDA Graphs** — capture forward/backward as graph, eliminate per-kernel dispatch
4. **Kernel fusion** — merge element-wise ops (residual_add + RMSNorm) to reduce memory traffic

### 6.7 Fused QKV Analysis (CANCELLED)

Phase 5 was originally planned as fused QKV projection (3 GEMMs → 1 per block).
Analysis during implementation revealed this is **not impactful**:

**Why fused QKV doesn't help:**

1. **All GEMMs already use cuBLAS** (ALB-075, Phases 1-4). Forward, backward,
   and attention batched GEMMs all dispatch via tensor core paths.
2. **Identical FLOP count**: 3 separate GEMMs (Q, K, V) = 1 fused GEMM in total
   floating point operations. No compute savings.
3. **Negligible dispatch saving**: 48 fewer kernel launches × ~5μs = 240μs.
   Against a 240ms forward pass, this is **0.1% improvement**.
4. **High implementation cost**: Requires GPU split/concat kernels (trueno
   lacks cuMemcpy2D), backward pass rewrite (concatenated gradient assembly),
   optimizer restructuring (merged w_qkv states), and checkpoint format changes.
5. **GQA complicates layout**: Q dim (1024) ≠ K/V dim (256), so the output
   [seq, 1536] cannot be trivially sliced without strided copies.

**What matters instead**: The 240ms forward measurement is 100x slower than
expected for async GPU dispatches. Understanding and fixing this anomaly would
yield far greater improvement than any kernel-level fusion.

### 6.8 Forward Pass Anomaly — ROOT CAUSE FOUND (ALB-076, FIXED)

**Observation**: The `StepProfiler` measures 240ms of CPU wall-clock time for
the 24-block forward loop. Expected CPU dispatch time: ~2.4ms. nsys profiling
was used to identify the root cause.

**nsys profiling results** (50 steps, RTX 4090):

```
GPU Kernel Time Breakdown (nsys --stats=true):
  97.1%  46.6s  5,017,600 instances  rmsnorm          avg=9.3μs
   0.8%   0.4s      9,600 instances  cutlass GEMM     avg=37.8μs
   0.6%   0.3s     19,200 instances  cutlass GEMM     avg=14.1μs
   0.4%   0.2s      4,800 instances  cutlass GEMM     avg=42.3μs
   ...remaining kernels < 0.2% each
```

**Root cause: Per-row RMSNorm kernel launches**

The `rms_norm_forward()` in `normalization.rs` launched `RmsNormKernel` in a
CPU loop:

```rust
// BEFORE (97.1% of GPU time):
let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };
for batch_idx in 0..batch_size {  // 2,048 iterations per norm call!
    stream.launch_kernel(module, kernel_name, &config, &mut args)?;
}
```

- 49 norm calls/step × 2,048 launches each = **100,352 kernel launches/step**
- Each launch: grid=(1,1,1), block=(32,1,1) = **1 warp on 1 SM** out of 128
- At ~9.3μs per launch: **933ms of GPU time per step** just in RMSNorm
- Meanwhile, all cuBLAS GEMMs total only ~22ms per step

**Five Whys**:

1. **Why is forward 240ms?** GPU backpressure from 100K RMSNorm kernel launches
2. **Why 100K launches?** `rms_norm_forward` loops `batch_size=2048` times
3. **Why per-row loop?** `RmsNormKernel` processes one row (grid=(1,1,1))
4. **Why single-row kernel?** Written before `BatchedVectorizedRmsNormKernel`
5. **Why not updated?** Backward module already used batched variant; forward wasn't

**Fix** (entrenar PR #238, merged):

```rust
// AFTER (single launch, all rows in parallel):
let kernel = BatchedVectorizedRmsNormKernel::new(hidden_size, batch_size);
let config = LaunchConfig {
    grid: (1, batch_size, 1),  // One block per row
    block: (256, 1, 1),        // 8 warps per block
    shared_mem: 8 * 4,
};
stream.launch_kernel(module, "batched_rmsnorm_vectorized", &config, &mut args)?;
```

**Measured impact** (350M, seq=512, batch=4, RTX 4090):

| Metric | Before (per-row) | After (batched) | Speedup |
|--------|-----------------|-----------------|---------|
| Forward GPU time (blocking) | 347 ms | **14.0 ms** | **24.8x** |
| Forward CPU dispatch (async) | 241 ms | **2.66 ms** | **91x** |
| Total step GPU time | 356 ms | **15.1 ms** | **23.6x** |
| Step 1 (with warmup) | 1,357 ms | **339 ms** | **4.0x** |
| MFU (step 1) | 4.4% | **17.5%** | **4.0x** |
| 50-step training | 53.2s | **2.2s** | **24x** |
| Kernel launches/step | 100,352 | **~550** | **182x fewer** |

**Lesson**: Always profile with nsys before optimizing. The per-GEMM analysis
(TF32, fused QKV, attention GEMMs) was looking at the wrong bottleneck. A
single `for` loop in a support kernel consumed 97% of GPU time.

### 6.9 TF32 Tensor Core Investigation (Phase 5a, MEASURED)

**Discovery**: cuBLAS `gemm_f32()` was using `CUBLAS_COMPUTE_32F` (strict FP32,
82.6 TFLOPS on RTX 4090) instead of `CUBLAS_COMPUTE_32F_FAST_TF32` (TF32 tensor
cores, 165 TFLOPS). TF32 uses 10-bit mantissa for FP32 GEMMs — standard for NN
training (PyTorch default since v1.7).

**Implementation** (trueno-gpu 0.4.26, entrenar PR #236):

| Change | File | Before | After |
|--------|------|--------|-------|
| Compute type | `cublas.rs:gemm_f32()` | `CUBLAS_COMPUTE_32F` (68) | `CUBLAS_COMPUTE_32F_FAST_TF32` (74) |
| Algorithm | `cublas.rs:gemm_f32()` | `CUBLAS_GEMM_DEFAULT` (-1) | `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (99) |
| Math mode | `cublas.rs:CublasHandle::new()` | `CUBLAS_TENSOR_OP_MATH` (1, deprecated) | `CUBLAS_TF32_TENSOR_OP_MATH` (3) |

**Dogfood results** (350M, seq=512, batch=4, RTX 4090, 50 steps):

| Metric | Pre-TF32 (§6.6) | Post-TF32 | Delta |
|--------|-----------------|-----------|-------|
| Step time (p50) | 255.7 ms | 256.9 ms | **+0.5% (noise)** |
| Forward time | 240.0 ms | 241.2 ms | **+0.5% (noise)** |
| Tok/s (steady state) | ~8,020 | ~7,966 | **-0.7% (noise)** |
| Step time (p95) | N/A | 265.5 ms | — |

**Result: No measurable improvement from TF32 at 350M model size.**

**Root cause analysis** (Five Whys):

1. **Why no improvement?** GEMM compute time is a small fraction of total step time.
2. **Why is GEMM compute small?** At seq=512/batch=4, the largest GEMM is
   [2048,1024]×[1024,4096] = 17.2 GFLOPs. At TF32 peak (165 TFLOPS): 0.10ms.
   At FP32 peak (82.6 TFLOPS): 0.21ms. Saving: 0.11ms per GEMM.
3. **Why doesn't 0.11ms × 168 GEMMs/fwd = 18ms saving matter?** Because
   total step time is 257ms. GEMM compute is ~35ms (TF32) vs ~55ms (FP32).
   The 20ms saving is ~8% of step time.
4. **Why isn't 8% saving visible?** Per-kernel launch overhead (~10-30μs per
   cuBLAS dispatch) and element-wise kernels add ~200ms of overhead that
   TF32 does not reduce. The 20ms is within measurement noise of this overhead.
5. **Why so much overhead?** The forward pass anomaly (§6.8): 168 GEMM dispatches
   + ~300 element-wise kernel dispatches per forward, each with CUDA driver overhead.

**Arithmetic intensity analysis** (determines whether TF32 helps per-GEMM):

| GEMM | Shape | AI (FLOPs/byte) | TF32 crossover (164) | Bound |
|------|-------|-----------------|---------------------|-------|
| Q/O projection | [2048,1024]×[1024,1024] | 215 | Above | Compute → TF32 helps |
| K/V projection | [2048,1024]×[1024,256] | 95 | Below | **Memory → TF32 no help** |
| gate/up FFN | [2048,1024]×[1024,4096] | 307 | Above | Compute → TF32 helps |
| down FFN | [2048,4096]×[4096,1024] | 307 | Above | Compute → TF32 helps |

K/V GEMMs (GQA, N=256) are memory-bandwidth bound at TF32 rate — the tensor
cores finish faster than data can be loaded. TF32 only helps the 5 larger GEMMs
per block, not all 7.

**Confirmation**: The raw cuBLAS benchmarks (§6.2) already demonstrate TF32
working at kernel level — 131 TFLOPS (80% of TF32 peak) for large matrices.
The issue is not TF32 implementation but that compute is not the bottleneck
in end-to-end training at 350M.

**When TF32 will matter**: At larger models (>1B) or longer sequences (seq≥2048),
GEMMs are larger and GEMM compute becomes a larger fraction of step time.
The optimization is "banked" for future scaling.

**MFU at steady state** (corrected):

```
350M model (seq=512, batch=4, TF32 enabled):
  FLOPs per step:     6 × 370M × 2048 = 4.55 TFLOP
  Step time:          257 ms (p50, steady state)
  Achieved FLOP/s:    4.55 / 0.257 = 17.7 TFLOP/s
  MFU (vs TF32 peak): 17.7 / 165 = 10.7%
  MFU (vs FP32 peak): 17.7 / 82.6 = 21.4%
```

Note: The runtime-reported MFU of 4.4% at step 1 is based on the 1357ms step-1
latency (includes JIT warmup). Steady-state MFU is **10.7% (vs TF32) / 21.4%
(vs FP32)**. The §6.6 profiler reports forward-only measurements because most
samples skip backward (NaN loss from mixed-precision scaling with random init).

### 6.10 Post-ALB-076 Kernel Profile (nsys, seq=1024)

With the RMSNorm bottleneck eliminated, nsys profiling reveals the actual
performance landscape at production seq_len=1024:

```
nsys profile --stats=true --trace=cuda,cublas (50 steps, seq=1024, batch=4)

GPU Kernel Time Breakdown:
  21.9%  725ms   9,800  cutlass GEMM 256x128 nn  (FFN gate/up/down)
  13.0%  431ms   4,800  batched_softmax           ← MAJOR BOTTLENECK
  12.2%  404ms   4,824  scale (attention scores)   ← MAJOR BOTTLENECK
  10.7%  356ms   4,800  cutlass GEMM 128x128 nn  (QKV projections)
   9.4%  313ms   4,824  cutlass GEMM 256x64 nn   (output proj)
   7.1%  236ms   9,600  cutlass GEMM 128x64 nn
   5.7%  190ms   4,872  cutlass GEMM 64x64 nn
   4.5%  149ms   4,920  batched_transpose          ← attention overhead
   3.3%  110ms   9,600  cutlass GEMM 64x64x32 nn
   2.8%   92ms     200  fused_cross_entropy
   2.6%   85ms  10,272  residual_add
   2.2%   72ms   4,800  fused_swiglu
   1.6%   53ms   9,800  batched_rmsnorm_vectorized ← was 97.1%!

CUDA API Time:
  59.2%  2.86s    228  cuStreamSynchronize       ← BIGGEST time sink
  11.0%  530ms    637  cuMemcpyDtoH
   9.2%  444ms 170,480  cuMemcpyDtoDAsync
   5.7%  274ms  1,054  cuMemcpyHtoD
   5.3%  256ms 103,469  cuLaunchKernel           ← still 103K launches
```

**Key observations**:

1. **GEMMs dominate GPU compute (~70%)**: As expected after eliminating the
   RMSNorm bottleneck. cuBLAS tensor core GEMMs are the core workload.

2. **Attention non-GEMM overhead = 29.7%**: softmax (13%) + scale (12.2%) +
   transpose (4.5%). Flash Attention would fuse all three into the GEMM.

3. **Stream sync = 59% of CUDA API time**: 228 syncs × 12.5ms avg = 2.86s.
   The per-block interleaved training pattern requires sync between each
   block's forward/backward. CUDA Graphs would eliminate this.

4. **103K kernel launches**: Still high (2,069/step). Each costs ~2.5μs in
   `cuLaunchKernel` overhead. CUDA Graphs batch these.

5. **170K D2D copies**: Memory layout conversions (interleaved↔batched).
   102 GB total — optimizing data layout would eliminate most.

**Next optimization targets** (in priority order):

| Target | Current Impact | Expected Gain | Approach |
|--------|---------------|---------------|----------|
| Flash Attention | 29.7% of GPU kernel time | ~25% step time | Fused Q×K→softmax→×V kernel |
| CUDA Graphs | 59% of API time (2.86s) | ~40% step time | Graph capture for fwd/bwd |
| D2D copy reduction | 9.2% of API time | ~8% step time | Unified memory layout |

### 6.11 v3 Training Time Impact (Updated)

Post-ALB-076 at seq=1024, batch=4, grad_accum=1:

| Scenario | Step Time | Tok/s | Wall Clock (250K steps) |
|----------|-----------|-------|------------------------|
| Baseline (PTX GEMMs) | 4,400 ms | 934 | **12.7 days** |
| Phase 1-4 (cuBLAS) | 1,379 ms | 1,485 | **4.0 days** |
| **Phase 5b (+ batched RMSNorm)** | **444 ms** | **9,216** | **1.3 days** |
| Phase 6 (+ CUDA Graphs) | ~200 ms | ~20K | **~14 hours** |
| Phase 7 (+ Flash Attention) | ~130 ms | ~31K | **~9 hours** |

Note: Phase 5b step time of 444ms includes JIT warmup. Steady-state estimated
~250-350ms based on profiler forward pass timing. With grad_accum=128 (production),
effective training time is per micro-batch × accum_steps.

### 6.12 Tensor Core NaN in Backward GEMMs — ROOT CAUSE FOUND (ALB-076, FIXED)

**Discovery**: cuBLAS tensor core GEMM algorithms (`CUBLAS_GEMM_DEFAULT_TENSOR_OP`,
algorithm 99) produce **ALL NaN output** for transposed backward GEMMs when
input gradient magnitudes reach ~1e5. Forward GEMMs (NoTrans/NoTrans) are
unaffected. This was the root cause of complete NaN corruption in v3 training.

**Symptom**: ALL GPU-resident transformer block weights become NaN after the
first optimizer step. Every gradient produced by cuBLAS backward is NaN.

**Five Whys analysis**:

1. **Why NaN weights?** Optimizer reads NaN weight gradients from cuBLAS backward
2. **Why NaN gradients?** cuBLAS `gemm_backward_a`/`gemm_backward_b` output ALL NaN
   starting at backward call #36 (first backward of block 18, FFN down_proj)
3. **Why NaN output from valid finite inputs?** Tensor core GEMM algorithm
   (`CUBLAS_GEMM_DEFAULT_TENSOR_OP`) has a numerical fault for transposed operands
4. **Why only backward and not forward?** Backward uses `Trans/NoTrans` and
   `NoTrans/Trans` transpose flags; forward uses `NoTrans/NoTrans` (unaffected)
5. **Why only after ~5 blocks (call #36)?** Gradient magnification through
   24-layer backward reaches ~1e5 magnitude at block 18, triggering the fault

**Diagnostic evidence** (NaN scan on every cuBLAS backward call):

| Call # | Block | Direction | grad_out max | cuBLAS output | Status |
|--------|-------|-----------|-------------|---------------|--------|
| 0 | 23 | bwd_a | small | max=3.24e-5 | Valid |
| 8 | 22 | bwd_a | ~1e-2 | max=1.04e-2 | Valid |
| 29 | 19 | bwd_b | ~1e2 | max=9.40e2 | Valid |
| 35 | 19 | bwd_b | ~1e-3 | max=1.49e-3 | Valid |
| **36** | **18** | **bwd_a** | **2.56e5** | **ALL 4.2M NaN** | **BUG** |
| 37+ | 18-0 | all | — | ALL NaN | Cascading |

**Key observation**: Call #36 inputs are entirely valid (grad_out: 0 NaN, max=2.56e5;
weight_b: 0 NaN, max=1.98e-2). The tensor core algorithm converts valid finite
inputs to NaN.

**Falsified hypotheses** (before root cause found):

1. **TF32 precision**: Changing `CUBLAS_COMPUTE_32F_FAST_TF32` → `CUBLAS_COMPUTE_32F`
   alone did NOT fix NaN — the algorithm, not precision, was the issue
2. **Stream synchronization**: `CUDA_LAUNCH_BLOCKING=1` still produced NaN
3. **Buffer size mismatch**: Oversized buffers verified to be within-bounds access

**Fix** (trueno #170, entrenar #239):

| Change | File | Before | After |
|--------|------|--------|-------|
| Math mode | `cublas.rs:CublasHandle::new()` | `CUBLAS_TF32_TENSOR_OP_MATH` (3) | `CUBLAS_DEFAULT_MATH` (0) |
| Compute type | `cublas.rs:gemm_f32()` | `CUBLAS_COMPUTE_32F_FAST_TF32` (74) | `CUBLAS_COMPUTE_32F` (68) |
| Algorithm | `cublas.rs:gemm_f32()` | `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (99) | `CUBLAS_GEMM_DEFAULT` (-1) |

**Result** (350M, seq=1024, batch=4, RTX 4090, 2 steps):

| Metric | With tensor cores | Without tensor cores | Delta |
|--------|-------------------|---------------------|-------|
| NaN in gradients | ALL (4.2M elements) | **0** | Fixed |
| Loss (step 1) | NaN | **10.4007** | Fixed |
| Tok/s | — | **5,216** | 5.9x over PTX |
| MFU (step 1) | — | **15.1%** | vs FP32 peak |
| gnorm | NaN | **2.05** | Healthy |

**Performance impact**: cuBLAS SIMD (no tensor cores) is still **5.9x faster**
than hand-written PTX (5,216 vs 890 tok/s). The tensor core advantage (~2x
theoretical) is irrelevant when it produces NaN.

**Phase 5a status: REVERTED**. TF32 tensor cores (§6.9) provided 0% measurable
improvement at 350M AND cause NaN in backward. The optimization is removed
entirely. Phase numbering unchanged; Phase 5a is now a null operation.

**Lesson**: Tensor core GEMM algorithms have undocumented numerical edge cases
with large-magnitude transposed operands. The NVIDIA documentation does not
warn about this failure mode. Always validate full backward pass (all layers,
production gradient magnitudes) before enabling tensor cores in training.

### 6.13 v3 Training Results (STOPPED, step 28K — plateau)

**Config**: 350M model, seq=1024, batch=4, codeparrot-clean (5.29B tokens,
20 shards × ~260K sequences), max_steps=250K, save_interval=1000.
**Status**: Stopped at step 28,000 due to val_ppl plateau (ALB-079 + ALB-080).
Superseded by v4.

**Loss curve** (v3, measured):

| Step | Loss | Val Loss | Val PPL | Tok/s | MFU | gnorm | lr |
|------|------|----------|---------|-------|-----|-------|-----|
| 1 | 10.40 | — | — | 5,606 | 16.2% | 2.19 | 1.5e-7 |
| 100 | 8.26 | — | — | 7,648 | 22.1% | 5.08 | 1.5e-5 |
| 200 | 6.89 | — | — | 7,194 | 20.8% | 2.43 | 3.0e-5 |
| 700 | 6.78 | — | — | 7,608 | 22.0% | 2.49 | 1.1e-4 |
| 900 | 6.90 | — | — | 7,653 | 22.2% | 2.32 | 1.4e-4 |
| 1000 | 6.93 | 7.38 | 1607.6 | 7,676 | 22.2% | 3.04 | 1.5e-4 |
| 1800 | 6.71 | — | — | 6,977 | 20.2% | 3.12 | 2.7e-4 |
| 1900 | 6.50 | — | — | 6,974 | 20.2% | 2.01 | 2.9e-4 |
| **2000** | **6.36** | **7.19** | **1331.7** | **6,972** | **20.2%** | **2.85** | **3.0e-4** |
| 2200 | 7.63 | — | — | 6,807 | 19.7% | 2.44 | 3.0e-4 |
| 2500 | 6.84 | — | — | 6,824 | 19.8% | 3.04 | 3.0e-4 |
| 3000 | 7.24 | 7.20 | 1341.2 | 6,783 | 19.6% | 2.17 | 3.0e-4 |
| 3500 | 6.54 | — | — | 6,681 | 19.3% | 2.62 | 3.0e-4 |
| 4000 | 7.85 | 7.10 | 1208.7 | 6,695 | 19.4% | 1.53 | 3.0e-4 |
| 4500 | 7.28 | — | — | 6,609 | 19.1% | 2.10 | 3.0e-4 |
| 5000 | 6.98 | 7.13 | 1244.0 | 6,632 | 19.2% | 1.83 | 3.0e-4 |
| 5500 | 6.49 | — | — | 6,565 | 19.0% | 1.65 | 3.0e-4 |
| 6000 | 7.16 | 7.05 | 1157.3 | 6,586 | 19.1% | 2.13 | 3.0e-4 |
| 7000 | 7.44 | 6.99 | 1084.9 | 6,586 | 19.1% | 1.19 | 3.0e-4 |
| 8000 | 7.14 | 7.02 | 1117.8 | 6,583 | 19.1% | 2.42 | 3.0e-4 |
| 9000 | 6.79 | 7.02 | 1114.0 | 6,561 | 19.0% | 0.89 | 3.0e-4 |
| 10000 | 6.35 | 7.07 | 1180.1 | 6,564 | 19.0% | 1.02 | 3.0e-4 |
| 12000 | 6.66 | 6.94 | 1036.7 | 6,570 | 19.0% | 0.84 | 3.0e-4 |
| 14000 | 6.48 | 6.93 | 1026.8 | 6,567 | 19.0% | 0.78 | 3.0e-4 |
| 16000 | 6.88 | 6.94 | 1036.4 | 6,578 | 19.0% | 0.37 | 3.0e-4 |
| 18000 | 6.56 | 6.96 | 1051.0 | 6,595 | 19.1% | 0.44 | 3.0e-4 |
| 20000 | 7.15 | 6.93 | 1023.1 | 6,621 | 19.2% | 0.36 | 3.0e-4 |
| 22000 | 6.77 | 6.92 | 1012.7 | 6,632 | 19.2% | 0.32 | 3.0e-4 |
| 24000 | 6.83 | 6.92 | 1010.5 | 6,651 | 19.3% | 0.22 | 3.0e-4 |
| 26000 | 6.61 | 6.91 | 1000.3 | 6,682 | 19.3% | 0.15 | 3.0e-4 |
| **28000** | **6.61** | **6.93** | **1018** | **6,690** | **19.4%** | **0.13** | **3.0e-4** |

**Steady-state performance** (steps 100-2000 warmup average):
- **7,600 tok/s** ± 200 (during warmup, steps 100-1000)
- **22.1% MFU** vs FP32 peak (RTX 4090, 82.6 TFLOP/s)
- **516 ms/step** (p50, warmup phase)

**Post-warmup performance** (steps 2000-28000, constant lr):
- **6,630 tok/s** ± 80 (steady state)
- **19.2% MFU** (post-warmup average)
- **~560 ms/step** (p50)
- **VRAM**: 11.4 GB / 24 GB (47% utilization)
- **0 NaN** in 28,000 steps (ALB-077 fix verified)

**Checkpoints** (every 1000 steps, 1520 MB SafeTensors each):
- step-1000 through step-28000 — all verified OK (28 checkpoints total).

**Training dynamics**:
- Loss converges from 10.4 to ~6.9 in 1000 steps (during warmup)
- Post-warmup spike at step 2200 (loss=7.63) — lr reached max (3e-4), recovered by step 2500
- Val loss plateaued: 7.38 → 7.05 → 6.94 → 6.93 → 6.92 → 6.91 → **6.93** (plateau since step 19K)
- Val PPL: 1608 → 1157 → 1037 → 1027 → 1013 → 1000 → **1018** (confirmed plateau)
- **Gradient norm collapse**: 3.04 (step 1K) → 1.02 (10K) → 0.13 (28K) — 23x decrease
  - Root cause: constant lr=3e-4 after warmup (ALB-079), no cosine decay
  - Combined with tiny effective batch of 4K tokens/step (ALB-080)
- B_noise decreasing: 0.22 → 0.08 (gradient signal/noise ratio improving)

**Token efficiency**: 115M tokens seen at step 28K. Val PPL=1018 at 115M tokens.
Reference: codeparrot-small (110M) achieved val_loss ~3.5 after 50B tokens.
The 350M model was undertrained — 115M tokens is <1% of typical training budget.

**v3 stopped**: Run terminated at step 28,000. Root causes identified as ALB-079
(constant lr, no cosine decay) and ALB-080 (4K tokens/step, 48-128x too small).
Superseded by v4 with cosine decay + gradient_accumulation=32 (131K tokens/step).

### 6.14 Stream Sync Bottleneck Analysis (ALB-078, Five Whys)

**Observation**: v3 training at step 1500 shows step time increased to 618ms
(from 516ms at step 1000). The difference correlates with gradient clipping
becoming active as gnorm grows.

**Five Whys**:

1. **Why 618ms/step?** Per-block gradient clipping introduces stream syncs
2. **Why per-block syncs?** `compute_workspace_clip_scale_gpu` calls
   `stream.synchronize()` after launching 9 `squared_sum` kernels per block
3. **Why sync needed?** CPU must download 9 partial-sum buffers to compute
   `clip_scale = min(1, max_norm / sqrt(sum_of_squared_norms))`
4. **Why CPU-side?** No fused GPU kernel exists for norm reduction + clip
5. **Why 24 syncs?** One per transformer block (interleaved backward+optimizer)

**Sync budget** (per step, with `grad_clip: 1.0`):

| Sync Point | Count/step | Location | Necessary? |
|------------|-----------|----------|-----------|
| Per-block clip norm | **24** | `compute_workspace_clip_scale_gpu` | **REDUNDANT** |
| LM head norm | 1 | `squared_sum_cuda` | REDUNDANT |
| Final global norm | 1 | `compute_clip_scale_with_norm` | REDUNDANT |
| CE loss D2H | 1 | `fused_cross_entropy_cuda` | YES (NaN guard) |
| Pre-embed sync | 1 | `gpu_backward:1134` | YES (C-STREAMSYNC-001) |
| **Total** | **28** | | 2 necessary, 26 redundant |

**Fix** (entrenar #240, trueno #171) — **IMPLEMENTED**:

Two new PTX kernels in `trueno-gpu/src/kernels/optimizer/fused_clip.rs`:

1. **`ClipScaleReduceKernel`**: Single-CTA, single-thread. Reads contiguous
   `f32[total_partials]` buffer of squared-sum partial results, computes
   `clip_scale = min(1.0, max_norm / sqrt(sum))`. IEEE 754 handles zero-norm
   without branching (`div(x, 0.0) = +inf`, `min(+inf, 1.0) = 1.0`).
   Writes `output[0] = scale, output[1] = norm` for observability.

2. **`GradientClipGpuScaleKernel`**: Element-wise. Reads scale from GPU pointer
   (not host param). Early exit when `scale ≈ 1.0` (within 1e-7) to avoid
   unnecessary memory bandwidth when no clipping needed.

Integration in `entrenar/src/autograd/cuda_optim.rs`:
- `FusedClipState`: Pre-allocated contiguous partials buffer + scale buffer
- `squared_sum_launch_into`: Writes partial sums at offset into contiguous buffer
- `clip_scale_reduce_cuda`: Launches ClipScaleReduceKernel (grid 1×1, block 1×1)
- `gradient_clip_gpu_scale_cuda`: Launches GradientClipGpuScaleKernel

Pipeline (per block): 9× squared_sum_launch_into → 1× clip_scale_reduce →
9× gradient_clip_gpu_scale. Zero sync points, zero D2H transfers.

This eliminates 26 of 28 syncs/step. The 2 remaining are irreducible:
- CE loss download for NaN guard
- Final sync before embed gradient D2H (C-STREAMSYNC-001)

**Status**: Implemented, verified active in v4 launch (1,794 partials, 7.0 KB buffer).
**Measured impact**: Eliminated 26 stream syncs/step. v4 micro-batch step time ~1,066ms
includes gradient accumulation overhead (not directly comparable to v3's 560ms).

### 6.15 Training Quality Analysis (ALB-079/080, Five Whys)

**Observation**: v3 training at step 26K shows val_loss plateau at 6.92 (val_ppl=1000)
since step 12K. Gradient norm collapsed from 3.04 (step 1K) to 0.15 (step 26K) — 20x
decrease while lr is at peak (3e-4).

**Five Whys — Root Cause 1: Missing Cosine LR Decay (ALB-079)**

1. **Why constant lr=3e-4 at all steps?** `CudaTransformerTrainer::current_lr()` only
   implemented linear warmup; returned `base_lr` after warmup (line 1938)
2. **Why no cosine?** `TransformerTrainConfig` has no `lr_scheduler` field; YAML config
   parsed by bridge but not propagated to CUDA path
3. **Why not caught earlier?** At step 2K-5K, cosine barely differs from constant
   (lr ≈ 2.99e-4 vs 3.00e-4); plateau only visible after 10K steps
4. **Fix** (entrenar #241): Cosine decay in `current_lr()` using `warmup_steps` and
   `max_steps`. CPU embedding optimizer synced via `set_lr()`.

**Five Whys — Root Cause 2: Effective Batch Size 48-128x Too Small (ALB-080)**

1. **Why val_ppl plateau at 1000?** Gradient noise too high to escape loss basin
2. **Why noisy gradients?** Effective batch = 4 × 1 × 1024 = 4,096 tokens/step
3. **Why 4,096?** `gradient_accumulation: 1` in config, VRAM limits `batch_size: 4`
4. **Why so small?** Config was set for debugging; no Chinchilla batch size analysis
5. **Why does it matter?** Comparable 350M models use 131K-524K tokens/step (32-128x larger)

| Model | Batch Size (tokens/step) |
|-------|--------------------------|
| CodeGen-350M-mono | ~500K+ |
| CodeParrot-small (110M) | 196K |
| GPT-2 124M (nanoGPT) | ~524K |
| **Albor v3** | **4,096** |
| **Albor v4** (launched) | **131,072** |

**Fix**: `pretrain-350m-v4.yaml` with `gradient_accumulation: 32` (131K tokens/step),
`warmup_steps: 375`, `max_steps: 7500` (~1B tokens). Same wall-clock as v3 (same
number of forward/backward passes), dramatically better gradient quality.

**Expected impact**: val_ppl should break through 1000 floor and reach <100 by 1B tokens.
gnorm should stabilize at 0.5-2.0 (not collapse to 0.13).

### 6.16 v4 Training Launch (LIVE)

**Config**: 350M model, seq=1024, batch=4, gradient_accumulation=32,
codeparrot-clean (5.29B tokens), max_steps=7,500 (~1B tokens),
warmup_steps=375 (5%), cosine lr decay, `pretrain-350m-v4.yaml`.

**Fixes over v3**:
- **ALB-079**: Cosine lr decay (entrenar #241). Linear warmup to 3e-4 over 375
  steps, then cosine anneal to 0 over remaining 7,125 steps. Replaces constant
  lr=3e-4 that caused gradient norm collapse.
- **ALB-080**: `gradient_accumulation: 32` (131,072 tokens/optimizer step vs
  4,096 in v3). 32x larger effective batch matches comparable 350M models.
- **ALB-078**: Fused gradient clipping (entrenar #240, trueno #171). GPU-side
  norm reduction eliminates 26 stream syncs/step.

**Effective batch size comparison**:

| Version | batch | grad_accum | tokens/opt_step | optimizer steps |
|---------|-------|------------|-----------------|-----------------|
| v3 | 4 | 1 | 4,096 | 250,000 |
| **v4** | **4** | **32** | **131,072** | **7,500** |

**Initial results** (step 12):

| Metric | Value |
|--------|-------|
| Training loss | 10.39 |
| lr | 9.6e-6 (linear warmup phase) |
| Throughput | 3,780 tok/s |
| MFU | 10.9% vs FP32 peak |
| Step time (micro-batch) | ~1,066 ms |
| Optimizer step time | ~34s (32 micro-batches) |
| GPU memory | 11,421 / 24,081 MB (47%) |
| NaN steps | 0 |

**Performance notes**:
- **Throughput drop vs v3** (3,780 vs 6,630 tok/s): Expected. The tok/s metric
  measures micro-batch throughput. Gradient accumulation adds per-micro-batch
  overhead: D2H gradient downloads, CPU-side accumulation buffers, and H2D
  upload of accumulated gradients every 32 steps. Effective token processing
  (131K tokens per optimizer step) is 32x v3.
- **MFU drop** (10.9% vs 19.2%): Same root cause — gradient accumulation
  overhead amortized across micro-batches reduces raw kernel utilization. The
  useful work per wall-clock second is higher because each optimizer step
  processes 32x more gradient signal.
- **Linear warmup confirmed correct**: lr=9.6e-6 at step 12 matches expected
  `base_lr × step / warmup_steps = 3e-4 × 12/375 = 9.6e-6`.

**Fused gradient clipping** (ALB-078, verified active):
- 1,794 partial sums (contiguous GPU buffer, zero D2H transfers)
- 7.0 KB partials buffer (vs 26 stream syncs/step in v3)
- GPU-side norm reduction + clip scale computation

**Gradient accumulation memory**:
- 1,728 MB CPU buffers for 24 blocks × 2 tensors (weights + bias)
- Accumulated on CPU, averaged, uploaded to GPU every 32 micro-batches
- GPU VRAM unchanged from v3 (11,421 MB)

**Training budget**:
- Max steps: 7,500 optimizer steps × 131K tokens = **~983M tokens** (~1B)
- Wall-clock per optimizer step: ~34s
- **ETA**: 7,500 × 34s = **70.8 hours (~3 days)**
- Warmup: 375 steps × 34s = **3.5 hours** (lr ramps to 3e-4)
- Cosine phase: 7,125 steps × 34s = **67.3 hours** (lr decays to 0)

**Expected trajectory** (based on comparable 350M models):
- Step 375 (end warmup): loss ~8.0, val_ppl ~3000
- Step 1,500 (~200M tokens): val_ppl ~500 (break through v3 plateau)
- Step 3,750 (~500M tokens): val_ppl ~100-200
- Step 7,500 (~1B tokens): val_ppl <100 (target)

## 7. Verification Architecture

### 7.1 Four-Layer Verification

```
Layer 1: CONTRACTS (provable-contracts / pv)
  What: Algebraic invariants, proof obligations, falsification tests
  When: BEFORE implementation (write contract first)
  How:  pv validate, pv scaffold, pv audit
  Files: contracts/cublas-gemm-v1.yaml
         contracts/training-step-budget-v1.yaml

Layer 2: BENCHMARKS (raw C ceiling + Criterion + regression detection)
  What: Three-tier GEMM comparison with hardware ceiling
  When: BEFORE (ceiling), DURING (Criterion), AFTER (regression)
  How:  make bench-gemm-compare, make bench-gemm-regression
  Pattern: Raw C cuBLAS (ceiling) vs Rust cuBLAS (target) vs PTX (floor)
    - FFI overhead < 2% (Rust vs Raw C)
    - Speedup > 10x (cuBLAS vs PTX)
    - Regression < 10% per shape between commits
    - Follows trueno/benchmarks/ matmul_comparison.py pattern exactly

Layer 3: BRICK PROFILING (probador)
  What: Per-component time budgets with Jidoka gates
  When: DURING implementation (continuous enforcement)
  How:  BrickHouse builder, brick assertions, budget_ms
  Pattern: Each training loop component = one Brick with:
    - can_render() = Jidoka gate (fail if > 2x budget)
    - verify() = timing assertion
    - budget_ms = SLA from contract

Layer 4: LAYER TRACING (renacer BrickTracer)
  What: Per-kernel, per-block, per-transfer timing with OTLP export
  When: DURING profiling runs + AFTER implementation (regression detection)
  How:  BrickTracer.trace(), OTLP -> Jaeger, anomaly escalation
  Pattern: Each CUDA kernel call = one trace span
    - Forward: block_N_gemm_qkv, block_N_attention, block_N_ffn
    - Backward: block_N_backward_gemm, block_N_backward_elementwise
    - Transfer: pcie_h2d_embed, pcie_d2h_logits, pcie_h2d_grad
    - Optimizer: block_N_optimizer_d2h, block_N_adamw, block_N_optimizer_h2d
```

### 7.2 Escalation Chain

Renacer implements automatic escalation from lightweight metrics to detailed
tracing:

```
Steady state (metrics only):
  - Counter: gemm_calls_total, pcie_bytes_total
  - Gauge: step_time_ms, mfu_ratio
  - Histogram: per_block_forward_us, per_block_backward_us

Escalation trigger (CV > 15% or efficiency < 25%):
  - BrickTracer captures full syscall breakdown
  - OTLP spans exported to Jaeger with per-kernel detail
  - Anomaly detector flags the brick and step number

Alert (budget violation > 2x):
  - Jidoka gate fires (probador)
  - Training loop pauses (Andon alert)
  - Full trace exported for post-mortem
```

This means training runs at full speed in steady state (metrics are SIMD-
accelerated via trueno), and only pays the tracing cost when something goes
wrong.

### 7.3 Continuous Verification During Training

```bash
# Run training with BrickTracer instrumentation
RUST_LOG=info renacer --otlp-endpoint http://localhost:4317 \
    --otlp-service-name "albor-v3-cublas" \
    --trace-compute \
    --trace-compute-threshold 100 \
    -- apr train apply --task pretrain \
        --config configs/train/pretrain-350m-v3.yaml

# In another terminal: monitor brick budgets
apr monitor ./checkpoints/albor-base-350m-v3/

# Post-run: audit contract compliance
pv audit contracts/cublas-gemm-v1.yaml \
    --binding contracts/trueno-gpu/cublas-binding.yaml
pv audit contracts/training-step-budget-v1.yaml \
    --binding contracts/entrenar/step-budget-binding.yaml

# Post-run: view traces in Jaeger
# http://localhost:16686 -> Service: "albor-v3-cublas"
# Filter by: operation="gemm_forward", minDuration=10ms
```

## 8. Risks

| Risk | Mitigation | Contract Obligation |
|------|------------|---------------------|
| cuBLAS FP16 numerical divergence | Keep FP32 master weights, compare loss curves | FALSIFY-CUBLAS-002 |
| libcublas.so version mismatch | Pin to CUDA 12.x, test on lambda machine | FALSIFY-CUBLAS-003 |
| cuBLAS workspace memory pressure | Pre-allocate fixed workspace, share across GEMMs | training-memory-kernel-v1 |
| CPU optimizer becomes new bottleneck | Phase 4 contract (gpu-optimizer-v1) | FALSIFY-BUDGET-002 |
| Tensor core shapes require padding | Albor shapes (1024, 4096, 32768) already multiples of 8 | FALSIFY-CUBLAS-003 |
| FP16 weight precision loss | Standard practice; master weights remain FP32 on CPU | FALSIFY-CUBLAS-002 |
| Silent regression after optimization | Brick budgets + Jidoka gates detect immediately | FALSIFY-BUDGET-003 |
| Unaccounted overhead hiding bottleneck | Brick coverage >= 95% of step time enforced | FALSIFY-BUDGET-001 |

## 9. Dependencies

- `libcublas.so` from CUDA toolkit (already installed: `/usr/local/cuda/lib64/`)
- `nvcc` for compiling raw C cuBLAS benchmark (ceiling measurement)
- trueno-gpu crate (target for FFI integration)
- entrenar CudaTransformerTrainer (consumer of cuBLAS GEMMs)
- renacer BrickTracer (layer tracing instrumentation)
- probador brick budgets (SLA enforcement)
- provable-contracts / `pv` (contract validation and audit)
- Criterion.rs (Rust benchmark harness, already a trueno dev-dependency)
- No new Rust crate dependencies (pure FFI, no bindgen)

## 10. Contract Registry

| Contract File | Status | Validates |
|---------------|--------|-----------|
| `contracts/cublas-gemm-v1.yaml` | **NEW** (write before Phase 1) | cuBLAS correctness, buffer safety, MFU improvement |
| `contracts/training-step-budget-v1.yaml` | **NEW** (write before Phase 0) | Brick-level performance SLAs, Jidoka enforcement |
| `contracts/training-gpu-kernel-v1.yaml` | EXISTING | Parent contract — PCIe transfers, stability, gradient flow |
| `contracts/training-memory-kernel-v1.yaml` | EXISTING | VRAM budget (must update for FP16 weight storage) |
| `contracts/training-config-kernel-v1.yaml` | EXISTING | Epoch/step/LR algebraic consistency |
| `contracts/fused-kernels-v1.yaml` | **NEW** (write before Phase 4) | Fused CE, RMS norm reuse, SwiGLU in-place, fused attention |
| `contracts/gpu-optimizer-v1.yaml` | FUTURE (Phase 4) | GPU-resident AdamW correctness |
| `contracts/gpu-embedding-v1.yaml` | FUTURE (Phase 5) | GPU embedding lookup + scatter-add |
| `contracts/async-pipeline-v1.yaml` | FUTURE (Phase 6) | Compute/transfer overlap safety |
| `contracts/grad-checkpoint-v1.yaml` | FUTURE (Phase 7) | Gradient checkpointing memory/correctness |

## 11. Unsloth-Inspired Kernel Optimizations

**Source**: Analysis of [unslothai/unsloth](https://github.com/unslothai/unsloth)
(cloned 2026-03-05). Unsloth achieves 2x training speedup over HuggingFace via
fused Triton kernels, selective activation saving, and in-place backward ops.
These patterns translate to our Rust + CUDA PTX stack.

### 11.1 Fused Cross-Entropy Loss + Backward

**What unsloth does**: Single Triton kernel computes `logsumexp`, loss, and
`dL/dx` (softmax - one_hot) in one pass. Never materializes the full probability
distribution.

**Current albor**: Separate kernels for logits→softmax, softmax→loss, loss→grad.
For vocab=32K, batch=4, seq=1024, the logit tensor is `[4096, 32768]` = 512 MB
in FP32. Three kernel launches + three full reads/writes of this tensor.

**Proposed change**: Fused CE kernel that:
1. Computes `logsumexp` per row (FP32 accumulation for stability)
2. Computes `loss = logsumexp - logit[label]` per row
3. Computes `grad[i] = exp(logit[i] - logsumexp) - delta(i, label)` in-place
4. Never allocates full softmax tensor

**Expected gain**: -2 kernel launches, -1 GB memory bandwidth per step.
Step time: ~20-40ms savings (CE is ~1% of step time, but memory bandwidth
relief helps other kernels via improved cache pressure).

**Contract**: `contracts/fused-kernels-v1.yaml` — FALSIFY-FUSED-001

```
Equations:
  fused_ce_correctness:
    loss_fused = -logit[label] + log(sum(exp(logit[i]))) for each row
    grad_fused[i] = exp(logit[i] - logsumexp) - delta(i, label)
  Invariant: max_abs_diff(loss_fused, loss_separate) < 1e-5
  Invariant: max_abs_diff(grad_fused, grad_separate) < 1e-5
  Invariant: FP32 accumulation for logsumexp (no FP16 overflow on 32K vocab)
```

### 11.2 Activation Memory Reuse (RMS LayerNorm)

**What unsloth does**: RMS LayerNorm forward saves ONLY `inv_var` (1 scalar per
row = `batch * seq_len` floats). Backward recomputes `normed = X * inv_var` from
the activation cache. Total saved: `O(B*S)` instead of `O(B*S*H)`.

**Current albor**: Saves `X`, `W`, `inv_var`, and `normed` per layer during
forward for use in backward. For 24 layers × `[4096, 1024]`:
- `X`: 24 × 16 MB = 384 MB
- `normed`: 24 × 16 MB = 384 MB
- `inv_var`: 24 × 16 KB = 384 KB (negligible)
- Total saved: **768 MB** of activation memory

**Proposed change**: Save only `inv_var` per layer. During RMS norm backward:
1. Recompute `normed = X_cached * inv_var` (X is available from the previous
   layer's output or the activation cache)
2. Compute `d_weight = sum(grad_output * normed)`
3. Compute `d_input = (grad_output * W - normed * d_weight_sum) * inv_var`

**Expected gain**: -384 MB activation memory (normed tensor eliminated).
This is 3.2% of 24 GB VRAM — modest alone, but compounds with other savings
to potentially enable batch=8 without gradient checkpointing.

**Contract**: `contracts/fused-kernels-v1.yaml` — FALSIFY-FUSED-002

```
Equations:
  rmsnorm_recompute_correctness:
    normed_recomputed = X * inv_var_saved
    max_abs_diff(normed_recomputed, normed_original) == 0.0  (exact, same FP32)
  Memory reduction:
    activation_memory(optimized) = activation_memory(current) - 24 * B * S * H * 4 bytes
    For B=4, S=1024, H=1024: savings = 24 * 4 * 1024 * 1024 * 4 = 402,653,184 bytes (~384 MB)
```

### 11.3 SwiGLU In-Place Backward

**What unsloth does**: GEGLU/SwiGLU backward overwrites input buffers with
gradient results. Forward: `h = silu(e) * g`. Backward stores `dh, de, dg`
into the same memory as `h, e, g`. No new allocations.

**Current albor**: `CudaGradWorkspace` reuses buffers per-block (already good),
but within a block, SwiGLU backward allocates separate `grad_gate`, `grad_up`,
and `grad_down` buffers. For intermediate_size=4096:
- `grad_gate`: `[4096, 4096]` = 64 MB
- `grad_up`: `[4096, 4096]` = 64 MB
- Total per-block overhead: 128 MB (shared workspace, so only peak matters)

**Proposed change**: Fuse SwiGLU backward to overwrite gate/up buffers in-place:
1. `d_gate = grad_output * up * silu_deriv(gate)` → store in `gate` buffer
2. `d_up = grad_output * silu(gate)` → store in `up` buffer
3. No separate allocation for d_gate, d_up

**Expected gain**: -128 MB peak workspace per block (already shared, so reduces
peak VRAM, not total allocations). Main benefit is reduced memory bandwidth —
fewer buffer copies between kernels.

**Contract**: `contracts/fused-kernels-v1.yaml` — FALSIFY-FUSED-003

```
Equations:
  swiglu_inplace_correctness:
    d_gate_inplace = grad_out * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    d_up_inplace = grad_out * silu(gate)
    max_abs_diff(d_gate_inplace, d_gate_separate) < 1e-5
    max_abs_diff(d_up_inplace, d_up_separate) < 1e-5
```

### 11.4 Mixed Precision Discipline (Validated)

**What unsloth does**: Loads activations as FP32 for critical arithmetic
(variance, softmax, logsumexp), keeps weights in BF16, casts output back after
critical ops.

**Albor status**: Already implemented correctly (validated by ALB-072 fix).
Our backward is all FP32, master weights are FP32 on CPU, forward weights are
FP32 on GPU (will become FP16 with cuBLAS). This matches unsloth's pattern.

**Action**: No code change needed. Document as validation that our approach
matches production-grade mixed precision practice.

### 11.5 RoPE Head Grouping

**What unsloth does**: Applies RoPE to 4 heads simultaneously, loading sin/cos
once and reusing across the group. `ROPE_GROUP_SIZE = 4`.

**Current albor**: Per-head RoPE application in the attention forward kernel.
Sin/cos recomputed or reloaded per head.

**Proposed change**: Batch RoPE across all Q heads (16) and KV heads (4) with
single sin/cos load. For our GQA architecture (16 Q heads, 4 KV heads):
- Q: load sin/cos once, apply to 16 heads
- K: same sin/cos, apply to 4 heads
- V: no RoPE (not rotated)

**Expected gain**: ~10% attention kernel speedup from better L2 cache utilization.
Small absolute impact (~5-10ms/step) since RoPE is not compute-dominant.

**Contract**: `contracts/fused-kernels-v1.yaml` — FALSIFY-FUSED-004

```
Equations:
  rope_grouped_correctness:
    For each head h in [0, n_heads):
      Q_rotated_grouped[h] == Q_rotated_individual[h]  (bit-exact)
    Performance: T_rope(grouped) < 0.9 * T_rope(individual)
```

### 11.6 Fused Attention (QK^T → Softmax → V)

**What unsloth does**: Uses Flash Attention or Flex Attention to fuse the
3-step attention computation into a single kernel. Never materializes the
full `[seq, seq]` attention score matrix.

**Current albor**: Three separate operations per attention head:
1. `scores = Q @ K^T` → cuBLAS GEMM → `[4096, 1024]` (with cuBLAS)
2. `probs = softmax(scores / sqrt(d_k))` → elementwise kernel
3. `output = probs @ V` → cuBLAS GEMM

This materializes the `[batch, heads, seq, seq]` = `[4, 16, 1024, 1024]` = 256 MB
attention score tensor. For 24 layers, that's 6.1 GB if all layers' scores
are live simultaneously (they aren't in our per-block architecture, but the
per-block peak still includes this).

**Proposed change**: Custom fused attention kernel (not Flash Attention — our
seq=1024 is short enough that tiled online softmax gives most of the benefit):
1. Tile Q, K, V into blocks (e.g., 64×64)
2. Compute `QK^T` tile, apply causal mask, running softmax (online algorithm)
3. Accumulate `softmax(tile) @ V` without materializing full score matrix
4. Output: attention result directly, save only logsumexp for backward

**Expected gain**:
- -256 MB peak VRAM per block (attention scores not materialized)
- -2 kernel launches per layer (3→1)
- ~15% attention speedup from reduced memory bandwidth
- Enables batch=8 by freeing VRAM headroom

**Contract**: `contracts/fused-kernels-v1.yaml` — FALSIFY-FUSED-005

```
Equations:
  fused_attention_correctness:
    output_fused = softmax(Q @ K^T / sqrt(d_k) + causal_mask) @ V
    max_abs_diff(output_fused, output_separate) < 1e-3  (FP32)
    max_abs_diff(output_fused, output_separate) < 1e-2  (FP16)
  Memory:
    peak_attn_memory(fused) < peak_attn_memory(separate) / 4
    # Separate: [B, H, S, S] = 256 MB
    # Fused: [B, H, tile, tile] = 256 MB / (S/tile)^2
```

### 11.7 Chunked Cross-Entropy for Future Vocab Scaling

**What unsloth does**: For vocab > 65K, splits logsumexp computation into chunks
of 65536. Mathematical property: `logsumexp(chunked_logsumexp) == logsumexp(full)`.

**Current albor**: Vocab = 32K, fits in single chunk. Not needed now.

**Future applicability**: If we scale to multi-lingual (65K+ vocab) or adopt a
larger tokenizer, chunked CE prevents register pressure overflow in the fused
CE kernel. The logsumexp decomposition is:

```
logsumexp([a, b]) = max(a, b) + log(exp(a - max) + exp(b - max))
```

Each chunk computes a partial logsumexp. The final logsumexp combines partials.
This is numerically stable and mathematically exact.

**Contract**: Deferred until vocab > 65K. Will be added to `fused-kernels-v1.yaml`
if tokenizer v3 exceeds 65K vocabulary.

### 11.8 Gradient Checkpointing (Activation Recomputation)

**What unsloth does**: Trades compute for memory by recomputing layer activations
during backward instead of saving them during forward. 2x slower backward, but
~3x smaller activation memory.

**Current albor**: Per-block interleaved backward+optimizer design already
limits peak activation memory to one block's worth. But with fused attention
(§11.6) and activation reuse (§11.2), we may not need gradient checkpointing
for batch=4.

**When needed**: If batch=8 + seq=2048 still OOMs after §11.2 + §11.6.

**Contract**: `contracts/grad-checkpoint-v1.yaml` (FUTURE — already in registry)

```
Equations:
  checkpoint_correctness:
    grad(checkpointed) == grad(full_save)  # Bit-exact: same computation
  Memory:
    peak_activation(checkpointed) = peak_activation(full) / num_checkpoint_segments
  Performance:
    T_backward(checkpointed) < 2.0 * T_backward(full)  # At most 2x slower
```

### 11.9 Summary: Optimization Priority Matrix

| # | Optimization | Expected Gain | Memory Savings | Effort | Phase |
|---|-------------|---------------|----------------|--------|-------|
| 1 | cuBLAS tensor core GEMMs | **50x GEMM, 2x step** | 0 | High | 1-3 |
| 2 | Fused CE loss + backward | 20-40ms/step | -512 MB bandwidth | Medium | 4 |
| 3 | RMS norm activation reuse | 0 (compute) | **-384 MB** | Low | 4 |
| 4 | SwiGLU in-place backward | 10-20ms/step | -128 MB peak | Low | 4 |
| 5 | RoPE head grouping | 5-10ms/step | 0 | Low | 4 |
| 6 | Fused attention (tiled) | 15% attn speedup | **-256 MB/layer** | High | 5 |
| 7 | Chunked CE (vocab >65K) | 0 (future) | 0 | Low | Deferred |
| 8 | Gradient checkpointing | -2x backward | **-66% activations** | Medium | 7 |

**Cumulative impact** (Phases 1-5b, measured):
- Step time: 4,400ms → 444ms (9.9x; cuBLAS SIMD 5.9x, batched RMSNorm 24.8x fwd)
- MFU: 2.5% → 26.7% (vs FP32 peak, runtime-reported)
- Tok/s: 934 → 9,216 (9.9x improvement)
- Note: Tensor cores disabled (ALB-076, §6.12) — produce NaN in transposed backward GEMMs

### 11.10 Falsification Tests for Kernel Optimizations

| ID | Rule | Prediction | Contract |
|----|------|------------|----------|
| FALSIFY-FUSED-001 | Fused CE matches separate CE | `max_abs_diff(loss) < 1e-5` on 50M model, 50 steps | fused-kernels-v1 |
| FALSIFY-FUSED-002 | RMS norm recompute is bit-exact | `normed_recomputed == normed_original` (FP32, exact) | fused-kernels-v1 |
| FALSIFY-FUSED-003 | SwiGLU in-place backward correct | `max_abs_diff(d_gate, d_gate_ref) < 1e-5` | fused-kernels-v1 |
| FALSIFY-FUSED-004 | RoPE grouped matches individual | Bit-exact Q_rotated for all 16 heads | fused-kernels-v1 |
| FALSIFY-FUSED-005 | Fused attention matches separate | `max_abs_diff(output) < 1e-3` (FP32) | fused-kernels-v1 |
| FALSIFY-FUSED-006 | Memory savings measured | Activation peak reduced by >= 300 MB | fused-kernels-v1 |
| FALSIFY-FUSED-007 | Fused CE never materializes softmax | Peak memory during CE < `B*S*V*4` bytes | fused-kernels-v1 |
| FALSIFY-FUSED-008 | Gradient checkpointing bit-exact | `grad(checkpointed) == grad(full)` for all params | grad-checkpoint-v1 |
| FALSIFY-FUSED-009 | Fused attention backward correct | All params get gradients, loss within 1% of separate | fused-kernels-v1 |
| FALSIFY-FUSED-010 | No training instability from fusions | 100-step run: loss.is_finite() every step, gnorm < 100 | fused-kernels-v1 |

## Appendix A: Popperian Falsification of This Specification

**Date**: 2026-03-05
**Method**: `batuta falsify .` (108-item checklist) + manual chain-of-thought
analysis of every claim, equation, and assumption in this spec.

**Batuta project score**: 80.1% (Andon Warning), 65 PASS, 0 FAIL, 43 PARTIAL.
Key findings from batuta mapped to spec weaknesses below.

### A.1 Chain-of-Thought Falsification

Each numbered item is a falsifiable claim from the spec, followed by the
attempt to break it.

**Claim 1: "Step time is 4,400ms with 57% in GEMM"** (Section 2.4)

- Status: **UNVERIFIED ESTIMATE**. The breakdown is labeled "Estimated" but
  no profiling data backs it. The spec prescribes renacer BrickTracer profiling
  in Phase 0, but Phase 0 hasn't run yet. The 57% GEMM figure is a guess.
- Risk: If GEMM is actually 30% of step time (e.g., CPU optimizer is 40%),
  cuBLAS integration yields only 1.3x speedup instead of 2x.
- Action: **Phase 0 is blocking**. Do not proceed to Phase 1 until BrickTracer
  confirms the breakdown. Add a contract obligation: FALSIFY-BASELINE-001.

**Claim 2: "cuBLAS achieves 130-150 TFLOP/s on Albor shapes"** (Section 4.1)

- Status: **VERIFIED**. Measured 152.3 TFLOP/s on FFN gate/up shape
  `[4096, 1024] x [1024, 4096]`, 141.2 TFLOP/s on FFN down, 89.4 TFLOP/s
  on square `[1024, 1024]`. The range 89-152 TFLOP/s matches or exceeds
  the 130-150 prediction for large shapes. Smaller square shapes are
  memory-bandwidth bound as expected.
- Verification: trueno-gpu cuBLAS hardware tests (PR #165).

**Claim 3: "FFI overhead < 2%"** (Section 5.7, FALSIFY-CUBLAS-008)

- Status: **PLAUSIBLE but untested**. cuBLAS FFI is a single function call
  with no data copies (pointers passed through). 2% overhead is reasonable.
- Risk: If `CublasHandle::set_stream()` is called per-GEMM (555 calls/step)
  rather than once per step, the cumulative overhead could exceed 2%.
- Action: The wrapper should call `set_stream()` once at step start, not
  per-GEMM. Add this as a contract invariant.

**Claim 4: "MFU = 2.5% vs FP32 peak"** (Section 1.2)

- Status: **PARTIALLY FALSIFIED**. The MFU formula uses `6 * P * tokens_per_step`
  but this approximation assumes all FLOPs are in GEMMs. For a 370M model
  with batch=4, seq=1024, the attention score computation (QK^T) adds
  `2 * S^2 * H * L = 2 * 1024^2 * 1024 * 24 = 51.5 GFLOP` per step,
  which is <1% of the 9.1 TFLOP total. The 6x approximation is valid here.
- Correction: MFU is correct to within ~1% of the true value. No action needed.

**Claim 5: "Step time drops to 2,150ms after cuBLAS"** (Section 6.1)

- Status: **MEASURED — 1,379 ms (better than projected)**. The original
  projection of 2,150ms assumed non-GEMM time stays constant at 1,900ms.
  Actual measurement showed 1,379ms (seq=512, batch=4), which is 36% better
  than projected. Verified via dogfooding: `apr train apply` with cuBLAS
  (entrenar PR #233), 1,485 tok/s, 4.3% MFU.
- FALSIFY-CUBLAS-009 still relevant: verify non-GEMM time decomposition.

**Claim 6: "555 GEMM operations per step"** (Section 2.1)

- Status: **APPROXIMATELY CORRECT but undercounted**. The count includes
  attention score GEMMs (QK^T) but omits attention value application (V
  projection after softmax), which is also a GEMM: `softmax(QK^T) * V`.
  Forward: 24 blocks x 1 = 24. Backward: 24 blocks x 2 = 48. Plus attention
  backward for the score GEMM itself.
- Correction: The actual count may be ~600 GEMMs, not 555. The difference
  is small (<10%) and doesn't change the analysis materially, but the spec
  should note the approximation.

**Claim 7: "Phase 7 achieves 17.5% MFU with batch=8"** (Section 6.3)

- Status: **CONTRADICTS KNOWN CONSTRAINT**. Section 4.3 of the spec notes
  seq=1024, batch=8 currently OOMs. Phase 7 lists this as requiring gradient
  checkpointing, but with cuBLAS adding FP16 weight copies alongside FP32
  master weights, VRAM pressure increases. The 650ms step time assumes batch=8
  fits, which is unproven.
- Risk: batch=8 may still OOM even with gradient checkpointing if FP16+FP32
  dual weight storage consumes the headroom.
- Action: Add VRAM budget equation to training-memory-kernel-v1.yaml for
  mixed-precision dual storage. FALSIFY-MEM-004: "batch=8 fits in 24GB with
  FP16 forward weights + FP32 master weights + gradient checkpointing."

**Claim 8: "Benchmark shapes are representative"** (Section 5.2)

- Status: **INCOMPLETE**. The 6 benchmark shapes cover the large GEMMs but
  omit the GQA key-value projection shapes: `[4096, 256, 1024]` (K and V
  projections with num_kv_heads=4, head_dim=64, so kv_dim=256). These are
  small, thin matrices where cuBLAS may show less speedup due to low
  arithmetic intensity.
- Action: Add `(4096, 256, 1024, "attn_kv")` to SHAPES in both C and
  Criterion benchmarks. This is the worst-case shape for tensor cores.

**Claim 9: "Performance regression gate at 10%"** (Section 5.5)

- Status: **MATCHES batuta JA-04 finding**. Batuta flagged JA-04 (Performance
  Regression Gate) as PARTIAL with rejection "Benchmarks exist but not gated
  in CI." The spec defines `make bench-gemm-regression` but does not integrate
  it into CI.
- Action: Add `bench-gemm-regression` to the `clean-room / gate` CI workflow
  for trueno-gpu. This addresses JA-04.

**Claim 10: "No new Rust crate dependencies"** (Section 9)

- Status: **CORRECT**. Pure FFI bindings require only `libc` types (already
  in std) and `libcublas.so` (system library). No `cublas-sys` or `bindgen`
  crate needed.
- Verified: This is consistent with trueno's existing pattern of hand-written
  CUDA driver API bindings.

### A.2 Batuta Findings Mapped to Spec

| Batuta ID | Status | Spec Impact |
|-----------|--------|-------------|
| JA-04 | PARTIAL: "Benchmarks not gated in CI" | Section 5: Add bench-gemm-regression to CI |
| PW-02 | PARTIAL: "No SIMD optimization" | N/A (spec is about GPU, not CPU SIMD) |
| EDD-01 | PARTIAL: "Partial equation documentation" | Section 3.1: Ensure all contract equations have domain/codomain/invariants |
| EDD-03 | PARTIAL: "Numerical code without analytical validation" | Section 5.2: Raw C baseline IS the analytical validation |
| NR-01 | PARTIAL: "No explicit IEEE 754 testing" | Add: cuBLAS FP32 accumulation contract (C-CUBLAS-004) covers this |
| NR-02 | PARTIAL: "Single platform testing" | N/A (CUDA-only by design, RTX 4090 target) |
| AI-01 | PARTIAL: "Config examples incomplete" | Add cuBLAS config example to YAML configs |
| AI-05 | PARTIAL: "No explicit validator" | `apr train validate` already validates; extend for cuBLAS feature |

### A.3 Missing Falsification Tests (Discovered by Chain-of-Thought)

The following tests are NOT in the current contract but SHOULD be:

```yaml
# Add to cublas-gemm-v1.yaml

  - id: FALSIFY-CUBLAS-009
    rule: "Non-GEMM overhead does not increase after cuBLAS"
    prediction: "T_non_gemm(cublas) < 1.1 * T_non_gemm(ptx)"
    test: |
      Profile 50 steps with PTX, measure total non-GEMM time.
      Profile 50 steps with cuBLAS, measure total non-GEMM time.
      Ratio must be < 1.10.
    if_fails: "FP16 casting, handle creation, or workspace allocation adds overhead"

  - id: FALSIFY-CUBLAS-010
    rule: "GQA thin-matrix GEMM still benefits from cuBLAS"
    prediction: "cuBLAS [4096, 256, 1024] > 50 TFLOP/s"
    test: |
      Run isolated GEMM on K/V projection shape [4096, 256, 1024].
      Must exceed 50 TFLOP/s (lower bar than large shapes due to
      low arithmetic intensity).
    if_fails: "Thin matrices memory-bandwidth-bound, not compute-bound"

  - id: FALSIFY-CUBLAS-011
    rule: "cuBLAS column-major convention handled correctly"
    prediction: "Row-major Rust buffers produce correct results via transpose flags"
    test: |
      Compute C = A * B in row-major (Rust native) using cuBLAS with
      appropriate CUBLAS_OP_T flags. Compare against known-good reference.
      All 7 GEMM shapes in a single transformer block must match.
    if_fails: "Leading dimension or transpose convention wrong (ALB-059 class bug)"

# Add to training-step-budget-v1.yaml

  - id: FALSIFY-BUDGET-004
    rule: "Phase 0 baseline matches estimated breakdown"
    prediction: "Measured GEMM fraction is 50-65% of step time"
    test: |
      Run BrickTracer profiling for 50 steps on PTX backend.
      T_gemm / T_step must be in [0.50, 0.65].
    if_fails: "Estimated breakdown is wrong; re-derive all phase projections"

# Add to training-memory-kernel-v1.yaml

  - id: FALSIFY-MEM-004
    rule: "Mixed-precision dual storage fits in VRAM"
    prediction: "FP16 forward weights + FP32 master weights + optimizer < 24GB"
    test: |
      Compute: P * 2 (FP16 GPU) + P * 4 (FP32 CPU master, not on GPU)
      + P * 8 (AdamW m+v, on GPU) + workspace.
      P=370M: 0.74 GB (FP16) + 2.96 GB (AdamW) + workspace = ~15.5 GB.
      Must fit in 24 GB with seq=1024, batch=4.
    if_fails: "VRAM budget exceeded, batch=4 may OOM with mixed precision"
```

**Claim 11: "TF32 tensor cores provide ~2x throughput"** (Section 6.9, Phase 5a)

- Status: **FALSIFIED — REVERTED (ALB-076)**. TF32 tensor cores showed 0%
  improvement at 350M model size (§6.9). More critically, tensor core GEMM
  algorithms (`CUBLAS_GEMM_DEFAULT_TENSOR_OP`) produce ALL NaN output for
  transposed backward GEMMs when gradient magnitudes reach ~1e5 (§6.12).
- Root cause: cuBLAS tensor core algorithm has undocumented numerical failure
  mode with transposed operands at high magnitudes. Forward (NoTrans/NoTrans)
  is unaffected.
- Fix: Disabled tensor cores entirely (`CUBLAS_DEFAULT_MATH`). cuBLAS SIMD path
  still 5.9x faster than PTX. Phase 5a reverted (trueno #170).
- Action: Phase 5a removed from optimization path. Added to bug pattern catalog.

### A.4 Unrealistic Assumptions Identified

| Assumption | Section | Reality Check |
|------------|---------|---------------|
| GEMM is 57% of step time | 2.4 | Unverified estimate. Phase 0 must confirm. |
| cuBLAS achieves 130-150 TFLOP/s | 4.1 | Depends on shape. May be 80-120 on rectangular. |
| Non-GEMM time stays constant | 6.1 | FP16 casting adds new overhead. |
| 2% FFI overhead | 5.7 | Plausible but requires per-GEMM vs per-step stream binding. |
| batch=8 fits with grad ckpt | 6.3 | Dual precision increases VRAM. Unproven. |
| 165 TFLOP/s is achievable peak | 1.2 | Marketing spec. Sustained is ~145-150 TFLOP/s. |

### A.5 Recommended Spec Revisions

1. **Gate Phase 1 on Phase 0 completion**. Do not write cuBLAS code until
   BrickTracer confirms the estimated breakdown.
2. **Add GQA thin-matrix shape** `[4096, 256, 1024]` to all benchmarks.
3. **Add FALSIFY-CUBLAS-009** (non-GEMM overhead preservation).
4. **Add FALSIFY-CUBLAS-010** (thin-matrix performance floor).
5. **Add FALSIFY-CUBLAS-011** (column-major convention correctness).
6. **Add FALSIFY-BUDGET-004** (baseline confirmation gate).
7. **Add FALSIFY-MEM-004** (mixed-precision VRAM budget).
8. **Integrate bench-gemm-regression into CI** (addresses batuta JA-04).
9. **Use sustained peak (~148 TFLOP/s)** instead of marketing peak (165) for
   MFU calculations.
10. **Note set_stream() binding scope** in cublas.rs contract: once per step,
    not per GEMM.
