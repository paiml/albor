# cuBLAS GEMM Integration Contract

**Contract**: `contracts/cublas-gemm-v1.yaml`
**Version**: 1.0.0
**Status**: NEW (ALB-075)
**Depends on**: training-gpu-kernel-v1, training-memory-kernel-v1

## Equations

### cublas_gemm_correctness

```
C_cublas = alpha * op(A) * op(B) + beta * C
where op(X) = X if transa=N, X^T if transa=T
A: FP16 [m, k], B: FP16 [k, n], C: FP16 [m, n]
Accumulation: FP32 (CUBLAS_COMPUTE_32F)
```

- max_abs_diff(C_cublas, C_ptx) < 1e-2 for identical inputs
- cuBLAS uses tensor cores when math mode is TENSOR_OP_MATH
- FP32 accumulation prevents catastrophic cancellation

### buffer_size_verification

```
For cublasGemmEx(m, n, k, A, B, C):
  A.len() >= m * k * 2  (FP16)
  B.len() >= k * n * 2  (FP16)
  C.len() >= m * n * 2  (FP16)
```

Verified at call site, not inside cuBLAS. Assertion failure = immediate panic.

### handle_lifecycle

```
create: cublasCreate_v2(&handle) -> CUBLAS_STATUS_SUCCESS
bind:   cublasSetStream_v2(handle, stream) once per training step
drop:   cublasDestroy_v2(handle) exactly once
```

- One handle per CudaContext (thread-safe within context)
- Stream set ONCE per step, not per GEMM (555 calls = measurable overhead)
- Handle destroyed on Drop (Rust RAII)

### ffi_overhead

```
overhead = T_rust_cublas / T_raw_c_cublas < 1.02
```

For identical GEMM shape, same GPU, same cuBLAS config. Measured via CUDA
events, not wall clock. Warmup: 50 iterations discarded before measurement.

### mfu_improvement

```
MFU = (6 * P * tokens_per_step) / (T_step * peak_flops)
P = 370M, tokens_per_step = 4096
peak_flops(FP16, sustained) = 148 TFLOP/s
```

- MFU(cublas) > MFU(ptx) (strict improvement)
- MFU(cublas) >= 0.025 (must beat current 2.5% FP32 baseline)

### mixed_precision_weight_flow

```
CPU master weights: FP32 (optimizer operates here)
GPU forward weights: FP16 (cast during upload)
GPU activation gradients: FP16 (cuBLAS backward output)
GPU weight gradients: FP32 (accumulated in FP32 buffer)
CPU gradient download: FP32 (for optimizer update)
```

- Master weights ALWAYS FP32 on CPU (no precision loss in optimizer)
- C-EMBED-GRAD-001 still holds: activation grad clipped before CPU scatter-add
- C-HYPERPARAMS-001 still holds: all optimizer params from YAML config

## Proof Obligations (8)

| ID | Type | Property |
|----|------|----------|
| 1 | equivalence | cuBLAS GEMM matches PTX GEMM (max_abs_diff < 1e-2) |
| 2 | invariant | Buffer sizes verified before every cublasGemmEx |
| 3 | invariant | cuBLAS handle lifecycle is RAII |
| 4 | bound | FFI overhead < 2% |
| 5 | bound | MFU improves over baseline |
| 6 | invariant | Training stability preserved (loss.is_finite()) |
| 7 | invariant | Gradient flow preserved (grad != 0 for all params) |
| 8 | invariant | FP32 accumulation enforced (CUBLAS_COMPUTE_32F) |

## Falsification Tests (11)

| ID | Rule | Prediction |
|----|------|------------|
| FALSIFY-CUBLAS-001 | Forward matches PTX | `max_abs_diff(logits) < 1e-2` on 50M |
| FALSIFY-CUBLAS-002 | Training stable 50 steps | Loss finite, within 5% of PTX baseline |
| FALSIFY-CUBLAS-003 | GEMM > 100 TFLOP/s | `[4096,1024] x [1024,4096]` isolated GEMM |
| FALSIFY-CUBLAS-004 | Step time improves | 350M < 3.0s (vs 4.4s PTX) |
| FALSIFY-CUBLAS-005 | Buffer overflow impossible | Undersized buffer panics, no silent corruption |
| FALSIFY-CUBLAS-006 | All params get gradients | `max(\|grad\|) > 0` for 110 params after 1 step |
| FALSIFY-CUBLAS-007 | C-EMBED-GRAD-001 preserved | Activation grad clipped before CPU scatter-add |
| FALSIFY-CUBLAS-008 | FFI overhead < 2% | `T_rust / T_raw_c < 1.02` for all shapes |
| FALSIFY-CUBLAS-009 | Non-GEMM overhead stable | `T_non_gemm(cublas) < 1.1 * T_non_gemm(ptx)` |
| FALSIFY-CUBLAS-010 | GQA thin-matrix benefits | `[4096,256,1024]` > 50 TFLOP/s |
| FALSIFY-CUBLAS-011 | Column-major convention | Row-major Rust buffers correct via transpose flags |

## Kani Harness

**KANI-CUBLAS-001**: Buffer size assertion prevents overflow for all valid GEMM
shapes (exhaustive, bound=8).

## QA Gate

**F-CUBLAS-001**: All 11 falsification tests must pass before cuBLAS backend
replaces PTX for training.
