# Training Step Budget Contract

**Contract**: `contracts/training-step-budget-v1.yaml`
**Version**: 1.0.0
**Status**: NEW (ALB-075)
**Depends on**: training-gpu-kernel-v1, cublas-gemm-v1

## Equations

### step_time_budget

```
T_step = T_gemm + T_optimizer + T_embedding + T_pcie + T_elementwise
       + T_cross_entropy + T_stream_sync + T_overhead
```

Every component maps to exactly one probador brick. Budget violation (> 2x)
triggers Jidoka alert.

### gemm_throughput

```
TFLOP_per_step = sum(2 * m * n * k / 1e12 for all ~555 GEMMs)
T_gemm = TFLOP_per_step / achieved_tflops
```

- PTX baseline: ~2 TFLOP/s
- cuBLAS target: >= 100 TFLOP/s

### mfu_definition

```
MFU = (6 * P * tokens_per_step) / (T_step * peak_flops)
P = 370M, tokens_per_step = 4096
peak_flops(FP16, sustained) = 148 TFLOP/s
```

## Proof Obligations (4)

| ID | Type | Property |
|----|------|----------|
| 1 | bound | Brick budgets cover >= 95% of step time |
| 2 | bound | GEMM dominates PTX baseline (> 50%) |
| 3 | bound | cuBLAS reduces GEMM time by >= 5x |
| 4 | bound | MFU improves monotonically across phases |

## Falsification Tests (4)

| ID | Rule | Prediction |
|----|------|------------|
| FALSIFY-BUDGET-001 | Brick coverage >= 95% | `T_step - sum(bricks) < 0.05 * T_step` |
| FALSIFY-BUDGET-002 | GEMM is primary bottleneck | `T_gemm > 50%` of step time |
| FALSIFY-BUDGET-003 | Jidoka gate fires | Injected delay pauses training |
| FALSIFY-BUDGET-004 | Baseline matches estimate | GEMM fraction in [50%, 65%] |

## QA Gate

**F-BUDGET-001**: All 4 falsification tests must pass before optimization
phase targets are considered valid.
