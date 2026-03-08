# Infrastructure

**Parent**: [albor-llm-spec.md](../albor-llm-spec.md) §6

---

## 1. Hardware

### 1.1 Compute

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X, sm_89, Ada Lovelace) |
| GPU TFLOPS | 82.6 FP32, 165.2 TF32, 330.3 FP16 |
| CPU | Intel Xeon (48 cores, 128 GB DDR4) |
| Storage | NVMe RAID-0 (`/mnt/nvme-raid0/`, ~2 TB) |
| PCIe | Gen 4 x16 (32 GB/s bidirectional) |

### 1.2 VRAM Budget

| Config | VRAM | Breakdown |
|--------|------|-----------|
| Student training (350M) | ~18 GB | Weights (1.4 GB) + AdamW (2.8 GB) + workspace (~14 GB) |
| Teacher inference (35B Q4) | ~20 GB | Q4 weights from APR (18.3 GB) + KV cache + activations |
| Student eval | ~5 GB | Forward only, no optimizer states |

**Constraint**: Teacher and student run sequentially, never simultaneously.
Single 4090 means no pipeline parallelism.

### 1.3 VRAM Breakdown for Training

| Component | Size | Notes |
|-----------|------|-------|
| 24 blocks (weights) | ~930 MB | 24 × (attn + FFN + norms) at f32 |
| 24 blocks (AdamW m,v) | ~1.86 GB | 2× weight size |
| CudaGradWorkspace | ~48 MB | Shared, sized for largest block (FFN) |
| GpuGradientAccumulator | ~930 MB | Same size as weights |
| Attention scores | ~1 GB | [batch, heads, seq, seq] at f32 |
| Activation buffers | ~8 GB | Forward/backward intermediates |
| cuBLAS workspace | ~256 MB | GEMM scratch space |
| **Total** | **~13 GB** | Leaves ~11 GB headroom |

Actual usage varies: v6 measured at 17.8 GB peak.

---

## 2. Software Stack

### 2.1 Sovereign Stack Components

| Tool | Role | Repo |
|------|------|------|
| `apr` (aprender) | CLI: train, eval, distill, quantize, export | `~/src/aprender` |
| `entrenar` | Training engine: CUDA trainer, GPU-resident grad accum | `~/src/entrenar` |
| `trueno` | Tensor ops: cuBLAS GEMMs, PTX kernels, RMSNorm | crates.io |
| `realizar` | Inference: teacher model, eval generation, MoE dispatch | `~/src/realizar` |
| `alimentar` | Data: import, tokenize, FIM transform, quality filter | — |
| `forjar` | Pipeline: DAG orchestration, multi-machine, idempotent | — |
| `batuta` | Mutation testing, falsification | — |
| `pmat` | Code quality: TDG grades, search, coverage analysis | — |
| `pv` | Contract validation (provable-contracts) | — |
| `renacer` | Tracing: BrickTracer, spans, metric events | `~/src/renacer` |
| `presentar` | Model serving | — |
| `certeza` | Quality gate: coverage ≥95%, mutation ≥80% | `~/src/certeza` |

### 2.2 Build System

```bash
# Build apr with local entrenar patches
cd ~/src/aprender && \
  CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/aprender \
  cargo build --release -p apr-cli

# Binary location
/mnt/nvme-raid0/targets/aprender/release/apr
```

**Critical**: aprender uses `[patch.crates-io]` to override entrenar and
realizar with local paths. `cargo install --path` bypasses these patches —
always use `cargo build` + run from target dir.

### 2.3 Dependency Chain

```
apr-cli
├── entrenar (local patch: ~/src/entrenar)
│   ├── trueno (crates.io, always latest)
│   │   └── trueno-gpu (CUDA kernels, cuBLAS FFI)
│   └── renacer (tracing)
├── realizar (local patch: ~/src/realizar)
│   ├── trueno
│   └── renacer
└── alimentar
```

---

## 3. Key Paths

| Path | Purpose |
|------|---------|
| `configs/train/` | Training YAML configs |
| `configs/pipeline/` | Forjar pipeline manifests |
| `contracts/` | Provable contract YAMLs (pv validate) |
| `docs/specifications/` | This spec + component files |
| `models/albor-tokenizer-v2/` | ByteLevel BPE tokenizer (32K vocab) |
| `data/pretokenized-1024-v3/` | codeparrot-clean (5.3B tokens, 20 shards) |
| `checkpoints/` | Model checkpoints (per training run) |
| `logs/` | Training logs (JSONL, per run) |

### 3.1 External Paths

| Path | Purpose |
|------|---------|
| `/mnt/nvme-raid0/targets/` | Cargo build targets (aprender, entrenar) |
| `/mnt/nvme-raid0/models/` | Large model files (`qwen35-moe.apr`, HF source) |
| `/mnt/nvme-raid0/data/` | Additional datasets (CSN, merged) |

---

## 4. CUDA Environment

### 4.1 Kernel Compilation

trueno compiles PTX kernels at build time via `build.rs`. Kernels embedded
in binary — no runtime compilation needed.

Key kernels:
- `BatchedRmsNormForwardKernel` / `BatchedRmsNormBackwardKernel`
- `SiluForwardKernel` / `SiluBackwardKernel`
- `ElementwiseMulForwardKernel` / `ElementwiseMulBackwardKernel`
- `RopeForwardKernel` / `RopeBackwardKernel`
- cuBLAS GEMMs (via FFI, not PTX)

### 4.2 Stream Management

All kernels launch on non-blocking CUDA streams (`CU_STREAM_NON_BLOCKING`).
**Critical**: `stream.synchronize()` required before every D2H transfer.
See ALB-065 — race condition between kernel execution and `cuMemcpyDtoH`.

### 4.3 Memory Management

GPU memory managed via trueno's `GpuBuffer`:
- `GpuBuffer::new(ctx, n)` — uninitialized (cuMemAlloc)
- `GpuBuffer::from_host(ctx, data)` — initialized from CPU data
- **Rule**: Always use `from_host` or call `.zero()` for optimizer states

---

## 5. Monitoring

### 5.1 Training Metrics (JSONL)

entrenar writes per-step metrics to JSONL log:
```json
{"step": 100, "loss": 7.83, "lr": 0.0003, "grad_norm": 1.42, "tok_s": 6700}
```

### 5.2 Andon Stops

Automatic training halt on anomalies:
```yaml
alerts:
  - condition: "loss > 10"
    action: "stop"
    message: "Loss exploded — Andon stop"
  - condition: "gradient_norm > 100"
    action: "stop"
    message: "Gradient explosion — Andon stop"
```

### 5.3 Noise Scale

Gradient noise scale tracked per optimizer step (deduped, not per micro-batch).
Computed from gnorm variance in rolling window (100 steps).

---

## 6. Known Infrastructure Issues

| Issue | Impact | Status |
|-------|--------|--------|
| All-shards-in-RAM data loading | 39 GB swap, 20% throughput loss | OPEN |
| `with_model()` resume broken | Can't resume from checkpoint | OPEN |
| Stale training_state.json | Rollback false positives | Workaround: clean dir |
| cargo-killer.service | May kill long builds | Under investigation |
