# Albor Project Rules

## Engineering Discipline

These rules are non-negotiable. They encode lessons from 6+ critical bugs
discovered during ALB-040 GPU-resident training dogfooding (ALB-041, ALB-043,
ALB-044, and 3 unnamed buffer overflows). Every rule maps to a specific failure.

### 1. Observability: Tracing, Not Printf

**NEVER use `eprintln!`, `println!`, or `dbg!` for debugging training code.**

The sovereign stack has structured tracing via **renacer** (BrickTracer, spans,
metric events). entrenar integrates it in `src/run.rs` (create_span, end_span,
emit_metric_event). Ad-hoc stderr prints:
- Are invisible to the tracing infrastructure
- Create cleanup debt (temporary code that stays forever)
- Lose brick profiling boundary isolation
- Cannot be filtered by log level at runtime

**Instead**: Use `tracing::debug!()`, `tracing::warn!()` with structured fields.
For training metrics, use `emit_metric_event()`. For performance, use renacer
BrickTracer spans. The `RUST_LOG` env var controls verbosity.

**Why**: ALB-044 diagnosis required ~150 lines of temporary `eprintln!` that had
to be manually removed. Proper tracing would have been permanent, filterable,
and integrated with the drift detector.

### 2. Buffer Sizes: Prove at Kernel Boundaries

**Every CUDA kernel call must have algebraically verifiable buffer sizes.**

Before calling `gemm_forward`, `silu_backward`, `rms_norm_backward`, etc.,
verify that input/output buffer sizes match the kernel's documented contract.
Mismatches cause `CUDA_ERROR_ILLEGAL_ADDRESS` (crash) or silent memory
corruption (wrong results, NaN).

**Contract**: `output.len() >= expected_elements` for every GPU buffer passed
to a kernel. Use `max_seq_len` (not runtime `seq_len`) for pre-allocated buffers.

**Why**: ALB-043 wrote `[S, intermediate_size]` into `[S, hidden_size]` buffer
(4x overflow). ALB-041 used `gate_out` (intermediate_size) as temp for
`grad_hidden` (hidden_size). Both were silent memory corruption that manifested
as wrong gradients, not crashes.

### 3. Hyperparameters: End-to-End Propagation

**Every optimizer hyperparameter must flow from YAML config to the actual
optimizer constructor. Never use `::default()` or `default_params()` for
training-critical code.**

The config chain is: `YAML → bridge → TrainConfig → TransformerTrainConfig →
optimizer constructor`. Every field (lr, beta1, beta2, weight_decay, epsilon,
grad_clip) must be explicitly passed. No implicit defaults.

**Contract (C-HYPERPARAMS-001)**: For every optimizer field in the YAML config,
assert that the corresponding runtime optimizer uses that exact value.

**Why**: ALB-044 used `AdamW::default_params(lr)` for the CPU embedding
optimizer, which set beta2=0.999 and wd=0.01 instead of the YAML's beta2=0.95
and wd=0.1. The 50x amplification in bias correction overflowed f32 → NaN.

### 4. Gradient Flow: Verify Every Trainable Parameter

**After implementing any new training path, verify that every trainable
parameter receives a non-zero gradient after one forward+backward step.**

This catches: missing backward ops, tensors created with `requires_grad: false`,
broken gradient chains through custom ops.

**Contract**: `grad(param) != 0` for all trainable parameters after one step
on a non-trivial batch. Zero gradient on a trainable parameter is always a bug
in a pretraining context.

**Why**: ALB-038 had `RMSNorm::forward_batched()` with no backward op —
blocked ALL gradient flow through the entire network. Training "worked" (loss
decreased via embedding gradients only) but produced garbage weights.

### 5. Activation Gradient Clipping at GPU-CPU Boundary

**When GPU backward produces activation gradients that flow to CPU code (e.g.,
embedding backward), those gradients must be clipped before the CPU optimizer
processes them.**

Per-block weight gradient clipping in `CudaGradWorkspace` does NOT clip the
activation gradient in `grad_buf_a`/`grad_buf_b`. For deep networks with
random init, this gradient can reach ~1e35 — far beyond f32 precision.

**Contract (C-EMBED-GRAD-001)**: Clip activation gradient norm to
`max_grad_norm` before scatter-adding to CPU embedding weight gradient.

**Why**: ALB-044 — 24-layer backward amplified activation gradients to ~1e35.
CPU AdamW's second moment overflowed f32 → 1298 NaN in 33.5M embedding table.

### 6. Provable Contracts: Write Before Code

**Every new training contract or kernel must have a YAML contract in
`contracts/` BEFORE implementation begins.**

The contract defines: equations, buffer size invariants, numerical bounds,
falsification tests. Implementation is then provably correct against the
contract. Bugs found during dogfooding become new contract obligations.

**Workflow**:
```
Write contract YAML → pv validate → implement → pv audit → dogfood → close gap
```

**The gap register (§11) is the feedback loop**: every bug found by profiling
becomes a permanent contract obligation that prevents recurrence.

### 7. Five Whys: Root Cause, Not Symptom

**When a training bug is found, trace through exactly 5 whys using brick
profiling boundaries before writing any fix.**

The brick architecture (per-kernel, per-block, per-transfer) isolates failures
to specific components. Use the boundaries:
1. Per-transfer: Which PCIe transfer carries corrupt data?
2. Per-block: Which transformer layer's backward produces NaN?
3. Per-kernel: Which GEMM/norm/activation kernel overflows?

**Never**: patch the symptom (e.g., "clamp NaN to 0"). Always: fix the root
cause at the deepest why.

## Build & Test

```bash
# Build apr with local entrenar patches
cd ~/src/aprender && CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/aprender \
    cargo build --release -p apr-cli

# Run 50M regression test (< 2 min)
/mnt/nvme-raid0/targets/aprender/release/apr train apply \
    --task pretrain --config configs/train/pretrain-50m-quick.yaml

# Run 350M CUDA test (< 1 min for 5 steps)
/mnt/nvme-raid0/targets/aprender/release/apr train apply \
    --task pretrain --config configs/train/pretrain-350m-cuda-test.yaml

# Validate contracts
pv validate contracts/*.yaml
```

## Key Paths

| Path | Purpose |
|------|---------|
| `configs/train/` | Training YAML configs |
| `contracts/` | Provable contract YAMLs |
| `docs/book/src/spec/` | mdBook specification chapters |
| `docs/book/src/spec/11-gaps.md` | Gap register |
| `docs/book/src/spec/12-quality-contracts.md` | Quality & contract spec |
| `models/albor-tokenizer-v2/` | ByteLevel BPE tokenizer |
| `data/pretokenized-2048/` | Pre-tokenized Parquet data |

## Upstream Repos

| Repo | Path | What |
|------|------|------|
| entrenar | `~/src/entrenar` | Training library (CUDA trainer, blocks) |
| aprender | `~/src/aprender` | CLI (`apr`) with `[patch.crates-io]` for entrenar |
| renacer | `~/src/renacer` | Tracing infrastructure (BrickTracer, spans) |
| provable-contracts | via `pv` | Contract validation |
