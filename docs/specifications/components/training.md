# Training Configuration

**Parent**: [albor-llm-spec.md](../albor-llm-spec.md) §6-7

---

## 1. Optimizer

### 1.1 AdamW Configuration

| Parameter | v28 (HPO) | v6 (original) | Rationale |
|-----------|-----------|---------------|-----------|
| lr | 7.35e-5 | 3e-4 | HPO-validated via 50M proxy (C-HPO-001) |
| beta1 | 0.9 | 0.9 | Default momentum |
| beta2 | 0.95 | 0.95 | Lower than 0.999 — avoids f32 overflow in bias correction |
| weight_decay | 0.012 | 0.1 | HPO-validated, ~8× lower than original |
| epsilon | 1e-8 | 1e-8 | Default |
| grad_clip | 1.0 | 1.0 | Max gradient norm (global, with ZClip) |

### 1.2 Learning Rate Schedule

**Cosine decay** with linear warmup:
```
warmup: 0 → lr over warmup_steps (linear)
decay:  lr → lr_min over remaining steps (cosine)
lr_min: lr / 10 = 3e-5
```

**ALB-079 fix**: Config field `lr_scheduler: "cosine"` was parsed but not
consumed by training loop. Now properly wired end-to-end.

### 1.3 Gradient Accumulation

GPU-resident accumulation (ALB-091): gradients stay on GPU across micro-batches.
No PCIe round-trip per micro-batch.

| Config | v28 (HPO) | v6 (original) |
|--------|-----------|---------------|
| batch_size | 4 (sequences per micro-batch) | 4 |
| gradient_accumulation | 32 (micro-batches per step) | 8 |
| tokens_per_step | 4 × 32 × 1024 = 131,072 | 4 × 8 × 1024 = 32,768 |

**ALB-080 fix**: Previous configs used 4K tokens/step (batch=4, ga=1).
HPO sweep (C-HPO-001) selected ga=32 for 131K tokens/step — matches 350M
scaling recommendations. ALB-078 fused gradient clipping enables this without
throughput penalty.

---

## 2. Training Configs

### 2.1 Active Config: pretrain-350m-v28.yaml

v28 uses HPO-validated hyperparameters (C-HPO-001) with corrected cosine
schedule horizon (ALB-129). Key insight: `max_steps` must match actual training
length, not be an arbitrary cap.

```yaml
model:
  architecture:
    hidden_size: 1024
    num_hidden_layers: 24
    num_attention_heads: 16
    num_key_value_heads: 4
    intermediate_size: 4096
    vocab_size: 32768
    max_position_embeddings: 1024
    rms_norm_eps: 1.0e-5
    rope_theta: 10000.0

data:
  train: "data/pretokenized-1024-v3/train/"
  val: "data/pretokenized-1024-v3/val/"
  batch_size: 4
  seq_len: 1024

optimizer:
  name: "adamw"
  lr: 7.35e-5
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.012

training:
  mode: "causal_lm"
  epochs: 1
  grad_clip: 1.0
  lr_scheduler: "cosine"
  warmup_steps: 93
  gradient_accumulation: 32
  max_steps: 38349    # = total_train_batches / GA = 1,227,172 / 32
  save_interval: 5000
  eval_interval: 500
  patience: 30
  seed: 789
```

**Changes from v6**: LR 4× lower (HPO), weight decay 8× lower (HPO), GA 4×
higher (131K tok/step), warmup ~93 steps (short), max_steps calibrated to
actual epoch length (ALB-129), patience extended to 30.

### 2.2 Test Config: pretrain-350m-cuda-test.yaml

Quick 5-step validation that CUDA training works end-to-end.
Uses same architecture but `max_steps: 5`, small data path.

### 2.3 Regression Config: pretrain-50m-quick.yaml

62M-param model, <2 min run. Validates training pipeline without needing GPU.

---

## 3. Historical Results

### 3.1 Run Summary

**Pre-HPO era (v3-v6):** Manual hyperparameters, cuBLAS integration, bug fixes.

| Run | Steps | Tokens | val_ppl | tok/s | MFU | Status |
|-----|-------|--------|---------|-------|-----|--------|
| v3 | 28K | 918M | 1018 | 6.7K | 19.3% | STOPPED: plateau |
| v4 | 2.4K | 79M | 918 | 6.7K | 19.3% | STOPPED: eval, no HumanEval pass |
| v5 | 3.4K | 112M | — | 7.9K | 22.9% | BROKEN: ALB-092 (norm grads) |
| v6 | 2K | 64M | 776 | 6.5K | 18.8% | STOPPED: strategic pivot |

**HPO era (v23-v28):** HPO-validated params (C-HPO-001), cosine horizon fix.

| Run | Steps | Tokens | val_ppl | tok/s | MFU | Status |
|-----|-------|--------|---------|-------|-----|--------|
| v23 | 10K | 1.3B | 50.38 | 8.2K | 25.7% | STOPPED: baseline |
| v25 | 6K | 786M | 48.09 | 8.1K | 25.4% | STOPPED: LR spike |
| v27 | 10.2K | 1.3B | 9.39 | 14.7K | 46.1% | STOPPED: diverged to 82 (ALB-129) |
| v28 (orig) | 5.4K | 708M | **5.88** | 14.7K | 46.1% | KILLED: experiment |
| **v28 (fresh)** | **11K** | **1.44B** | **38.53** | **11K** | **36.3%** | **STOPPED** |

**v28 fresh** stopped at step 11K (2026-04-04). Best val_ppl=38.53 at step 6K.
Regressed to 75.65 at step 11K (same pattern as v27 — peaks early, then
diverges on raw data). Confirms data quality is the ceiling on raw codeparrot.
Best checkpoint: `model-best.apr` (step 6K).

### 3.2 v28 Fresh Loss Curve

```
Step    val_ppl  pred_final  Notes
500     138.70               Warmup
1000    98.56                First sub-100
1500    74.68    12.6        Rapid improvement
2000    59.34    10.3
2500    53.52    10.2        Plateau begins
3000    52.07
3500    46.91    41.9        Scaling law slope flattens
4000    47.04    36.0
4500    48.96    41.4        Minor regression
5000    42.99    32.4
5500    44.13    31.7
6000    38.53    26.3        ★ Best
6500    40.09    25.6
7000    41.73    26.7        Plateau at 38-42
7500    42.81    28.4
8000    41.78    29.3
8500    40.47    29.6
9000    44.41    31.6        Regression begins
9500    44.62    33.3
10000   45.72    35.1
10500   52.40    38.6        Spike
11000   75.65    47.5        Diverging — STOPPED
```

### 3.3 v28 Original vs Fresh

v28 original achieved val_ppl=5.88 at step 3.5K — best ever — but was killed
for experimentation. The fresh run starts from scratch and shows a different
convergence pattern (slower initial descent, more stable plateau). The original
run's rapid descent to 5.88 may reflect favorable seed or data ordering.

### 3.4 Convergence Analysis

**v6 era conclusion** (validated): Architecture and optimizer work. Data
quality is the bottleneck.

**v28 era update**: HPO + cosine horizon fix (ALB-129) reduced val_ppl from
776 (v6) to 38.53 (v28 fresh, step 6K). Key factors:
1. **LR right-sizing**: 7.35e-5 vs 3e-4 (HPO validated on 50M proxy)
2. **Cosine horizon**: `max_steps=38349` matches actual epoch (ALB-129)
3. **Fused gradient clipping** (ALB-078): MFU 19% → 38.7%, tok/s 6.7K → 12.3K
4. **Gradient accumulation**: ga=32 → 131K tokens/step

**v28 post-mortem**: Model peaked at step 6K (val_ppl=38.53), plateaued at
38-45 from steps 6K-10K, then diverged to 75.65 at step 11K. Scaling law
slope went negative (-0.0167) — model getting worse, not better. Same pattern
as v27 (peaked at step 2K, diverged to 82). Raw codeparrot data runs out of
useful signal after ~800M tokens on this architecture.

**Next**: v29 on filtered data (2.04B clean tokens, higher signal density).
HumanEval eval on v28 best checkpoint. Distillation with teacher completions.

---

## 4. CUDA Trainer Architecture

### 4.1 GPU-Resident Training

entrenar's `CudaTransformerTrainer` keeps all transformer weights on GPU.
Only embeddings + LM head are on CPU (too large for 24GB + optimizer states).

```
CPU:  embedding weights + embedding optimizer (AdamW)
      LM head weights + LM head optimizer (AdamW)

GPU:  24 transformer blocks (weights + AdamW states)
      CudaGradWorkspace (shared across blocks)
      Activation buffers

PCIe: 3 transfers per step
      1. embed(input_ids) → GPU  [H2D, batch×seq×d]
      2. logits ← GPU            [D2H, batch×seq×vocab]
      3. grad_embed → GPU        [H2D, batch×seq×d]
```

### 4.2 Per-Block Backward + Optimize

Backward pass processes blocks in reverse order. After each block's backward:
1. Compute weight gradients for that block
2. Accumulate into GPU-resident gradient accumulator
3. After all micro-batches: apply AdamW update for that block
4. Zero gradient accumulator

This means only ONE block's gradients are in flight at any time.
`CudaGradWorkspace` is reused across blocks — it holds the largest block's
workspace (FFN: 3 × d × intermediate = 3 × 1024 × 4096 = 48MB at f32).

### 4.3 Known Issues

1. **`with_model()` broken**: Loading pre-trained weights doesn't properly
   upload to GPU. Model reverts to random init on resume. Needs investigation.
2. **All-shards-in-RAM**: Data loader loads all parquet shards at startup
   (19 shards = ~25 GB). Causes 39 GB swap, 20% throughput loss.
3. **Noise-scale logging**: Fixed in `entrenar@eb1b584` (dedup per optimizer step).

---

## 5. Throughput Analysis

### 5.1 Current Performance (v28)

| Metric | v28 | v6 | Change |
|--------|-----|-----|--------|
| tok/s | 12.3K | 6.5-7.9K | +55-89% |
| MFU | 38.7% | 18.8-22.9% | +2× |
| FLOPS/tok | ~2.1 GFLOP | ~2.1 GFLOP | — |
| GPU VRAM | 13.7 GB | 17.8 GB peak | -23% |
| Peak TFLOPS | 82.6 (RTX 4090 FP32) | — | — |

**ALB-078** (fused gradient clipping) was the key throughput breakthrough:
eliminated per-block D2H gradient transfers. Single fused clip pass on GPU
replaces 24 individual block clips + host round-trips.

### 5.2 Bottlenecks (Updated)

1. ~~**Swap pressure** (20%)~~: Mitigated by v28's larger ga (fewer steps, same data)
2. **PCIe transfers** (10%): 3 transfers per step for embeddings
3. **CPU optimizer** (5%): Embedding + LM head AdamW on CPU
4. **Kernel launch overhead** (5%): Per-kernel launch latency
5. **ZClip overhead** (~2%): Per-step z-score spike detection (acceptable)

### 5.3 cuBLAS Status (ALB-075)

cuBLAS GEMMs fully integrated. Using `CUBLAS_DEFAULT_MATH` (not tensor ops)
due to NaN issues with tensor cores on transposed GEMMs at high magnitude
(ALB-077). TF32 tensor cores available via `trueno-gpu@0.4.26` for compatible
operations.

---

## 6. Checkpoint Format

```
checkpoints/albor-base-350m-v28/
├── model.safetensors       # All weights (SafeTensors format)
├── config.json             # Architecture config
├── final_model.json        # Training metadata
├── training_state.json     # Step, loss EMA, optimizer state refs
└── step-{N}/               # Intermediate checkpoints (per save_interval)
    ├── model.safetensors
    └── config.json
```

Checkpoints saved every `save_interval` steps (5000 in v28). Training state
saved for rollback detection (loss EMA comparison).

---

## 7. Distillation Training Configs (Planned)

### 7.1 Stage A: Pre-train on Curated Data

```yaml
# configs/train/pretrain-350m-curated.yaml
training:
  lr: 3.0e-4
  warmup_steps: 2000
  max_steps: 60000
  gradient_accumulation: 8
  # tokens_per_step: 32K
  # total: ~2B tokens
```

### 7.2 Stage B: SFT on Teacher Completions

```yaml
# configs/train/sft-350m-distill.yaml
training:
  lr: 1.0e-5
  warmup_steps: 200
  max_steps: 5000
  gradient_accumulation: 4
  # tokens_per_step: 16K
  # total: ~80M tokens (3-5 epochs over synthetic data)
```
