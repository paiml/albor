# Training Configuration

**Parent**: [albor-llm-spec.md](../albor-llm-spec.md) §6-7

---

## 1. Optimizer

### 1.1 AdamW Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| lr | 3e-4 | Standard for 350M (Chinchilla scaling) |
| beta1 | 0.9 | Default momentum |
| beta2 | 0.95 | Lower than 0.999 — avoids f32 overflow in bias correction |
| weight_decay | 0.1 | Standard regularization |
| epsilon | 1e-8 | Default |
| grad_clip | 1.0 | Max gradient norm (global) |

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

| Config | Value |
|--------|-------|
| batch_size | 4 (sequences per micro-batch) |
| gradient_accumulation | 8 (micro-batches per step) |
| tokens_per_step | 4 × 8 × 1024 = 32,768 |

**ALB-080 fix**: Previous configs used 4K tokens/step (batch=4, ga=1).
Rule: ≥64K tokens/step ideal for 350M, but 32K is acceptable given VRAM
constraints.

---

## 2. Training Configs

### 2.1 Active Config: pretrain-350m-v6.yaml

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

data:
  train: "data/pretokenized-1024-v3/train/"
  val: "data/pretokenized-1024-v3/val/"
  batch_size: 4
  seq_len: 1024

optimizer:
  name: "adamw"
  lr: 3.0e-4
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1

training:
  mode: "causal_lm"
  epochs: 1
  grad_clip: 1.0
  lr_scheduler: "cosine"
  warmup_steps: 500
  gradient_accumulation: 8
  max_steps: 20000
  save_interval: 500
  eval_interval: 250
  patience: 10
```

### 2.2 Test Config: pretrain-350m-cuda-test.yaml

Quick 5-step validation that CUDA training works end-to-end.
Uses same architecture but `max_steps: 5`, small data path.

### 2.3 Regression Config: pretrain-50m-quick.yaml

62M-param model, <2 min run. Validates training pipeline without needing GPU.

---

## 3. Historical Results

### 3.1 Run Summary

| Run | Steps | Tokens | val_ppl | tok/s | MFU | Status |
|-----|-------|--------|---------|-------|-----|--------|
| v3 | 28K | 918M | 1018 | 6.7K | 19.3% | STOPPED: plateau |
| v4 | 2.4K | 79M | 918 | 6.7K | 19.3% | STOPPED: eval, no HumanEval pass |
| v5 | 3.4K | 112M | — | 7.9K | 22.9% | BROKEN: ALB-092 (norm grads) |
| v6 | 2K | 64M | **776** | 6.5K | 18.8% | STOPPED: strategic pivot |

### 3.2 v6 Loss Curve

```
Step    Loss    val_ppl  Notes
0       10.44            Random init
50      8.92             Warmup phase
100     7.83
150     7.07             Loss plateau begins
200     6.82
250     6.73   829       First eval (val from same distribution)
500     6.53   965       Post-warmup
750     6.42   865
1000    6.35   849
1250    6.27   862
1500    6.19   838
1750    6.11   813
2000    6.03   776       Best val_ppl (killed here)
```

### 3.3 Convergence Analysis

Loss decreasing at ~0.02/250 steps = 0.08/1000 steps. At this rate:
- val_ppl < 500: ~5K more steps (~160M tokens)
- val_ppl < 100: ~60K more steps (~2B tokens)
- Competitive HumanEval: likely never without data quality improvement

**Conclusion**: Architecture works. Optimizer works. Training is stable.
The bottleneck is 100% data quality. Distillation addresses this directly.

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

### 5.1 Current Performance

| Metric | Value |
|--------|-------|
| tok/s | 6.5-7.9K (varies by swap pressure) |
| MFU | 18.8-22.9% |
| FLOPS/tok | ~2.1 GFLOP (370M × 6) |
| Peak TFLOPS | 82.6 (RTX 4090 FP32) |
| Theoretical max tok/s | ~39K |

### 5.2 Bottlenecks

1. **Swap pressure** (20%): All shards loaded into RAM at startup
2. **PCIe transfers** (10%): 3 transfers per step for embeddings
3. **CPU optimizer** (5%): Embedding + LM head AdamW on CPU
4. **Kernel launch overhead** (5%): Per-kernel launch latency

### 5.3 cuBLAS Status (ALB-075)

cuBLAS GEMMs fully integrated. Using `CUBLAS_DEFAULT_MATH` (not tensor ops)
due to NaN issues with tensor cores on transposed GEMMs at high magnitude
(ALB-077). TF32 tensor cores available via `trueno-gpu@0.4.26` for compatible
operations.

---

## 6. Checkpoint Format

```
checkpoints/albor-base-350m-v6/
├── model.safetensors       # All weights (SafeTensors format)
├── config.json             # Architecture config
├── final_model.json        # Training metadata
├── training_state.json     # Step, loss EMA, optimizer state refs
└── step-{N}/               # Intermediate checkpoints (per save_interval)
    ├── model.safetensors
    └── config.json
```

Checkpoints saved every `save_interval` steps (500). Training state saved
for rollback detection (loss EMA comparison).

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
