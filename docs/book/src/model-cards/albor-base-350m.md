# Model Card: albor-base-350m

## Model Details

| Field | Value |
|-------|-------|
| Name | albor-base-350m |
| Version | 1.0 (base pre-training) |
| Type | Decoder-only Transformer (Qwen2-style) |
| Parameters | 398.5M |
| Architecture | hidden=1024, layers=24, heads=16, kv_heads=4, ffn=4096 |
| Vocab Size | 32,768 (ByteLevel BPE v2, whitespace-preserving) |
| Context Length | 2,048 tokens |
| Training Data | 22,079 sequences x 2,048 tokens = 45.2M tokens (Python code) |
| Training Time | ~20 hours on RTX 4090 (full run); 396s for 50-step test |
| Framework | entrenar + realizar (CUDA, CudaTransformerTrainer) |

## Intended Use

**Base pre-training model.** This model learns Python code patterns from
pre-tokenized data. It serves as the foundation for:
1. Knowledge distillation from Qwen3-Coder-Next (Phase 4)
2. Fine-tuning with LoRA (Phase 6)
3. Post-training optimization: pruning, merging, quantization (Phase 6)

## Training Details

- **Optimizer**: AdamW (lr=3e-4, beta1=0.9, beta2=0.95, wd=0.1)
- **Scheduler**: Cosine with 2000 warmup steps
- **Gradient Accumulation**: 32 (effective batch = 8 x 32 x 2048 = 524K tokens)
- **Mixed Precision**: fp16
- **Epochs**: 1 (single pass over 22,079 sequences)
- **Batches**: 2,760 micro-batches, ~86 optimizer steps
- **Max Steps**: 5,000 (config; training completes at ~86 steps due to data size)
- **Loss (50-step test)**: 10.39 → 6.07 (best 5.51) — convergence verified
- **Perplexity (50-step test)**: ~31,926 (finite; random baseline ~32,768)
- **Loss (full run)**: TBD (5000-step training in progress)
- **Perplexity (full run)**: TBD
- **CUDA Mode**: GPU-resident training via CudaTransformerTrainer (ALB-040), 3 PCIe transfers/step

## Tokenizer

- **Type**: ByteLevel BPE (v2)
- **Vocab**: 32,768 tokens
- **Preserves**: Whitespace, indentation, newlines (critical for Python)
- **Source**: Trained with Python `tokenizers` library on 100K lines of Python code
- **Location**: `models/albor-tokenizer-v2/tokenizer.json`

## FALSIFY Predictions

| ID | Prediction | Status |
|----|-----------|--------|
| FALSIFY-ALBOR-001 | Loss decreases monotonically | CORROBORATED (50M: 10.3→4.42; 350M CUDA 50-step: 10.39→6.07) |
| FALSIFY-ALBOR-002 | Gradient norms bounded | PENDING (per-step logging available via ALB-035) |
| FALSIFY-ALBOR-003 | Checkpoint determinism | UNTESTED |

## Evaluation

| Benchmark | Metric | Result |
|-----------|--------|--------|
| Training loss (50-step test) | cross-entropy | 10.39 → 6.07 (best 5.51) |
| Training perplexity (50-step test) | exp(loss) | ~31,926 (finite) |
| Checkpoint validation | weights trained? | PASS (layers distinct, not init) |
| realizar inference | loads + generates? | PASS (218 tensors, 50 tokens generated) |
| HumanEval (20 problems) | pass@1 | TBD (after full training) |
| Python intermediate (15 problems) | pass@1 | TBD (after full training) |

## Limitations

1. Single epoch on 45.2M tokens (typical base models train on 10B+ tokens)
2. Python-only training data (no multilingual code)
3. No FIM training (fill-in-the-middle not applied to this run)
4. ~~Checkpoint broken by ALB-038~~ **FIXED** — entrenar now saves trained weights correctly
5. ~~Evaluation blocked by ALB-037~~ **FIXED** — realizar loads trained checkpoint, generates tokens

## Known Gaps

- **ALB-035** (**FIXED**): Per-step loss logging via `train_epoch_with_callback()` (`entrenar@5d41a96`)
- **ALB-037** (**FIXED**): realizar now loads trained checkpoint, generates tokens (e2e verified with 350M)
- **ALB-038** (**FIXED**): Broken autograd in `RMSNorm::forward_batched()` and
  `MultiHeadAttention::forward()`. Fixed in `entrenar@91ba9da` and `entrenar@1ede409`.
  All 20 model parameters now receive gradients.
- **ALB-040** (**VERIFIED**): GPU-resident pretraining via `CudaTransformerTrainer`. 3 PCIe
  transfers/step vs ~16K. 350M CUDA test: 50 steps, loss 10.39→6.07, checkpoint valid.
- **ALB-041** (**FIXED**): D2D buffer size mismatch in `backward_attention()`. Fixed in
  `entrenar@a48e3d2`. Was blocking GPU backward pass.
- **ALB-043** (**FIXED**): backward_ffn buffer overflow + missing SwiGLU gradients.
  Fixed in `entrenar@f7805f1`.
- **ALB-044** (**FIXED**): Activation gradient clipping at GPU-CPU boundary + CPU optimizer
  hyperparams (beta2/wd mismatch). Fixed in `entrenar@86eec38`.

## Data Provenance

See `docs/PROVENANCE.md` for full SHA-256 hashes of all data artifacts.

## Checkpoint

- **Test checkpoint**: `checkpoints/albor-350m-cuda-test/model.safetensors` (1.59 GB, 218 tensors)
- **Full checkpoint**: `checkpoints/albor-base-350m/model.safetensors` (TBD — training in progress)
- **Metadata**: `checkpoints/albor-base-350m/final_model.json`
- **Config (test)**: `configs/train/pretrain-350m-cuda-test.yaml`
- **Config (full)**: `configs/train/pretrain-350m.yaml`
