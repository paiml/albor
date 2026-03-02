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
| Training Time | TBD (estimated ~20 hours on RTX 4090) |
| Framework | entrenar 0.7.5 + realizar (CUDA) |

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
- **Loss**: TBD → TBD
- **Perplexity**: TBD (target: < 30)

## Tokenizer

- **Type**: ByteLevel BPE (v2)
- **Vocab**: 32,768 tokens
- **Preserves**: Whitespace, indentation, newlines (critical for Python)
- **Source**: Trained with Python `tokenizers` library on 100K lines of Python code
- **Location**: `models/albor-tokenizer-v2/tokenizer.json`

## FALSIFY Predictions

| ID | Prediction | Status |
|----|-----------|--------|
| FALSIFY-ALBOR-001 | Loss decreases monotonically | PENDING (350M run) |
| FALSIFY-ALBOR-002 | Gradient norms bounded | UNTESTED (no per-step reporting, ALB-035) |
| FALSIFY-ALBOR-003 | Checkpoint determinism | UNTESTED |

## Evaluation

| Benchmark | Metric | Result |
|-----------|--------|--------|
| Training perplexity | exp(loss) | TBD |
| HumanEval (20 problems) | pass@1 | TBD (blocked: ALB-037) |
| Python intermediate (15 problems) | pass@1 | TBD (blocked: ALB-037) |

## Limitations

1. Single epoch on 45.2M tokens (typical base models train on 10B+ tokens)
2. Python-only training data (no multilingual code)
3. No FIM training (fill-in-the-middle not applied to this run)
4. Evaluation blocked by ALB-037 (realizar weight loading bug)
5. No real-time training metrics (ALB-035)

## Known Gaps

- **ALB-035**: No `training_state.json` during training (only `final_model.json` at end)
- **ALB-037**: `apr eval` ignores loaded weights (blocks perplexity/code evaluation)

## Data Provenance

See `docs/PROVENANCE.md` for full SHA-256 hashes of all data artifacts.

## Checkpoint

- **Path**: `checkpoints/albor-base-350m/model.safetensors` (TBD size)
- **Metadata**: `checkpoints/albor-base-350m/final_model.json`
- **Config**: `configs/train/pretrain-350m.yaml`
