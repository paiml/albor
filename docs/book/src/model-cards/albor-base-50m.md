# Model Card: albor-base-50m

## Model Details

| Field | Value |
|-------|-------|
| Name | albor-base-50m |
| Version | 1.0 (pipeline validation) |
| Type | Decoder-only Transformer (LLaMA-style) |
| Parameters | ~62M (hidden=512, layers=12 — "50M" is approximate label) |
| Architecture | hidden=512, layers=12, heads=8, kv_heads=2, ffn=2048 |
| Vocab Size | 32,768 (BPE, whitespace-split v1; later upgraded to ByteLevel BPE v2) |
| Context Length | 128 tokens (validation run; architecture supports 2048) |
| Training Data | 500 rows Python code, 64K tokens |
| Training Time | 110.7 seconds (CUDA on RTX 4090) |
| Framework | entrenar + realizar (CUDA, CudaTransformerTrainer) |

## Intended Use

**Pipeline validation only.** This model validates that the albor training stack
(alimentar → entrenar → realizar) works end-to-end. It is NOT intended for
code completion or any production use.

## Training Details

- **Optimizer**: AdamW (lr=6e-4, β1=0.9, β2=0.95, wd=0.1)
- **Steps**: 31 optimizer steps (125 batches, gradient_accumulation=4)
- **Mixed Precision**: fp16
- **Loss**: 10.335 → 4.423 (perplexity 30,802 → 5.4)
- **Compute**: 76.8s CUDA matmul (69%), 32.9s transpose (30%), 0.9s alloc (1%)

## Tokenizer

- **Type**: BPE with `split_whitespace()` pre-tokenizer + `</w>` suffix
- **Vocab**: 32,768 tokens
- **Known Limitation**: Normalizes whitespace (loses Python indentation)
- **Source**: Trained with `apr tokenize apply` on 100K lines of Python code

## FALSIFY Predictions

| ID | Prediction | Status |
|----|-----------|--------|
| FALSIFY-ALBOR-001 | Loss decreases monotonically | CORROBORATED (10.3→4.42) |
| FALSIFY-ALBOR-002 | Gradient norms bounded | PENDING (per-step logging now available, ALB-035 FIXED) |
| FALSIFY-ALBOR-003 | Checkpoint determinism | UNTESTED |

## Limitations

1. Whitespace normalization in tokenizer makes output invalid Python
2. Only 500 training rows (not representative of target distribution)
3. Short context (128 tokens, not production 2048)
4. No evaluation on code completion benchmarks (structural eval only)

## Data Provenance

See `docs/PROVENANCE.md` for full SHA-256 hashes of all data artifacts.

## Checkpoint

- **Path**: `checkpoints/albor-base-50m/model.safetensors` (249 MB)
- **Metadata**: `checkpoints/albor-base-50m/final_model.json`
