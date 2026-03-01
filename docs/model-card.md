# Albor-350M Model Card

> **Status**: Template — to be populated after training completes.

## Model Description

**Albor** is a 350M-parameter decoder-only transformer for Python code completion,
trained entirely using the [Sovereign AI Stack](https://github.com/paiml) — a pure-Rust
ML toolkit with no Python, PyTorch, or cloud dependencies.

| Property | Value |
|----------|-------|
| Architecture | LLaMA-style (24L, 1024H, 16A, 4KV) |
| Parameters | ~354M |
| Context Length | 2048 tokens |
| Vocabulary | 32,768 BPE tokens |
| Language | Python only |
| License | Apache 2.0 |
| Training Stack | entrenar + trueno + alimentar (Rust) |
| Distillation Teacher | Qwen3-Coder-Next (80B MoE) |

## Intended Use

- Python code completion (left-to-right and fill-in-the-middle)
- On-device inference (laptops, Raspberry Pi, browsers via WASM)
- Educational artifact demonstrating first-principles LLM training

## Training Data

| Source | Tokens | Weight |
|--------|--------|--------|
| StarCoder Python | ~4B | 40% |
| FineWeb-Edu | ~2B | 20% |
| Local ground truth (10x upsampled) | ~1B effective | 10% |
| Python docs + PEPs | ~1B | 10% |

Total: ~10B tokens, 80% Python/Python-adjacent.

## Improvement Ladder

| Stage | Model | HumanEval | MBPP |
|-------|-------|-----------|------|
| 1. Pre-train | albor-base | TBD | TBD |
| 2. Distill | albor-distill | TBD | TBD |
| 3. Fine-tune | albor-instruct | TBD | TBD |
| 4. Merge | albor-merged | TBD | TBD |
| 5. Prune | albor-pruned | TBD | TBD |
| 6. Quantize | albor-q4 | TBD | TBD |

## Limitations

- Python only — no other language support
- Not a chat model or instruction follower
- 2048 token context window
- MoE-to-dense distillation is uncharted at 350M scale

## Quality Verification

All computational kernels verified via [provable-contracts](https://github.com/paiml/provable-contracts):
- Level 4 (Kani proof): softmax, attention, cross-entropy, KD loss, gradient accumulation
- Level 3 (property tests): all other kernels
- 9 falsification tests (FALSIFY-ALBOR-001 through 009)

## How to Use

```bash
# Download and run (GGUF format, ~90MB)
apr serve --model paiml/albor-350m --format gguf

# Or load in Rust
apr eval apply --model paiml/albor-350m --tasks humaneval
```

## Reproducibility

```bash
# Reproduce from scratch using only apr commands
git clone https://github.com/paiml/albor
cd albor
apr pipeline plan configs/pipeline/albor.yaml
apr pipeline apply configs/pipeline/albor.yaml
```

Full reproducibility protocol in [spec §16](https://paiml.github.io/albor/spec/16-reproducibility.html).
