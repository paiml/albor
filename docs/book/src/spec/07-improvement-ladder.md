# 7. Post-Training Improvement Ladder

Each stage improves the model and exercises a different `entrenar` / `apr`
capability. Every stage produces a benchmarked checkpoint.

### 7.1 Stage 1: Pre-Train Base Model

```bash
apr train plan configs/train/pretrain-350m.yaml          # Validate + VRAM estimate
apr train apply configs/train/pretrain-350m.yaml --seed 42
```

**Produces**: `albor-base-350m` — raw pre-trained model
**Exercises**: entrenar, trueno (CUDA), alimentar (data streaming)
**Expected**: OPT-350M class on general benchmarks (~48% avg). On HumanEval,
target >8% (above random, below CodeGen-350M's 12.8% due to less training data).
v28 is training on 5.08B tokens (73% Chinchilla-optimal): val_ppl=38.53 at step 6K,
predicted ~29 at completion. v29 ready on filtered data (2.04B clean tokens).

### 7.2 Stage 2: Synthetic Data Distillation from Qwen3-Coder-30B

```bash
# Phase 1: Generate synthetic Python completions with teacher (Q4K GPU)
apr distill apply configs/train/distill-synthetic.yaml --stage generate

# Phase 2: Train student on synthetic + original data
apr train apply configs/train/pretrain-350m-distill.yaml
```

**Produces**: `albor-distill-350m` — distilled model with teacher knowledge
**Exercises**: realizar (Q4K MoE inference), alimentar (data pipeline)
**Expected**: Meaningful HumanEval improvement over pretraining-only baseline.
Synthetic data approach avoids vocab mismatch between teacher (151K Qwen BPE)
and student (32K Albor BPE). Student initializes from v8 pretrained checkpoint.
See §4 for detailed architecture and data budget.

### 7.3 Stage 3: Instruction Fine-Tuning (LoRA/QLoRA)

```bash
apr finetune plan configs/train/finetune-lora.yaml        # Validate LoRA config + VRAM
apr finetune apply configs/train/finetune-lora.yaml
```

**Produces**: `albor-instruct-350m` — instruction-following model
**Exercises**: apr finetune, entrenar LoRA, alimentar (JSONL instruction data)
**Expected**: Better IFEval scores, improved structured output, chat capability.

### 7.4 Stage 4: Model Merging

```bash
apr merge plan \
  --models albor-distill-350m,albor-instruct-350m \
  --method slerp --weight 0.6 \
  --output ./checkpoints/albor-merged/
# Plan checks: architectures compatible, method valid, output size estimate

apr merge apply \
  --models albor-distill-350m,albor-instruct-350m \
  --method slerp --weight 0.6 \
  --output ./checkpoints/albor-merged/
```

**Produces**: `albor-merged-350m` — best-of-all-worlds model
**Exercises**: apr merge (SLERP, TIES, DARE algorithms)
**Expected**: Cherry-picks strengths from each variant. Potentially better
than any single model on diverse benchmarks.

### 7.5 Stage 5: Pruning

```bash
apr prune plan \
  --model ./checkpoints/albor-merged-350m/ \
  --method wanda --sparsity 0.5 \
  --output ./checkpoints/albor-pruned/
# Plan checks: model exists, sparsity in [0,1], output size estimate

apr prune apply \
  --model ./checkpoints/albor-merged-350m/ \
  --method wanda --sparsity 0.5 \
  --output ./checkpoints/albor-pruned/
```

**Produces**: `albor-pruned-175m` — half the parameters, similar performance
**Exercises**: apr prune (WANDA, SparseGPT, magnitude, depth pruning)
**Expected**: ~2-5% benchmark degradation at 50% sparsity. WANDA is well-studied
at larger scales (7B+) but less validated at 350M where there is less redundancy.
Depth pruning to ~18 layers yields ~260M params.

### 7.6 Stage 6: Quantization

```bash
apr quantize plan \
  --model ./checkpoints/albor-merged-350m/ \
  --method q4_k \
  --output ./checkpoints/albor-q4/
# Plan checks: model exists, format valid, output size estimate (~90MB)

apr quantize apply \
  --model ./checkpoints/albor-merged-350m/ \
  --method q4_k \
  --output ./checkpoints/albor-q4/

# Export for broad compatibility
apr export plan --model ./checkpoints/albor-q4/ --format gguf
apr export apply \
  --model ./checkpoints/albor-q4/ \
  --format gguf \
  --output ./release/albor-350m-q4_k.gguf
```

**Produces**: `albor-q4-350m` — 4-bit quantized, ~90MB on disk
**Exercises**: apr quantize, apr export (GGUF, SafeTensors)
**Expected**: <1% benchmark loss from Q4_K quantization. Model runs on any
device — phones, Raspberry Pi, browsers (WASM via trueno).

### 7.7 Benchmark Trajectory

Every stage is benchmarked. The trajectory itself is a key result.
Code completion metrics (HumanEval, FIM) are primary; general benchmarks are secondary.

| Stage | Model | Params | Size | HumanEval | MBPP | CPU tok/s |
|-------|-------|--------|------|-----------|------|-----------|
| 1 | albor-base (v28/v29) | 350M | ~700MB | TBD (predicted 8-15%) | TBD | — |
| 2 | albor-distill | 350M | ~700MB | ~10-20% | ~5-10% | — |
| 3 | albor-instruct | 350M | ~700MB | ~12-22% | ~7-12% | — |
| 4 | albor-merged | 350M | ~700MB | ~13-23% | ~8-13% | — |
| 5 | albor-pruned | ~175M | ~350MB | ~10-18% | ~6-10% | — |
| 6 | albor-q4 | 350M | ~90MB | ~12-22% | ~7-12% | >50 |

*Stage 1: v28 RUNNING (step 8.3K/38K), val_ppl=38.53 at step 6K. 28 training runs,
9 root causes found and fixed (RC1-RC9). HPO-validated hyperparameters (C-HPO-001).
129+ gaps filed, 52 contracts validated. Teacher pilot: 330/1K completions at 100%
pass rate (Qwen3-8B interim). Distillation uses sequence-level synthetic data (not
logit-level KD) due to vocab mismatch (teacher 151K vs student 32K). Any non-zero
HumanEval from a 350M sovereign-stack model is a meaningful result.*
