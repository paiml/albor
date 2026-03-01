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
target >8% (above random, below CodeGen-350M's 12.8% due to less training data)

### 7.2 Stage 2: Knowledge Distillation from Qwen3-Coder-Next

```bash
# Plan: check teacher fits in RAM, estimate logit disk usage
apr distill plan configs/train/distill.yaml

# Apply phase 1: Pre-compute teacher logits on intel (300GB RAM, CPU inference)
apr distill apply configs/train/distill.yaml --stage precompute

# Apply phase 2: Distill into student on lambda (4090)
apr distill apply configs/train/distill.yaml --stage train
```

**Produces**: `albor-distill-350m` — distilled model with teacher knowledge
**Exercises**: realizar (teacher inference), apr distill, alimentar (logit storage)
**Expected**: Moderate improvement — absorbs coding patterns from 80B teacher.
Estimated +2-7 points on HumanEval via logit-level KD. Note: MoE→dense
distillation is uncharted at this scale; the architecture mismatch (DeltaNet+MoE
teacher → LLaMA-style dense student) may limit transfer compared to dense→dense
distillation (e.g., GPT-3.5→phi-1).

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
| 1 | albor-base | 350M | ~700MB | ~8% | ~8% | — |
| 2 | albor-distill | 350M | ~700MB | ~13-15% | ~10-12% | — |
| 3 | albor-instruct | 350M | ~700MB | ~14-16% | ~11-13% | — |
| 4 | albor-merged | 350M | ~700MB | ~15-17% | ~12-14% | — |
| 5 | albor-pruned | ~175M | ~350MB | ~12-14% | ~10-12% | — |
| 6 | albor-q4 | 350M | ~90MB | ~14-16% | ~11-13% | >50 |

*Numbers are estimates. The distillation gain (+2-7 points over base) assumes
500M-2B tokens of teacher logits. This is conservative — published distillation
results show larger gains with dense teachers (phi-1 used GPT-3.5, a dense
model). Our MoE→dense distillation path is uncharted at 350M scale. The FIM
column is removed because there is no standardized FIM benchmark — we will
define our own eval and report absolute numbers, not targets.
CPU tok/s measured on Xeon at Q4.*
