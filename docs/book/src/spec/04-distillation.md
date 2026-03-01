# 4. Distillation Teacher: Qwen3-Coder-Next

### 4.1 Teacher Model Profile

| Property | Value |
|----------|-------|
| Model | [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) |
| Parameters | 80B total, 3B activated (MoE) |
| Architecture | Hybrid: DeltaNet + Gated Attention + MoE (512 experts, 10 active) |
| Hidden dim | 2048 |
| Context | 256K tokens |
| License | Apache 2.0 |
| Specialization | Coding, agentic reasoning, tool use |

### 4.2 Why This Teacher

- **Apache 2.0**: Legally clean for distillation, no license contamination
- **MoE → Dense distillation**: Compresses 512 experts' collective knowledge
  into a small dense student. The student benefits from all experts without
  the MoE routing overhead.
- **Coding focus**: Distilled student inherits strong code capabilities,
  making it competitive on HumanEval/MBPP — benchmarks where tiny models
  normally fail.
- **Novel architecture** (DeltaNet + MoE): Exercising realizar's model
  loading on a non-standard architecture is exactly the kind of gap-finding
  that validates the stack.

### 4.3 Distillation Architecture

```
┌───────────────────────────────┐                          ┌───────────────────────────┐
│  intel (300 GB RAM)           │    pre-computed logits    │  lambda (RTX 4090)        │
│                               │    via Parquet shards     │                           │
│  Qwen3-Coder-Next 80B fp16   │ ────────────────────────► │  Student: albor-350M      │
│  Running on CPU (32 threads)  │                           │  KD loss + CE loss        │
│  ~160 GB model in RAM         │                           │  Training on GPU          │
│  ~140 GB for KV cache         │                           │                           │
│                               │                           │  entrenar distill         │
│  realizar inference           │                           │    --teacher-logits       │
└───────────────────────────────┘                           └───────────────────────────┘
```

### 4.4 Pre-Computed Logits Strategy

Rather than running teacher inference online during distillation (bottlenecked
by CPU inference speed), we **pre-compute teacher logits offline**:

1. Intel box runs Qwen3-Coder-Next on all training data batches
2. Teacher top-k logits (k=128) saved as sharded Parquet via `alimentar`
3. Lambda loads pre-computed logits during distillation at full GPU speed
4. No network bottleneck during training — all data is local

```bash
# Step 0: Plan — check teacher fits in intel RAM, estimate logit disk usage
apr distill plan configs/train/distill.yaml

# Step 1: Pre-compute teacher logits on intel (offline, can run for days)
apr distill apply configs/train/distill.yaml --stage precompute

# Step 2: Train student on lambda using pre-computed logits
apr distill apply configs/train/distill.yaml --stage train --seed 42
```

**Estimated teacher throughput on CPU (Xeon 32T, 80B fp16)**:
- ~2-5 tok/s for full 80B in fp16
- For 10B tokens of training data: ~23-58 days at full precision
- At Q8 (~80GB, faster): ~5-15 tok/s → ~8-23 days
- Strategy: pre-compute on a representative subset (~1-2B tokens), not full corpus

### 4.5 Distillation Data Budget

| Approach | Teacher Tokens | Time (est.) | Quality |
|----------|---------------|-------------|---------|
| Full corpus (10B tokens) | 10B | ~30-60 days | Best |
| Representative subset (2B) | 2B | ~6-12 days | Good — focus on diverse/hard examples |
| Curated hard examples (500M) | 500M | ~2-3 days | Targeted — highest knowledge density |

**Recommended**: Start with the local ground truth corpora (~50-100M raw tokens)
plus curated hard examples from StarCoder Python (~400M tokens) for ~500M total.
The ground truth corpora should be distilled **first** — they are our highest
quality data and benefit most from teacher knowledge. Scale to 2B with broader
StarCoder data if benchmarks justify the compute. Python-only focus means all
teacher compute goes toward the language we care about.
