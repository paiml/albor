# 9. Distributed Training Architecture

### 9.1 Machine Roles (Revised)

With 300 GB RAM on the intel box, the architecture is asymmetric:

| Machine | Primary Role | Secondary Role |
|---------|-------------|----------------|
| lambda (4090) | Student training (GPU) | — |
| intel (300GB RAM) | Teacher inference (CPU), logit pre-computation | Eval runner, data pipeline, checkpoint backup |

### 9.2 Distillation Split (Primary Distributed Architecture)

The natural multi-machine split is **teacher on intel, student on lambda**:

```
┌───────────────────────────────┐                          ┌───────────────────────────┐
│  intel (300 GB RAM)           │    pre-computed logits    │  lambda (RTX 4090)        │
│                               │    as sharded Parquet     │                           │
│  Qwen3-Coder-Next 80B fp16   │ ────────────────────────► │  albor-350M student       │
│  Full model in CPU RAM        │    (rsync / NFS)          │  KD loss + CE loss        │
│  realizar CPU inference       │                           │  Full GPU speed training  │
│  ~5-15 tok/s                  │                           │                           │
│                               │ ◄──── checkpoints ─────  │  apr distill apply    │
│  Concurrent eval runner       │    (rsync / NFS)          │                           │
└───────────────────────────────┘                           └───────────────────────────┘
```

This requires **no gradient sync, no ring all-reduce, no distributed training
framework** for the distillation stage. The teacher pre-computes logits offline;
the student trains at full GPU speed against stored logits. Simple and effective.

### 9.3 Gradient-Parallel Training (Future / Stretch)

For pure pre-training (Stage 1), distributed gradient-parallel across both
machines remains a stretch goal. The gaps are significant:

**Gap ALB-002**: Implement ring all-reduce in repartir.
**Gap ALB-003**: Wire repartir gradient sync into entrenar's training loop.
**Gap ALB-004**: Unified CUDA + wgpu backend dispatch in entrenar.
**Gap ALB-005**: trueno wgpu backward pass (gradient WGSL shaders).

These are deferred to a later phase. The distillation architecture (Section 9.2)
achieves multi-machine utilization without them.

### 9.4 W5700X Role

The W5700X GPUs (2x 8GB each) can assist with:
- **Eval inference**: Run benchmarks on latest checkpoint via wgpu/Vulkan
- **Partial KV cache offload**: Assist CPU-based teacher inference
- **Future**: Participate in gradient-parallel training once ALB-005 is resolved
