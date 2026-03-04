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

### 9.3 Entrenar Native DDP (Complete)

entrenar has full distributed data parallelism infrastructure
([entrenar#133](https://github.com/paiml/entrenar/issues/133)), superseding
the repartir approach:

**Implemented (all wired end-to-end):**
- **Wire protocol v2**: TCP-based message framing with `BlockGradientPayload`,
  `AveragedBlockGradient`, `NonBlockGradientPayload`, `AveragedNonBlockGradient`
- **GradientServer**: Coordinator that collects gradients from N workers, averages
  them (per-block AllReduce), and broadcasts averaged gradients back
- **WorkerClient**: Worker-side TCP client that sends/receives gradient payloads
- **PerBlockGradientAccumulator**: CPU-side gradient accumulator for AllReduce
  (same one used by ALB-066 single-GPU gradient accumulation)
- **RingAllReduce**: Ring-based averaging for N workers
- **DistributedCudaTrainer**: `train_batch()` → forward+backward → per-block
  AllReduce → optimizer step. Wraps `CudaTransformerTrainer` with distributed comm
- **`train_loop_cuda_distributed()`**: Full training loop with data sharding by rank,
  coordinator thread auto-spawn (rank 0), worker connection, epoch iteration
- **`spawn_coordinator_thread()`**: Background thread running `GradientServer` for
  rank 0 process
- **CLI flags**: `--distributed --world-size N --rank R` inject distributed config
  into YAML at runtime
- **11 integration tests**: C-DDP-001 weight consistency via BLAKE3, 4-worker ring
  AllReduce, per-block reverse-order AllReduce

**Architecture**:
```
Process 0 (rank=0):                     Process 1 (rank=1):
  GradientServer (bg thread)
  DistributedCudaTrainer                  DistributedCudaTrainer
    └─ CudaTransformerTrainer (GPU 0)       └─ CudaTransformerTrainer (GPU 1)
    └─ WorkerClient → TCP ─────────────────── WorkerClient → TCP
```

### 9.4 Original Repartir Gaps (Stretch)

The original plan for distributed training via a standalone `repartir` crate
is now partially superseded by entrenar's native DDP, but some gaps remain
relevant for cross-vendor GPU support:

**Gap ALB-002**: Ring all-reduce (now partially implemented in entrenar itself).
**Gap ALB-004**: Unified CUDA + wgpu backend dispatch in entrenar.
**Gap ALB-005**: trueno wgpu backward pass (gradient WGSL shaders).

The distillation architecture (Section 9.2) achieves multi-machine utilization
without any of these.

### 9.5 W5700X Role

The W5700X GPUs (2x 8GB each) can assist with:
- **Eval inference**: Run benchmarks on latest checkpoint via wgpu/Vulkan
- **Partial KV cache offload**: Assist CPU-based teacher inference
- **Future**: Participate in gradient-parallel training once ALB-005 is resolved
