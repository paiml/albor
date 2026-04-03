# Albor LLM Specification

**Version**: 1.0.0
**Date**: 2026-03-08
**Author**: Noah Gift / Pragmatic AI Labs

> *Albor* (Spanish: "dawn") — A sovereign 350M Python code completion model.
> Distilled from Qwen3-Coder-30B-A3B, trained exclusively with the Sovereign AI stack.
> The goal: produce a **usable Python code assist** that runs anywhere Rust compiles,
> **and** validate every layer of the stack end-to-end.

---

## Status

| Metric | Value |
|--------|-------|
| Best val_ppl | **5.88** (v28 orig, step 3.5K); **38.53** (v28 fresh, step 6K, running) |
| HumanEval pass@1 | 0% (v4 baseline, pre-HPO) |
| Target HumanEval | ≥30% pass@1 |
| Training throughput | **12.3K tok/s, 38.7% MFU** (RTX 4090, v28) |
| Contracts validated | 50 |

**v28 era (v1.2)**: HPO-validated hyperparameters (C-HPO-001) + cosine horizon fix
(ALB-129) + fused gradient clipping (ALB-078) pushed val_ppl from 776 (v6) to 38.53
(v28 fresh, step 6K, predicted ~26 at completion). Data filtering complete: 850K files,
2.04B clean tokens for v29. Teacher pilot running: Qwen3-8B on gx10, 330/1K completions.

**Strategy**: Complete v28 full epoch → v29 on filtered data → distill with teacher
completions → target HumanEval pass@1 > 0%.

**Teacher selection (v1.1)**: Falsification analysis revealed Qwen3.5-35B-A3B is the
wrong teacher — no FIM support, no HumanEval benchmarks, designed for agentic reasoning
not code completion. Switched to Qwen3-Coder-30B-A3B-Instruct: Apache 2.0, FIM support,
code-specialized, simpler MoE architecture (`qwen3_moe`, 128 experts, standard GQA).

---

## 1. Objectives

### 1.1 Primary Goal
Train a **350M-parameter decoder-only transformer** for Python code completion using
only the Sovereign AI stack (`apr`, `entrenar`, `trueno`, `realizar`, `alimentar`,
`forjar`, `batuta`, `pmat`, `pv`, `renacer`, `presentar`, `certeza`).

### 1.2 Secondary Goal
Identify and fix every gap in the stack that blocks end-to-end LLM development.
The model is the proof; the stack improvements are the lasting value.

### 1.3 Success Criteria
| Criterion | Target | Method |
|-----------|--------|--------|
| HumanEval pass@1 | ≥30% | `apr eval --task humaneval` |
| MBPP pass@1 | ≥40% | `apr eval --task mbpp` |
| Inference latency | <50ms/tok CPU, <10ms GPU | `apr bench` |
| Model size | <200 MB (Q4) | `apr quantize` |
| Reproducible | Full retrain from `apr` commands | Seed + config |

---

## 2. Architecture

**Model**: LLaMA-style decoder-only transformer, 350M parameters.

| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| num_hidden_layers | 24 |
| num_attention_heads | 16 |
| num_kv_heads | 4 (GQA) |
| intermediate_size | 4096 (SwiGLU) |
| vocab_size | 32768 |
| max_position_embeddings | 1024 |
| rms_norm_eps | 1e-5 |
| rope_theta | 10000 |
| actual_params | ~370M |

**Reference**: phi-1-small (350M, h=1024, 20 layers) achieved **45% HumanEval** with
7B tokens of curated data. Albor has 24 layers (slightly larger) — architecturally
capable of matching this result with the right data and training strategy.

**Tokenizer**: ByteLevel BPE, 32K vocab (`models/albor-tokenizer-v2/tokenizer.json`).

→ Details: [components/architecture.md](components/architecture.md)

---

## 3. Strategy: Distillation-First Pipeline

The critical insight from phi-1: **data quality dominates model scale**. A 350M model
trained on 7B curated tokens beats a 350M model trained on 577B raw tokens. Our
strategy has three phases:

### Phase 1: Teacher Inference (3-5 days)
Generate high-quality Python completions from Qwen3-Coder-30B-A3B (MoE, 30.5B total,
3.3B active per token) running at Q4 on the RTX 4090. FIM-capable.

```
Qwen3-Coder-30B-A3B (Q4, 18.6 GB VRAM)
    │
    ├── Sequence-level: Generate 100-500K code completions
    │   Filter by execution correctness (rejection sampling)
    │   → 25-128M tokens of verified synthetic data
    │
    ├── FIM completions: <|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>
    │   → Code infill training data matching student's target task
    │
    └── Logit-level: Precompute top-k soft targets on training data
        → Richer gradient signal for student optimization
```

**Teacher throughput**: ~100-140 tok/s at Q4 on RTX 4090.
- 100K completions × 256 tok avg = ~71 hours
- 500K completions = ~15 days
- Start with 100K, scale if benchmarks justify

### Phase 2: Data Curation (parallel with Phase 1)
Quality-filter the existing 5.3B token codeparrot-clean corpus:
1. **Classifier filtering**: Train a quality classifier (random forest on GPT-4
   annotations) to score "textbook quality". Keep top 1-2B tokens.
2. **Synthetic textbooks**: Generate Python teaching material with explanations
   and code using the teacher model.
3. **CodeExercises**: Generate HumanEval-style function stubs + solutions.

### Phase 3: Student Training (2-3 days)
Two-stage training of the 350M student:

**Stage A — Pre-train on curated data**:
- 1-2B tokens of quality-filtered codeparrot-clean
- Standard causal LM loss, cosine LR decay
- Goal: teach basic Python fluency (val_loss < 4.0)

**Stage B — Distill from teacher**:
- SFT on teacher-generated completions (sequence-level distillation)
- Optionally: KD loss with precomputed soft targets (T=2, α=0.5)
- Finetune on CodeExercises-style synthetic data
- Goal: transfer teacher's coding knowledge (HumanEval ≥ 30%)

→ Details: [components/distillation.md](components/distillation.md)

---

## 4. Teacher: Qwen3-Coder-30B-A3B-Instruct

| Property | Value |
|----------|-------|
| Parameters | 30.5B total, 3.3B active (MoE: 128 experts, top-8) |
| Architecture | Standard GQA, 48 layers, h=2048, SwiGLU (moe_intermediate=768) |
| License | Apache 2.0 |
| VRAM (Q4) | 18.6 GB → fits RTX 4090 (5.4 GB headroom) |
| Throughput | ~100-140 tok/s generation on RTX 4090 |
| FIM | Yes (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) |
| Code capability | Code-specialized (SWE-bench 50.3%, FIM trained) |
| model_type | `qwen3_moe` (standard, NOT `qwen3_5_moe`) |

**Why this teacher**: Code-specialized with FIM support — directly matches our
target task. Apache 2.0 license. 30.5B knowledge at 3.3B inference cost.
Simpler MoE than Qwen3.5 (standard GQA, no DeltaNet, no shared expert).

**Previous teacher (Qwen3.5-35B-A3B) rejected**: Falsification revealed it has
no FIM support, no HumanEval benchmarks, and is designed for agentic reasoning
rather than code completion. The right teacher must match the student's task.

**Fallback**: Qwen2.5-Coder-3B (dense, 84.1% HumanEval, already supported by
realizar). Lower distillation ceiling but zero implementation risk. License:
Qwen Research (not Apache 2.0).

**Implementation status** (ALB-010, re-scoped for `qwen3_moe`):
- MoE routing, expert dispatch from Steps 1-3: reusable (architecture-agnostic)
- Config parsing: needs adaptation (`qwen3_moe` vs `qwen3_5_moe`)
- Tensor mapping: simpler (128 experts, no shared expert, standard attn names)
- Model files: need download + APR import

→ Details: [components/distillation.md](components/distillation.md)

---

## 5. Data Pipeline

### 5.1 Available Data

| Dataset | Tokens | Seq Len | Quality | Status |
|---------|--------|---------|---------|--------|
| codeparrot-clean-2M | 5.3B | 1024 | Raw (unfiltered) | Ready |
| CodeSearchNet Python | 133M | 2048 | Medium | Ready |
| Synthetic completions | 0 | — | High (teacher-generated) | **Phase 1** |
| Synthetic textbooks | 0 | — | High (teacher-generated) | **Phase 2** |
| CodeExercises | 0 | — | High (HumanEval-style) | **Phase 2** |

### 5.2 Data Quality Strategy

The phi-1 recipe:
1. **Filter** 35B raw → 6B "textbook quality" via GPT-4-trained classifier
2. **Generate** 1B synthetic textbook tokens via GPT-3.5
3. **Generate** 180M exercise tokens (function stubs + solutions)
4. **Two-stage**: Pretrain on (1)+(2), finetune on (3)

Our adaptation:
1. **Filter** 5.3B codeparrot-clean → ~1-2B via teacher-annotated classifier
2. **Generate** synthetic completions from Qwen3-Coder-30B-A3B (100-500K samples)
3. **Generate** CodeExercises-style data + FIM completions from teacher
4. **Two-stage**: Pretrain on filtered data, SFT on synthetic data

→ Details: [components/data.md](components/data.md)

---

## 6. Training Infrastructure

### 6.1 Hardware
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM, sm_89)
- **CPU**: Intel Xeon (48 cores, 128 GB RAM)
- **Storage**: NVMe RAID-0 (`/mnt/nvme-raid0/`)

### 6.2 Software Stack
| Tool | Role |
|------|------|
| `apr` (aprender) | CLI: train, eval, distill, quantize, export |
| `entrenar` | Training engine: CUDA trainer, GPU-resident grad accum |
| `trueno` | Tensor ops: cuBLAS GEMMs, PTX kernels, RMSNorm |
| `realizar` | Inference: teacher model, eval generation, MoE dispatch |
| `alimentar` | Data: import, tokenize, FIM transform, quality filter |
| `forjar` | Pipeline: DAG orchestration, multi-machine, idempotent |

### 6.3 VRAM Budget

| Config | VRAM | Notes |
|--------|------|-------|
| Student training (350M) | ~18 GB | Weights + AdamW + workspace |
| Teacher inference (30.5B Q4) | ~19 GB | Q4 weights (18.6 GB) + KV cache + activations |
| Student eval | ~5 GB | Forward only |

Teacher and student run sequentially, never simultaneously.

### 6.4 Known Issues
- All-shards-in-RAM data loading: 19 shards → 39 GB swap → 20% throughput loss
- Noise-scale logging per micro-batch (fixed in entrenar@eb1b584, not yet deployed)
- Checkpoint resume: weight loading via `with_model()` broken for CUDA trainer

→ Details: [components/infrastructure.md](components/infrastructure.md)

---

## 7. Training Results (Historical)

| Run | Steps | Tokens | val_ppl | tok/s | MFU | Notes |
|-----|-------|--------|---------|-------|-----|-------|
| v3 | 28K | 918M | 1018 | 6.7K | 19.3% | No cosine decay, small batch (ALB-079/080) |
| v6 | 2K | 64M | 776 | 6.5K | 18.8% | ALB-092 fixed, matched val data |
| v27 | 10.2K | 1.3B | 9.39 | 14.7K | 46.1% | HPO-validated, ALB-129 (diverged to 82) |
| v28 orig | 5.4K | 708M | **5.88** | 14.7K | 46.1% | ALB-129 fix confirmed. Killed. |
| **v28 fresh** | **6.8K** | **891M** | **38.53** | **12.3K** | **38.7%** | **RUNNING** (ETA ~3.5 days) |

**Lessons**:
1. v6 era: Data quality is the bottleneck (64M tokens → val_ppl 776)
2. v28 era: HPO + correct cosine schedule + fused clipping → 20× val_ppl improvement
3. Distillation remains the multiplier for HumanEval — raw pre-training alone won't suffice

→ Details: [components/training.md](components/training.md)

---

## 8. Evaluation

### 8.1 Primary Benchmarks
| Benchmark | Problems | Metric | Our Target | phi-1-small |
|-----------|----------|--------|------------|-------------|
| HumanEval | 164 | pass@1 | ≥30% | 45% |
| MBPP | 974 | pass@1 | ≥40% | 55% |

### 8.2 Secondary Metrics
| Metric | Target |
|--------|--------|
| val_ppl (same-dist) | <100 |
| FIM accuracy | >60% |
| Inference speed (CPU Q4) | <50ms/tok |
| Model size (Q4) | <200 MB |

### 8.3 Evaluation Protocol
```bash
apr eval --task humaneval --model checkpoints/albor-distill/
apr eval --task mbpp --model checkpoints/albor-distill/ --k 1,5,10
```

→ Details: [components/evaluation.md](components/evaluation.md)

---

## 9. Improvement Ladder

Post-distillation optimization path:

```
albor-base     → Pre-trained on curated data (Phase 3A)
albor-distill  → SFT on teacher completions (Phase 3B)    ← primary target
albor-instruct → Instruction fine-tune (LoRA)
albor-merged   → Merge with complementary checkpoint
albor-pruned   → Structured pruning for efficiency
albor-q4       → 4-bit quantization for deployment
```

Each stage exercises a different `apr` subcommand and may reveal new stack gaps.

---

## 10. Gap Register Summary

**129+ gaps filed, 50 contracts validated.** Key gaps:

| ID | Gap | Status | Notes |
|----|-----|--------|-------|
| ALB-010 | `qwen3_moe` teacher loading | IN PROGRESS | Steps 4-5 DONE, 6-8 remaining |
| ALB-078 | Fused gradient clipping | **FIXED** | MFU 19%→38.7% |
| ALB-129 | Cosine schedule horizon | **FIXED** | val_ppl 82→5.88 |
| ALB-089 | GPU-accelerated inference | DOGFOODING | For fast eval |

→ Full register: [components/gaps.md](components/gaps.md)
→ Bug patterns: [components/bugs.md](components/bugs.md)

---

## 11. Execution Plan

### 11.1 Critical Path

```
Download → Import → Q4K → Adapt config → Dogfood teacher → Pilot → Scale → SFT → Eval
  2h        30m     30m    1-3 days      1-2 days        4h    3-5 days  12h    2h
```

Steps 5-6 (prompt extraction + rejection pipeline) run parallel with Steps 3-4.

### 11.2 Milestones

| Milestone | Target Date | Criterion |
|-----------|-------------|-----------|
| M0: Teacher downloaded + imported | 2026-03-09 | APR + Q4K files on disk |
| M1: Teacher inference works | 2026-03-13 | Generates valid Python + FIM via realizar |
| M1.5: Pilot generation | 2026-03-14 | 1K prompts, rejection rate > 30% |
| M2: 100K completions generated | 2026-03-19 | Rejection-sampled, execution-verified |
| M3: Student SFT complete | 2026-03-21 | Trained on teacher completions |
| M4: HumanEval > 0% | 2026-03-21 | At least 1 problem solved |
| M5: HumanEval ≥ 30% | 2026-03-28 | With iteration + CodeExercises |
| M6: Q4 model published | 2026-04-04 | HuggingFace model card |

### 11.3 Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| realizar Q4K+MoE untested | M1 slips 1-2 weeks | Medium | Fallback: Qwen2.5-Coder-3B (dense, already works) |
| Throughput < 50 tok/s | M2 slips 1-2 weeks | Medium | Batch prompts, optimize kernel dispatch |
| Rejection rate < 10% | Bad SFT data | Low | Adjust temperature, inspect failure modes |
| Student doesn't hit 30% | M5 fails | Medium | More data, CodeExercises FT, logit KD |
| `apr export` missing SafeTensors | M6 blocked | Unknown | New gap — verify before M5 |

---

## 12. Component Specifications

| Component | File | Lines |
|-----------|------|-------|
| Distillation Strategy | [components/distillation.md](components/distillation.md) | ≤500 |
| Model Architecture | [components/architecture.md](components/architecture.md) | ≤500 |
| Training Configuration | [components/training.md](components/training.md) | ≤500 |
| Data Pipeline | [components/data.md](components/data.md) | ≤500 |
| Evaluation Framework | [components/evaluation.md](components/evaluation.md) | ≤500 |
| Infrastructure | [components/infrastructure.md](components/infrastructure.md) | ≤500 |
| Gap Register | [components/gaps.md](components/gaps.md) | ≤500 |

---

## 13. Engineering Discipline

Eight hard rules from 52 bugs found during ALB-040 dogfooding. See `CLAUDE.md`.

1. **Tracing not printf** — Use renacer, never eprintln!
2. **Buffer sizes provable** — Algebraic verification at every kernel boundary
3. **Hyperparams end-to-end** — YAML → constructor, never default_params()
4. **Gradient flow verified** — Every param gets non-zero gradient after 1 step
5. **Activation grad clipped** — At GPU→CPU boundary
6. **Stream sync before D2H** — CU_STREAM_NON_BLOCKING requires explicit sync
7. **Contracts before code** — pv validate → implement → pv audit → dogfood
8. **Five Whys root cause** — Trace through brick boundaries before any fix

---

## 14. References

- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) — phi-1 methodology
- [MiniLLM](https://arxiv.org/abs/2306.08543) — Reverse KL for generative KD
- [Sparse Logit Sampling](https://arxiv.org/abs/2503.16870) — Efficient logit storage
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — Sequence-level distillation
- [Magicoder](https://github.com/ise-uiuc/magicoder) — OSS-Instruct synthetic data
- [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) — Teacher model
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) — Previous teacher (rejected: no FIM)
