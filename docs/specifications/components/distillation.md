# Distillation Strategy

**Parent**: [albor-llm-spec.md](../albor-llm-spec.md) §3

---

## 1. Why Distillation

Pre-training 350M params from scratch on a single RTX 4090 hit val_ppl=776
after 64M tokens (v6, step 2000). Extrapolating: competitive results require
7B+ tokens of curated data at current convergence rate. The phi-1 result proves
a 350M model CAN reach 45% HumanEval — but only with data quality far beyond
raw codeparrot-clean.

**Distillation is the multiplier**: transfer a 35B model's coding knowledge
into 350M parameters via high-quality synthetic data, not brute-force token
volume.

---

## 2. Teacher: Qwen3.5-35B-A3B

| Property | Value |
|----------|-------|
| Total params | 35B (256 experts + 1 shared) |
| Active params | 3B per token (top-8 routing) |
| Architecture | 40 layers: 30 Gated DeltaNet + 10 full GQA |
| FFN (experts) | SwiGLU, intermediate_size=512 |
| License | Apache 2.0 |
| VRAM (Q4) | ~19.7 GB → fits RTX 4090 |
| Throughput | ~100 tok/s generation at Q4 |
| APR model | `/mnt/nvme-raid0/models/qwen35-moe.apr` (67 GB, mmap zero-copy) |
| HF source | `/mnt/nvme-raid0/models/Qwen3.5-35B-A3B/` (14 SafeTensors shards, imported) |

### 2.1 Why This Teacher

1. **Knowledge density**: 256 MoE experts encode rich distributional info.
   Soft targets from MoE are more informative than dense model logits.
2. **Inference efficiency**: 3B active → ~100 tok/s on 4090. Fast enough
   for 100K+ completion generation in 3-4 days.
3. **License**: Apache 2.0 — no restrictions on derivative models.
4. **Single-GPU**: Fits at Q4 with 4.3 GB headroom for KV cache.

### 2.2 Implementation Status (ALB-010)

| Step | Description | Status |
|------|-------------|--------|
| 1 | MoE routing (top-k gating, load balancing) | MERGED |
| 2 | Expert dispatch (scatter/gather) | MERGED |
| 3 | Forward pass integration | MERGED |
| 4 | Config parsing (`Qwen3_5MoeForConditionalGeneration`) | MERGED |
| 5a | Unit tests (routing correctness) | MERGED |
| 5b | Integration tests (15,053+ pass) | MERGED |
| 6 | APR tensor→MoE slot mapping | **IN PROGRESS** |
| 7 | End-to-end generation dogfood | BLOCKED on 6 |

### 2.3 Weight Loading: APR, Not SafeTensors

The model is already imported to APR format (`qwen35-moe.apr`, 67 GB). realizar
has `AprV2ReaderRef` — mmap-based, zero-copy, 10.9 MB RSS. The Five Whys
analysis revealed PR #135 conflated "read a file format" with "map tensors to
MoE slots." The file format is solved (APR). The real work is Step 6: parsing
tensor names like `model.language_model.layers.{L}.mlp.experts.{E}.gate_proj`
and loading each into the correct expert slot in the MoE runtime.

This is sovereign: we use our own format end-to-end. No SafeTensors dependency
at inference time.

### 2.4 Fallback Teacher

**Qwen2.5-Coder-3B** (dense, already supported by realizar). Lower quality
ceiling but zero implementation risk. If Step 6 blocks for >3 days, switch to
this fallback and begin generating data immediately.

---

## 3. Distillation Methods

### 3.1 Sequence-Level Distillation (Primary)

Generate completions from teacher, filter by correctness, SFT student on
verified completions. This is the simplest and most proven approach.

```
Teacher generates completion → Execute → Pass? → Keep for SFT
                                           ↓
                                        Discard
```

**Pipeline**:
1. Prepare prompts: function signatures, docstrings, partial implementations
2. Generate N completions per prompt (temperature=0.8, top-p=0.95)
3. Execute each completion in sandboxed Python (timeout=10s)
4. Keep only completions that pass execution + type checks
5. Deduplicate by AST similarity
6. Package as SFT dataset: `(prompt, verified_completion)` pairs

**Rejection sampling rate**: Expect ~30-50% pass rate for code completions.
With 100K generations → ~40K verified samples → ~10M tokens.

### 3.2 Logit-Level Distillation (Secondary)

Precompute teacher's soft targets on training data. Student trains with
combined loss: `L = α·KD(soft_targets, student_logits) + (1-α)·CE(labels, student_logits)`

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Temperature (T) | 2.0 | Standard for code; smooths distribution |
| Alpha (α) | 0.5 | Equal weight KD/CE |
| Top-k storage | 12 | Importance sampling sufficient |

**Storage**: Top-12 logits per token × 1B training tokens × 4 bytes × 2 (idx+val)
= ~96 GB. Feasible on NVMe RAID-0 (~2 TB available).

**Throughput**: Teacher forward pass on training data at ~100 tok/s.
1B tokens / 100 tok/s = ~116 days. NOT feasible on single GPU.
→ Only apply logit distillation to the curated subset (1-2B tokens after filtering).

### 3.3 Decision: Start Sequence-Level

Sequence-level distillation:
- Simpler to implement (no logit storage infrastructure)
- Execution filtering guarantees functional code
- Proven at scale (DeepSeek-R1, phi-1)
- Can begin generating data as soon as teacher works (M1)

Add logit-level distillation later if sequence-level alone doesn't hit ≥30%
HumanEval.

---

## 4. Data Generation Pipeline

### 4.1 Prompt Sources

| Source | Prompts | Style |
|--------|---------|-------|
| HumanEval-style stubs | ~500 | Function signature + docstring |
| MBPP-style tasks | ~1000 | Natural language → code |
| codeparrot-clean headers | ~50K | Extract first function/class from samples |
| Synthetic textbook exercises | ~10K | Multi-step problems with hints |
| OSS-Instruct (Magicoder-style) | ~20K | Real code → inspired problem |

### 4.2 Generation Config

```yaml
teacher:
  model: "qwen35-moe.apr"
  quantization: "q4"
  temperature: 0.8
  top_p: 0.95
  max_tokens: 512
  num_completions: 5  # per prompt, pick best

execution:
  sandbox: "bubblewrap"  # or nsjail
  timeout_seconds: 10
  python_version: "3.11"
  allowed_imports: ["math", "collections", "itertools", "typing", "re", "string"]

filtering:
  require_execution_pass: true
  require_type_check: false  # aspirational
  dedup_by_ast: true
  min_tokens: 20
  max_tokens: 512
```

### 4.3 Generation Schedule

| Phase | Prompts | Completions | ~Tokens | Wall Time |
|-------|---------|-------------|---------|-----------|
| Pilot (M1) | 1K | 5K | ~1.3M | ~3.5h |
| Scale 1 (M2) | 20K | 100K | ~26M | ~71h |
| Scale 2 (M2+) | 100K | 500K | ~128M | ~15 days |

Start with pilot to validate pipeline and measure rejection rate.

---

## 5. Student Training Strategy

### 5.1 Stage A: Pre-train on Curated Data

Train student from random init on quality-filtered codeparrot-clean.

| Config | Value |
|--------|-------|
| Data | 1-2B tokens (filtered from 5.3B raw) |
| Loss | Causal LM (cross-entropy) |
| LR | 3e-4, cosine decay |
| Warmup | 2000 steps |
| Batch | 32K tokens/step |
| Steps | ~40K-60K |
| Target | val_loss < 4.0 |

**Quality classifier**: Train random forest on Qwen3.5 annotations.
Annotate ~10K samples as "textbook quality" (1) or not (0). Apply classifier
to full 5.3B tokens, keep samples scoring > 0.5.

### 5.2 Stage B: SFT on Teacher Completions

Finetune Stage A checkpoint on teacher-generated, execution-verified data.

| Config | Value |
|--------|-------|
| Data | 10-128M tokens (verified completions) |
| Loss | Causal LM on completions only |
| LR | 1e-5 (10x lower than Stage A) |
| Warmup | 200 steps |
| Batch | 16K tokens/step |
| Epochs | 3-5 over synthetic data |
| Target | HumanEval ≥ 20% |

### 5.3 Stage C: CodeExercises Finetuning (Optional)

If Stage B < 30% HumanEval, finetune on HumanEval-style exercises.

| Config | Value |
|--------|-------|
| Data | 5-20M tokens (exercise stubs + solutions) |
| Loss | Causal LM on solutions only |
| LR | 5e-6 |
| Epochs | 5-10 |
| Target | HumanEval ≥ 30% |

---

## 6. phi-1 Recipe Reference

Microsoft's phi-1-small (350M, h=1024, 20 layers) achieved 45% HumanEval:

| Stage | Data | Tokens | Description |
|-------|------|--------|-------------|
| Pre-train | CodeTextbooks | 6B | GPT-4-filtered web data ("textbook quality") |
| Pre-train | Synthetic textbooks | 1B | GPT-3.5-generated Python teaching material |
| Finetune | CodeExercises | 180M | Function stubs + solutions |

**Key lessons**:
1. Data quality > data quantity (7B curated > 577B raw)
2. Two-stage (pretrain + finetune) outperforms single-stage
3. Exercise-style data critical for HumanEval performance
4. 350M architecture is sufficient — no need for larger student
5. Filtering by "textbook quality" classifier is the highest-leverage step

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| ALB-010 Step 6 (tensor mapping) blocks | Can't use best teacher | Fallback to Qwen2.5-Coder-3B |
| Low rejection sampling rate | Fewer verified samples | Increase temperature, use more prompts |
| Student doesn't converge in Stage B | Wasted compute | Monitor val_ppl every 100 steps, early stop |
| Execution sandbox escape | Security | Use bubblewrap/nsjail, restricted imports |
| Teacher generates memorized code | Copyright risk | Deduplicate against training data |
| Logit storage too large | Disk pressure | Top-k=12 with importance sampling |

---

## 8. Success Metrics by Phase

| Phase | Metric | Target | Measurement |
|-------|--------|--------|-------------|
| Phase 1 | Teacher generates valid Python | Yes/No | Manual inspection of 100 samples |
| Phase 1 | Rejection sampling rate | > 30% | Automated execution pipeline |
| Phase 2 | Quality classifier accuracy | > 80% F1 | Hold-out validation set |
| Phase 3A | Pre-train val_loss | < 4.0 | entrenar eval |
| Phase 3B | HumanEval pass@1 | ≥ 20% | `apr eval --task humaneval` |
| Phase 3C | HumanEval pass@1 | ≥ 30% | `apr eval --task humaneval` |

---

## 9. References

- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) — phi-1 recipe
- [MiniLLM](https://arxiv.org/abs/2306.08543) — Reverse KL for generative KD
- [Sparse Logit Sampling](https://arxiv.org/abs/2503.16870) — Top-k logit storage
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — Sequence-level distillation
- [Magicoder](https://github.com/ise-uiuc/magicoder) — OSS-Instruct synthetic data
