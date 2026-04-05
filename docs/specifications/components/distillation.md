# Distillation Strategy

**Parent**: [albor-llm-spec.md](../albor-llm-spec.md) §3

---

## 1. Why Distillation

Pre-training 350M params on raw codeparrot-clean with HPO-validated
hyperparameters reached val_ppl=38.53 at step 6K (v28 fresh, running). The
v28 original run achieved val_ppl=5.88 at step 3.5K — best ever — but
HumanEval pass@1 remains 0% on raw pre-training data alone. The phi-1 result
proves a 350M model CAN reach 45% HumanEval — but only with curated data +
SFT on teacher completions.

**Distillation is the multiplier**: transfer a 30B MoE teacher's coding
knowledge into 350M parameters via high-quality synthetic data. Pre-training
on filtered data (v29, 2.04B clean tokens) provides the foundation; teacher
completions provide the HumanEval-targeting signal.

---

## 2. Teacher: Qwen3-Coder-30B-A3B-Instruct

| Property | Value |
|----------|-------|
| Total params | 30.5B (128 experts, top-8 routing) |
| Active params | 3.3B per token |
| Architecture | 48 layers, h=2048, 32 Q-heads, 4 KV-heads, standard GQA |
| FFN (experts) | SwiGLU, moe_intermediate_size=768 |
| License | Apache 2.0 |
| model_type | `qwen3_moe` |
| VRAM (Q4) | ~18.6 GB → fits RTX 4090 (5.4 GB headroom) |
| Throughput | ~100-140 tok/s generation at Q4 |
| FIM tokens | `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` |
| HF source | `Qwen/Qwen3-Coder-30B-A3B-Instruct` (16 shards, 61.1 GB BF16) |
| SWE-bench | 50.3% verified |
| vocab_size | 151936 |

### 2.1 Why This Teacher

1. **Code-specialized**: Trained specifically for code generation + FIM.
   Matches our student's target task (Python code completion).
2. **FIM support**: Native fill-in-middle tokens — can generate infill training
   data directly. Qwen3.5-35B-A3B lacks this entirely.
3. **Apache 2.0**: No restrictions on derivative models.
4. **Inference efficiency**: 3.3B active → ~100-140 tok/s on 4090.
5. **Simpler MoE**: Standard GQA, 128 experts, no shared expert, no DeltaNet.
   Significantly less engineering than `qwen3_5_moe`.

### 2.2 Teacher Selection: Falsification Record

Qwen3.5-35B-A3B was the original teacher choice. Falsification revealed three
fatal flaws:

1. **No FIM support** — cannot generate code infill training data
2. **No HumanEval/MBPP benchmarks** — code capability unverifiable
3. **Designed for agentic reasoning** — thinking mode (`<think>...</think>`),
   not direct code completion

The `qwen3_5_moe` architecture is also more complex (Gated DeltaNet, 256
experts + shared expert, hybrid linear attention) with no benefit for our
code completion use case. ALB-010 Steps 1-5b MoE routing/dispatch code is
reusable; config parsing needs adaptation for `qwen3_moe`.

### 2.3 Implementation Status (ALB-010, re-scoped)

| Step | Description | Status |
|------|-------------|--------|
| 1-3 | MoE routing, dispatch, forward (architecture-agnostic) | MERGED (PR #133) |
| 4 | Config: `has_qk_norm=true`, QK norm tensor names | DONE (`0c495ef`) |
| 5 | Per-expert weight loading (Layout 2 fallback) | DONE (`0c495ef`) |
| 6 | Download model + APR import + tensor→slot mapping | **IN PROGRESS** |
| 7 | Q4K quantization via `apr quantize` | TODO |
| 8 | End-to-end generation dogfood | BLOCKED on 6-7 |

**Interim teacher**: Qwen3-8B (dense) serving on gx10 via `realizar serve`
(2 instances, ports 8090/8091). Teacher completions pilot: 330/1K prompts
completed at 36/h throughput before connection reset. 100% pass rate on
generated completions — no failures in first 330 samples.

### 2.4 Weight Loading: APR Format

Same sovereign approach: download HF shards → `apr import` → APR file →
`apr quantize --scheme q4k` → Q4K APR → realizar loads via
`load_quantized_weights_with_type()`. Q4K GEMV kernels already exist in
realizar (multi-warp vectorized, DP4A, fused RMSNorm+Q4K).

### 2.5 Fallback Teacher

**Qwen2.5-Coder-3B** (dense, 84.1% HumanEval, 85.7% HumanEval-FIM).
Already supported by realizar. Zero implementation risk.
Caveats: Qwen Research license (not Apache 2.0), lower distillation ceiling
(dense 3B vs MoE 30.5B with 128 expert banks).

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

| Source | Prompts | Style | Status |
|--------|---------|-------|--------|
| codeparrot-clean headers | **50,000** | Function/method/class from filtered files | **DONE** |
| HumanEval-style stubs | ~500 | Function signature + docstring | TODO |
| MBPP-style tasks | ~1000 | Natural language → code | TODO |
| Synthetic textbook exercises | ~10K | Multi-step problems with hints | TODO |
| OSS-Instruct (Magicoder-style) | ~20K | Real code → inspired problem | TODO |

**Actual prompt distribution** (`data/distill/prompts-filtered-50k.jsonl`):
- function: 12,895 (25.8%)
- method: 28,891 (57.8%)
- class: 8,214 (16.4%)

Extracted from first shard of filtered codeparrot via
`scripts/extract_distill_prompts.py`.

### 4.2 Generation Config

**Actual config** (interim teacher, Qwen3-8B on gx10):
```yaml
teacher:
  model: "Qwen3-8B-Q4_K_M.gguf"  # Interim (ALB-010 MoE not ready)
  server: "gx10:8090"             # realizar serve --gpu --openai-api
  temperature: 0.0                # Greedy for pilot, 0.8 for scale
  max_tokens: 512
  num_completions: 1              # Single completion per prompt

filtering:
  min_completion_length: 10       # chars, stripped (C-TEACHER-QUALITY-001)
  dedup_by_prompt_hash: true      # C-TEACHER-DEDUP-001
```

**Contract**: `teacher-completions-pipeline-v1.yaml` (C-TEACHER-*).
Resilient pipeline with retry, resume, dedup. See
`scripts/generate_teacher_completions_api.py`.

**Target config** (full teacher, Qwen3-Coder-30B-A3B):
```yaml
teacher:
  model: "qwen3-coder-30b.apr"
  quantization: "q4k"
  temperature: 0.8
  top_p: 0.95
  max_tokens: 512
  num_completions: 5  # per prompt, pick best

execution:
  sandbox: "bubblewrap"
  timeout_seconds: 10
  python_version: "3.11"
  allowed_imports: ["math", "collections", "itertools", "typing", "re", "string"]

filtering:
  require_execution_pass: true
  require_type_check: false
  dedup_by_ast: true
  min_tokens: 20
  max_tokens: 512
```

### 4.3 Generation Schedule

| Phase | Prompts | Completions | ~Tokens | Wall Time | Status |
|-------|---------|-------------|---------|-----------|--------|
| **Pilot (M1)** | **1K** | **982** | **~400K** | **5h** | **DONE** (98.3% pass) |
| Scale 1 (M2) | 10K | ~9.8K | ~4M | ~38h (GPU) | TODO |
| Scale 2 (M2+) | 50K | ~49K | ~20M | ~8 days (GPU) | TODO |

**Pilot results** (Qwen3-8B-Q4K, greedy, 1 completion per prompt):
- **982/1000 completed**, 17 failures, 132 completions/hour (GPU mode)
- Completion distribution: 406 function, 434 method, 142 class
- Completion length: min=11, median=1046, max=5622, mean=1635 chars
- Total: ~1.6M chars, ~400K estimated tokens
- Infrastructure: realizar serve on gx10 GB10 GPU (realizar#184 fix enabled
  GGUF→OpenAI completions; arXiv:2306.11644 sequence-level distillation)
- Pipeline: `scripts/generate_teacher_completions_api.py` with retry/resume
  (C-TEACHER-*). Runs directly on gx10 via `scripts/resume-teacher-completions.sh`.

**Scaling projection**: GPU mode at 132/h means 10K completions in ~3.2 days.
With parallel servers (2× realizar instances) or upgraded teacher
(Qwen3-Coder-30B at 100+ tok/s): ~1.5 days for 10K.

---

## 5. Student Training Strategy

### 5.1 Stage A: Pre-train on Curated Data

Two parallel paths, best checkpoint from either feeds into Stage B:

**Path 1 (v28)**: Raw codeparrot-clean (5.3B tokens). **STOPPED** at step 11K.
Best val_ppl=38.53 at step 6K, then diverged to 75.65 at step 11K. Raw data
ceiling at ~800M tokens — same regression pattern as v27 (arXiv:2203.15556
Chinchilla scaling: model needs higher-quality tokens, not more raw ones).

**Path 2 (v29)**: Train on AST-filtered subset (2.04B tokens, 850K files,
`data/pretokenized-1024-v4/`). Config: `configs/train/pretrain-350m-v29.yaml`
(15,530 steps, ~2.4 days). Expected to converge faster due to higher data
quality (28.7% pass rate, valid AST + docstrings + import diversity).

| Config | v28 (raw) | v29 (filtered) |
|--------|-----------|----------------|
| Data | 5.3B tokens | 2.04B tokens |
| LR | 7.35e-5, cosine | 7.35e-5, cosine |
| Warmup | 93 steps | TBD |
| Batch | 131K tokens/step | 131K tokens/step |
| Steps | 38,349 | 15,530 |
| Target | val_ppl < 30 | val_ppl < 25 |

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
| ALB-010 qwen3_moe loading blocks | Can't use best teacher | Fallback to Qwen2.5-Coder-3B |
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
