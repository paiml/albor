# 4. Distillation: Synthetic Data from Qwen3-Coder-30B

### 4.1 Why Distillation, Not More Pretraining

Pure pretraining hit a wall. Eight training runs (v1-v8) on codeparrot-clean
show the same pattern: loss converges to ~6.5 (val_ppl ~800) then plateaus.

**Root cause: Chinchilla scaling mismatch.**

| Parameter | Albor | Chinchilla Optimal |
|-----------|-------|-------------------|
| Model size | 350M | 350M |
| Training tokens | 655M (20K steps) | ~7B (20x params) |
| Ratio (tokens/params) | 1.9x | 20x |

We're 10x undertrained. More steps on the same data would overfit, not
improve. More data requires 10x the GPU time (~10 days on a single 4090).

**Distillation is the efficient path**: a strong teacher can transfer knowledge
in far fewer tokens than pure pretraining requires. phi-1 (1.3B) achieved
51% HumanEval using synthetic data from GPT-3.5 — dramatically better than
any model of that size trained on raw code alone.

### 4.2 Teacher Model: Qwen3-Coder-30B-A3B-Instruct

| Property | Value |
|----------|-------|
| Model | [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Parameters | 30.5B total, 3.3B active per token (MoE) |
| Architecture | Standard GQA, 48 layers, h=2048, 128 experts top-8 |
| License | Apache 2.0 |
| HumanEval | Strong code generation (SWE-bench 50.3% verified) |
| FIM support | `<\|fim_prefix\|>`, `<\|fim_suffix\|>`, `<\|fim_middle\|>` |
| Q4K size | 17 GB — fits RTX 4090 (7 GB headroom) |
| Local file | `/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.apr` |

### 4.3 Why Synthetic Data, Not Logit-Level KD

The original spec planned logit-level knowledge distillation: pre-compute
teacher's top-k logits per token, train student with KL divergence loss.

**This doesn't work because of the vocab mismatch:**

| | Teacher (Qwen3-Coder) | Student (Albor) |
|--|----------------------|-----------------|
| Vocab size | 151,936 | 32,768 |
| Tokenizer | Qwen BPE | Albor ByteLevel BPE |
| Token boundaries | Different | Different |

The output distributions live in different spaces. A token in Qwen's vocab
doesn't map 1:1 to a token in Albor's vocab. Logit-level KD requires either:
- Same tokenizer (changes student architecture)
- Token alignment layer (complex, lossy, unproven at this scale)
- Shared vocab subset (throws away teacher knowledge)

**Synthetic data generation sidesteps all of this.** The teacher generates
Python code as text. The student tokenizes it with its own tokenizer and
trains on it as normal causal LM data. No vocab alignment needed.

### 4.4 Distillation Architecture

```
Phase 1: Generate synthetic training data (teacher on GPU, ~17 GB)
┌─────────────────────────────────────────────────────────────────┐
│  Prompts (from codeparrot)  ──►  Qwen3-Coder-30B (Q4K GPU)    │
│                                        │                        │
│                                   generates Python              │
│                                   completions                   │
│                                        │                        │
│                                        ▼                        │
│                              synthetic_data/*.parquet            │
│                              (prompt + completion pairs)        │
└─────────────────────────────────────────────────────────────────┘

Phase 2: Train student on synthetic data (student on GPU, ~13 GB)
┌─────────────────────────────────────────────────────────────────┐
│  synthetic_data/*.parquet  ──►  Albor 350M (CUDA trainer)      │
│  + codeparrot originals         causal LM training              │
│                                 cosine LR, AdamW                │
│                                        │                        │
│                                        ▼                        │
│                              albor-distill-350m checkpoint      │
└─────────────────────────────────────────────────────────────────┘
```

**Sequential, not simultaneous.** Teacher and student never share GPU.
Phase 1 fills VRAM with the 17 GB teacher. Phase 2 uses 13 GB for the
student. No VRAM contention.

### 4.5 Synthetic Data Strategy

#### 4.5.1 Prompt Sources

| Source | Prompts | Strategy |
|--------|---------|----------|
| codeparrot-clean | ~2M files | Extract function signatures, class headers, docstrings as prompts |
| HumanEval | 164 problems | Include all — these ARE the benchmark |
| MBPP | 974 problems | Include all — these ARE the benchmark |
| Custom hard cases | ~500 | Hand-crafted edge cases (recursion, generators, decorators) |

#### 4.5.2 Generation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.8 | Diverse but coherent completions |
| Top-p | 0.95 | Standard nucleus sampling |
| Max tokens | 512 | Function-length completions |
| Samples per prompt | 1-3 | More for hard prompts, 1 for easy |
| Format | Parquet (prompt, completion, source) | Alimentar-compatible |

#### 4.5.3 Data Budget

| Tier | Synthetic Tokens | Generation Time (est.) | Student Training |
|------|-----------------|----------------------|-----------------|
| Minimum viable | 50M | ~6 hours | ~3 hours (1,500 steps) |
| Target | 200M | ~24 hours | ~12 hours (6,000 steps) |
| Full | 500M | ~60 hours | ~30 hours (15,000 steps) |

Start with minimum viable (50M tokens), evaluate on HumanEval/MBPP,
scale up if results justify the compute.

### 4.6 Expected Throughput

Qwen3-Coder-30B at Q4K on RTX 4090:

| Mode | Throughput | Notes |
|------|-----------|-------|
| Prefill (prompt) | ~200-500 tok/s | Batch-1, Q4K GEMV |
| Decode (generation) | ~30-80 tok/s | Autoregressive, KV cache |

At 50 tok/s decode average, 50M tokens of synthetic data takes:
50M / 50 = 1M seconds = ~12 days. That's too slow.

**Optimization: batch prefill + greedy decode.**
- Process prompts in batches where possible
- Use greedy decode (temperature=0) for deterministic, faster generation
- Focus on short completions (128-256 tokens, not 512)
- 50M tokens at 128 avg length = 390K completions

**Revised estimate**: With prompt batching and short completions,
~2-4 days for 50M tokens is realistic.

### 4.7 Blocker: Teacher Inference (ALB-095)

ALB-095 (Q4K GPU inference in realizar) is the critical dependency.
The Q4K APR file exists, the GPU adapter code exists, but end-to-end
inference hasn't been dogfooded yet.

**Unblock sequence:**
1. Dogfood realizar Q4K GPU inference on Qwen3-Coder-30B
2. Verify text generation quality (sample outputs)
3. Build prompt extraction pipeline (from codeparrot)
4. Generate synthetic data (Phase 1)
5. Train student (Phase 2)

### 4.8 Fallback: Qwen2.5-Coder-3B (Dense)

If ALB-095 (MoE Q4K inference) proves too complex, fall back to
**Qwen2.5-Coder-3B** as a dense teacher:

| Property | Value |
|----------|-------|
| Model | [Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) |
| Parameters | 3B (dense) |
| HumanEval | 84.1% (pass@1) |
| License | Qwen Research License |
| Status | Already works in realizar (standard Qwen2 architecture) |

**Trade-off**: 3B dense has ~10x less knowledge capacity than 30B MoE,
but it already works. If MoE inference is blocked, a 3B teacher still
produces useful synthetic data — just lower quality.

### 4.9 Student Training on Synthetic Data

The student trains with standard causal LM loss on synthetic data,
mixed with original codeparrot data for regularization:

| Parameter | Value |
|-----------|-------|
| Data mix | 70% synthetic + 30% codeparrot original |
| Optimizer | AdamW (lr=1e-4, lower than pretraining) |
| Schedule | Cosine with 100-step warmup |
| Init | From v8 checkpoint (step 1000, loss 6.69) |
| Max steps | 6,000-15,000 (depending on data tier) |

Starting from the v8 checkpoint (not random init) means the student
already has basic language modeling ability. The synthetic data teaches
it the teacher's coding patterns on top of that foundation.

### 4.10 Success Criteria

| Metric | Pretraining Only (v8) | Distillation Target | Stretch |
|--------|----------------------|--------------------|---------|
| val_ppl | ~800 | < 200 | < 100 |
| HumanEval pass@1 | 0/164 | > 5/164 (3%) | > 15/164 (9%) |
| MBPP pass@1 | 0/974 | > 20/974 (2%) | > 50/974 (5%) |

Any non-zero HumanEval score from a 350M model trained on sovereign
stack synthetic data would be a meaningful result.
