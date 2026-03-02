# 8. Evaluation & Benchmarks

### 8.1 Evaluation Strategy

**Leaderboard target**: [Big Code Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)
— the standard HuggingFace leaderboard for code generation models. Uses
HumanEval (pass@1) and MultiPL-E (18 languages). Currently tracks ~60 models.
**No sub-1B model has ever appeared on this leaderboard.** The smallest entries
are 1.0B (DeciCoder-1B at 19.3%, phi-1 at 50.6%, SantaCoder at 18.1%).
Albor would be the first sub-1B entry — and the only model trained in Rust.

**Secondary**: Classic `lm-evaluation-harness` benchmarks (zero-shot) for
general capability comparison against Pythia, OPT, GPT-2.

**NOT targeting**: Open LLM Leaderboard v2 (IFEval, BBH, MATH Level 5, GPQA,
MuSR, MMLU-PRO). These benchmarks were designed for large models — a 350M model
scores near random on MATH Level 5 (~0%), GPQA (~25%), and MMLU-PRO (~10%).

**Also NOT targeting**: EvalPlus Leaderboard (HumanEval+, MBPP+). A secondary
submission target if results are strong, but the Big Code leaderboard is the
primary scoreboard.

### 8.2 Benchmark Suite

**Python Code Completion Benchmarks (Primary — matches use case)**

| Benchmark | Type | Metric | What It Tests | Leaderboard? |
|-----------|------|--------|---------------|-------------|
| HumanEval | Function generation | pass@1, pass@10 | Complete a Python function given docstring | Big Code LB |
| MultiPL-E | Multilingual code gen | pass@1 | HumanEval translated to 18 languages (Python-only for us) | Big Code LB |
| MBPP | Basic programming | pass@1 | Solve simple Python programming tasks (3-shot) | — |
| DS-1000 | Data science | pass@1 | Pandas/NumPy/sklearn code generation | — |
| FIM (custom) | Fill-in-the-middle | exact match | Infill Python code between prefix and suffix | — |
| Latency | Inference speed | tok/s | Tokens per second on CPU (Q4) and GPU (fp16) | Big Code LB |

**General Capability Benchmarks (Secondary — validates base model quality)**

| Benchmark | Type | Shots | Random | What It Tests |
|-----------|------|-------|--------|---------------|
| ARC-Easy | Science reasoning | 0 | 25% | Elementary science knowledge |
| HellaSwag | Commonsense completion | 0 | 25% | Sentence completion with physical intuition |
| PIQA | Physical intuition | 0 | 50% | Physical interaction Q&A |
| LAMBADA | Next-word prediction | 0 | 0% | Long-range dependency in text |

### 8.3 Competitive Baselines

**Python Code Completion Baselines (Primary Competition)**

| Model | Params | HumanEval pass@1 | MBPP pass@1 | FIM | Data | Notes |
|-------|--------|-----------------|-------------|-----|------|-------|
| phi-1 | 1.3B | 50.6% | 55.5% | No | 7B (textbooks) | Our direct inspiration — same playbook |
| phi-1-small | 350M | 45%† | — | No | 7B (textbooks) | **Same param count as Albor** (†never released — see note) |
| SantaCoder | 1.1B | 18% | 35% | Yes | 236B (The Stack) | FIM-trained, multi-language |
| StarCoderBase-1B | 1B | 15.2% | — | Yes | 1T (The Stack v2) | Multi-language code model |
| CodeGen-350M-mono | 350M | 12.8% | — | No | 577B (mixed) | Same param count, no distillation |
| **albor-base (target)** | **350M** | **>8%** | **>8%** | **Yes** | **10B** | **Pre-distillation baseline** |
| **albor-distill (target)** | **350M** | **>15%** | **>12%** | **Yes** | **10B + distill** | **Post-distillation from 80B teacher** |

**†phi-1-small caveat**: phi-1-small was never publicly released — it exists
only as an ablation study in "Textbooks Are All You Need" (Gunasekar et al.,
2023). The 45% HumanEval claim is self-reported and has never been independently
reproduced. We treat it as an aspirational ceiling, not a verified baseline.

The benchmark to beat is **CodeGen-350M-mono** (same param count, no
distillation, no FIM, 12.8% HumanEval). The realistic target for distillation
is **+2-5 points** over the base model. Albor uses a stronger teacher (80B MoE)
but faces a significant architecture mismatch (MoE teacher → dense student)
and uses a first-generation Rust training stack instead of PyTorch.

**Big Code Models Leaderboard — where Albor would land**

CodeGen-350M-mono is not on the leaderboard (never submitted). The smallest
models currently on the board are 1B-class. If albor-distill hits >15%
HumanEval, it would sit just below the 1B tier — at 1/3 the parameter count:

| Model | Params | HumanEval | On Leaderboard? |
|-------|--------|-----------|-----------------|
| phi-1 | 1.3B | 50.6% | Yes |
| DeciCoder-1B | 1.0B | 19.3% | Yes (smallest entry) |
| SantaCoder | 1.1B | 18.1% | Yes |
| StarCoderBase-1B | 1.0B | 15.2% | Yes |
| **albor-distill (target)** | **350M** | **>15%** | **Submission target** |
| CodeGen-350M-mono | 350M | 12.8% | No (never submitted) |

**Submission protocol**: Run `bigcode-evaluation-harness` with standard params
(top-p=0.95, temperature=0.2, n_samples=50), submit PR to the leaderboard's
`community_results/` folder. Results marked as "non-verified" (community).

**General Capability Baselines (Secondary)**

| Model | Params | ARC-E | HellaSwag | PIQA | Avg |
|-------|--------|-------|-----------|------|-----|
| Pythia-410M | 410M | 47.1 | 40.1 | 67.2 | 51.5 |
| OPT-350M | 350M | 41.9 | 36.2 | 64.8 | 47.6 |
| GPT-2 Medium | 345M | ~43 | ~34 | ~66 | ~48 |
| **albor-distill (target)** | **350M** | **>42** | **>36** | **>65** | **>48** |

*Note: General capability targets are conservative. Albor is 80% Python code
data with a coding teacher — distillation from Qwen3-Coder-Next will not
improve general reasoning (ARC-E, HellaSwag). The target is OPT-350M parity,
not Pythia-410M. Code benchmarks are the real scoreboard.*

### 8.4 Evaluation Protocol

```bash
# Plan: validate model exists, tasks recognized, output writable
apr eval plan \
  --model ./checkpoints/albor-distill-350m/ \
  --tasks humaneval,humaneval_fim,mbpp,ds1000

# Python code completion benchmarks (primary — run after every stage)
apr eval apply \
  --model ./checkpoints/albor-distill-350m/ \
  --tasks humaneval,humaneval_fim,mbpp,ds1000 \
  --output ./eval/python-code-results.json \
  --seed 42

# General capability benchmarks (secondary)
apr eval apply \
  --model ./checkpoints/albor-350m-final/ \
  --tasks arc_easy,hellaswag,piqa,lambada \
  --batch-size 32 \
  --output ./eval/general-results.json \
  --seed 42

# Latency benchmark (critical for code completion use case)
apr bench plan --model ./checkpoints/albor-q4/
apr bench apply \
  --model ./checkpoints/albor-q4/ \
  --prompt "def fibonacci(n):" \
  --max-tokens 128 \
  --device cpu --device cuda \
  --output ./eval/latency-results.json

# Perplexity on held-out Python code
apr eval apply \
  --model ./checkpoints/albor-350m-final/ \
  --perplexity \
  --data ./data/eval/held-out-python.parquet

# ── Big Code Leaderboard submission eval ──
# Must use bigcode-evaluation-harness with standard params for comparability
# This runs OUTSIDE the sovereign stack (Python, not Rust) — it is the
# leaderboard's own eval tool, not ours. Our apr eval results are the
# primary record; this is for leaderboard submission only.
#
# bigcode-evaluation-harness \
#   --model ./release/albor-350m.safetensors \
#   --tasks humaneval,multiple-py \
#   --temperature 0.2 --top_p 0.95 \
#   --n_samples 50 --max_length_generation 512 \
#   --output ./eval/bigcode-leaderboard/
```

### 8.5 Continuous Evaluation During Training

The intel box runs eval on the latest checkpoint concurrently with training:

```bash
# On intel (300GB RAM), polling for new checkpoints
apr eval apply \
  --model ./checkpoints/latest/ \
  --tasks arc_easy,hellaswag \
  --batch-size 16 \
  --output ./eval/step-$(cat ./checkpoints/latest/step.txt).json
```

**Gap ALB-006**: ~~Verify `apr eval plan/apply` supports these benchmark tasks
natively.~~ FIXED: `apr eval` supports perplexity and classification eval.

**Gap ALB-037**: `apr eval` loads SafeTensors models but ignores weights during
inference (identical perplexity regardless of weight content). Workaround:
`scripts/eval-perplexity.py` implements pure-Python transformer inference.

**Gap ALB-038** (**FIXED**): entrenar previously saved initialization weights
instead of trained weights due to broken autograd gradient flow. Root cause:
`RMSNorm::forward_batched()` created tensors with no backward op, and
`MultiHeadAttention::forward()` broke Q/K/V gradient chain. Fixed in
`entrenar@91ba9da` (RMSNorm backward) and `entrenar@1ede409` (attention
backward). All 20 model parameters now receive gradients during training.
See [GitHub #36](https://github.com/paiml/albor/issues/36).

### 8.6 Local Evaluation Infrastructure

The following scripts provide model evaluation independently of `apr eval`:

```bash
# Validate checkpoint integrity (fast, detects ALB-038)
python scripts/eval-perplexity.py checkpoints/albor-base-350m/ --validate-checkpoint

# Validate all canonical solutions (no model needed)
python scripts/eval-code.py configs/eval/python-intermediate.jsonl --validate-only
python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --validate-only

# Full evaluation suite (orchestrates all steps)
bash scripts/run-eval-suite.sh checkpoints/albor-base-350m/

# Perplexity on pre-tokenized validation data
python scripts/eval-perplexity.py checkpoints/albor-base-350m/ \
    --data data/pretokenized-2048/val/val.parquet \
    --max-sequences 100 --seq-len 2048 --threshold 30

# Evaluate via apr serve API (when ALB-037 is fixed)
python scripts/eval-code.py configs/eval/humaneval-subset.jsonl \
    --api http://localhost:8080 --samples 10

# Training convergence validation (FALSIFY-ALBOR-001)
python scripts/validate-training-convergence.py \
    checkpoints/albor-base-350m/training.log

# Convert entrenar checkpoint format for realizar
python scripts/convert-checkpoint.py checkpoints/albor-base-350m/ \
    --config configs/train/pretrain-350m.yaml
```

**Benchmark datasets:**
- `configs/eval/python-intermediate.jsonl` — 15 intermediate Python problems
- `configs/eval/humaneval-subset.jsonl` — 20 HumanEval-format problems

### 8.7 Weight Convention & Checkpoint Format

entrenar stores linear layer weights as **[in_features, out_features]** in
row-major (C) order, and computes forward pass as `x @ W` (no transpose).
This differs from the HuggingFace convention of **[out_features, in_features]**
with `x @ W.T`.

| Component | Convention | Forward | Example: gate_proj |
|-----------|-----------|---------|-------------------|
| entrenar (training) | [in, out] | `x @ W` | [512, 2048] |
| HuggingFace (standard) | [out, in] | `x @ W.T` | [2048, 512] |
| realizar (inference) | [out, in] | `x @ W.T` | [2048, 512] |

The `convert-checkpoint.py` script handles the conversion:
1. Reads 1D flat tensors from entrenar SafeTensors
2. Reshapes as [in, out] (entrenar convention)
3. Transposes to [out, in] (HuggingFace/realizar convention)
4. Writes new SafeTensors with proper 2D shapes

Embeddings (`model.embed_tokens.weight`) are stored as [vocab, hidden] in
both conventions (indexed by token ID for row lookup).
