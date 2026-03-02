# Albor LLM Specification

**Version**: 0.3.0-draft
**Date**: 2026-03-01
**Status**: Draft — Brainstorm Phase
**Author**: Noah Gift / Pragmatic AI Labs

> *Albor* (Spanish: "dawn") — A sovereign Python code completion model trained
> from first principles using only the Sovereign AI stack. Python-only following
> the phi-1 playbook: maximum concentration on one language, distilled from
> Qwen3-Coder-Next (80B), then optimized through fine-tuning, merging, pruning,
> and quantization into a fast, local, zero-dependency code completion engine.
> The goal is twofold: produce a **usable Python code assist model** that runs
> anywhere Rust compiles, **and** identify + fix every gap in the stack that
> blocks end-to-end LLM development.

---

## 1. Objectives

### 1.1 Primary Goal
Train, distill, and optimize a **350M-parameter decoder-only transformer** using
exclusively the Sovereign AI stack:
- `apr` for training, distillation, merging, pruning, quantization, eval, export
- `alimentar` for data loading and preprocessing
- `forjar` for pipeline orchestration (DAG engine, multi-machine, state tracking)
- `bashrs` (Rash) for shell fragment validation, Makefile linting (KING of linting), and pipeline command purification
- `repartir` for distributed compute
- `entrenar` for the training engine (autograd, optimizers, checkpointing)
- `trueno` for SIMD/GPU tensor operations
- `realizar` for inference (teacher model, eval, serving)
- `presentar` for training visualization (TUI dashboards, experiment browser, WASM)
- `batuta` for orchestration, stack coordination, and falsification
- `pv` (provable-contracts) for design-by-contract verification of every kernel
- `pmat` for TDG scoring, compliance, fault pattern analysis, and coverage gaps
- `certeza` for three-tier test effectiveness (unit → property → formal)

### 1.2 Secondary Goal (Stack Validation)
Identify every implementation gap that blocks the primary goal. Fix each gap in
the correct upstream component. The model is the proof; the stack improvements
are the lasting value.

### 1.3 Multi-Stage Improvement Ladder
The model is not a single training run — it is iteratively improved through every
post-training technique available in `apr`. Each stage exercises a different
part of the stack, produces a benchmarked checkpoint, and may reveal new gaps.

```
Stage 1: Pre-train base model         → albor-base
Stage 2: Distill from Qwen3-Coder-Next → albor-distill
Stage 3: Instruction fine-tune (LoRA)  → albor-instruct
Stage 4: Merge with complementary model → albor-merged
Stage 5: Prune for efficiency          → albor-pruned
Stage 6: Quantize for deployment       → albor-q4
```

### 1.4 Target Use Cases

**Primary: Sovereign Code Assist**

A tiny, fast, zero-dependency code completion model that runs entirely locally.
No API calls, no Python runtime, no telemetry, no cloud. Distillation from
Qwen3-Coder-Next gives it coding capability far above what 350M parameters
normally achieve.

| Capability | Description |
|------------|-------------|
| Python code completion | Left-to-right next-token prediction in `.py` files |
| Fill-in-the-middle (FIM) | Insert Python code between existing prefix and suffix (PSM/SPM) |
| Single-line infill | Complete the current line given surrounding context |
| Multi-line body generation | Generate function bodies, loop contents, comprehensions, decorators |
| On-device inference | Runs on laptops, Raspberry Pi, browsers (WASM via trueno) |
| Latency target | <50ms per token on CPU (Q4), <10ms on GPU |

**Language**: Python only. Following the phi-1 playbook — maximum concentration
on a single language produces dramatically better results at small param counts
than spreading tokens across many languages. A 350M model that completes Python
well is more useful than a 350M model that completes 10 languages poorly.

**What Albor is NOT**: It is not a chat model, not an instruction follower, not a
reasoning engine, not a polyglot code model. It is a fast, local Python code
completion kernel — the kind of model that lives inside an editor extension and
fires on every keystroke.

**Secondary: Stack Demonstration & Teaching Artifact**

The model exists equally to prove the Sovereign AI stack can train, distill,
optimize, and serve an LLM end-to-end in pure Rust. The HuggingFace model card
is a tour of the stack. The reproducibility protocol means anyone can retrain
from scratch using only `apr` commands.

| Audience | What They Get |
|----------|---------------|
| Developers | A code completion model they can self-host with zero dependencies |
| Researchers | A fully reproducible training recipe with provable quality contracts |
| Stack users | Proof that aprender/entrenar/trueno/realizar handle real LLM workloads |
| Educators | A case study in first-principles LLM training (data → deploy in Rust) |

### 1.5 What Albor Builds

Albor is a **project repo**, not a library. It contains no production Rust code.
All Rust changes happen upstream in the sovereign stack components. Albor drives
the upstream work, validates it end-to-end, and produces the model.

#### 1.5.1 What Lives in Albor (This Repo)

```
albor/
├── docs/
│   ├── specifications/albor-llm-spec.md    # This spec
│   ├── model-card.md                       # HuggingFace model card
│   └── falsification-report.md             # batuta falsify output
├── configs/
│   ├── train/
│   │   ├── pretrain-50m.yaml              # 50M: model arch + training (pipeline validation)
│   │   ├── pretrain-125m.yaml             # 125M: model arch + training (intermediate)
│   │   ├── pretrain-350m.yaml             # 350M: model arch + training (final)
│   │   ├── distill.yaml                   # Distillation config
│   │   └── finetune-lora.yaml             # LoRA fine-tuning config
│   ├── pipeline/
│   │   └── albor.yaml                      # THE manifest: infra + data + train + eval + publish
│   ├── dashboard/
│   │   └── albor-dashboard.yaml            # presentar dashboard (TUI + WASM)
│   └── data-mix.yaml                       # Data source weights + upsampling
├── contracts/
│   ├── knowledge-distillation-kernel-v1.yaml  # ALB-013
│   ├── bpe-tokenizer-kernel-v1.yaml           # ALB-014
│   ├── model-merging-kernel-v1.yaml           # ALB-015
│   ├── pruning-kernel-v1.yaml                 # ALB-016
│   └── gradient-accumulation-kernel-v1.yaml   # ALB-017
├── tests/
│   ├── falsify/                            # FALSIFY-ALBOR-001 through 009
│   ├── integration/                        # End-to-end pipeline tests
│   └── smoke/                              # Quick sanity checks (50M model)
├── state/                                  # (gitignored) forjar state + locks
│   ├── lambda/state.lock.yaml              # Per-machine resource state
│   ├── intel/state.lock.yaml
│   └── forjar.lock.yaml                    # Global pipeline state
├── data/                                   # (gitignored) Training data
├── checkpoints/                            # (gitignored) Model checkpoints
└── eval/                                   # (gitignored) Evaluation results
```

#### 1.5.2 `apr` as Unified Entry Point

`apr` is the **single CLI** for all model operations. It delegates to
sibling projects (entrenar, alimentar, realizar, etc.) under the hood. If a
subcommand doesn't exist yet, we file a GitHub issue, implement it in the
correct upstream repo, wire it into `apr`, dogfood it in albor, and close
the issue.

##### Design Principle: Plan/Apply Everywhere

Every `apr` subcommand that touches data, compute, or infrastructure follows
a **plan/apply** contract inspired by Terraform and forjar:

```
plan   → Validate inputs, estimate cost, show what WILL happen. No side effects.
apply  → Execute the plan. Mutates state (files, models, infrastructure).
```

This is not optional. It is the **unifying design principle** of the CLI.
Every expensive operation gets a free dry-run. Every destructive operation
shows you what it will do before it does it. Users never commit GPU hours,
disk space, or network bandwidth without seeing the plan first.

**The contract**:
1. `apr <cmd> plan <config>` — Parse config, validate paths, estimate
   resources (VRAM, disk, time, tokens), print a human-readable execution
   plan. Exit 0 if valid, exit 1 with diagnostics if not. **No GPU, no
   writes, no network.**
2. `apr <cmd> apply <config>` — Execute. Reads the same config, does the
   work. Can be interrupted and resumed.
3. `apr <cmd> validate <config>` — Alias for `plan` with `--strict`
   schema-only checking (no resource estimation). Fast enough for CI.

**Why this matters for albor**: Training a 350M model for 7 days on a 4090
is not something you retry casually. A config typo caught at `plan` time
saves days. A VRAM overestimate caught at `plan` time prevents OOM crashes
at step 15,000. Plan/apply turns "hope it works" into "prove it will work,
then run it."

##### Dispatch Table

```
apr <subcommand>
├── pipeline plan/apply      → forjar DAG engine (THE entry point — runs everything)
├── tokenize plan/apply      → aprender BPE tokenizer
├── train plan/apply         → entrenar TransformerTrainer
├── distill plan/apply       → entrenar + realizar (precompute + student training)
├── finetune plan/apply      → entrenar LoRA/QLoRA
├── eval plan/apply          → aprender eval harness
├── merge plan/apply         → entrenar SLERP/TIES/DARE
├── prune plan/apply         → entrenar WANDA/magnitude
├── quantize plan/apply      → entrenar Q4/Q8
├── export plan/apply        → entrenar SafeTensors/GGUF
├── publish plan/apply       → entrenar HuggingFace Hub
├── bench plan/apply         → realizar latency benchmarks
├── provision plan/apply     → forjar infrastructure convergence
├── experiment view/export   → presentar TUI + entrenar SQLite
└── monitor                  → presentar live TUI (reads training_state.json)
```

`apr pipeline` is the **top-level command**. It reads a single YAML manifest
that describes infrastructure resources AND training tasks in one DAG. Forjar's
engine resolves dependencies (Kahn's toposort), tracks state (BLAKE3 hashes),
and dispatches each step — calling back into `apr` subcommands for ML tasks.
Individual subcommands (`apr train`, `apr eval`, etc.) still work standalone
for development and debugging.

##### Plan Output Format

Every `plan` subcommand prints a structured summary:

```
$ apr train plan configs/train/pretrain-350m.yaml

  Albor Train Plan
  ─────────────────────────────────────────────
  Model:        llama (24L, 1024H, 16A, 4KV)
  Parameters:   354,267,136 (~354M)
  Precision:    fp16 mixed
  ─────────────────────────────────────────────
  VRAM Budget:
    Weights       700 MB
    Optimizer   2,800 MB   (AdamW fp32 m+v)
    Gradients     700 MB
    Activations 9,200 MB   (grad ckpt, batch=8, seq=2048)
    Total      13,400 MB   (55.8% of 24,576 MB)
    Headroom   11,176 MB   ✓
  ─────────────────────────────────────────────
  Data:
    Train shards  data/tokenized/train/ (47 files, 8.2 GB)
    Val shards    data/tokenized/val/   (3 files, 410 MB)
    Tokenizer     models/albor-tokenizer/tokenizer.json ✓
    Vocab match   32,768 = model.vocab_size ✓
  ─────────────────────────────────────────────
  Training:
    Global batch  524,288 tokens (8 × 32 × 2048)
    Total tokens  10,000,000,000 (~10B)
    Total steps   19,073
    Warmup        2,000 steps (10.5%)
    Checkpoints   19 (every 1,000 steps)
    Disk est.     ~13.3 GB (19 × 700 MB)
  ─────────────────────────────────────────────
  Estimated wall time: 5.2 days on RTX 4090
  ─────────────────────────────────────────────
  ✓ Plan valid. Run `apr train apply configs/train/pretrain-350m.yaml` to start.
```

**Forjar already does this** (`forjar plan -f albor.yaml`). Entrenar has the
`TrainingPlan` module (`training_plan.rs`) that mirrors forjar's architecture.
Albor's job is to close the loop: every `apr` subcommand gets plan/apply,
and every gap (ALB-XXX) that adds a new subcommand must implement both phases.

##### What Plan Validates Per Subcommand

| Subcommand | Plan Checks |
|------------|-------------|
| `tokenize` | Input Parquet exists, vocab size valid, output dir writable, estimated time |
| `train` | YAML schema, model arch sanity (divisibility, KV ratio), VRAM budget, data paths, tokenizer vocab match, checkpoint disk estimate |
| `distill` | Teacher model loadable (RAM check), student checkpoint exists, logit output dir writable, temperature/alpha valid |
| `finetune` | Base model exists, LoRA rank/alpha valid, dataset format, VRAM with adapters |
| `eval` | Model checkpoint exists, benchmark tasks recognized, output dir writable |
| `merge` | All input models exist and have compatible architectures, merge method valid |
| `prune` | Model exists, sparsity ratio in [0,1], method recognized, output size estimate |
| `quantize` | Model exists, target format valid (Q4/Q8), output size estimate |
| `export` | Model exists, format valid (SafeTensors/GGUF), output path writable |
| `publish` | Model + model card exist, HF token valid, repo name available |
| `provision` | forjar plan: SSH reachable, packages installable, GPU drivers, disk space |

#### 1.5.3 Development Workflow: Issue-Driven Dogfooding

When albor hits a wall — a missing subcommand, a broken feature, a gap in
a sibling project — the workflow is:

```
1. Hit wall       → apr <subcommand> doesn't exist or fails
2. File issue     → GitHub issue on correct repo (aprender, entrenar, alimentar, etc.)
3. Implement      → Fix upstream in the correct component
4. Wire into apr  → Add/update apr subcommand if needed
5. Dogfood        → Run the blocked albor pipeline step
6. Prove          → Tests pass, FALSIFY test passes, pmat comply check
7. Close issue    → Link to albor gap ID (ALB-XXX)
```

Every ALB-XXX gap in the gap register (§11) maps to a GitHub issue. The gap
is not "closed" until the `apr` subcommand works end-to-end in the albor
pipeline.

#### 1.5.4 What Lives Upstream (Other Repos)

| Upstream Repo | What Albor Adds to It | Gaps |
|---------------|----------------------|------|
| **aprender** (apr) | `pipeline plan/apply`, `tokenize plan/apply`, `distill plan/apply`, `eval plan/apply`, `train plan/apply`, plan/apply contract enforcement | ALB-001, 006, 009, 011, 023, 028 |
| **alimentar** | `import local`, `mix` with upsampling, FIM transforms, streaming to entrenar | ALB-007, 018, 019, 020 |
| **realizar** | Qwen3-Coder-Next / DeltaNet / MoE architecture support | ALB-010 |
| **entrenar** | Training engine, model merging, pruning, quantization, LoRA, custom YAML model arch, human-readable config values | ALB-003, 004, 021, 022 |
| **forjar** | `task` resource type for ML pipeline orchestration, DAG engine for `apr pipeline` | ALB-027 |
| **presentar** | SQLite experiment viewer, live training TUI, WASM dashboard, `apr experiment` CLI | ALB-024, 025, 026 |
| **bashrs** | Shell fragment validation for all `task` resource `command:` fields | (used by ALB-027) |
| **trueno** | wgpu backward pass (stretch) | ALB-005 |
| **repartir** | Ring all-reduce (stretch), heterogeneous balancing | ALB-002, 008 |
| **provable-contracts** | 5 new kernel contracts (KD, BPE, merging, pruning, grad accum) | ALB-013–017 |

#### 1.5.5 Where Quality Constraints Apply

| Constraint | Applies To | NOT To |
|------------|-----------|--------|
| 95% test coverage | Upstream Rust code we modify (aprender, entrenar, alimentar, etc.) | Albor's shell scripts and YAML configs |
| 85% mutation score | Upstream Rust code we modify | Albor configs |
| 500-line file limit | ALL files: upstream Rust, albor scripts, YAML configs, contracts | Generated output (eval results, logs) |
| TDG grade A | Upstream Rust code via `pmat` | Albor shell scripts |
| Zero clippy warnings | Upstream Rust code | N/A |
| pmat comply check | Each upstream repo after modification | Albor repo itself |
| Contract verification | Upstream kernel implementations | Albor orchestration |
| FALSIFY-ALBOR tests | The albor pipeline end-to-end | Individual upstream unit tests |

**The albor repo has no Rust code to cover.** Its quality is measured by:
- Do the configs work? (integration tests)
- Do the FALSIFY tests pass? (end-to-end validation)
- Are the contracts complete? (`pv status`)
- Does the pipeline reproduce? (deterministic re-run)

### 1.6 Constraints
- **Zero Python dependencies** — Pure Rust from data to deployment
- **Scientifically reproducible** — Fixed seeds, versioned data, deterministic training
- **Publicly auditable** — All data, code, hyperparameters, and training logs published
- **`apr` only** — Every model operation uses an `apr <subcommand>`. Missing commands are gaps to implement.
- **Plan/apply everywhere** — Every `apr` subcommand implements `plan` (dry-run, no side effects) and `apply` (execute). No GPU time without a passing plan.
- **One manifest, one DAG** — `apr pipeline plan/apply configs/pipeline/albor.yaml` orchestrates the entire pipeline. No Makefiles, no shell scripts. Forjar's DAG engine handles dependency resolution, state tracking, multi-machine dispatch, and resumability.
- **bashrs linted** — All shell fragments in forjar task resources are validated by bashrs (Rash). No unvalidated shell.
- **No file over 500 lines** — Applies to all code, scripts, configs, and contracts (not docs/specs)
- **Provably correct** — Every kernel has a YAML contract with falsification tests and Kani proofs
- **pmat compliant** — Upstream changes: TDG grade A, 95% coverage, 85% mutation score, zero SATD
- **Falsifiable** — Every claim in this spec has a concrete test that could disprove it

---

## 2. Hardware Inventory

### 2.1 Machine: `lambda` (Threadripper)
| Property | Value |
|----------|-------|
| CPU | AMD Threadripper (high core count) |
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X) |
| GPU Backend | CUDA 12.x |
| FP32 TFLOPS | 82.6 |
| FP16 TFLOPS | 165 (with tensor cores) |
| Role | **Primary trainer, student model** |

### 2.2 Machine: `intel` (Mac Pro 2019 chassis, Linux)
| Property | Value |
|----------|-------|
| CPU | Intel Xeon W-3245 @ 3.20 GHz (16C/32T) |
| RAM | **~300 GB** |
| GPU | 2x AMD Radeon Pro W5700X (8 GB GDDR6 each) |
| GPU Backend | wgpu/Vulkan (ROCm unsupported for RDNA 1 / gfx1010) |
| FP32 TFLOPS | ~9 per card (~18 total) |
| Role | **Teacher inference (Qwen3-Coder-Next in CPU RAM), data pipeline, eval** |

### 2.3 Network
- SSH connectivity (`ssh intel`) with ControlMaster multiplexing (forjar FJ-252)
- LAN bandwidth assumed ≥1 Gbps

### 2.4 Key Insight: 300 GB RAM Enables CPU-Based Teacher Inference

The intel box's 300 GB RAM fundamentally changes the distillation architecture.
Qwen3-Coder-Next (80B params) fits entirely in CPU RAM:

| Model Format | Size in RAM | Fits in 300 GB? | Headroom |
|-------------|-------------|-----------------|----------|
| fp16 | ~160 GB | Yes | ~140 GB for KV cache + buffers |
| Q8 | ~80 GB | Easily | ~220 GB |
| Q4 | ~40 GB | Trivially | ~260 GB |

No quantization-induced quality loss needed. The teacher runs at full fp16
precision, producing the highest-quality soft targets for distillation.

---

## 3. Model Architecture

### 3.1 Architecture: LLaMA-Style Decoder-Only Transformer

entrenar's transformer is a pre-norm LLaMA-style architecture with RMSNorm,
SwiGLU FFN, Grouped-Query Attention, and RoPE. This is hardcoded in the
`Transformer` struct — we configure it via YAML, we don't build it from scratch.

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Parameters | ~350M | Fits in 4090 VRAM with optimizer state in fp16 |
| Layers | 24 | GPT-2 Medium proven at this depth |
| Hidden dim (d_model) | 1024 | Standard for this param count |
| Attention heads | 16 | d_head = 64, well-studied |
| KV heads | 4 | GQA with 4:1 ratio (memory efficient) |
| FFN dim (intermediate) | 4096 | ~4x hidden dim (SwiGLU gate + up + down) |
| Vocab size | 32,768 | BPE trained on corpus (power of 2 for GPU efficiency) |
| Context length | 2048 | Sufficient for benchmarks, manageable VRAM |
| Position encoding | RoPE | Built into entrenar's `MultiHeadAttention` |
| Attention | GQA | Built into entrenar, fewer KV heads than Q heads |
| Normalization | RMSNorm | Built into entrenar, pre-norm (before attn + FFN) |
| FFN activation | SwiGLU | Built into entrenar (gate_proj, up_proj, down_proj) |
| Dropout | 0.0 | Modern practice for pre-training (regularize via data) |

### 3.2 Progressive Model Sizing

To validate the pipeline quickly, we train progressively larger models.
Each gets its own YAML config file (see §6.2 for full config format).

| Model | Config | Params | Layers | Hidden | Heads | Purpose |
|-------|--------|--------|--------|--------|-------|---------|
| albor-50M | `pretrain-50m.yaml` | ~50M | 12 | 512 | 8 | Pipeline validation (hours) |
| albor-125M | `pretrain-125m.yaml` | ~125M | 16 | 768 | 12 | Intermediate, first benchmarks (1-2 days) |
| albor-350M | `pretrain-350m.yaml` | ~350M | 24 | 1024 | 16 | Final base model (3-7 days) |

The 50M model proves the entire stack works end-to-end before committing
days of GPU time to the 350M run.

### 3.3 VRAM Budget (fp16 mixed precision, RTX 4090)

| Component | Size |
|-----------|------|
| Model weights (fp16) | ~700 MB |
| Adam optimizer states (fp32 m, v) | ~2.8 GB |
| Gradients (fp16) | ~700 MB |
| Activations (grad checkpoint, batch=8, seq=2048) | ~8-12 GB |
| **Total estimated** | **~13-16 GB** |
| **4090 headroom** | **~8-11 GB** |

Fits comfortably. Batch size tunable up to ~16 with gradient checkpointing.

---

## 4. Distillation Teacher: Qwen3-Coder-Next

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

---

## 5. Training Data

### 5.1 Data Philosophy
- All datasets either locally owned (MIT/Apache 2.0) or publicly available with permissive licenses
- **Local-first**: Sovereign ground truth corpora are our highest-quality data — curated,
  tested, type-annotated, and owned. They are upsampled to punch above their token weight.
- Exact download URLs, versions, and SHA-256 hashes recorded for all external data
- Preprocessing pipeline is deterministic (fixed seed, recorded transforms)
- Quality validated by `alimentar quality check`

### 5.2 Data Mix (Target: ~10B tokens)

Following the phi-1 playbook: maximum concentration on Python. phi-1 proved that
a small model (1.3B) with focused data and distillation can hit 50% HumanEval —
outperforming models 10x its size trained on diluted multi-language corpora.

**Key insight from phi-1**: Data quality matters more than quantity at small
param counts. A 350M model trained on 1B tokens of textbook-quality code can
outperform a 350M model trained on 100B tokens of raw GitHub scrapes. We have
~71K curated Python files locally — this is our unfair advantage.

| Source | Tokens (est.) | Weight | License | Rationale |
|--------|--------------|--------|---------|-----------|
| StarCoder Python subset (HF) | ~4B | 40% | Apache 2.0 | Bulk Python code diversity; aligns with Qwen3-Coder teacher |
| **Local ground truth corpora** (upsampled 10x) | ~50-100M raw → ~500M-1B effective | **10%** | MIT | Highest-quality anchor — see §5.2.1 |
| **Local ML framework code** | ~200-400M | **10%** | MIT / Apache 2.0 | ML/AI Python patterns — see §5.2.2 |
| FineWeb-Edu (subset) | ~2B | 20% | ODC-BY | Educational web text for docstring understanding |
| Python textbooks + tutorials (HF) | ~1B | 10% | Apache 2.0 / CC | "Textbooks Are All You Need" — public educational code |
| Python docs + PEPs + Stack Overflow | ~1B | 10% | CC BY-SA | API knowledge, idiomatic patterns |

Total: ~10B tokens. Chinchilla-optimal for 350M params is ~7B; we slightly
overtrain for benchmark performance (common practice in SmolLM, Phi-1.5).

**Python concentration**: 80% of training data is Python or Python-adjacent
(code, textbooks, docs). The remaining 20% (FineWeb-Edu) provides general
language understanding for docstrings, comments, and natural language prompts.

#### 5.2.1 Local Ground Truth Corpora (Tier 1 — Upsampled)

These are our "textbook-quality" data — the phi-1 equivalent. Every file has
been curated, tested to 98%+ coverage, and validated by CI. They are upsampled
10x during training because their per-token teaching signal is 10-100x higher
than raw GitHub code.

| Corpus | Path | Files | Lines (est.) | Quality Signal |
|--------|------|-------|-------------|----------------|
| depyler examples + tdd-book | `../depyler/examples/`, `../depyler/tdd-book/` | 1,845 | ~219K | Type-annotated, transpiler-validated, 27 stdlib modules, property-tested |
| hf-ground-truth-corpus | `../hf-ground-truth-corpus/` | 11,928 | ~500K+ | 98%+ test coverage, zero lint violations, production HF recipes |
| jax-ground-truth-corpus | `../jax-ground-truth-corpus/` | 2,697 | ~200K+ | 100% test coverage, full type checking, numerical computing |
| vllm-ground-truth-corpus | `../vllm-ground-truth-corpus/` | 1,118 | ~100K+ | Production inference optimization code |
| **Total** | | **17,588** | **~1M+** | **All MIT licensed, all CI-validated** |

**Why upsampling works**: phi-1's "textbook" data was <10% of total tokens but
had outsized impact on HumanEval. Our ground truth corpora share the same
properties: clean types, complete docstrings, tested correctness, educational
structure. The model sees these examples multiple times, reinforcing correct
patterns over noisy GitHub code.

**depyler corpus is uniquely valuable**: Every Python function in the depyler
corpus was validated by a transpiler — it has clear types, clean control flow,
and provably correct semantics. The tdd-book covers 27 stdlib modules
(json, datetime, collections, itertools, os, pathlib, re, etc.) with
property-based tests. This teaches the model Python's standard library idioms
at a depth no scraped dataset matches.

#### 5.2.2 Local ML Framework Code (Tier 2)

Large, high-quality Python codebases from our local repos. Not upsampled —
used at natural frequency for pattern diversity.

| Corpus | Path | Files | Notes |
|--------|------|-------|-------|
| huggingface-fine-tuning | `../huggingface-fine-tuning/` | 12,274 | Fine-tuning recipes and examples |
| llms-with-huggingface | `../llms-with-huggingface/` | 13,869 | LLM integration patterns |
| HF-Hub-Ecosystem | `../HF-Hub-Ecosystem/` | 16,978 | Comprehensive HF Hub code |
| pytorch | `../pytorch/` | 4,217 | ML framework fundamentals |
| vllm | `../vllm/` | 2,400 | Inference serving |
| databricks-data-engineering | `../databricks-data-engineering/` | 3,038 | Data engineering patterns |
| algorithm-competition-corpus | `../algorithm-competition-corpus/` | 201 | Algorithms + data structures |
| coursera-stats | `../coursera-stats/` | 430 | Statistical modeling |
| cuda-python | `../cuda-python/` | 161 | GPU computing |
| **Total** | | **53,568** | **All MIT / Apache 2.0** |

#### 5.2.3 Pre-Built Local Datasets

| File | Path | Format | Size |
|------|------|--------|------|
| hf_gtc_corpus.parquet | `../hf-ground-truth-corpus/hf_gtc_corpus.parquet` | Parquet | 2 MB |
| corpus_manifest_v1.json | `../depyler/corpus_manifest_v1.json` | JSON | Tier metadata |
| corpus_tiers.json | `../depyler/corpus_tiers.json` | JSON | Complexity metrics |

#### 5.2.4 Data Sourcing Summary

```
Local owned data (~71K files, ~1-2M lines):
├── Tier 1: Ground truth corpora (17,588 files) → upsampled 10x
├── Tier 2: ML framework code   (53,568 files) → natural frequency
└── Pre-built: Parquet + JSON manifests

External data (HuggingFace, ~8B tokens):
├── StarCoder Python subset     (~4B tokens)   → bulk diversity
├── FineWeb-Edu                 (~2B tokens)   → general language
├── Python textbooks/tutorials  (~1B tokens)   → educational code
└── Python docs + PEPs + SO     (~1B tokens)   → API knowledge
```

**Sovereign data advantage**: 20% of training tokens come from data we own,
curate, and can improve. Unlike scraped web data, we know the provenance,
license, and quality of every file. If benchmarks reveal weaknesses in specific
Python patterns, we can add targeted examples to our ground truth corpora and
retrain — a feedback loop no public-dataset-only approach can match.

### 5.3 Fill-in-the-Middle (FIM) Training

Code completion requires fill-in-the-middle capability, not just left-to-right
generation. During training, a fraction of code sequences are transformed using
the PSM (Prefix-Suffix-Middle) format:

```
<fim_prefix>def fibonacci(n):<fim_suffix>    return fib_sequence<fim_middle>
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FIM rate | 50% of code sequences | SantaCoder/StarCoder standard |
| FIM format | PSM (Prefix-Suffix-Middle) | Most common, best tooling support |
| Special tokens | `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>` | Added to BPE vocabulary |
| Context split | Random split point per sequence | Uniform distribution over valid positions |

**Gap ALB-018**: Verify `entrenar` / `alimentar` support FIM data transforms.
If not, implement PSM/SPM sequence transformation in the data pipeline.

This is critical — without FIM, the model is a text generator, not a code
completion engine.

### 5.4 Data Pipeline

```bash
# ── Step 1: Ingest local ground truth corpora (Tier 1 — highest quality) ──
alimentar import local ../depyler/examples/ ../depyler/tdd-book/tests/ \
  --lang python --output ./data/local/depyler.parquet
alimentar import local ../hf-ground-truth-corpus/ \
  --lang python --output ./data/local/hf-gtc.parquet
alimentar import local ../jax-ground-truth-corpus/ \
  --lang python --output ./data/local/jax-gtc.parquet
alimentar import local ../vllm-ground-truth-corpus/ \
  --lang python --output ./data/local/vllm-gtc.parquet

# ── Step 2: Ingest local ML framework code (Tier 2) ──
alimentar import local \
  ../huggingface-fine-tuning/ ../llms-with-huggingface/ ../HF-Hub-Ecosystem/ \
  ../pytorch/ ../vllm/ ../databricks-data-engineering/ \
  ../algorithm-competition-corpus/ ../coursera-stats/ ../cuda-python/ \
  --lang python --output ./data/local/ml-frameworks.parquet

# ── Step 3: Download external data (on intel — 300GB RAM) ──
alimentar import hf bigcode/starcoderdata --lang python --output ./data/starcoder-python/
alimentar import hf HuggingFaceFW/fineweb-edu --output ./data/fineweb-edu/

# ── Step 4: Quality validation ──
alimentar quality check ./data/local/ --profile ml-training
alimentar quality check ./data/starcoder-python/ --profile ml-training
alimentar quality check ./data/fineweb-edu/ --profile ml-training

# ── Step 5: Filter, dedup, shard ──
alimentar filter ./data/starcoder-python/ --lang python --min-tokens 32 --max-tokens 8192 \
  --dedup --output ./data/processed/starcoder-python.parquet
alimentar convert ./data/fineweb-edu/ ./data/processed/fineweb-edu.parquet

# ── Step 6: Build training mix with upsampling ──
alimentar mix \
  --input ./data/processed/starcoder-python.parquet --weight 0.40 \
  --input ./data/local/depyler.parquet --weight 0.025 --upsample 10 \
  --input ./data/local/hf-gtc.parquet --weight 0.025 --upsample 10 \
  --input ./data/local/jax-gtc.parquet --weight 0.025 --upsample 10 \
  --input ./data/local/vllm-gtc.parquet --weight 0.025 --upsample 10 \
  --input ./data/local/ml-frameworks.parquet --weight 0.10 \
  --input ./data/processed/fineweb-edu.parquet --weight 0.20 \
  --input ./data/processed/textbooks.parquet --weight 0.10 \
  --input ./data/processed/python-docs.parquet --weight 0.10 \
  --output ./data/mixed/ \
  --seed 42 --shuffle

# ── Step 7: Record provenance ──
alimentar provenance ./data/mixed/ --output ./data/provenance.json
```

**Gap ALB-019**: Verify `alimentar import local` supports recursive Python file
ingestion from local directories. If not, implement local filesystem source.

**Gap ALB-020**: Verify `alimentar mix` supports weighted multi-source mixing
with upsampling. If not, implement mix command with upsample parameter.

### 5.5 Tokenizer

**Existing capability**: `aprender::text::tokenize::BpeTokenizer` with full
`train()` / `encode()` / `decode()` support. `entrenar::tokenizer::BPETokenizer`
provides the training-pipeline integration.

```bash
# Plan: validate inputs, estimate vocab training time
apr tokenize plan \
  --input ./data/processed/*.parquet \
  --vocab-size 32768 \
  --algorithm bpe \
  --output ./models/albor-tokenizer/

# Apply: train the tokenizer
apr tokenize apply \
  --input ./data/processed/*.parquet \
  --vocab-size 32768 \
  --algorithm bpe \
  --output ./models/albor-tokenizer/ \
  --seed 42
```

**Gap ALB-001**: Verify `apr tokenize plan/apply` exists as a CLI subcommand.
If not, wire `aprender::text::tokenize::BpeTokenizer::train()` into apr with
the plan/apply contract (see §1.5.2).

---

## 6. Training Configuration

### 6.1 Optimizer & Schedule

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard; in aprender/entrenar |
| Learning rate | 3e-4 | Chinchilla-recommended for 350M |
| Weight decay | 0.1 | Standard AdamW regularization |
| Beta1, Beta2 | 0.9, 0.95 | LLaMA/GPT-3 standard |
| Epsilon | 1e-8 | Standard |
| LR schedule | Cosine annealing with warmup | `CosineAnnealingLR` in aprender |
| Warmup steps | 2000 | ~0.2% of total steps |
| Min LR | 3e-5 | 10% of peak (standard) |
| Gradient clipping | 1.0 (global norm) | Stability |
| Batch size (global) | 512K tokens | ~256 sequences x 2048 tokens |
| Micro-batch (4090) | 8-16 | Tuned to VRAM |
| Gradient accumulation | 16-32 steps | Reach global batch size |
| Total training tokens | 10B | ~19,531 steps at 512K tokens/step |
| Mixed precision | fp16 (CUDA) | Hardware-appropriate |

### 6.2 Training Config: `configs/train/pretrain-350m.yaml`

A single YAML file defines **everything** — model architecture and training
hyperparameters. This is the industry standard (Axolotl, torchtune, HuggingFace
Trainer). One file, one truth. `apr train validate` lints it before GPU time.

```yaml
# configs/train/pretrain-350m.yaml — Albor 350M pre-training config

model:
  preset: "llama"                          # LLaMA-style decoder-only transformer
  hidden_size: 1024                        # d_model
  num_layers: 24
  num_attention_heads: 16                  # d_head = 64
  num_kv_heads: 4                          # GQA 4:1 ratio
  intermediate_size: 4096                  # SwiGLU FFN (gate + up + down)
  vocab_size: 32_768                       # BPE, power of 2 for GPU efficiency
  max_position_embeddings: 2048            # Context length
  rms_norm_eps: 1.0e-5
  tie_word_embeddings: false
  dropout: 0.0                             # Modern practice: regularize via data

data:
  train: "data/tokenized/train/"
  val: "data/tokenized/val/"
  batch_size: 8                            # Micro-batch per step
  seq_len: 2048
  tokenizer: "models/albor-tokenizer/tokenizer.json"
  input_column: "text"

optimizer:
  name: "adamw"
  lr: 3.0e-4
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1

training:
  mode: "causal_lm"
  epochs: 1                                # Pre-training uses max_steps, not epochs
  grad_clip: 1.0
  lr_scheduler: "cosine"
  warmup_steps: 2000
  gradient_accumulation: 32                # Global batch = 8 * 32 * 2048 = 512K tokens
  mixed_precision: "fp16"
  output_dir: "./checkpoints/albor-base-350m"
  save_interval: 1000
```

**Note on YAML numeric formatting**: YAML supports underscore notation natively
(`32_768`, `1_000_000`) for human-readable large numbers. All albor configs use
this convention. For shorthand like `10B` or `512K`, see gap ALB-021.

### 6.3 Training Workflow (Plan/Apply)

```bash
# Step 1: Plan — validate config, estimate VRAM, show execution plan (no GPU)
apr train plan configs/train/pretrain-350m.yaml

# Step 2: Apply — execute the training run
apr train apply configs/train/pretrain-350m.yaml --seed 42

# Step 3: Resume if interrupted (apply with --resume)
apr train apply configs/train/pretrain-350m.yaml \
  --resume checkpoints/albor-base-350m/checkpoint-step-5000.json \
  --seed 42
```

**Plan phase** (`apr train plan`):
- Schema validation: required keys, correct types, valid enum values
- Architecture sanity: `hidden_size` divisible by `num_attention_heads`, `num_kv_heads` divides `num_attention_heads`
- VRAM budget: computes model size + optimizer + activations, warns if > GPU capacity
- Data paths: confirms `train:` and `val:` directories exist with Parquet/tokenized shards
- Tokenizer: loads tokenizer, checks vocab size matches `model.vocab_size`
- Time estimate: estimated wall time based on model size and hardware
- Prints structured plan summary (see §1.5.2 for output format)
- **No GPU, no writes, no network.** Runs on CPU in seconds.

**Apply phase** (`apr train apply`):
- Reads the same YAML, builds a random-initialized `Transformer` with the
  `model:` section architecture, runs the causal LM training loop via entrenar
- Checkpoints every `save_interval` steps — resumable on crash
- No Rust code needed — just one config file

`apr train validate` is an alias for `apr train plan --strict` — schema-only
checking without resource estimation. Fast enough for CI.

### 6.4 Checkpointing Strategy

| Aspect | Design |
|--------|--------|
| Format | SafeTensors (primary) + JSON metadata |
| Frequency | Every 1000 steps (~512M tokens) |
| Content | Model weights, optimizer state, LR scheduler state, RNG state, step count |
| Storage | Local on lambda, rsync to intel (300GB RAM box) for backup |
| Resume | `--resume checkpoint-step-5000.json` |
| Export | `apr publish --format safetensors` for HuggingFace |

### 6.5 Experiment Tracking & Training Monitoring

entrenar has a full monitoring stack built in, and presentar provides rich
terminal visualization. Albor uses both — no external tools (no W&B, no
MLflow, no TensorBoard). Sovereign monitoring, sovereign visualization.

#### 6.5.1 Monitoring Config: `configs/train/pretrain-350m.yaml` (monitoring section)

```yaml
monitoring:
  terminal:
    enabled: true
    refresh_rate: 1000              # TUI refresh in ms
    metrics: ["loss", "learning_rate", "gradient_norm"]
    charts:
      - type: "loss_curve"
        metric: "loss"
        window: 100                 # Smoothing window
        show_eta: true

  tracking:
    enabled: true
    backend: "sqlite"               # .entrenar/experiments.db (WAL mode)
    experiment: "albor-pretrain-350m"
    tags:
      model: "albor-350m"
      stage: "pretrain"
      data: "10B-python-80pct"

  system:
    enabled: true
    interval: 5000                  # System metrics every 5s
    metrics: ["gpu_utilization", "memory", "temperature"]

  alerts:
    - condition: "loss > 10"
      action: "stop"
      message: "Loss exploded — Andon stop"
    - condition: "gradient_norm > 100"
      action: "stop"
      message: "Gradient explosion — Andon stop"
```

#### 6.5.2 What Entrenar Monitors Automatically

| Component | What It Does | Already Built? |
|-----------|-------------|----------------|
| `MetricsCollector` | Records loss, LR, gradient norms per step (SIMD-accelerated) | Yes (entrenar) |
| `ExperimentTracker` | Tracks run_id, params, metrics, artifacts, status | Yes (entrenar) |
| `SqliteBackend` | Durable experiment store: runs, params, metrics, artifacts in `.entrenar/experiments.db` (WAL mode) | Yes (entrenar) |
| `ProgressCallback` | Kalman-filtered ETA, Unicode progress bars | Yes (entrenar) |
| `MonitorCallback` | Integrates metrics into training, detects NaN/Inf → Andon alert | Yes (entrenar) |
| `CheckpointCallback` | Saves best model + metadata (epoch, is_best, timestamp) | Yes (entrenar) |
| `EarlyStopping` | Patience-based stopping on loss plateau | Yes (entrenar) |
| `Andon alerts` | Toyota Way: Critical/Error/Warning/Info severity levels | Yes (entrenar) |
| `TuiMonitor` | Detached terminal dashboard with Braille loss curves | Yes (entrenar) |
| `DriftDetector` | PSI, KS, Wasserstein distribution shift detection | Yes (entrenar) |
| `JsonFileStore` | Real-time metrics to `training_state.json` (atomic writes) | Yes (entrenar) |
| `LossCurve` widget | Training loss over epochs with EMA smoothing | Yes (presentar) |
| `ConfusionMatrix` widget | Multi-class classification evaluation | Yes (presentar) |
| `Braille/Sparkline` | High-resolution terminal charts (2x4 dots/cell, 8-level sparklines) | Yes (presentar) |
| `Heatmap` widget | 2D matrix with CIELAB perceptual color gradients | Yes (presentar) |

#### 6.5.3 Live Monitoring During Training

```bash
# Terminal 1 (lambda): Run training
apr train apply configs/train/pretrain-350m.yaml --seed 42

# Terminal 2 (lambda or ssh): Attach live monitor (presentar TUI)
apr monitor ./checkpoints/albor-base-350m/

# Terminal 3: Browse past experiments from SQLite
apr experiment view --db .entrenar/experiments.db

# Compare loss curves across runs
apr experiment view --db .entrenar/experiments.db \
  --runs albor-pretrain-50m,albor-pretrain-350m \
  --metric loss --chart loss_curve

# One-shot profiler (GPU utilization, per-layer timing)
apr cbtop ./checkpoints/albor-base-350m/latest.safetensors

# Inference latency profiling
apr profile ./checkpoints/albor-base-350m/ --prompt "def fibonacci(n):"

# Stack-level health (from batuta)
batuta stack status
```

#### 6.5.4 Experiment Lifecycle

Each training run creates two data streams:

**Real-time (JSON file IPC)** — for live TUI monitoring:
```
checkpoints/albor-base-350m/
├── training_state.json         # Live metrics (loss, lr, grad_norm, GPU telemetry)
├── checkpoint-step-1000.safetensors
├── checkpoint-step-1000.json   # Checkpoint metadata (epoch, is_best)
├── checkpoint-step-2000.safetensors
├── checkpoint-step-2000.json
├── checkpoint-best.safetensors
└── checkpoint-best.json
```

**Durable (SQLite experiment store)** — for post-hoc analysis and comparison:
```
.entrenar/experiments.db        # WAL mode, concurrent-safe
├── experiments                 # Experiment metadata (name, description, config)
├── runs                        # Training runs per experiment (status, timestamps)
├── params                      # Hyperparameters per run (key/value/type)
├── metrics                     # All metrics per run (key, step, value, timestamp)
├── artifacts                   # Model artifacts (path, size, SHA-256, binary data)
└── span_ids                    # Distributed trace integration
```

**Two consumers, zero contention**:
- `apr monitor` reads `training_state.json` (atomic write-then-rename) for
  live dashboards. Multiple monitors attach simultaneously.
- `apr experiment` reads `.entrenar/experiments.db` (WAL mode) for
  cross-run comparison, metric queries, artifact tracking. Read-only during
  training — no lock contention with the writer.

#### 6.5.5 Presentar Visualization: Rich Terminal Dashboards

presentar (`presentar-terminal`) provides **ML-specific visualization widgets**
that go far beyond entrenar's built-in `TuiMonitor`. The connection point is
entrenar's SQLite experiment store (`.entrenar/experiments.db`), which holds
all metrics, params, and artifacts across runs.

**Live training dashboard** (`apr monitor` — reads `training_state.json`):

```
╭─ Albor Pre-Train: albor-base-350m ─── Step 12,847 / 19,073 ──── 67.4% ─╮
│                                                                          │
│  Loss                                          GPU (RTX 4090)            │
│  3.2 ⣀⣀                                       ████████████░░░ 82%       │
│      ⠈⠉⠉⠑⠒⠒⠤⣀                                VRAM: 14.2 / 24.0 GB      │
│               ⠈⠉⠑⠒⠤⣀⣀                        Temp: 72°C                │
│  1.8                  ⠈⠉⠒⠒⣀⣀⣀⣀               Power: 312W               │
│                              ⠉⠉⠉              Tokens/s: 18,432          │
│  0 ──────────────────────────────── 12K                                  │
│                                                                          │
│  Learning Rate              Gradient Norm       ETA: 1d 14h 22m          │
│  ⣿⣿⣿⣷⣶⣶⣤⣤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀     ▁▁▂▁▁▃▁▂▁▁▁▂▁▁    Throughput: 5.2B / 10B   │
│  3e-4 → 2.1e-4              0.42 (norm)        Checkpoint: step-12000    │
╰──────────────────────────────────────────────────────────────────────────╯
```

**Post-hoc experiment comparison** (`apr experiment view` — reads SQLite):

```bash
# Compare loss curves across all pre-training runs
apr experiment view --db .entrenar/experiments.db \
  --runs albor-pretrain-50m,albor-pretrain-350m \
  --metric loss --chart loss_curve

# Hyperparameter comparison table
apr experiment view --db .entrenar/experiments.db \
  --experiment albor-pretrain-350m --params

# Export metrics for external analysis (Parquet for alimentar)
apr experiment export --db .entrenar/experiments.db \
  --run albor-pretrain-350m --format parquet --output ./eval/metrics.parquet
```

**Presentar widgets used by albor**:

| Widget | Use Case | Data Source |
|--------|----------|-------------|
| `LossCurve` | Training loss over steps with EMA smoothing | `training_state.json` (live) or SQLite `metrics` table (post-hoc) |
| `Sparkline` | Compact LR schedule, gradient norm history | `training_state.json` lr_history, grad_norm |
| `Heatmap` | Attention pattern visualization, weight distribution | Model checkpoint tensors |
| `Gauge` | GPU utilization, VRAM usage, training progress | `training_state.json` gpu telemetry |
| `BrailleGraph` | High-resolution loss/metric curves over SSH | `training_state.json` loss_history |
| `Histogram` | Weight distribution per layer (pre/post distillation) | Model checkpoint tensors |
| `BarChart` | Benchmark scores across model stages | `eval/*.json` results |

**Two rendering targets, same widgets, same data**:

presentar compiles the same widget tree to **two targets** — terminal and
WASM. The dashboard YAML is written once. `presentar-terminal` renders it
via crossterm (works over SSH). `presentar` renders it via WebGPU in the
browser (60fps, GPU-accelerated). Both read from the same data sources.

| Mode | Command | Renderer | Data Source | Use Case |
|------|---------|----------|-------------|----------|
| **Live TUI** | `apr monitor ./checkpoints/` | `presentar-terminal` (crossterm) | `training_state.json` (polling) | Watch training over SSH |
| **Experiment TUI** | `apr experiment view` | `presentar-terminal` (crossterm) | SQLite `.entrenar/experiments.db` | Compare runs in terminal |
| **Web dashboard** | `presentar serve --config albor-dashboard.yaml` | `presentar` (WebGPU/WASM) | SQLite + checkpoints | Rich browser dashboard |

Both TUI and WASM are **first-class deliverables**, not stretch goals.
The terminal TUI is the primary interface (SSH to lambda/intel). The WASM
dashboard is the shareable artifact for model cards and teaching.

#### 6.5.6 No External Dependencies

| What Others Use | What Albor Uses Instead | Why |
|-----------------|------------------------|-----|
| Weights & Biases | entrenar `SqliteBackend` + presentar dashboards | Sovereign — no cloud, no API keys, all data local |
| TensorBoard | presentar `LossCurve` + `BrailleGraph` over SSH | No Python, no browser required, works over SSH |
| MLflow | entrenar `ExperimentTracker` + SQLite + `apr experiment` | Self-hosted SQLite, no server process, query via CLI |
| nvidia-smi polling | entrenar system metrics + `apr cbtop` | Integrated into training loop, not bolted on |
| Streamlit dashboards | presentar WASM dashboard (10x faster rendering) | GPU-accelerated, 60fps, zero Python |

---

## 7. Post-Training Improvement Ladder

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

---

## 8. Evaluation & Benchmarks

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

**Gap ALB-006**: Verify `apr eval plan/apply` supports these benchmark tasks
natively. If not, implement benchmark harness integration with plan/apply
contract (see §1.5.2).

---

## 9. Distributed Training Architecture

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

---

## 10. Pipeline Orchestration (`apr pipeline` + forjar DAG)

### 10.1 Architecture: One Manifest, One DAG

The entire albor pipeline — from bare metal to published model — lives in a
single YAML manifest: `configs/pipeline/albor.yaml`. Forjar's DAG engine
resolves dependencies, tracks state, and dispatches steps across machines.
`apr pipeline` wraps forjar, so the user never calls forjar directly.

```
apr pipeline plan configs/pipeline/albor.yaml    # Show full DAG, estimate everything
apr pipeline apply configs/pipeline/albor.yaml   # Execute (resumable)
apr pipeline status                              # Show what's converged/pending/failed
apr pipeline drift                               # Detect unauthorized state changes
```

**How it works**:

```
                     configs/pipeline/albor.yaml
                              │
                    apr pipeline plan/apply
                              │
                     forjar DAG engine
                    (Kahn's toposort)
                              │
         ┌────────────┬───────┴───────┬────────────┐
         │            │               │            │
    infra resources   │          task resources    │
    (package, gpu,    │          (run apr cmds,    │
     file, mount,     │           track output)    │
     model)           │               │            │
         │            │               │            │
    forjar native     │     apr train apply        │
    convergence       │     apr distill apply      │
                      │     apr eval apply         │
                      │     apr publish apply      │
                      │               │            │
                 state/lambda/     state/intel/
                 state.lock.yaml   state.lock.yaml
```

**Key properties**:
- **Resumable**: BLAKE3 hashes per resource. Re-run skips converged steps.
- **Multi-machine**: Infra + tasks dispatch to lambda or intel via SSH.
- **Plan/apply**: `apr pipeline plan` shows the full DAG with estimates before
  committing any resources. Exit 0 if valid, exit 1 with diagnostics.
- **Idempotent**: Same manifest, same state → zero changes (all NoOp).
- **bashrs linted**: All shell fragments in task `command:` fields are validated
  by bashrs (Rash v6.65) at plan time. No unvalidated shell reaches execution.
  bashrs is KING of linting — `bashrs make lint` validates Makefiles, `bashrs lint`
  validates shell scripts, `bashrs classify` classifies safety.

**Dual orchestration**:
- **forjar manifest** (`configs/pipeline/albor.yaml`): Infrastructure provisioning
  (GPU drivers, packages, directories, mounts, teacher model download). Blocked on
  `type: task` (ALB-027) for ML steps.
- **batuta playbook** (`configs/pipeline/albor-playbook.yaml`): ML pipeline orchestration
  (data prep, train, distill, finetune, merge, prune, quantize, eval, publish).
  19-stage deterministic DAG with BLAKE3 caching. Validates successfully.

### 10.2 Pipeline Manifest: `configs/pipeline/albor.yaml`

```yaml
version: "1.0"
name: albor-training-pipeline
description: "Sovereign Python code completion model — full pipeline"

machines:
  lambda:
    hostname: lambda
    addr: 127.0.0.1
    user: noah
    arch: x86_64
    roles: [gpu-train, student]

  intel:
    hostname: intel
    addr: intel
    user: noah
    ssh_key: ~/.ssh/id_ed25519
    arch: x86_64
    roles: [teacher-inference, data-pipeline, eval, checkpoint-backup]

resources:
  # ═══════════════════════════════════════════════════════════
  # INFRASTRUCTURE (forjar native resources)
  # ═══════════════════════════════════════════════════════════

  cuda-driver:
    type: gpu
    machine: lambda
    gpu_backend: nvidia
    driver_version: "550"
    cuda_version: "12.4"
    persistence_mode: true
    compute_mode: exclusive_process

  vulkan-driver:
    type: package
    machine: intel
    provider: apt
    state: present
    packages: [mesa-vulkan-drivers, vulkan-tools, libvulkan-dev]

  data-dir:
    type: file
    machine: [lambda, intel]
    path: /data/albor
    state: directory
    mode: "0755"

  teacher-model:
    type: model
    machine: intel
    name: Qwen/Qwen3-Coder-Next
    state: present
    cache_dir: /data/albor/models/teacher
    depends_on: [data-dir]

  checkpoint-share:
    type: mount
    machine: intel
    source: "lambda:/data/albor/checkpoints"
    path: /data/albor/checkpoints
    fstype: nfs
    options: "rw,sync,no_subtree_check"
    depends_on: [data-dir]

  logit-share:
    type: mount
    machine: lambda
    source: "intel:/data/albor/teacher-logits"
    path: /data/albor/teacher-logits
    fstype: nfs
    options: "ro,sync"
    depends_on: [data-dir]

  # ═══════════════════════════════════════════════════════════
  # DATA PIPELINE (task resources — call apr subcommands)
  # ═══════════════════════════════════════════════════════════

  ingest-local:
    type: task
    machine: lambda
    command: >
      alimentar import local ../depyler/examples/ ../depyler/tdd-book/tests/
        --lang python --output ./data/local/depyler.parquet &&
      alimentar import local ../hf-ground-truth-corpus/
        --lang python --output ./data/local/hf-gtc.parquet &&
      alimentar import local ../jax-ground-truth-corpus/
        --lang python --output ./data/local/jax-gtc.parquet &&
      alimentar import local ../vllm-ground-truth-corpus/
        --lang python --output ./data/local/vllm-gtc.parquet
    output_artifacts: ["./data/local/*.parquet"]
    depends_on: [data-dir]

  ingest-external:
    type: task
    machine: lambda
    command: >
      alimentar import hf bigcode/starcoderdata --lang python
        --output ./data/starcoder-python/ &&
      alimentar import hf HuggingFaceFW/fineweb-edu
        --output ./data/fineweb-edu/
    output_artifacts: ["./data/starcoder-python/", "./data/fineweb-edu/"]
    depends_on: [data-dir]

  data-mix:
    type: task
    machine: lambda
    command: >
      alimentar quality check ./data/ --profile ml-training &&
      alimentar mix
        --input ./data/local/depyler.parquet --weight 0.025 --upsample 10
        --input ./data/local/hf-gtc.parquet --weight 0.025 --upsample 10
        --input ./data/local/jax-gtc.parquet --weight 0.025 --upsample 10
        --input ./data/local/vllm-gtc.parquet --weight 0.025 --upsample 10
        --input ./data/starcoder-python/ --weight 0.40
        --input ./data/fineweb-edu/ --weight 0.20
        --input ./data/processed/python-docs.parquet --weight 0.10
        --output ./data/mixed/ --seed 42 --shuffle
    output_artifacts: ["./data/mixed/"]
    depends_on: [ingest-local, ingest-external]

  tokenize:
    type: task
    machine: lambda
    command: >
      apr tokenize plan --input ./data/mixed/*.parquet --vocab-size 32768
        --output ./models/albor-tokenizer/ &&
      apr tokenize apply --input ./data/mixed/*.parquet --vocab-size 32768
        --output ./models/albor-tokenizer/ --seed 42 &&
      apr tokenize apply --tokenizer ./models/albor-tokenizer/
        --input ./data/mixed/*.parquet --output ./data/tokenized/
        --max-seq-len 2048
    output_artifacts: ["./models/albor-tokenizer/", "./data/tokenized/"]
    depends_on: [data-mix]

  # ═══════════════════════════════════════════════════════════
  # TRAINING (task resources — long-running, checkpoint-aware)
  # ═══════════════════════════════════════════════════════════

  train-50m:
    type: task
    machine: lambda
    command: >
      apr train plan configs/train/pretrain-50m.yaml &&
      apr train apply configs/train/pretrain-50m.yaml --seed 42
    output_artifacts: ["./checkpoints/albor-base-50m/"]
    completion_check: "test -f ./checkpoints/albor-base-50m/checkpoint-best.safetensors"
    depends_on: [tokenize, cuda-driver]

  train-350m:
    type: task
    machine: lambda
    command: >
      apr train plan configs/train/pretrain-350m.yaml &&
      apr train apply configs/train/pretrain-350m.yaml --seed 42
    output_artifacts: ["./checkpoints/albor-base-350m/"]
    completion_check: "test -f ./checkpoints/albor-base-350m/checkpoint-best.safetensors"
    depends_on: [train-50m]

  # ═══════════════════════════════════════════════════════════
  # DISTILLATION (cross-machine: intel produces logits, lambda trains)
  # ═══════════════════════════════════════════════════════════

  distill-logits:
    type: task
    machine: intel
    command: >
      apr distill plan configs/train/distill.yaml &&
      apr distill apply configs/train/distill.yaml --stage precompute
    output_artifacts: ["./data/teacher-logits/"]
    completion_check: "test -d ./data/teacher-logits/ && ls ./data/teacher-logits/*.parquet"
    depends_on: [train-350m, teacher-model, logit-share]

  distill:
    type: task
    machine: lambda
    command: >
      apr distill apply configs/train/distill.yaml --stage train --seed 42
    output_artifacts: ["./checkpoints/albor-distill/"]
    completion_check: "test -f ./checkpoints/albor-distill/checkpoint-best.safetensors"
    depends_on: [distill-logits]

  # ═══════════════════════════════════════════════════════════
  # POST-TRAINING LADDER (sequential, each depends on previous)
  # ═══════════════════════════════════════════════════════════

  finetune:
    type: task
    machine: lambda
    command: >
      apr finetune plan configs/train/finetune-lora.yaml &&
      apr finetune apply configs/train/finetune-lora.yaml
    output_artifacts: ["./checkpoints/albor-instruct/"]
    depends_on: [distill]

  merge:
    type: task
    machine: lambda
    command: >
      apr merge plan --models albor-distill-350m,albor-instruct-350m
        --method slerp --weight 0.6 --output ./checkpoints/albor-merged/ &&
      apr merge apply --models albor-distill-350m,albor-instruct-350m
        --method slerp --weight 0.6 --output ./checkpoints/albor-merged/
    output_artifacts: ["./checkpoints/albor-merged/"]
    depends_on: [finetune]

  prune:
    type: task
    machine: lambda
    command: >
      apr prune plan --model ./checkpoints/albor-merged-350m/
        --method wanda --sparsity 0.5 --output ./checkpoints/albor-pruned/ &&
      apr prune apply --model ./checkpoints/albor-merged-350m/
        --method wanda --sparsity 0.5 --output ./checkpoints/albor-pruned/
    output_artifacts: ["./checkpoints/albor-pruned/"]
    depends_on: [merge]

  quantize:
    type: task
    machine: lambda
    command: >
      apr quantize plan --model ./checkpoints/albor-merged-350m/
        --method q4_k --output ./checkpoints/albor-q4/ &&
      apr quantize apply --model ./checkpoints/albor-merged-350m/
        --method q4_k --output ./checkpoints/albor-q4/
    output_artifacts: ["./checkpoints/albor-q4/"]
    depends_on: [merge]

  # ═══════════════════════════════════════════════════════════
  # EVALUATION (can run on intel concurrently with training)
  # ═══════════════════════════════════════════════════════════

  eval-code:
    type: task
    machine: lambda
    command: >
      apr eval plan --model ./checkpoints/albor-merged-350m/
        --tasks humaneval,humaneval_fim,mbpp,ds1000 &&
      apr eval apply --model ./checkpoints/albor-merged-350m/
        --tasks humaneval,humaneval_fim,mbpp,ds1000
        --output ./eval/python-code-results.json --seed 42
    output_artifacts: ["./eval/python-code-results.json"]
    depends_on: [merge]

  eval-general:
    type: task
    machine: intel
    command: >
      apr eval apply --model ./checkpoints/albor-merged-350m/
        --tasks arc_easy,hellaswag,piqa,lambada
        --output ./eval/general-results.json --seed 42
    output_artifacts: ["./eval/general-results.json"]
    depends_on: [merge, checkpoint-share]

  # ═══════════════════════════════════════════════════════════
  # RELEASE
  # ═══════════════════════════════════════════════════════════

  export:
    type: task
    machine: lambda
    command: >
      apr export plan --model ./checkpoints/albor-q4/ --format gguf &&
      apr export apply --model ./checkpoints/albor-q4/ --format gguf
        --output ./release/albor-350m-q4_k.gguf &&
      apr export apply --model ./checkpoints/albor-merged-350m/
        --format safetensors
        --output ./release/albor-350m.safetensors
    output_artifacts: ["./release/"]
    depends_on: [quantize, eval-code]

  publish:
    type: task
    machine: lambda
    command: >
      apr publish plan --model ./release/ --hub paiml/albor-350m &&
      apr publish apply --model ./release/ --hub paiml/albor-350m
    depends_on: [export, eval-general]

policy:
  failure: stop_on_first
  parallel_machines: true
  retry: 2
  bashrs_lint: true            # Validate all task command: fields via bashrs
```

### 10.3 Pipeline Workflow

```bash
# Show full DAG with time/resource estimates (no side effects)
apr pipeline plan configs/pipeline/albor.yaml

# Execute everything (resumable — skips converged steps)
apr pipeline apply configs/pipeline/albor.yaml

# Check what's done, what's pending, what failed
apr pipeline status

# Detect unauthorized changes to converged resources
apr pipeline drift

# Re-run only failed steps (everything else is NoOp)
apr pipeline apply configs/pipeline/albor.yaml

# Force re-run a specific resource and its dependents
apr pipeline apply configs/pipeline/albor.yaml --target train-350m --force
```

### 10.4 The `task` Resource Type (ALB-027)

The `task` resource is what makes forjar a pipeline orchestrator, not just an
infrastructure tool. It runs an arbitrary command, tracks completion, and
hashes output artifacts for idempotency.

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Shell command to execute (bashrs-validated at plan time) |
| `output_artifacts` | list[string] | Paths to hash for idempotency (glob-supported) |
| `completion_check` | string | Optional shell expression to verify completion (e.g., checkpoint exists) |
| `timeout` | duration | Max wall time before Andon stop (default: none) |
| `resume_command` | string | Optional command for resuming interrupted long-running tasks |

**Idempotency for ML tasks**: A `task` resource is considered converged when:
1. The `command` exited 0 on a previous run, AND
2. The BLAKE3 hash of `output_artifacts` matches the lock file, AND
3. The `completion_check` (if set) passes

If any of these fail, the task is re-run. For training jobs that crashed
mid-run, the `command` itself includes `--resume` logic (e.g., `apr train
apply` auto-detects and resumes from the latest checkpoint).

### 10.5 Why Not Makefile / Shell Scripts

| Approach | DAG | State | Resume | Multi-Machine | Lint |
|----------|-----|-------|--------|---------------|------|
| **`apr pipeline` (forjar)** | Kahn's toposort | BLAKE3 lock files | Automatic (skip converged) | Native SSH dispatch | bashrs at plan time |
| Makefile | File timestamps only | None | Manual | None (SSH in recipes) | None |
| Shell scripts | Sequential only | None | Manual | Manual SSH | ShellCheck (external) |

The Makefile and shell scripts are eliminated. One manifest. One DAG. One tool.

---

## 11. Gap Register

Every gap discovered during development is tracked here. Each gap maps to a
specific upstream component, a GitHub issue, and a clear acceptance criterion.

**Lifecycle**: Gap discovered → GitHub issue filed → implemented upstream →
wired into `apr` → dogfooded in albor pipeline → FALSIFY/pmat verified → closed.

| Status | Meaning |
|--------|---------|
| OPEN | Gap identified, not yet implemented |
| IN PROGRESS | GitHub issue filed, work underway |
| DOGFOODING | Implemented, being validated in albor pipeline |
| CLOSED | Verified working end-to-end, issue closed |

### 11.1 Critical Path Gaps (Block the Improvement Ladder)

| ID | Issue | Component | Gap | Severity | Status | Acceptance Criterion |
|----|-------|-----------|-----|----------|--------|---------------------|
| ALB-001 | [#6](https://github.com/paiml/albor/issues/6) | apr (aprender) | `apr tokenize plan/apply` subcommand | Medium | **FIXED** | `apr tokenize plan` validates inputs + estimates time; `apr tokenize apply` trains BPE/WordPiece/Unigram tokenizer (`aprender@90427205`). Writes vocab.json + merges.txt. |
| ALB-006 | [#7](https://github.com/paiml/albor/issues/7) | apr (aprender) | `apr eval plan/apply` benchmark harness | High | **FIXED** | `apr eval --task code --data benchmark.jsonl` evaluates code completion with pass@1 scoring. `apr eval --task plan` validates model + data exist. JSONL format with prompt/test/canonical_solution. Phase 1: structural validation. Phase 2: full inference (ALB-009 prerequisite). (`aprender@4e61297e`) |
| ALB-007 | [#8](https://github.com/paiml/albor/issues/8) | entrenar | Parquet→LMBatch bridge via alimentar | Medium | **FIXED** | `load_lm_batches_from_parquet()` reads text or pre-tokenized Parquet (single file or directory of shards) via alimentar. Text columns tokenized with HfTokenizer. Column auto-detection. (`entrenar@a5a2fb7`) |
| ALB-009 | [#1](https://github.com/paiml/albor/issues/1) | apr (entrenar) | `apr train plan/apply` for pre-training from scratch | Critical | **FIXED** | `apr train plan --task pretrain --config <yaml>` validates config via entrenar, shows architecture and training params. `apr train apply --task pretrain --config <yaml>` runs full pre-training (TransformerTrainer + CausalLMLoss). (`aprender@d79ed943`) |
| ALB-010 | [#2](https://github.com/paiml/albor/issues/2) | realizar | Qwen3-Coder-Next / DeltaNet / MoE architecture support | Critical | OPEN | `realizar` loads and runs inference on Qwen3-Coder-Next (80B MoE with DeltaNet layers) |
| ALB-011 | [#3](https://github.com/paiml/albor/issues/3) | apr (entrenar + realizar) | `apr distill plan/apply` (precompute + train stages) | Critical | **FIXED** | `apr distill --config <yaml> --plan` validates config, shows teacher/student/training params. `apr distill --config <yaml> --stage precompute` inspects teacher, writes manifest. `apr distill --config <yaml> --stage train` validates precompute manifest, sets up KD training. Local DistillYamlConfig matches entrenar schema. (`aprender@81dd4432`) |
| ALB-018 | [#19](https://github.com/paiml/albor/issues/19) | entrenar/alimentar | Fill-in-the-Middle (FIM) data transform (PSM/SPM) | High | **FIXED** | `alimentar fim` transform with PSM/SPM formats, configurable rate/seed (`alimentar@290582d`). `Fim` struct implements `Transform` trait for pipeline integration. |
| ALB-019 | [#20](https://github.com/paiml/albor/issues/20) | alimentar | `alimentar import local` for local Python files | Medium | **FIXED** | `alimentar import local` subcommand now available (commit `alimentar@265541b`). Supports CSV/JSON/JSONL/Parquet format conversion. |
| ALB-020 | [#21](https://github.com/paiml/albor/issues/21) | alimentar | `alimentar mix` with weighted upsampling | Medium | **FIXED** | `alimentar mix` with weighted sampling and upsampling now available (`alimentar@64b1e92`). Syntax: `alimentar mix a.parquet:0.8 b.parquet:0.2 -o out.parquet`. |
| ALB-021 | [#22](https://github.com/paiml/albor/issues/22) | entrenar | Custom model architecture params in YAML | High | **FIXED** | `ArchitectureOverrides` struct carries YAML manifest `architecture:` params through bridge converter to `TransformerConfig`. Supports all fields: `hidden_size`, `num_layers`, `num_heads`, `num_kv_heads`, `intermediate_size`, `vocab_size`, `max_seq_length`, `rms_norm_eps`, `rope_theta`, `use_bias`. (`entrenar@a414861`) |
| ALB-022 | [#23](https://github.com/paiml/albor/issues/23) | entrenar | Human-readable value shorthand in YAML configs | Low | **FIXED** | `parse_human_usize()` and `deserialize_human_usize_opt` support SI suffixes (32K, 1M, 10B, 1T), scientific notation (1e6), and fractional suffixes (1.5K). Applied to ArchitectureConfig and DataConfig fields. (`entrenar@1cb0950`) |
| ALB-023 | [#24](https://github.com/paiml/albor/issues/24) | apr (aprender) | Plan/apply contract for all subcommands | High | **FIXED** | Every `apr <cmd>` action command now exposes plan mode: `merge --plan`, `export --plan`, `publish --plan` added to join existing `train plan/apply`, `tokenize plan/apply`, `quantize --plan`, `finetune --plan`, `prune --plan`, `distill --plan`, `eval --task plan`. Pre-dispatch contract validation skipped in plan mode. (`aprender@526a1e4b`) |
| ALB-024 | [#25](https://github.com/paiml/albor/issues/25) | presentar + apr | `apr experiment view` — SQLite experiment browser (TUI + WASM) | Medium | OPEN | `apr experiment view --db .entrenar/experiments.db` renders presentar `LossCurve`, metric comparison, run diff. Terminal via `presentar-terminal`, browser via `presentar serve`. Both read same SQLite tables. |
| ALB-025 | [#26](https://github.com/paiml/albor/issues/26) | presentar + apr | `apr monitor` upgrade — presentar widgets for live training TUI | Medium | OPEN | `apr monitor` uses presentar `LossCurve`, `Gauge`, `Sparkline`, `BrailleGraph` instead of entrenar's built-in `TuiMonitor` renderer. Same widget tree compiles to terminal and WASM. |
| ALB-026 | [#27](https://github.com/paiml/albor/issues/27) | presentar | WASM training dashboard — `albor-dashboard.yaml` | Medium | OPEN | Declarative YAML dashboard config that renders training metrics, experiment comparison, and model card via `presentar serve`. Embeddable in HuggingFace model card as static WASM artifact. |
| ALB-027 | [#4](https://github.com/paiml/albor/issues/4) | forjar | `task` resource type for pipeline orchestration | Critical | **FIXED** | New forjar resource type: runs arbitrary command, tracks exit code, hashes `output_artifacts` for idempotency via b3sum, supports `completion_check` and `timeout`. Handlers: check_script, apply_script, state_query_script. Validation: command required, timeout > 0. (`forjar@d14e633`) |
| ALB-028 | [#5](https://github.com/paiml/albor/issues/5) | apr (aprender) | `apr pipeline plan/apply` wrapping forjar DAG engine | Critical | **FIXED** | `apr pipeline plan` shows full DAG with 23 resources across 2 machines. `apr pipeline apply` converges via forjar engine. `apr pipeline status` shows state. `apr pipeline validate` checks manifest. Shells out to forjar binary (decoupled). (`aprender@e653d5ca`) |

### 11.2 Distributed Training Gaps (Stretch / Future)

| ID | Issue | Component | Gap | Severity | Status | Acceptance Criterion |
|----|-------|-----------|-----|----------|--------|---------------------|
| ALB-002 | [#9](https://github.com/paiml/albor/issues/9) | repartir | Ring all-reduce implementation | High | OPEN | Gradient tensors synchronized across 2+ workers with <5% overhead |
| ALB-003 | [#10](https://github.com/paiml/albor/issues/10) | entrenar | repartir integration for distributed training | High | OPEN | Training loop calls `repartir::GradientSync` for multi-worker training |
| ALB-004 | [#11](https://github.com/paiml/albor/issues/11) | entrenar | Unified CUDA + wgpu backend dispatch | Medium | OPEN | Same training config runs on CUDA (4090) and wgpu (W5700X) |
| ALB-005 | [#12](https://github.com/paiml/albor/issues/12) | trueno | wgpu backward pass (gradient WGSL shaders) | High | OPEN | Compute shaders for matmul_backward, gelu_backward, rmsnorm_backward, attention_backward |
| ALB-008 | [#13](https://github.com/paiml/albor/issues/13) | repartir | Heterogeneous worker throughput balancing | Medium | OPEN | Workers with different GPU speeds get proportional workload |

### 11.3 Quality & Verification Gaps

| ID | Issue | Component | Gap | Severity | Status | Acceptance Criterion |
|----|-------|-----------|-----|----------|--------|---------------------|
| ALB-013 | [#14](https://github.com/paiml/albor/issues/14) | provable-contracts | Knowledge distillation contract | High | DOGFOODING | `knowledge-distillation-kernel-v1.yaml` — committed and passes `pv validate`. 3 equations, 6 obligations, 5 falsification tests, 2 Kani harnesses. Needs binding to entrenar implementation. |
| ALB-014 | [#15](https://github.com/paiml/albor/issues/15) | provable-contracts | BPE tokenizer contract | Medium | DOGFOODING | `bpe-tokenizer-kernel-v1.yaml` — committed and passes `pv validate`. Roundtrip invariant, FIM sentinel tests. Needs binding to aprender BPE. |
| ALB-015 | [#16](https://github.com/paiml/albor/issues/16) | provable-contracts | Model merging contract (SLERP, TIES, DARE) | Medium | DOGFOODING | `model-merging-kernel-v1.yaml` — committed and passes `pv validate`. SLERP bound, DARE unbiased estimator. Needs binding. |
| ALB-016 | [#17](https://github.com/paiml/albor/issues/17) | provable-contracts | Pruning contract (WANDA, magnitude) | Medium | DOGFOODING | `pruning-kernel-v1.yaml` — committed and passes `pv validate`. Sparsity invariant, score ordering. Needs binding. |
| ALB-017 | [#18](https://github.com/paiml/albor/issues/18) | provable-contracts | Gradient accumulation contract | High | DOGFOODING | `gradient-accumulation-kernel-v1.yaml` — committed and passes `pv validate`. Numerical equivalence, gradient zeroing. Needs binding. |

**Contract coverage report** (`pv coverage contracts`): 5 contracts, 13 equations, 29 obligations, 19 falsification tests, 7 Kani harnesses, **100% obligation coverage**. All contracts at impl=0/N — waiting for upstream bindings.

### 11.4 Dogfooding-Discovered Gaps

| ID | Issue | Component | Gap | Severity | Status | Acceptance Criterion |
|----|-------|-----------|-----|----------|--------|---------------------|
| ALB-029 | [#28](https://github.com/paiml/albor/issues/28) | batuta | `batuta falsify` false positives on project repos | Medium | **FIXED** | Fixed upstream in `batuta@905a862`: AI-01 searches `configs/`, AI-04 excludes `book-output/`, AI-05 detects pv/forjar validation. Score: 72.2% → 73.1%. |
| ALB-030 | [#29](https://github.com/paiml/albor/issues/29) | batuta | `batuta stack status` fails without Cargo.toml | Low | **FIXED** | Fixed upstream in `batuta@371557a`: Falls back to binary detection, discovers 11 installed PAIML tools with versions. |
| ALB-031 | [#30](https://github.com/paiml/albor/issues/30) | batuta | `batuta hf search` returns mock/placeholder data | Low | OPEN | `batuta hf search model "code completion"` returns live HuggingFace Hub results instead of placeholder models. |

*Gaps are added as they are discovered during implementation and dogfooding.*

---

## 12. Provable Quality & Design by Contract

Every computational kernel used in Albor must have a provable-contracts YAML
specification with Popperian falsification tests, property-based probar tests,
and Kani bounded model checking harnesses. This is not optional — it is a
first-class deliverable alongside the model.

### 12.1 Verification Ladder

Four levels of assurance, from cheapest to most rigorous:

```
Level 4: Kani bounded model check    ─── PROOF (exhaustive for inputs ≤ N)
Level 3: probar property tests       ─── HIGH CONFIDENCE (10,000+ random inputs)
Level 2: Falsification tests         ─── TARGETED (specific edge cases)
Level 1: Type system                 ─── BY CONSTRUCTION (Rust compiler)
Level 0: Code review                 ─── HUMAN (necessary but insufficient)
```

**Requirement**: Every kernel reaches at least Level 3. Critical kernels
(softmax, attention, cross-entropy, KD loss) reach Level 4.

### 12.2 Contract Registry for Albor

Albor requires contracts for every kernel in the training + post-training pipeline.
Many already exist in provable-contracts; new ones must be written.

#### Existing Contracts (bind to aprender implementations)

| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| `softmax-kernel-v1.yaml` | softmax | 6 (normalization, positivity, monotonicity, SIMD parity, translation invariance, bound) | Exists, 289 bindings |
| `rmsnorm-kernel-v1.yaml` | RMSNorm | 5 (finiteness, scale invariance, SIMD parity, idempotency) | Exists |
| `attention-kernel-v1.yaml` | scaled dot-product attention | Multiple (causal mask, score bounds, gradient flow) | Exists |
| `rope-kernel-v1.yaml` | Rotary Position Embedding | Multiple (rotation invariant, frequency spectrum) | Exists |
| `gelu-kernel-v1.yaml` | GELU activation | Bound, monotonicity, SIMD parity | Exists |
| `matmul-kernel-v1.yaml` | matrix multiplication | Associativity, SIMD parity, bound | Exists |
| `cross-entropy-kernel-v1.yaml` | cross-entropy loss | Non-negativity, gradient correctness | Exists |
| `adamw-kernel-v1.yaml` | AdamW optimizer | Bias correction, weight decay decoupling | Exists |
| `gqa-kernel-v1.yaml` | Grouped Query Attention | Equivalence to MHA when groups=heads | Exists |
| `swiglu-kernel-v1.yaml` | SwiGLU FFN | Gating invariants | Exists |

#### New Contracts Required for Albor (ALB-013 through ALB-017)

| Contract (NEW) | Key Equations | Key Obligations | Priority |
|----------------|---------------|-----------------|----------|
| `knowledge-distillation-kernel-v1.yaml` | KD_loss = α·KL(σ(z_t/T) ∥ σ(z_s/T))·T² + (1-α)·CE(y, z_s) | KL non-negativity, temperature scaling invariant, gradient correctness, α interpolation bound | Critical |
| `bpe-tokenizer-kernel-v1.yaml` | BPE merge rules, byte-pair encoding | Roundtrip invariant: decode(encode(x)) = x, vocab coverage, merge ordering | High |
| `model-merging-kernel-v1.yaml` | SLERP: interp(θ, w₁, w₂) on unit sphere; TIES: trim + elect + disjoint merge | SLERP interpolation bound (‖result‖ ≈ 1), TIES sparsity guarantee | Medium |
| `pruning-kernel-v1.yaml` | WANDA: score = |w| · ‖x‖₂; magnitude: score = |w| | Sparsity invariant (exactly k% weights zeroed), score ordering preserved | Medium |
| `gradient-accumulation-kernel-v1.yaml` | G_accum = (1/N)·Σ g_i ≈ g_full | Numerical equivalence within tolerance, loss scaling correctness | High |

### 12.3 Contract Workflow for Each Kernel

```bash
# 1. Write or validate YAML contract
pv validate contracts/knowledge-distillation-kernel-v1.yaml

# 2. Generate trait stubs + failing tests
pv scaffold contracts/knowledge-distillation-kernel-v1.yaml

# 3. Generate property-based tests (wired to actual aprender code)
pv probar contracts/knowledge-distillation-kernel-v1.yaml \
  --binding contracts/aprender/binding.yaml

# 4. Generate Kani bounded model checking harnesses
pv kani contracts/knowledge-distillation-kernel-v1.yaml

# 5. Run falsification sweep
pv audit contracts/knowledge-distillation-kernel-v1.yaml \
  --binding contracts/aprender/binding.yaml

# 6. Verify full contract status
pv status contracts/knowledge-distillation-kernel-v1.yaml
```

### 12.4 Falsification Tests: Albor-Specific

Every claim in this specification must be falsifiable. Below are the concrete
falsification tests for Albor's key properties.

#### Training Correctness

```yaml
# FALSIFY-ALBOR-001: Loss decreases monotonically (smoothed)
- id: FALSIFY-ALBOR-001
  rule: "Training convergence"
  prediction: "EMA(loss, window=100) is monotonically decreasing after warmup"
  test: "Load training log, compute EMA, assert no sustained increase >5% over 500 steps"
  if_fails: "Learning rate too high, data corruption, or gradient computation bug"

# FALSIFY-ALBOR-002: Gradient norms are bounded
- id: FALSIFY-ALBOR-002
  rule: "Training stability"
  prediction: "Global gradient norm < 10.0 after clipping for all steps"
  test: "Parse training log, assert max gradient norm across all steps"
  if_fails: "Gradient clipping not applied, loss spike, or NaN propagation"

# FALSIFY-ALBOR-003: Checkpoint determinism
- id: FALSIFY-ALBOR-003
  rule: "Reproducibility"
  prediction: "Two runs with seed=42 produce identical checkpoints at step 1000"
  test: "Train twice, BLAKE3 hash both checkpoints, assert equality"
  if_fails: "Non-deterministic operation (async GPU, HashMap ordering, etc.)"
```

#### Distillation Correctness

```yaml
# FALSIFY-ALBOR-004: KL divergence is non-negative
- id: FALSIFY-ALBOR-004
  rule: "KD loss validity"
  prediction: "KL(teacher || student) >= 0 for all batches"
  test: "proptest with 10000 random logit pairs, assert KL >= -1e-7"
  if_fails: "Log-domain computation error or softmax numerical instability"

# FALSIFY-ALBOR-005: Distillation improves over base
- id: FALSIFY-ALBOR-005
  rule: "Distillation value"
  prediction: "albor-distill avg benchmark > albor-base avg benchmark"
  test: "Run full eval suite on both, paired t-test with p < 0.05"
  if_fails: "Teacher logits corrupted, temperature too high/low, or alpha miscalibrated"

# FALSIFY-ALBOR-006: Teacher logit integrity
- id: FALSIFY-ALBOR-006
  rule: "Data pipeline integrity"
  prediction: "Pre-computed teacher logits match live teacher inference within 1e-4"
  test: "Sample 100 batches, run live teacher inference, compare against stored logits"
  if_fails: "Serialization precision loss, wrong batch ordering, or teacher model mismatch"
```

#### Post-Training Invariants

```yaml
# FALSIFY-ALBOR-007: Merge interpolation bound
- id: FALSIFY-ALBOR-007
  rule: "SLERP correctness"
  prediction: "‖SLERP(w1, w2, t)‖ ≈ ‖w1‖ for t ∈ [0,1] (unit sphere)"
  test: "proptest with 10000 random weight pairs and t values"
  if_fails: "SLERP implementation uses LERP instead, or normalization missing"

# FALSIFY-ALBOR-008: Pruning sparsity guarantee
- id: FALSIFY-ALBOR-008
  rule: "WANDA correctness"
  prediction: "Exactly 50% of weights are zero after prune --sparsity 0.5"
  test: "Count zero weights, assert within ±0.1% of target sparsity"
  if_fails: "Pruning threshold computation error or layer exclusion bug"

# FALSIFY-ALBOR-009: Quantization round-trip
- id: FALSIFY-ALBOR-009
  rule: "Q4 fidelity"
  prediction: "Perplexity(Q4 model) < 1.05 × Perplexity(fp16 model)"
  test: "Evaluate both on held-out set, assert ratio < 1.05"
  if_fails: "Quantization calibration data insufficient or block size wrong"
```

### 12.5 Verification DAG (Albor End-to-End)

Like the Qwen 3.5 verification DAG in provable-contracts, Albor composes
sub-contracts into a full model verification:

```
softmax ← attention ← gqa
                        ↑
rmsnorm ──────────────── albor-forward ← training-loop
                        ↑                      ↑
gelu ← swiglu ──────────┘                     │
                                               │
rope ──────────────────── albor-forward        │
                                               │
matmul ← gqa                                   │
                                               │
cross-entropy ─────────── training-loss ────────┘
                              ↑
adamw ─────────── optimizer-step ──────── training-loop
                                               │
gradient-accumulation ─────────────────────────┘
                                               │
knowledge-distillation ── distill-loss ── distill-loop
                              ↑
bpe-tokenizer ─── data-pipeline ─── training-loop

model-merging ─── post-training ─── albor-merged
pruning ────────── post-training ─── albor-pruned
```

Each node in this DAG is a contract. `pv graph contracts/ --format mermaid`
renders the full dependency graph. A change to any sub-contract triggers
re-verification of all dependents.

---

## 13. pmat Compliance & Quality Gates

### 13.1 Scope: Where Quality Applies

Albor is a project repo (configs, scripts, contracts, docs). It produces no
Rust library code. All quality gates apply to **upstream Rust changes** made
in service of Albor's gaps — not to albor's shell scripts or YAML configs.

```bash
# Run on all modified stack components (NOT on albor itself)
pmat comply check --strict ../aprender      # ALB-001, 006, 009, 011
pmat comply check --strict ../entrenar      # ALB-003, 004
pmat comply check --strict ../trueno        # ALB-005
pmat comply check --strict ../realizar      # ALB-010
pmat comply check --strict ../alimentar     # ALB-007, 018, 019, 020
pmat comply check --strict ../repartir      # ALB-002, 008
```

### 13.2 Quality Gate Thresholds (Upstream Rust Code)

| Gate | Threshold | Applies To | Enforcement |
|------|-----------|-----------|-------------|
| TDG Grade | A (score ≤ 1.0) | Upstream Rust | `pmat analyze tdg --include-components` |
| Test Coverage | ≥ 95% line coverage | Upstream Rust | `cargo llvm-cov --summary-only` |
| Mutation Score | ≥ 85% | Upstream Rust | `cargo mutants --no-times` |
| Cyclomatic Complexity | ≤ 15 per function | Upstream Rust | `pmat analyze complexity` |
| **File Length** | **≤ 500 lines** | All Rust files (upstream) | `find . -name '*.rs' \| xargs wc -l` |
| SATD | Zero (no TODO/FIXME/HACK) | Upstream Rust | `pmat analyze satd` |
| Unwrap Calls | Zero in new code | Upstream Rust | `pmat query --literal "unwrap()" --faults` |
| Clippy | Zero warnings | Upstream Rust | `cargo clippy -- -D warnings` |

### 13.3 Quality Gate Thresholds (Albor Repo)

| Gate | Threshold | Applies To | Enforcement |
|------|-----------|-----------|-------------|
| **File Length** | **≤ 500 lines** | Scripts, YAML, contracts (not specs/docs) | `wc -l` on non-doc tracked files |
| FALSIFY-ALBOR tests | All 9 pass | Pipeline end-to-end | `batuta falsify .` |
| Contract completeness | All 5 new contracts at Level 3+ | `contracts/` | `pv status contracts/` |
| Config validity | All YAML parses and `plan` passes | `configs/` | `apr pipeline plan` (validates all configs in one DAG pass) |
| Reproducibility | Same seed → same checkpoint hash | Full pipeline | FALSIFY-ALBOR-003 |

### 13.3 pmat Quality Commands for Albor

```bash
# TDG analysis of all Albor-touched code
pmat analyze tdg ../aprender --include-components
pmat analyze tdg ../entrenar --include-components

# Find coverage gaps (highest ROI targets)
pmat query --coverage-gaps --limit 30 --exclude-tests

# Fault pattern audit (unwrap, panic, unsafe)
pmat query "training" --faults --exclude-tests

# Full quality audit on distillation code
pmat query "distill" --churn --duplicates --entropy --faults -G

# Complexity check on new kernels
pmat query "knowledge_distillation" --max-complexity 15 --include-source

# Create quality baseline before Albor work begins
pmat tdg baseline create

# Check for regressions after each phase
pmat tdg check-regression --baseline
```

### 13.5 Certeza Three-Tier Testing (Upstream Repos)

When modifying upstream Rust code for gap fixes, follow certeza tiers:

**Tier 1: On-Save (sub-second)**
```bash
cargo check && cargo test --lib -- --quiet    # Type check + unit tests
```

**Tier 2: On-Commit (1-5 minutes)**
```bash
cargo test                                     # Full test suite
cargo llvm-cov --summary-only                  # Coverage ≥ 95%
pmat analyze tdg                               # TDG regression check
pv audit contracts/ --binding                  # Contract compliance
```

**Tier 3: On-Merge / Nightly (hours)**
```bash
cargo mutants --no-times                       # Mutation score ≥ 85%
cargo kani                                     # Formal verification
batuta falsify . --min-grade toyota-standard   # 108-item checklist
pmat rust-project-score --full                 # Comprehensive quality score
```

### 13.6 Albor Pipeline Commands

Since albor is a project repo, its primary interface is `apr pipeline`.
No Makefiles, no shell scripts. One manifest, one DAG.

```bash
# ── Pipeline (the only entry point you need) ──
apr pipeline plan configs/pipeline/albor.yaml     # Full DAG dry-run (no GPU, no writes)
apr pipeline apply configs/pipeline/albor.yaml    # Execute everything (resumable)
apr pipeline status                               # What's converged / pending / failed
apr pipeline drift                                # Detect unauthorized state changes

# ── Targeted execution (run one step + its dependencies) ──
apr pipeline apply configs/pipeline/albor.yaml --target train-350m
apr pipeline apply configs/pipeline/albor.yaml --target eval-code
apr pipeline apply configs/pipeline/albor.yaml --target publish

# ── Force re-run (ignore converged state) ──
apr pipeline apply configs/pipeline/albor.yaml --target distill --force

# ── Individual subcommands (for development / debugging) ──
apr train plan configs/train/pretrain-350m.yaml   # Plan one step standalone
apr train apply configs/train/pretrain-350m.yaml --seed 42
apr monitor ./checkpoints/albor-base-350m/        # Live TUI
apr experiment view --db .entrenar/experiments.db  # Browse experiments

# ── Quality (upstream repos — run independently of pipeline) ──
pmat tdg baseline create                          # TDG baseline across all components
pmat comply check --strict ../aprender
pmat comply check --strict ../entrenar
pv validate contracts/*.yaml                      # Contract schema validation
pv status contracts/                              # Contract completeness
batuta falsify . --min-grade toyota-standard      # 108-item falsification checklist
```

---

## 14. Batuta Falsification Checklist

### 14.1 108-Item Popperian Assessment

The Albor project itself is subject to batuta's 108-item falsification checklist:

```bash
# Full assessment
batuta falsify . --verbose --format markdown --output docs/falsification-report.md

# Critical-only (blocks release)
batuta falsify . --critical-only

# CI-friendly output
batuta falsify . --format github-actions --min-grade kaizen-required
```

### 14.2 Key Sections Applied to Albor

**Section 1: Sovereign Data Governance (SDG)**
- All training data has documented provenance (HuggingFace commit SHAs)
- No PII in training corpus (alimentar quality check)
- Data residency: all data stored on owned hardware (lambda + intel)
- Teacher model license verified (Apache 2.0)

**Section 3: Hypothesis-Driven Development (HDD)**
- Each improvement stage has a falsifiable hypothesis:
  - "Distillation improves avg benchmark by >5%" (FALSIFY-ALBOR-005)
  - "Pruning at 50% sparsity degrades benchmarks by <2%" (FALSIFY-ALBOR-008)
  - "Q4 quantization degrades perplexity by <5%" (FALSIFY-ALBOR-009)
- Reproducibility standard: **Gold** (deterministic seeds, versioned data,
  BLAKE3 checkpoint hashes, Cargo.lock pinning)

**Section 4: Numerical Reproducibility (NR)**
- Float determinism enforced via fixed seeds and operator ordering
- Cross-platform consistency: checkpoint trained on lambda loads on intel
- SIMD parity: all kernels have provable-contracts SIMD equivalence obligations

**Section 5: Performance & Waste Elimination (PW)**
- Seven Wastes (Muda) applied to training pipeline:
  - No redundant data copies (alimentar streaming)
  - No idle GPU time (pre-computed teacher logits)
  - No over-processing (progressive model sizing: 50M → 125M → 350M)

**Section 6: Safety & Formal Verification (SF)**
- Critical kernels have Kani proofs (softmax, attention, cross-entropy)
- New kernels (KD loss, gradient accumulation) get Kani harnesses

**Section 10: Architectural Invariants (AI) — CRITICAL**
- AI-01: All model operations use apr (no manual weight manipulation)
- AI-02: Every checkpoint is BLAKE3-hashed and version-tracked
- AI-03: Training config is immutable once committed (no runtime overrides)
- AI-04: Eval results are reproducible (fixed seed, deterministic batching)
- AI-05: No undeclared dependencies (Cargo.lock enforced)

### 14.3 Target Grade

**Toyota Standard (90-100%)** — the highest tier. This means:
- All 5 Critical items pass (Section 10)
- All Major items pass or have documented remediation
- Overall score ≥ 90/108

---

## 15. Implementation Phases

### Phase 0: Pipeline Manifest, Contracts & Quality Baseline (Week 1)
- [x] Write `configs/pipeline/albor.yaml` — full pipeline manifest (infra + data + train + eval + publish) (ALB-028 FIXED)
- [x] `apr pipeline plan` — validate entire DAG, estimate resources (23 resources, 2 machines)
- [ ] `apr pipeline apply --target cuda-driver --target vulkan-driver --target data-dir` — provision infra
- [ ] Verify `trueno` wgpu on W5700X via Vulkan (not Metal — Linux)
- [x] Verify `trueno` CUDA on 4090 (CUDA executor confirmed working via `apr train apply`)
- [ ] Download Qwen3-Coder-Next to intel box, verify it loads in realizar (ALB-010 blocker)
- [ ] `pmat tdg baseline create` on all stack components
- [x] `pv coverage contracts/ --binding` — 5 contracts, 100% obligation coverage, 0/13 impl bindings
- [x] `batuta falsify . --critical-only` — 70% (3/5 pass, AI-01 partial, AI-04 fail)

### Phase 1: Data Pipeline + Tokenizer Contract (Week 1-2)
- [x] Ingest local ground truth corpora via `alimentar import local` (~~fix ALB-019~~ FIXED)
  - [x] depyler: examples/ + tdd-book/tests/ (1,843 files → 6MB Parquet)
  - [x] hf-ground-truth-corpus (11,493 files → 197MB Parquet)
  - [x] jax-ground-truth-corpus (2,637 files → 50MB Parquet)
  - [x] vllm-ground-truth-corpus (1,100 files → 18MB Parquet)
- [ ] Ingest local ML framework code (Tier 2, ~53K files)
- [ ] Download external datasets via `alimentar import hf` (StarCoder Python, FineWeb-Edu)
- [ ] Quality validation via `alimentar quality check` on all sources
- [x] Build weighted training mix with 3.7x upsampling on Tier 1 (17,070 rows, depyler:0.4 hf:0.3 jax:0.15 vllm:0.15)
- [x] Write `bpe-tokenizer-kernel-v1.yaml` contract (ALB-014 — DOGFOODING, passes `pv validate`)
- [x] `pv probar` + `pv kani` on tokenizer contract (roundtrip, FIM sentinel tests generated)
- [x] Train BPE tokenizer on mixed corpus — `apr tokenize apply --vocab-size 32768 --algorithm bpe` (17,070 docs, 197MB)
- [ ] Verify FALSIFY roundtrip: `decode(encode(text)) = text` for all test data
- [x] Tokenize all data into sharded Parquet (~~fix ALB-007~~ FIXED — Parquet→LMBatch bridge working, `entrenar@a5a2fb7`)
- [x] Apply FIM transforms to code sequences — `alimentar fim` PSM 50% rate (17,070 rows → mixed-fim.parquet)
- [x] Create train/val/test splits via `alimentar` — train: 17,070 / val: 500 / test: 200
- [x] Record SHA-256 hashes + provenance manifest for all data artifacts (docs/PROVENANCE.md)
- [ ] `pmat comply check --strict` on alimentar changes

### Phase 2: Pipeline Validation — 50M Model (Week 2)
- [x] Write `gradient-accumulation-kernel-v1.yaml` contract (ALB-017 — DOGFOODING, passes `pv validate`)
- [x] Write `configs/train/pretrain-50m.yaml` (model arch + training + monitoring)
- [ ] Train albor-50M on 4090 (hours, not days)
- [ ] Validate `apr monitor` attaches to running training (live TUI)
- [ ] Validate Andon alerts fire on NaN/Inf (inject a bad batch to test)
- [x] ~~Fix ALB-009~~ FIXED: `apr train plan/apply` with causal LM pre-training (`aprender@d79ed943`)
- [ ] Verify FALSIFY-ALBOR-001 (loss decreases) and FALSIFY-ALBOR-002 (gradient bounds)
- [x] ~~Fix ALB-006~~ FIXED: `apr eval --task code` with pass@1 scoring (`aprender@4e61297e`)
- [ ] `pv audit` on all existing kernel contracts used in training
- [ ] **Milestone**: Training loop converges, monitoring works, all kernel contracts pass

### Phase 3: Base Model — 350M Pre-Training (Week 2-4)
- [ ] Write `configs/train/pretrain-350m.yaml` (model arch + training + monitoring)
- [ ] Train albor-base-350m on 4090, checkpoint every 1000 steps
- [ ] Monitor training via `apr monitor` from intel box over SSH
- [ ] Run eval on intel concurrently
- [ ] Validate loss curve, perplexity convergence (from `training_state.json`)
- [ ] Tune hyperparameters (LR, batch size, warmup)
- [ ] Verify FALSIFY-ALBOR-003 (checkpoint determinism)
- [ ] `pmat tdg check-regression` on all touched components
- [ ] **Milestone**: Perplexity < 30, TDG grade A maintained

### Phase 4: Teacher Setup & Logit Pre-Computation (Week 3-5)
- [ ] Fix ALB-010: Add Qwen3-Coder-Next support to realizar
- [ ] Validate teacher inference on intel (CPU, fp16, 300GB RAM)
- [ ] Write `knowledge-distillation-kernel-v1.yaml` contract (ALB-013)
- [ ] `pv kani` on KD loss contract (KL non-negativity, temperature scaling)
- [x] Fix ALB-011: Implement `apr distill --config --stage precompute|train` (`aprender@81dd4432`)
- [ ] Pre-compute teacher logits on curated subset (~500M-2B tokens)
- [ ] Verify FALSIFY-ALBOR-006 (teacher logit integrity)
- [ ] Store as sharded Parquet via alimentar
- [ ] `pmat comply check --strict` on realizar changes
- [ ] **Milestone**: Teacher logits verified, KD contract at Level 4

### Phase 5: Knowledge Distillation (Week 5-6)
- [ ] Implement `apr distill apply` with KD loss
- [ ] Distill albor-base-350m → albor-distill-350m
- [ ] Verify FALSIFY-ALBOR-004 (KL non-negativity in production)
- [ ] Verify FALSIFY-ALBOR-005 (distillation improves benchmarks)
- [ ] Benchmark: measure improvement over base
- [ ] `pv probar --binding` on KD contract with actual training data
- [ ] **Milestone**: >5% avg benchmark improvement, KD contract fully wired

### Phase 6: Post-Training Optimization (Week 6-8)
- [ ] Write `model-merging-kernel-v1.yaml` contract (ALB-015)
- [ ] Write `pruning-kernel-v1.yaml` contract (ALB-016)
- [ ] Fine-tune with LoRA: `apr finetune` → albor-instruct
- [ ] Merge variants: `apr merge --method slerp` → albor-merged
- [ ] Verify FALSIFY-ALBOR-007 (SLERP interpolation bound)
- [ ] Prune: `apr prune --method wanda` → albor-pruned
- [ ] Verify FALSIFY-ALBOR-008 (sparsity guarantee)
- [ ] Quantize: `apr quantize --method q4_k` → albor-q4
- [ ] Verify FALSIFY-ALBOR-009 (quantization fidelity)
- [ ] Benchmark every variant
- [ ] `pv coverage contracts/ --binding` — final contract coverage report
- [ ] **Milestone**: Full ladder complete, all post-training contracts pass

### Phase 7: Quality Assurance & Falsification Sweep (Week 8)
- [ ] `batuta falsify . --min-grade toyota-standard --verbose` — full 108-item assessment
- [ ] `pmat rust-project-score --full` on all touched components
- [ ] `pmat tdg check-regression --baseline` — no quality regressions
- [ ] `pv graph contracts/ --format mermaid` — publish verification DAG
- [ ] `pv status contracts/` — all contracts at Level 3+, critical at Level 4
- [ ] `cargo mutants --no-times` on all new code — mutation score ≥ 85%
- [ ] `cargo llvm-cov` — coverage ≥ 95% on all new code
- [ ] Address any falsification failures or contract violations
- [ ] **Milestone**: Toyota Standard grade, all quality gates green

### Phase 8: Evaluation, Leaderboard Submission & Publication (Week 8-9)
- [ ] Final eval on all benchmark tasks (all 6 model variants)
- [ ] Run `bigcode-evaluation-harness` with leaderboard-standard params on best model
- [ ] Submit PR to Big Code Models Leaderboard (`community_results/` folder)
- [ ] Export all models: SafeTensors + GGUF
- [ ] `apr publish` to HuggingFace Hub as `paiml/albor-*`
- [ ] Write model card with full reproducibility details + leaderboard results
- [ ] Publish training logs, loss curves, eval trajectories
- [ ] Publish verification report (contract status, falsification results)
- [ ] `batuta falsify . --format markdown --output docs/falsification-report.md`
- [ ] **Milestone**: Models on HuggingFace, leaderboard submission live, quality evidence published

### Phase 9: Distributed Training — Stretch (Week 9+)
- [ ] Implement ring all-reduce in repartir (ALB-002)
- [ ] Wire into apr training loop (ALB-003)
- [ ] wgpu backward pass in trueno (ALB-005)
- [ ] Full distributed training: 4090 + W5700X x2
- [ ] **Milestone**: Multi-GPU training demonstrated

---

## 16. Reproducibility Protocol

| Artifact | Recorded |
|----------|----------|
| Random seed | 42 (global), per-component seeds derived |
| Data versions | HuggingFace dataset commit SHAs + local repo git SHAs |
| Data provenance | `alimentar provenance` manifest (source path, git SHA, file count, token count per source) |
| Data checksums | SHA-256 of every Parquet shard |
| Tokenizer | Vocabulary + merge rules saved in APR format |
| Training config | YAML checked into git |
| Checkpoint hashes | BLAKE3 of every checkpoint (forjar tripwire) |
| Software versions | Cargo.lock pinning all crate versions |
| Hardware | nvidia-smi + vulkaninfo + free -h captured |
| Training logs | Stdout/stderr + structured JSON logs |
| Eval results | JSON per checkpoint per stage, diffable |
| Teacher logits | BLAKE3 hashes of all pre-computed Parquet shards |

---

## 17. Success Criteria

### Minimum Viable (Phase 3 complete)
- [ ] 350M base model trained on 4090 to convergence (~10B tokens, 80% Python)
- [ ] FIM (fill-in-the-middle) training implemented and validated (ALB-018)
- [ ] **HumanEval pass@1 > 8%** (baseline Python capability, beat random)
- [ ] **HumanEval-FIM working** (model can infill Python code)
- [ ] Entire pipeline uses only sovereign stack components
- [ ] All training artifacts reproducible from spec
- [ ] All existing kernel contracts pass `pv audit` (Level 2+)
- [ ] `pmat comply check` passes on all modified components

### Good (Phase 5 complete)
- [ ] Distillation from Qwen3-Coder-Next demonstrated
- [ ] albor-distill-350m outperforms albor-base-350m on all code benchmarks
- [ ] **HumanEval pass@1 > 15%** (beat CodeGen-350M-mono's 12.8% via distillation)
- [ ] **MBPP pass@1 > 12%**
- [ ] **FIM infill working** (qualitatively: model can complete Python between prefix and suffix)
- [ ] KD contract at Level 4 (Kani-proved KL non-negativity)
- [ ] All FALSIFY-ALBOR tests pass (001-006)

### Full Success (Phase 8 complete)
- [ ] All 6 model variants benchmarked (base → distill → instruct → merged → pruned → q4)
- [ ] Benchmark trajectory published showing improvement at each stage
- [ ] **Submitted to Big Code Models Leaderboard** — first sub-1B model on the board
- [ ] **Q4 model: <50ms/token on CPU, <10ms/token on GPU** (code completion latency)
- [ ] Critical path gaps (ALB-001, 006, 009, 010, 011, 018) closed with upstream fixes
- [ ] Models published on HuggingFace as `paiml/albor-python-*`
- [ ] Q4 quantized model < 100MB, runs on consumer hardware
- [ ] **All 5 new contracts written and verified** (ALB-013 through ALB-017)
- [ ] **batuta falsify: Toyota Standard grade (≥90/108)**
- [ ] **pmat TDG: Grade A on all touched components**
- [ ] **Test coverage ≥ 95%, mutation score ≥ 85% on all new code**
- [ ] **All 9 FALSIFY-ALBOR tests pass**
- [ ] **Verification DAG published via `pv graph`**

### Stretch Goals
- [ ] **HumanEval pass@1 > 20%** (strong distillation result at 350M)
- [ ] **DS-1000 pass@1 > 10%** (data science code generation)
- [ ] Editor integration: VS Code / Neovim / Helix extension using realizar as backend
- [ ] Distributed gradient-parallel training across 4090 + W5700X demonstrated
- [ ] `apr pipeline apply` reproduces entire ladder from bare metal to published model
- [ ] BabyLM 2026 submission using constrained data variant
- [ ] All critical kernels at Level 4 (Kani formal proofs)
- [ ] Lean 4 theorem stubs generated for core training loop invariants

---

## 18. Reference Commands

```bash
# ═══════════════════════════════════════════════════════════
# THE PIPELINE (two orchestrators working together)
# ═══════════════════════════════════════════════════════════

# Infrastructure provisioning (forjar — bare metal to ready state)
forjar validate -f configs/pipeline/infra-only.yaml   # Validate
forjar apply -f configs/pipeline/infra-only.yaml       # Provision

# ML pipeline orchestration (batuta playbook — data to published model)
batuta playbook validate configs/pipeline/albor-playbook.yaml  # Validate DAG
batuta playbook run configs/pipeline/albor-playbook.yaml       # Execute (resumable)
batuta playbook status configs/pipeline/albor-playbook.yaml    # Check progress

# Future: unified pipeline (apr pipeline wraps forjar + batuta)
apr pipeline plan configs/pipeline/albor.yaml      # (blocked: ALB-028)
apr pipeline apply configs/pipeline/albor.yaml
apr pipeline status

# ═══════════════════════════════════════════════════════════
# MONITORING (run in a separate terminal during training)
# ═══════════════════════════════════════════════════════════

apr monitor ./checkpoints/albor-base-350m/        # Live training TUI
apr experiment view --db .entrenar/experiments.db  # Browse past experiments
apr cbtop ./checkpoints/albor-base-350m/           # GPU profiler

# ═══════════════════════════════════════════════════════════
# QUALITY (bashrs is KING of linting)
# ═══════════════════════════════════════════════════════════

# bashrs — sovereign linter for all shell artifacts
bashrs make lint Makefile                          # Makefile quality
bashrs classify Makefile                           # Safety classification
bashrs make purify Makefile                        # Deterministic output
bashrs lint scripts/*.sh                           # Shell script safety

# provable-contracts — kernel correctness
pv validate contracts/*.yaml                       # Contract schemas
pv coverage contracts                              # Obligation coverage
pv generate contracts/*.yaml                       # Scaffold + tests + harnesses
pv book contracts/                                 # mdBook pages
pv status contracts/                               # Contract completeness
pv graph contracts/ --format mermaid               # Verification DAG

# batuta — falsification
batuta falsify . --format markdown                 # 108-item checklist
batuta oracle --list                               # Stack components
batuta oracle --local                              # Local workspace status

# pmat — code quality (upstream repos)
pmat tdg baseline create                           # TDG baseline
pmat comply check --strict ../aprender
pmat comply check --strict ../entrenar

# ═══════════════════════════════════════════════════════════
# INDIVIDUAL SUBCOMMANDS (for development / debugging only)
# ═══════════════════════════════════════════════════════════

# These are what the pipeline calls under the hood.
# Use them directly when developing/debugging a single step.
apr train plan configs/train/pretrain-350m.yaml
apr train apply configs/train/pretrain-350m.yaml --seed 42
apr distill plan configs/train/distill.yaml
apr distill apply configs/train/distill.yaml --stage precompute
apr eval apply --model ./checkpoints/albor-merged-350m/ \
  --tasks humaneval,mbpp --output ./eval/results.json --seed 42
```

---

## Appendix A: Batuta Oracle Consultation

**Query**: "distributed LLM training across heterogeneous GPUs using sovereign AI stack"

**Response** (2026-03-01):
- Primary: `repartir` (95% confidence) — distributed computing primitives
- Supporting: `entrenar` (70%) — distributed_training pattern
- Supporting: `trueno` (80%) — SIMD/GPU backend for compute acceleration

## Appendix B: Stack Version Matrix

| Component | Version | Role in Albor |
|-----------|---------|---------------|
| aprender | 0.27.2 | ML library, BPE tokenizer, transformer layers |
| entrenar | 0.7.5 | Training engine, autograd, optimizers, LoRA |
| trueno | 0.16.1 | SIMD/GPU tensor backend |
| realizar | 0.8.0 | Inference engine (teacher model, eval, serving) |
| alimentar | 0.2.6 | Data pipeline, Parquet I/O, HF Hub import |
| repartir | 2.0.3 | Distributed compute (future: gradient sync) |
| forjar | 1.0.0 | Pipeline orchestration (DAG engine, infra + task resources) |
| presentar | 0.3.2 | Training visualization (TUI dashboards, WASM, experiment browser) |
| bashrs (Rash) | 6.65.0 | KING of linting: Makefile lint/purify/classify, shell safety, pipeline command validation |
| batuta | 0.7.2 | Stack orchestration, oracle, falsification (108 checks), playbook DAG engine, HF Hub |
| provable-contracts | latest | Design-by-contract YAML specs, Kani proofs, falsification tests |
| pmat | latest | TDG scoring, comply check, fault patterns, coverage gaps |
| certeza | latest | Three-tier test effectiveness (unit → property → formal) |

## Appendix C: Qwen3-Coder-Next Architecture Details

| Layer Pattern | Count | Description |
|---------------|-------|-------------|
| Gated DeltaNet → MoE | 36 (3 per block × 12 blocks) | Linear attention with gating, routed to 10/512 experts |
| Gated Attention → MoE | 12 (1 per block × 12 blocks) | Standard GQA with gating, routed to 10/512 experts |
| **Total layers** | **48** | |

This hybrid architecture means realizar needs to support:
- DeltaNet (linear attention variant) — likely a new gap
- MoE routing (top-k expert selection) — may partially exist
- Gated variants of both attention types

## Appendix D: W5700X Vulkan Validation

The W5700X has been validated with trueno's wgpu backend on **Metal** (macOS)
with documented performance numbers (trueno book, 2026-01-03). The intel box
runs **Linux**, so the backend will be **Vulkan** (not Metal). Vulkan support
for RDNA 1 on Linux via Mesa RADV is mature and well-tested.

**Action item**: Run trueno GPU tests on intel via Vulkan to confirm parity
with Metal benchmarks before relying on W5700X for compute tasks.

## Appendix E: Leaderboard Strategy

### E.1 Target: Big Code Models Leaderboard

**URL**: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard

The Big Code Models Leaderboard is the standard HuggingFace scoreboard for code
generation models. It evaluates HumanEval (Python pass@1) and MultiPL-E
(18 languages) with throughput measurements. ~60 models currently listed.

**Why this leaderboard**:
- Code generation focus — matches Albor's use case exactly
- HumanEval is our primary benchmark
- Accepts community submissions via PR
- **No sub-1B model has ever appeared** — Albor would be the first

**Current smallest entries (1B tier)**:

| Model | Params | HumanEval pass@1 |
|-------|--------|-------------------|
| phi-1 | 1.3B | 50.6% |
| DeciCoder-1B | 1.0B | 19.3% |
| SantaCoder | 1.1B | 18.1% |
| StarCoderBase-1B | 1.0B | 15.2% |

**Albor's position**: At >15% HumanEval with 350M params, Albor would be
competitive with the 1B tier at 1/3 the size. Even at >8% (base model), it
would establish the sub-1B category on the board.

**Submission process**:
1. Run `bigcode-evaluation-harness` (Python tool — the one exception to our
   zero-Python rule, because it is the leaderboard's own eval framework)
2. Standard params: top-p=0.95, temperature=0.2, n_samples=50,
   max_length_generation=512
3. Submit PR to `community_results/PAIML_ALBOR350M_noahgift/`
4. Include: scores JSON, generations folder, metrics folder
5. Results appear as "non-verified" (community submission)

### E.2 Why NOT Other Leaderboards

**Open LLM Leaderboard v2**: Benchmarks (IFEval, BBH, MATH L5, GPQA, MuSR,
MMLU-PRO) were designed for models >7B. A 350M model scores near random on
MATH Level 5 (~0%), GPQA (~25%), and MMLU-PRO (~10%). Waste of eval compute.

**EvalPlus Leaderboard**: Uses HumanEval+ and MBPP+ (80x more tests than
vanilla HumanEval). Secondary submission target if Big Code results are strong.
Currently no sub-1B models either. URL: https://evalplus.github.io/leaderboard.html

**BigCodeBench Leaderboard**: 1,140 software-engineering tasks. Designed for
7B+ models. A 350M model would score near zero. Not appropriate.

### E.3 General Capability Eval (Not a Leaderboard — Internal Only)

ARC-Easy, HellaSwag, PIQA, LAMBADA are the standard for sub-1B general model
comparison (Pythia, OPT, GPT-2 all publish on these). We evaluate on them for
internal comparison, but they have no dedicated leaderboard worth targeting.
Code benchmarks are the real scoreboard.

### E.4 FIM Evaluation

There is no canonical FIM benchmark. SantaCoder used a custom FIM evaluation;
other models use MultiPL-E or proprietary internal evals. Albor will define its
own FIM evaluation protocol (exact match on held-out Python functions) and
report absolute numbers rather than targeting a specific percentage.

### E.5 Falsification Risks for the Leaderboard Targets

1. **MoE→Dense distillation gap**: No published work demonstrates distilling
   an 80B MoE model into a 350M dense model. The architecture mismatch
   (DeltaNet+MoE routing → vanilla LLaMA) may limit knowledge transfer.
   If distillation gains are <2 points on HumanEval, the "Good" success
   criterion is at risk.

2. **Teacher inference bottleneck**: At ~2-5 tok/s (fp16 on Xeon), producing
   2B tokens of teacher logits takes ~12 days. If 500M tokens of logits
   proves insufficient, the timeline extends by weeks.

3. **Rust training stack maturity**: entrenar has never trained a model from
   scratch at 350M scale. Bugs in gradient accumulation, mixed precision,
   or checkpointing could cause silent correctness issues that only surface
   as poor benchmark scores.

4. **Data quality ceiling**: The local ground truth corpora (~71K files) are
   high quality but narrow. If the BPE tokenizer or data mix doesn't
   generalize well to HumanEval-style problems, the base model ceiling
   is lower than projected.

5. **bigcode-evaluation-harness compatibility**: The leaderboard eval tool is
   Python-based and expects HuggingFace-format models. Our SafeTensors export
   must be compatible with the harness's model loading. If not, we need a
   thin adapter — this is a potential gap not yet tracked.

### E.6 The Real Story

"A Python code completion model that was trained entirely in Rust with zero
Python dependencies — from data pipeline to on-device inference." The irony is
deliberate: a Rust ML stack producing a Python code assistant. The model is
the proof; the stack is the lasting value. Publishable
regardless of exact benchmark numbers.
