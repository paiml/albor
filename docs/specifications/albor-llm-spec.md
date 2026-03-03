# Albor LLM Specification

**Version**: 0.6.0
**Date**: 2026-03-03
**Status**: Phase 3 — 350M Base Model Retraining (ALB-060 fix, v2 data)
**Author**: Noah Gift / Pragmatic AI Labs

> *Albor* (Spanish: "dawn") — A sovereign Python code completion model trained
> from first principles using only the Sovereign AI stack. Python-only following
> the phi-1 playbook: maximum concentration on one language, distilled from
> Qwen3-Coder-Next (80B), then optimized through fine-tuning, merging, pruning,
> and quantization into a fast, local, zero-dependency code completion engine.
> The goal is twofold: produce a **usable Python code assist model** that runs
> anywhere Rust compiles, **and** identify + fix every gap in the stack that
> blocks end-to-end LLM development.

**Latest milestone**: 350M CUDA test training verified — 50 steps, loss
10.39→5.92 (best 5.53), checkpoint loads in realizar, all training stability
contracts pass. First full training run failed (ALB-060: epochs=1 only ran
43/5000 steps). Fixed with C-TRAINCFG-001 contract + v2 config (67,977
sequences, 139M tokens, epochs=38). Qwen2.5-Coder-3B interim teacher validated
for distillation. 24+ upstream gaps fixed across 8 sovereign stack components.

---

---


# 1. Objectives

### 1.1 Primary Goal
Train, distill, and optimize a **350M-parameter decoder-only transformer** using
exclusively the Sovereign AI stack:
- `apr` for training, distillation, merging, pruning, quantization, eval, export
- `alimentar` for data loading and preprocessing
- `forjar` for pipeline orchestration (DAG engine, multi-machine, state tracking)
- `bashrs` (Rash) for shell fragment validation in pipeline task resources
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


# 2. Hardware Inventory

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


# 3. Model Architecture

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
| Context length | 2048 (spec) / 1024 (training) | 2048 OOMs at batch≥4 on 4090; training uses 1024 |
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

**Speculative estimates** (pre-dogfooding):

| Component | Size |
|-----------|------|
| Model weights (fp16) | ~700 MB |
| Adam optimizer states (fp32 m, v) | ~2.8 GB |
| Gradients (fp16) | ~700 MB |
| Activations (grad checkpoint, batch=8, seq=2048) | ~8-12 GB |
| **Total estimated** | **~13-16 GB** |

**Actual measurements** (from ALB-040 dogfooding with CudaTransformerTrainer):

| Config | VRAM Used | Status |
|--------|-----------|--------|
| seq=512, batch=4 | ~18 GB | PASS |
| seq=1024, batch=4 | ~19.5 GB | PASS (production config) |
| seq=2048, batch=4 | OOM | FAIL — logits [4,2048,32768] = 1 GB exceeds budget |
| seq=2048, batch=8 | OOM | FAIL — OOM at block 21 upload |

The GPU-resident `CudaTransformerTrainer` keeps all 24 blocks in VRAM (weights +
AdamW states ≈ 5 GB) plus a shared workspace for activations (~10-12 GB). This
is tighter than the speculative estimate because the shared workspace includes
attention score matrices that scale as O(heads × seq² × batch). Batch size is
fixed at 4; gradient accumulation (128 steps) achieves the effective batch size.
See §6.4 for detailed breakdown.

---


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

### 4.6 Interim Teacher: Qwen2.5-Coder-3B (Unblocking Distillation)

ALB-010 (Qwen3-Coder-Next 80B MoE support in realizar) is a 3-4 week blocker
requiring DeltaNet attention + MoE routing implementation. To unblock
distillation NOW, we use **Qwen2.5-Coder-3B** as an interim teacher:

| Property | Value |
|----------|-------|
| Model | [Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) |
| Parameters | 3B (dense) |
| Architecture | Qwen2 (standard transformer — already supported by realizar) |
| Compression ratio | 8.6× (3B → 350M) — within recommended 5-20× range |
| CPU inference | ~12 GB RAM, ~2 tok/s on 48 cores |
| License | Apache 2.0 |

**Why this works**:
- Already supported by realizar's Qwen2 architecture loader (no MoE/DeltaNet)
- `apr distill --stage precompute` verified working with 3B teacher (2026-03-03)
- Sharded SafeTensors handled by RosettaStone in apr (not by realizar directly)
- CPU precompute feasible on lambda box (~12 GB RAM)

**Config**: `configs/train/distill-qwen3b.yaml` — teacher: Qwen2.5-Coder-3B,
student: albor-base-350m, temperature=4.0, alpha=0.5, LoRA rank 16.

**Limitation**: 3B teacher has weaker code capabilities than 80B. Distillation
quality ceiling is lower. ALB-010 (80B teacher) remains the stretch goal for
maximum HumanEval improvement.

---


# 5. Training Data

### 5.1 Data Philosophy
- All datasets either locally owned (MIT/Apache 2.0) or publicly available with permissive licenses
- **Local-first**: Sovereign ground truth corpora are our highest-quality data — curated,
  tested, type-annotated, and owned. They are upsampled to punch above their token weight.
- Exact download URLs, versions, and SHA-256 hashes recorded for all external data
- Preprocessing pipeline is deterministic (fixed seed, recorded transforms)
- Quality validated by `alimentar quality check`

### 5.2 Data Mix (Target: ~10B tokens)

**Current status (2026-03-03)**: 139M tokens in v2 dataset (67,977 pretokenized
sequences × 2048). Composed of Tier 1 upsampled 10x + 8 Tier 2 repos at 1x +
50% FIM PSM. External data (StarCoder, FineWeb-Edu) not yet downloaded. See
§5.4.1 for the actual pipeline used.

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

~~**Gap ALB-018**~~: **FIXED** — `alimentar fim` supports PSM/SPM transforms.
Verified: `alimentar fim mixed.parquet -o out.parquet --rate 0.5 --format psm --seed 42`
produces correct FIM-encoded sequences. Used in v2 data pipeline.

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

~~**Gap ALB-019**~~: **FIXED** — `alimentar import local` expects data files
(CSV/JSON/Parquet), not source code directories. Workaround:
`scripts/source-to-parquet.py` converts Python source repos to Parquet with the
Tier 1 schema (file, source, text columns). Used for all Tier 2 imports.

~~**Gap ALB-020**~~: **FIXED** — `alimentar mix` supports weighted proportional
sampling. Syntax: `alimentar mix file1.parquet:10.0 file2.parquet:1.0 -o out.parquet`.

#### 5.4.1 Actual Pipeline (v2 Dataset — 2026-03-03)

The pipeline below produced the v2 dataset (139M tokens, 67,977 sequences):

```bash
# ── Step 1: Convert Tier 2 repos to Parquet (alimentar can't read source dirs) ──
for repo in pytorch hf-repos mlflow vllm-full tgi algo-corpus cuda-python llms-with-hf; do
    python3 scripts/source-to-parquet.py ~/src/$repo $repo data/parquet/tier2/$repo.parquet
done
# Result: 28,553 Python files across 8 repos

# ── Step 2: Mix Tier 1 (10x) + Tier 2 (1x) ──
alimentar mix \
  data/parquet/depyler/shard_0000.parquet:10.0 \
  data/parquet/hf-ground-truth/shard_0000.parquet:10.0 \
  data/parquet/jax/shard_0000.parquet:10.0 \
  data/parquet/vllm/shard_0000.parquet:10.0 \
  data/parquet/tier2/pytorch.parquet:1.0 \
  data/parquet/tier2/hf-repos.parquet:1.0 \
  data/parquet/tier2/mlflow.parquet:1.0 \
  data/parquet/tier2/vllm-full.parquet:1.0 \
  data/parquet/tier2/tgi.parquet:1.0 \
  data/parquet/tier2/algo-corpus.parquet:1.0 \
  data/parquet/tier2/cuda-python.parquet:1.0 \
  data/parquet/tier2/llms-with-hf.parquet:1.0 \
  -o data/staging/mixed-expanded.parquet --seed 42
# Result: 45,420 mixed rows

# ── Step 3: Apply FIM (50% PSM) ──
alimentar fim data/staging/mixed-expanded.parquet \
  -o data/staging/mixed-expanded-fim.parquet --rate 0.5 --format psm --seed 42
# Result: 45,420 rows with ~50% FIM-encoded

# ── Step 4: Pretokenize into 2048-length sequences ──
python3 scripts/pretokenize.py \
  --input data/staging/mixed-expanded-fim.parquet \
  --tokenizer models/albor-tokenizer-v2/tokenizer.json \
  --seq-len 2048 \
  --output data/pretokenized-2048-v2/train/train.parquet
# Result: 67,977 sequences × 2048 = 139,218,944 tokens (191 MiB)

# Validation set: reuse v1
cp data/pretokenized-2048/val/val.parquet data/pretokenized-2048-v2/val/val.parquet
```

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


# 6. Training Configuration

### 6.1 Optimizer & Schedule

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard; in aprender/entrenar |
| Learning rate | 3e-4 | Chinchilla-recommended for 350M |
| Weight decay | 0.1 | Standard AdamW regularization |
| Beta1, Beta2 | 0.9, 0.95 | LLaMA/GPT-3 standard |
| Epsilon | 1e-8 | Standard |
| LR schedule | Cosine annealing with warmup | `CosineAnnealingLR` in aprender |
| Warmup steps | 2000 (v1) / 500 (v2) | **ALB-060**: 2000/5000 = 40%, not 0.2%. v2 config uses 500 (10%) per C-TRAINCFG-001 |
| Min LR | 3e-5 | 10% of peak (standard) |
| Gradient clipping | 1.0 (global norm) | Stability |
| Batch size (global) | 512K tokens | ~512 sequences x 1024 tokens |
| Micro-batch (4090) | 4 | GPU-resident (batch=8 OOM at seq≥1024) |
| Gradient accumulation | 128 steps | Reach global batch size |
| Total training tokens | Target 10B; current 139M (v2 dataset) | ~5000 steps at 512K tokens/step (v2: 67K seqs) |
| Mixed precision | fp16 (CUDA) | Hardware-appropriate |

### 6.2 Training Config: `configs/train/pretrain-350m.yaml`

A single YAML file defines **everything** — model architecture and training
hyperparameters. This is the industry standard (Axolotl, torchtune, HuggingFace
Trainer). One file, one truth. `apr train validate` lints it before GPU time.

```yaml
# configs/train/pretrain-350m.yaml — Albor 350M pre-training config

model:
  path: "."                                  # From scratch (random init)
  mode: transformer                         # LLM transformer mode
  architecture:
    hidden_size: 1024                       # d_model
    num_hidden_layers: 24
    num_attention_heads: 16                 # d_head = 64
    num_key_value_heads: 4                  # GQA 4:1 ratio
    intermediate_size: 4096                 # SwiGLU FFN (gate + up + down)
    vocab_size: 32768                       # ByteLevel BPE (v2 tokenizer)
    max_position_embeddings: 1024           # Context length (2048 OOM'd on 4090)
    rms_norm_eps: 1.0e-5

data:
  train: "data/pretokenized-2048/train/"    # Pre-tokenized ByteLevel BPE v2
  val: "data/pretokenized-2048/val/"
  batch_size: 4                             # Micro-batch (batch=8 OOM'd)
  seq_len: 1024
  tokenizer: "models/albor-tokenizer-v2/tokenizer.json"
  input_column: "input_ids"                 # Pre-tokenized: List<u32> column

optimizer:
  name: "adamw"
  lr: 3.0e-4
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1

training:
  mode: "causal_lm"
  epochs: 117                               # ALB-060: epochs=1 was FATAL (only 43/5000 steps)
                                             # ceil(5000 / floor(22079/4/128)) = ceil(5000/43) = 117
  grad_clip: 1.0
  lr_scheduler: "cosine"
  warmup_steps: 2000
  gradient_accumulation: 128               # Global batch = 4 * 128 * 1024 = 512K tokens
  mixed_precision: "fp16"
  output_dir: "./checkpoints/albor-base-350m"
  save_interval: 25
  max_steps: 5000
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

### 6.4 GPU-Resident Training (CudaTransformerTrainer)

The `CudaTransformerTrainer` (ALB-040) keeps all 24 transformer blocks
GPU-resident, reducing PCIe transfers from ~16K/step to exactly 3:

```
Transfer 1 (H2D): embedding hidden states   ~S×H×4 bytes
Transfer 2 (D2H): logits for cross-entropy  ~S×V×4 bytes
Transfer 3 (H2D): grad_logits to GPU        ~S×V×4 bytes
```

Each `CudaTransformerBlock` holds its own weights, AdamW optimizer states
(m + v), and shares a `CudaGradWorkspace` for forward/backward activation
buffers. The per-block interleaved backward+optimizer pattern overwrites
the shared workspace each layer — memory cost is O(1 block), not O(24 blocks)
for activations.

**VRAM budget (actual, RTX 4090 24GB):**

| Component | Memory |
|-----------|--------|
| 24 blocks (weights + AdamW m + v) | ~5 GB |
| Shared workspace (activation/gradient buffers) | ~10-12 GB (depends on seq_len) |
| LM head (weights + AdamW + logits buffer) | ~1-2.5 GB |
| System (Xorg/desktop) | ~1 GB |

At `seq_len=512, batch=4`: fits comfortably (~18 GB used).
At `seq_len=1024, batch=4`: fits (~19.5 GB used).
At `seq_len=2048, batch=4`: OOM at LM head alloc (logits [4,2048,32768] too large).
At `seq_len=2048, batch=8`: OOM at block 21 upload.

**Dogfooding results:**

| Config | Steps | Loss | Time | Status |
|--------|-------|------|------|--------|
| 50M quick (seq=512, batch=4) | 5 | 10.42→9.45 | ~10s | PASS (post ALB-059 fix) |
| 350M test (seq=512, batch=4) | 50 | 10.39→5.92 (best 5.53) | ~400s | PASS (post ALB-059 fix) |
| 350M full (seq=1024, batch=4, accum=128) | 43/5000 | 10.39 flat | ~12s | **FAIL (ALB-060)**: epochs=1 exhausted data |
| 350M full v2 (seq=1024, batch=4, accum=128) | 5000 | TBD | ~20h | PENDING (v2 data, epochs=38) |

**ALB-060: Training Configuration Epoch/Step Mismatch (Critical)**

The first 350M full training run (2026-03-02) ran only 43 of 5000 steps because
`epochs: 1` caps total steps to `floor(num_sequences / batch_size / grad_accum)`.
With 22,079 sequences, batch=4, accum=128: `steps_per_epoch = 43`. Warmup (2000
steps) never completed — LR peaked at 6.45e-6 vs target 3e-4. Loss stayed flat
at ~10.39 for all 43 steps (never exited warmup). Root cause: no pre-flight
algebraic validation of epoch/step consistency.

Fix: C-TRAINCFG-001 contract (`contracts/training-config-kernel-v1.yaml`) +
`epochs: 117` for v1 data, or v2 config (`pretrain-350m-v2.yaml`) with expanded
dataset (67,977 sequences, `epochs: 38`, `warmup_steps: 500`).

**Training stability contracts verified (ALB-044, ALB-059, ALB-060):**
- C-EMBED-GRAD-001: Activation gradient clipped at GPU→CPU boundary
- C-HYPERPARAMS-001: All optimizer params flow from YAML config
- C-BUFSIZE-001: Buffer sizes algebraically verified (ALB-043 fix)
- C-GRADFLOW-001: All trainable parameters receive gradients (ALB-038 fix)
- C-GEMMARGS-001: GEMM backward constructor args match documented order (ALB-059 fix)
- C-GPUINIT-001: All optimizer m/v buffers zero-initialized (ALB-059 fix)

### 6.5 Checkpointing Strategy

| Aspect | Design |
|--------|--------|
| Format | SafeTensors (primary) + JSON metadata |
| Frequency | Every 1000 steps (~512M tokens) |
| Content | Model weights, optimizer state, LR scheduler state, RNG state, step count |
| Storage | Local on lambda, rsync to intel (300GB RAM box) for backup |
| Resume | `--resume checkpoint-step-5000.json` |
| Export | `apr publish --format safetensors` for HuggingFace |

### 6.6 Experiment Tracking & Training Monitoring

entrenar has a full monitoring stack built in, and presentar provides rich
terminal visualization. Albor uses both — no external tools (no W&B, no
MLflow, no TensorBoard). Sovereign monitoring, sovereign visualization.

#### 6.6.1 Monitoring Config: `configs/train/pretrain-350m.yaml` (monitoring section)

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
      data: "python-code-v2"                 # 139M tokens (v2 dataset)

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

#### 6.6.2 What Entrenar Monitors Automatically

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
| `TuiMonitor` | Detached terminal dashboard composing presentar widgets (ALB-057) | Yes (entrenar + presentar) |
| `DriftDetector` | PSI, KS, Wasserstein distribution shift detection | Yes (entrenar) |
| `JsonFileStore` | Real-time metrics to `training_state.json` (atomic writes) | Yes (entrenar) |
| `LossCurve` widget | Training loss over epochs with EMA smoothing | Yes (presentar) |
| `ConfusionMatrix` widget | Multi-class classification evaluation | Yes (presentar) |
| `Braille/Sparkline` | High-resolution terminal charts (2x4 dots/cell, 8-level sparklines) | Yes (presentar) |
| `Heatmap` widget | 2D matrix with CIELAB perceptual color gradients | Yes (presentar) |

#### 6.6.3 Live Monitoring During Training

```bash
# Terminal 1 (lambda): Run training
apr train apply --task pretrain --config configs/train/pretrain-350m.yaml

# Terminal 2 (lambda or ssh): Attach live monitor (presentar TUI)
apr monitor ./checkpoints/albor-base-350m/

# Terminal 2 (alternative): JSON output for LLM agents / CI
apr monitor --json ./checkpoints/albor-base-350m/

# Discover all active training runs (reads global SQLite registry)
apr monitor

# List past experiments from SQLite registry
apr runs ls --global

# Show detailed metrics for a specific run
apr runs show <run-id> --global --json

# Browse past experiments from SQLite
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

#### 6.6.4 Experiment Lifecycle

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

**Durable (dual SQLite experiment stores)** — for post-hoc analysis and comparison:
```
checkpoints/albor-base-350m/.entrenar/
└── experiments.db              # Local per-experiment store (WAL mode)
    ├── experiments             # Experiment metadata (name, description, config)
    ├── runs                    # Training runs (status, timestamps)
    ├── params                  # Hyperparameters (key/value/type)
    ├── metrics                 # Per-step metrics (loss, lr, tok/s, timestamp)
    ├── artifacts               # Model artifacts (path, size, SHA-256)
    └── span_ids                # Distributed trace integration

~/.entrenar/
└── experiments.db              # Global cross-machine registry (WAL mode)
    └── (same schema)           # All runs across all experiments
```

**`PretrainTracker`** (ALB-055/056) writes to both stores on every log interval.
All operations are best-effort — storage failures never block training.

**Three consumers, zero contention**:
- `apr monitor` reads `training_state.json` (atomic write-then-rename) for
  live dashboards. Multiple monitors attach simultaneously.
- `apr runs ls` reads `~/.entrenar/experiments.db` (global registry) for
  cross-experiment history. Supports `--json` for LLM agent consumption.
- `apr experiment` reads local `.entrenar/experiments.db` (WAL mode) for
  per-run metric queries and artifact tracking. Read-only during
  training — no lock contention with the writer.

#### 6.6.5 Presentar Visualization: Rich Terminal Dashboards

presentar (`presentar-terminal`) provides **ML-specific visualization widgets**
that entrenar's `TrainingDashboard` now composes directly (ALB-057). The
dashboard builds a widget tree from `Layout::rows()` of `Border`-wrapped
section panels, each containing `Meter`, `GpuPanel`, `Sparkline`, or `Text`
widgets. The connection point for historical data is entrenar's SQLite
experiment store (`.entrenar/experiments.db`).

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

#### 6.6.6 No External Dependencies

| What Others Use | What Albor Uses Instead | Why |
|-----------------|------------------------|-----|
| Weights & Biases | entrenar `SqliteBackend` + presentar dashboards | Sovereign — no cloud, no API keys, all data local |
| TensorBoard | presentar `LossCurve` + `BrailleGraph` over SSH | No Python, no browser required, works over SSH |
| MLflow | entrenar `ExperimentTracker` + SQLite + `apr experiment` | Self-hosted SQLite, no server process, query via CLI |
| nvidia-smi polling | entrenar system metrics + `apr cbtop` | Integrated into training loop, not bolted on |
| Streamlit dashboards | presentar WASM dashboard (10x faster rendering) | GPU-accelerated, 60fps, zero Python |

---


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

---


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

**Gap ALB-037** (**FIXED**): `apr eval` previously ignored loaded weights during
inference. Now fixed — `realizar run` loads trained SafeTensors checkpoints and
generates from learned weights. Verified end-to-end with 350M test checkpoint
(218 tensors loaded, tokens generated). `scripts/eval-perplexity.py` provides
independent pure-Python perplexity evaluation.

**Gap ALB-038** (**FIXED**): entrenar previously saved initialization weights
instead of trained weights due to broken autograd gradient flow. Root cause:
`RMSNorm::forward_batched()` created tensors with no backward op, and
`MultiHeadAttention::forward()` broke Q/K/V gradient chain. Fixed in
`entrenar@91ba9da` (RMSNorm backward) and `entrenar@1ede409` (attention
backward). All 20 model parameters now receive gradients during training.
See [GitHub #36](https://github.com/paiml/albor/issues/36).

**Gap ALB-059** (**FIXED**): GEMM backward constructor args n/k swapped in
entrenar — baked wrong compile-time stride constants into PTX. Output rows
overflowed into optimizer state buffers, causing NaN in AdamW. The 50-step
test model trained with this bug had loss 10.39→6.07; after the fix, loss
improved to 10.39→5.92. All evaluation results should use the post-fix
checkpoint (`entrenar@846ae0c`). Additionally, all optimizer m/v buffers
are now zero-initialized (cuMemAlloc returns uninitialized VRAM).

**Gap ALB-060** (OPEN): The "full" 350M training run completed only 43 of 5000
optimizer steps because `epochs: 1` exhausted the 22K-sequence dataset before
`max_steps` was reached. Steps per epoch = floor(22079 / 4 / 128) = 43. With
warmup_steps=2000, the LR never progressed past 6.45e-6 (vs target 3e-4), so
loss remained flat at ~10.39. The `checkpoints/albor-base-350m/` checkpoint
contains effectively untrained weights. Fix: `epochs: 117` (proven by
C-TRAINCFG-001 contract, FALSIFY-CFG-001/002). All evaluation of the 350M base
model must wait for a corrected training run.

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

# Evaluate via apr serve API (ALB-037 FIXED — realizar loads trained checkpoints)
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

---


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

---


# 10. Pipeline Orchestration (`apr pipeline` + forjar DAG)

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


# 11. Gap Register

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
| ALB-007 | [#8](https://github.com/paiml/albor/issues/8) | entrenar | Parquet→LMBatch bridge via alimentar | Medium | **FIXED** | `load_lm_batches_from_parquet()` reads text or pre-tokenized Parquet (single file or directory of shards) via alimentar. Text columns tokenized with HfTokenizer. Column auto-detection (input_ids/token_ids for pre-tokenized, text/content/code for text). Gated behind `parquet` feature. (`entrenar@a5a2fb7`) |
| ALB-009 | [#1](https://github.com/paiml/albor/issues/1) | apr (entrenar) | `apr train plan/apply` for pre-training from scratch | Critical | **FIXED** | `apr train plan --task pretrain --config <yaml>` validates config via entrenar, shows model architecture and training params. `apr train apply --task pretrain --config <yaml>` runs full pre-training via `train_from_yaml()` (TransformerTrainer + CausalLMLoss). Config updated to match entrenar TrainSpec schema. (`aprender@d79ed943`) |
| ALB-010 | [#2](https://github.com/paiml/albor/issues/2) | realizar | Qwen3-Coder-Next / DeltaNet / MoE architecture support | Critical | OPEN | `realizar` loads and runs inference on Qwen3-Coder-Next (80B MoE with DeltaNet layers) |
| ALB-011 | [#3](https://github.com/paiml/albor/issues/3) | apr (entrenar + realizar) | `apr distill plan/apply` (precompute + train stages) | Critical | **FIXED** | `apr distill --config <yaml> --plan` validates config, shows teacher/student/training params. `apr distill --config <yaml> --stage precompute` inspects teacher, writes manifest. `apr distill --config <yaml> --stage train` validates precompute manifest, sets up KD training. Local DistillYamlConfig matches entrenar schema. (`aprender@81dd4432`) |
| ALB-018 | [#19](https://github.com/paiml/albor/issues/19) | entrenar/alimentar | Fill-in-the-Middle (FIM) data transform (PSM/SPM) | High | **FIXED** | `alimentar fim` transform with PSM/SPM formats, configurable rate/seed (`alimentar@290582d`). `Fim` struct implements `Transform` trait for pipeline integration. |
| ALB-019 | [#20](https://github.com/paiml/albor/issues/20) | alimentar | `alimentar import local` for local Python files | Medium | **FIXED** | `alimentar import local` subcommand now available (`alimentar@265541b`). Supports CSV/JSON/JSONL/Parquet format conversion. |
| ALB-020 | [#21](https://github.com/paiml/albor/issues/21) | alimentar | `alimentar mix` with weighted upsampling | Medium | **FIXED** | `alimentar mix` with weighted sampling and upsampling now available (`alimentar@64b1e92`). Syntax: `alimentar mix a.parquet:0.8 b.parquet:0.2 -o out.parquet`. |
| ALB-021 | [#22](https://github.com/paiml/albor/issues/22) | entrenar | Custom model architecture params in YAML | High | **FIXED** | `ArchitectureOverrides` struct carries YAML manifest `architecture:` params through bridge converter to `TransformerConfig`. Supports all fields: `hidden_size`, `num_layers`, `num_heads`, `num_kv_heads`, `intermediate_size`, `vocab_size`, `max_seq_length`, `rms_norm_eps`, `rope_theta`, `use_bias`. (`entrenar@a414861`) |
| ALB-022 | [#23](https://github.com/paiml/albor/issues/23) | entrenar | Human-readable value shorthand in YAML configs | Low | **FIXED** | `parse_human_usize()` and `deserialize_human_usize_opt` support SI suffixes (32K, 1M, 10B, 1T), scientific notation (1e6), and fractional suffixes (1.5K). Applied to ArchitectureConfig and DataConfig fields. (`entrenar@1cb0950`) |
| ALB-023 | [#24](https://github.com/paiml/albor/issues/24) | apr (aprender) | Plan/apply contract for all subcommands | High | **FIXED** | Every `apr <cmd>` action command now exposes plan mode: `merge --plan`, `export --plan`, `publish --plan` added to join existing `train plan/apply`, `tokenize plan/apply`, `quantize --plan`, `finetune --plan`, `prune --plan`, `distill --plan`, `eval --task plan`. Pre-dispatch contract validation skipped in plan mode. (`aprender@526a1e4b`) |
| ALB-024 | [#25](https://github.com/paiml/albor/issues/25) | apr (aprender) | `apr experiment view` — interactive SQLite experiment browser | Medium | **FIXED** | `apr experiment view --global` opens ratatui TUI with run table, sparkline, and braille loss chart. `--json` mode for CI. Reads local or global `~/.entrenar/experiments.db`. (`aprender@1196d244`) |
| ALB-025 | [#26](https://github.com/paiml/albor/issues/26) | presentar + apr | `apr monitor` upgrade — presentar widgets for live training TUI | Medium | **FIXED** | `TrainingDashboard` composes presentar-terminal `Meter`, `GpuPanel`, `Sparkline`, `Text`, `Border`, `Layout` (ALB-057). `TuiApp` handles resize/Ctrl+C/diffing (ALB-047/048). WASM compilation deferred to ALB-026. (`entrenar@0ad416e`) |
| ALB-026 | [#27](https://github.com/paiml/albor/issues/27) | presentar | WASM training dashboard — `albor-dashboard.yaml` | Medium | OPEN | Declarative YAML dashboard config that renders training metrics, experiment comparison, and model card via `presentar serve`. Embeddable in HuggingFace model card as static WASM artifact. |
| ALB-027 | [#4](https://github.com/paiml/albor/issues/4) | forjar | `task` resource type for pipeline orchestration | Critical | **FIXED** | New forjar resource type: runs arbitrary command, tracks exit code, hashes `output_artifacts` for idempotency via b3sum, supports `completion_check` and `timeout`. Handlers: `check_script` (completion_check or artifact existence), `apply_script` (set -euo pipefail, working_dir, timeout), `state_query_script` (b3sum artifacts). Validation: command required, timeout > 0. (`forjar@d14e633`) |
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

**Contract coverage report** (`pv coverage contracts`): 8 contracts, 31 equations, 51 obligations, 34 falsification tests, 10 Kani harnesses, **100% obligation coverage**. All contracts at impl=0/N — waiting for upstream bindings.

### 11.4 Dogfooding-Discovered Gaps

| ID | Issue | Component | Gap | Severity | Status | Acceptance Criterion |
|----|-------|-----------|-----|----------|--------|---------------------|
| ALB-029 | [#28](https://github.com/paiml/albor/issues/28) | batuta | `batuta falsify` false positives on project repos | Medium | **FIXED** | Fixed upstream in `batuta@905a862`: AI-01 searches `configs/`, AI-04 excludes `book-output/`, AI-05 detects pv/forjar validation. Score: 72.2% → 73.1%. |
| ALB-030 | [#29](https://github.com/paiml/albor/issues/29) | batuta | `batuta stack status` fails without Cargo.toml | Low | **FIXED** | Fixed upstream in `batuta@371557a`: Falls back to binary detection, discovers 11 installed PAIML tools with versions. |
| ALB-031 | [#30](https://github.com/paiml/albor/issues/30) | batuta | `batuta hf search` returns mock/placeholder data | Low | OPEN | `batuta hf search model "code completion"` returns live HuggingFace Hub results instead of placeholder models. |
| ALB-033 | [#31](https://github.com/paiml/albor/issues/31) | apr (aprender) | `apr tokenize` → entrenar tokenizer.json format gap | Medium | DOGFOODING | `apr tokenize apply` produces vocab.json + merges.txt but entrenar expects HuggingFace tokenizer.json. Workaround: Python `tokenizers` lib. |
| ALB-034 | [#32](https://github.com/paiml/albor/issues/32) | entrenar | `max_steps` config not respected in training loop | Medium | **FIXED** | `max_steps` wired through YAML manifest → bridge → TrainingParams → TransformerTrainConfig → trainer loop. Training stops when optimizer step count reaches limit (`entrenar@07db101`). |
| ALB-035 | [#33](https://github.com/paiml/albor/issues/33) | entrenar | Does not write `training_state.json` during training | Medium | **FIXED** | Added `train_epoch_with_callback()` and per-step logging (~100 lines/epoch) in `entrenar@5d41a96`. |
| ALB-036 | [#34](https://github.com/paiml/albor/issues/34) | apr (aprender) | BPE tokenizer normalizes whitespace | Medium | DOGFOODING | `split_whitespace()` pre-tokenizer destroys Python indentation. Workaround: ByteLevel BPE v2. |
| ALB-037 | [#35](https://github.com/paiml/albor/issues/35) | realizar | SafeTensors inference ignores loaded weights | High | **FIXED** | Root cause chain: ALB-038 (no gradient flow) → ALB-043 (backward_ffn buffer overflow + wrong SwiGLU gradients). Secondary: entrenar didn't save config.json (`entrenar@6097780`). Verified e2e: `realizar run` loads 350M trained checkpoint (218 tensors), generates tokens from learned weights. |
| ALB-038 | [#36](https://github.com/paiml/albor/issues/36) | entrenar | Saves initialization weights, not trained weights | Critical | **FIXED** | Root cause: `RMSNorm::forward_batched()` created tensors with no backward op, blocking all gradient flow. Attention `forward()` also broke Q/K/V gradients. Fixed in `entrenar@91ba9da` (norm backward) and `entrenar@1ede409` (attention backward). All 20 model parameters now receive gradients. |
| ALB-040 | [#38](https://github.com/paiml/albor/issues/38) | entrenar | GPU-resident pretraining — wire CudaTransformerBlock into TransformerTrainer | Critical | **VERIFIED** | `CudaTransformerTrainer` in `cuda_trainer.rs` follows classify_pipeline.rs pattern. 3 PCIe transfers/step vs 16K. Auto-detect CUDA with graceful CPU fallback. Contract: `training-gpu-kernel-v1.yaml`. 350M verified: 50-step test loss 10.39→6.07, checkpoint valid, realizar loads + generates. Full training running (seq=1024, batch=4, accum=128). |
| ALB-041 | [#39](https://github.com/paiml/albor/issues/39) | entrenar | D2D buffer size mismatch in CudaTransformerBlock backward_attention | High | **FIXED** | `backward_attention()` used `gate_out` (intermediate_size) as temp buffer for `grad_hidden` accumulation, but D2D copy requires exact size match. Fixed: use `o_proj_out` (hidden_size). Also added seq_len truncation and error logging in `CudaTransformerTrainer`. (`entrenar@a48e3d2`) |
| ALB-042 | [#40](https://github.com/paiml/albor/issues/40) | entrenar | CudaTransformerTrainer runtime errors → silent loss=0.0 instead of CPU fallback | Medium | OPEN | When CUDA operations fail during training (e.g., VRAM contention), trainer should detect N consecutive failures and gracefully fall back to CPU mode. Currently reports loss=0.0 and saves garbage checkpoint. Workaround: `CUDA_VISIBLE_DEVICES=""`. |
| ALB-043 | [#41](https://github.com/paiml/albor/issues/41) | entrenar | backward_ffn buffer overflow + missing SwiGLU gradients | Critical | **FIXED** | Two bugs: (1) `silu_backward` wrote [S,I] output into [S,H] buffer (4× overflow → `CUDA_ERROR_ILLEGAL_ADDRESS`). (2) SwiGLU backward missing `×up` factor in gate gradient; `grad_up`/`grad_w_up` completely absent (w_up never trained). Fixed with correct 10-step decomposition using `elementwise_mul_forward`, `silu_forward`, `silu_backward`. (`entrenar@f7805f1`) |
| ALB-044 | [#42](https://github.com/paiml/albor/issues/42) | entrenar | Unclipped activation gradients + CPU optimizer hyperparameter mismatch cause 350M NaN | Critical | **FIXED** | Two bugs: (1) Activation gradient from block[0] backward (~1e35) unclipped — per-block clipping only applies to weight gradients in CudaGradWorkspace. (2) CPU AdamW used `default_params(lr)` (β₂=0.999, wd=0.01) instead of YAML config (β₂=0.95, wd=0.1) — 50× bias correction amplification overflows f32. Fixed: C-EMBED-GRAD-001 clips activation gradient before scatter-add; CPU optimizer matches YAML hyperparams. 350M now trains without NaN. |
| ALB-045 | — | entrenar | `train_loop_cuda` does not write `training_state.json` — `apr monitor` blind to pretraining | Critical | **FIXED** | `write_training_snapshot()` helper in `src/config/train/loader.rs` writes `TrainingSnapshot` to `training_state.json` on every log interval. Both `train_loop_cuda` and `train_loop_cpu` now emit Initializing→Running→Completed snapshots. Verified: `apr monitor checkpoints/albor-base-350m/` shows live TUI with loss curve, GPU name, tok/s, progress during CUDA 350M pretraining. (`entrenar@2ddc11c`) |
| ALB-046 | — | entrenar | GPU telemetry all zeros in `training_state.json` — no live NVML/nvidia-smi data | High | **FIXED** | `query_gpu_telemetry()` shells out to `nvidia-smi --query-gpu` with CSV output, populates all GpuTelemetry fields. Wired into `write_training_snapshot()`. Verified: util=5%, VRAM=12.0G/24.0G, temp=41°C, power=94W/480W during 350M training (`entrenar@9b53c13`). |
| ALB-047 | — | entrenar | TUI monitor hardcodes width=80, no terminal resize handling | Medium | **FIXED** | Replaced hand-rolled renderer with presentar-terminal `TuiApp`. Gets terminal resize detection for free from crossterm backend + presentar's smart diffing. `TuiMonitorConfig.width/height` retained for headless mode only (`entrenar@9b53c13`). |
| ALB-048 | — | entrenar | No signal handling in TUI monitor — Ctrl+C leaves cursor hidden | Medium | **FIXED** | presentar-terminal `TuiApp::run()` handles Ctrl+C/`q` with clean cursor restore, screen cleanup, and status message. No raw signal handlers needed — crossterm event loop + Drop impl (`entrenar@9b53c13`). |
| ALB-049 | — | entrenar | No keyboard input in TUI monitor — can't scroll/pause/interact | Low | **FIXED** | presentar-terminal `TuiApp` provides crossterm event loop with `q` quit and Ctrl+C. Scroll/pause deferred to presentar widget-level interaction (GpuPanel, LossCurve already support focus). |
| ALB-050 | — | apr (aprender) | No `apr runs ls` — can't list past training experiments | High | **FIXED** | `apr runs ls` reads local/global SQLite registry, shows table of runs with status, final loss, tok/s, duration. `apr runs show <id>` shows detailed metrics + hyperparameters. Supports `--global`, `--json`, `--status` filter. (`aprender@91641f2e`) |
| ALB-051 | — | apr (aprender) | No run comparison — can't overlay loss curves from two runs | Medium | **FIXED** | `apr runs diff <a> <b>` shows side-by-side comparison: inline sparklines, loss trajectory overlay, config diff (only changed params), final metric comparison with verdict (winner by final loss). Supports `--json` for LLM agents. (`aprender@9f9e9f63`) |
| ALB-052 | — | entrenar | SQLite experiment tracking exists but not wired to pretraining | Medium | **FIXED** | `PretrainTracker` in `config/train/loader.rs` writes to both local and global SQLite stores. Uses existing `SqliteBackend` with `ExperimentStorage` trait. Logs experiment metadata, hyperparameters, and per-step metrics (loss, lr, tok/s). Best-effort — storage failures never block training. (`entrenar@daa0afc`) |
| ALB-053 | — | entrenar | HeadlessOutput JSON missing fields present in TUI | High | **FIXED** | HeadlessOutput now has full field parity with TUI: `global_step`, `progress_percent`, `loss_history`, `lr_history`, `elapsed_seconds`, `optimizer_name`, `batch_size`, `model_path`, `checkpoint_path`, `executable_path`, `accuracy`, `samples_per_second`, `HeadlessSample`. `From<&TrainingSnapshot>` populates all fields. All 6 headless tests pass. (`entrenar@9b53c13`) |
| ALB-054 | — | entrenar + apr | No multi-job monitoring — can't watch multiple concurrent training runs | High | **FIXED** | `apr monitor` (no args) discovers active training runs from global SQLite registry (`~/.entrenar/experiments.db`). Checks for live `training_state.json` in registered output dirs. Lists active runs with experiment name, directory, run ID, start time. `apr monitor <dir>` attaches to specific run. Supports `--json` output for LLM agents. (`aprender@91641f2e`) |
| ALB-055 | — | entrenar | No local SQLite experiment DB per training run | High | **FIXED** | `PretrainTracker` opens `<output_dir>/.entrenar/experiments.db` for local per-experiment metrics history. Logs experiment metadata, hyperparameters (task, model, optimizer, lr, epochs, batch_size, seq_len, max_steps, device), and per-step metrics (loss, lr, tok/s). All best-effort via `SqliteBackend`. (`entrenar@daa0afc`) |
| ALB-056 | — | entrenar | No global SQLite experiment registry | High | **FIXED** | `PretrainTracker` opens `~/.entrenar/experiments.db` for global cross-machine experiment registry. Same schema as local: experiment + run + hyperparams + per-step metrics. `apr runs ls --global` reads it. `apr monitor` (no args) discovers active runs from it. (`entrenar@daa0afc`) |
| ALB-057 | — | entrenar | Dashboard paints raw text instead of composing presentar widgets | Medium | **FIXED** | `TrainingDashboard` composes presentar-terminal widgets via `Layout::rows()`: `Border` for section panels, `Meter` for progress bar, `GpuPanel` for GPU telemetry (with `GpuDevice`/`GpuProcess` conversion from entrenar types), `Sparkline` for loss history, `Text` for info lines. Widget tree rebuilt each frame from snapshot. Panel verification wired into `Brick::verify()` via `layout_can_render()`. (`entrenar@0ad416e`) |
| ALB-058 | — | apr (aprender) | `apr monitor --json` flag missing | Medium | **FIXED** | `apr monitor --json <dir>` streams headless JSON output with full TUI parity (ALB-053). `apr monitor --format text <dir>` for human-readable log lines. `--json` flag overrides `--format`. Routes to `HeadlessMonitor` for JSON/text, `TuiMonitor` for TUI. (`aprender@91641f2e`) |
| ALB-059 | — | entrenar | GEMM backward constructor args n/k swapped — buffer overflow into optimizer states | Critical | **FIXED** | `GemmBackwardAKernel::tiled_unrolled(m, k, n, tile)` called with k and n swapped vs trueno constructor `(m, n, k, tile_size)`. Bakes wrong stride constants into PTX: output stride = vocab_size (32768) instead of hidden_size (512) for LM head backward. Rows overflow 64× into adjacent VRAM (m_w_k, v_w_k of block 0). Negative values in v_w_k → sqrt(negative) = NaN in AdamW. Same bug in backward_b. Also zero-initialized all optimizer m/v buffers (cuMemAlloc returns uninitialized VRAM). (`entrenar@846ae0c`) |
| ALB-060 | — | entrenar / albor config | `epochs: 1` exhausts data before `max_steps` reached — 350M trains only 43/5000 steps | Critical | OPEN | With 22K sequences, batch_size=4, grad_accum=128: one epoch = 22079/4/128 = 43 steps. `max_steps: 5000` never reached. LR at step 43 still in warmup (6.45e-6 vs target 3e-4), loss flat at ~10.39. Fix: either set epochs=117 (5000/43) to cycle data enough times, or add data-looping in entrenar when `max_steps > steps_per_epoch`. The 350M checkpoint at `checkpoints/albor-base-350m/` contains effectively untrained weights. |

| ALB-061 | [#43](https://github.com/paiml/albor/issues/43) | albor docs | Monolithic spec stale — diverges from mdBook chapters | Medium | **FIXED** | `scripts/generate-spec.sh` regenerates `docs/specifications/albor-llm-spec.md` from mdBook chapters. `make spec` target added. |
| ALB-062 | [#44](https://github.com/paiml/albor/issues/44) | albor docs | Stale spec chapters — §3 VRAM, §15/18 blockers, §16 repro, model card, intro | Medium | **FIXED** | All chapters updated to match reality: VRAM budget, ALB-025/037 no longer blockers, v2 pipeline in §16, ALB-060 context in model card and introduction. |
| ALB-063 | [#45](https://github.com/paiml/albor/issues/45) | albor training | Retrain 350M with v2 config (corrected epochs + expanded data) | Critical | IN PROGRESS | C-TRAINCFG-001 pre-flight passes. Training started with `train-guard.sh`. |
| ALB-064 | [#46](https://github.com/paiml/albor/issues/46) | albor / entrenar | Training process dies silently — no crash detection, no watchdog, no recovery | Critical | **FIXED** | `scripts/train-guard.sh`: crash-resilient supervisor with exit code classification, GPU state capture, structured JSON crash reports, exponential backoff restart, heartbeat monitoring, pre-flight GPU health checks. Auto-diagnostic mode: detects async CUDA crash pattern, enables `CUDA_LAUNCH_BLOCKING=1` on restart. Five Whys: CUDA driver crash → SIGABRT/SIGSEGV → bypasses Rust panic handler → no stderr output → no diagnosis. Root cause: ALB-065. |
| ALB-065 | [#47](https://github.com/paiml/albor/issues/47) | entrenar / trueno | Missing `stream.synchronize()` before D2H gradient transfers — async CUDA crash | Critical | **FIXED** | `compute_workspace_clip_scale()` and `compute_clip_scale()` call `cuMemcpyDtoH` without synchronizing the non-blocking CUDA stream. `cuMemcpyDtoH` only synchronizes with the default stream, but trueno creates streams with `CU_STREAM_NON_BLOCKING`. Result: backward kernels not finished when gradient buffers are read → garbage clip scale → NaN/crash. Fix: `stream.synchronize()` at 3 locations before D2H transfers (`entrenar@d3a3d26`). |

*Gaps are added as they are discovered during implementation and dogfooding.*


---


# 12. Provable Quality & Design by Contract

Every computational kernel used in Albor must have a provable-contracts YAML
specification with Popperian falsification tests, property-based probar tests,
and Kani bounded model checking harnesses. This is not optional — it is a
first-class deliverable alongside the model.

## 12.1 Verification Ladder

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

## 12.2 Contract Registry for Albor

Albor requires contracts for every kernel in the training + post-training pipeline.
Many already exist in provable-contracts; new ones must be written.

### Existing Contracts (bind to aprender implementations)

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

### New Contracts Required for Albor (ALB-013 through ALB-017)

| Contract (NEW) | Key Equations | Key Obligations | Priority |
|----------------|---------------|-----------------|----------|
| `knowledge-distillation-kernel-v1.yaml` | KD_loss = α·KL(σ(z_t/T) ∥ σ(z_s/T))·T² + (1-α)·CE(y, z_s) | KL non-negativity, temperature scaling invariant, gradient correctness, α interpolation bound | Critical |
| `bpe-tokenizer-kernel-v1.yaml` | BPE merge rules, byte-pair encoding | Roundtrip invariant: decode(encode(x)) = x, vocab coverage, merge ordering | High |
| `model-merging-kernel-v1.yaml` | SLERP: interp(θ, w₁, w₂) on unit sphere; TIES: trim + elect + disjoint merge | SLERP interpolation bound (‖result‖ ≈ 1), TIES sparsity guarantee | Medium |
| `pruning-kernel-v1.yaml` | WANDA: score = |w| · ‖x‖₂; magnitude: score = |w| | Sparsity invariant (exactly k% weights zeroed), score ordering preserved | Medium |
| `gradient-accumulation-kernel-v1.yaml` | G_accum = (1/N)·Σ g_i ≈ g_full | Numerical equivalence within tolerance, loss scaling correctness | High |
| `training-config-kernel-v1.yaml` | steps_per_epoch, total_achievable_steps, LR warmup coverage, Chinchilla tokens | Epoch sufficiency for max_steps, warmup completion, peak LR reached, data sufficiency | Critical |

## 12.3 Contract Workflow for Each Kernel

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

## 12.4 Falsification Tests: Albor-Specific

Every claim in this specification must be falsifiable. Below are the concrete
falsification tests for Albor's key properties.

### Training Correctness

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

### Distillation Correctness

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

### Post-Training Invariants

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

## 12.5 Brick Profiling Architecture

Training a 350M model on a single 4090 is a systems engineering problem, not
a scaling problem. Every watt of GPU silicon must be accounted for. The
architecture achieves this by treating each component as a **brick** — a
self-contained unit with measurable inputs, outputs, and a provable contract.

### 12.5.1 Three Granularities of Profiling

**Per-kernel.** Every CUDA kernel (`gemm_forward`, `silu_backward`,
`rms_norm_forward`, `batched_transpose_forward`, etc.) is individually
measurable via `compute-sanitizer`, `nsys`, or `nvprof`. When a kernel
misbehaves, the brick boundary isolates the failure to a single function with
known input/output shapes. The contract for each kernel specifies buffer size
invariants that can be checked statically.

**Per-block.** `CudaTransformerBlock` encapsulates one transformer layer's
forward, backward, and optimizer step as a single GPU-resident unit. Diagnostic
sampling after backward (downloading 1K elements from each gradient buffer)
immediately distinguishes "math is wrong" (NaN in gradients) from "math is
right but magnitudes are wrong" (gradient explosion). The brick boundary
separates kernel correctness from training dynamics.

**Per-transfer.** The 3-transfer-per-step contract (`C-GPUTRAIN-002`) fixes
the PCIe budget:

```
Transfer 1 (H2D): embedding hidden states   ~S×H×4 bytes
Transfer 2 (D2H): logits for cross-entropy  ~S×V×4 bytes
Transfer 3 (H2D): grad_logits to GPU        ~S×V×4 bytes
```

Any deviation from 3 transfers is a bug, not a tuning knob. For 350M at
seq=2048: total ~544 MB/step, overhead ~17 ms on PCIe 4.0 x16 — under 5%
of compute time.

### 12.5.2 Chain of Thought: How Brick Boundaries Diagnose Bugs

When a training run fails, the brick architecture converts "something is
broken" into a structured diagnosis:

1. **Which granularity?** Check per-transfer (D2D size mismatch?), per-block
   (which layer's backward fails?), per-kernel (which GEMM overflows?).
2. **Local or global?** If one block fails and others succeed, the bug is in
   that block's kernels. If all blocks succeed but loss diverges, the bug is in
   training dynamics (LR, grad clipping, optimizer config).
3. **Static or dynamic?** Buffer overflow is a static invariant violation
   (detectable by algebraic dimension checking). Gradient explosion is a
   dynamic stability issue (detectable by runtime sampling).

### 12.5.3 Five Whys: From Symptom to Root Cause

The brick architecture enforces a disciplined root-cause chain. Concrete
example from dogfooding:

| Why | Finding | Brick boundary |
|-----|---------|----------------|
| **Why does 350M training produce NaN at step 2?** | Gradients reach 1e35, AdamW produces NaN weights | Per-block sampling: `grad_gate max=3.28e35` |
| **Why are gradients 1e35?** | 24-layer backward amplifies without clipping | Per-transfer: config has `grad_clip: 1.0` but CUDA path ignores it |
| **Why no gradient clipping in CUDA path?** | `CudaTransformerTrainer` copied from finetuning (pre-trained weights, small grads) | Brick mismatch: finetuning brick assumed well-conditioned weights |
| **Why wasn't this caught by the GPU training contract?** | Contract validates kernel correctness + transfer count, not training stability | Contract gap: no `C-TRAINSTABLE-001` obligation |
| **Why doesn't the contract cover stability?** | Contracts target kernel-level (local) correctness, not loop-level (global) dynamics | **Action**: add training-stability contract bridging kernel and loop levels |

This same pattern resolved four bugs during ALB-040 dogfooding:

| Bug | Profiling diagnosis | Contract that prevents recurrence |
|-----|--------------------|------------------------------------|
| ALB-043: `silu_backward` writes `[S,I]` into `[S,H]` buffer (4x overflow) | `compute-sanitizer` pinpoints illegal address in `silu_backward` | Buffer size invariant: output must be `[S, intermediate_size]` |
| ALB-041: D2D copy size mismatch in `backward_attention` | Error logged at exact block index; `gate_out` used as `grad_hidden` temp | D2D invariant: `src.len() == dst.len()` for `copy_from_buffer_async` |
| `backward_attention`: transpose `attn_scores [H,S,S]` into `attn_kv_temp2 [H,S,hd]` | Algebraic trace: `16×512×512 = 4.2M` into 524K buffer = 8x overflow | Transpose output buffer invariant: `output.len() >= batch × rows × cols` |
| `gpu_forward`: D2D copy fails when `seq_len < max_seq_len` | All forwards return None; traced to `PAR-023` size mismatch | Forward buffer invariant: input/output buffers at `max_seq_len` size |
| ALB-044: Unclipped activation gradient (~1e35) overflows CPU AdamW | Per-boundary sampling: embed weights have 1298 NaN after optimizer step | C-EMBED-GRAD-001: clip activation gradient at GPU→CPU boundary |
| ALB-044: CPU AdamW beta2=0.999 vs YAML beta2=0.95 (50x amplification) | Traced bias correction: v_hat = v/0.001 with beta2=0.999 vs v/0.05 with 0.95 | C-HYPERPARAMS-001: all optimizer fields must match YAML config |
| ALB-059: GEMM backward constructor args n/k swapped — output stride 64× too large | Per-kernel: v_w_k[block0] corrupted during `gemm_backward_a(LM head)`. Pointer analysis: 3 contiguous 256KB allocs. Stride 32768 writes rows into m_w_k/v_w_k. | C-GEMMARGS-001: kernel constructor args must match documented parameter order |
| ALB-059: Uninitialized optimizer m/v buffers (cuMemAlloc returns garbage) | Per-block: v_w_k nonzero before any backward op (not from overflow). `GpuBuffer::new()` ≠ zero-init. | C-GPUINIT-001: all optimizer state buffers must be zero-initialized |

### 12.5.4 How Bricks and Contracts Interlock

The gap register (§11) is the feedback loop between profiling and contracts:

```
Brick profiling finds anomaly
  → File gap (ALB-0XX)
    → Write or update contract obligation
      → Fix upstream brick
        → Verify contract passes (`pv audit`)
          → Dogfood in albor pipeline
            → Close gap
```

Profiling finds bugs that contracts miss (runtime-only issues like gradient
explosion). Contracts prevent bugs that profiling misses (the 50M model's 2x
buffer overflow "worked" through undefined behavior — only a static size
invariant would have caught it). Together they form a ratchet: every bug found
by profiling becomes a permanent contract obligation that prevents recurrence.

## 12.6 Verification DAG (Albor End-to-End)

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
training-config ─── config-validation ─────────┘
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

## 12.7 Training Stability Contracts

The kernel-level contracts in §12.2 verify *local* correctness — each kernel
produces the right output for its input. They do NOT verify *global* training
stability — that the training loop converges without NaN, that hyperparameters
propagate correctly, or that gradients flow to all parameters.

ALB-038, ALB-041, ALB-043, and ALB-044 all passed kernel-level contracts
while producing training failures. These contracts bridge the gap between
kernel correctness and training correctness.

### C-TRAINSTABLE-001: Training Stability

All weights and loss must remain finite for the entire training run.

```yaml
obligations:
  - "loss.is_finite() for all steps"
  - "weight[i].is_finite() for all i, all steps"
  - "grad[i].is_finite() for all i after clipping, all steps"
falsification: |
  FALSIFY-STABLE-001: Train 100 steps on random init.
  Assert loss.is_finite() at every step.
  Assert no NaN in any model weight after every optimizer step.
```

### C-EMBED-GRAD-001: Activation Gradient Clipping at GPU-CPU Boundary

When GPU backward produces activation gradients that flow to a CPU optimizer,
those gradients must be clipped to `max_grad_norm` before the CPU processes
them.

**Status: VERIFIED** — 350M CUDA test (50 steps) produces zero NaN in embedding
weights. Fix in `entrenar@86eec38`.

```yaml
motivation: |
  Per-block gradient clipping in CudaGradWorkspace only clips WEIGHT gradients.
  The ACTIVATION gradient in grad_buf_a/b flows unclipped to the CPU embedding
  optimizer. For 24-layer random init, this gradient reaches ~1e35 — overflowing
  the CPU AdamW second moment buffer.
obligation: |
  Before scatter-adding activation gradients into CPU embedding weight gradient:
    grad_norm = L2_norm(activation_grad)
    if grad_norm > max_grad_norm:
        activation_grad *= max_grad_norm / grad_norm
falsification: |
  FALSIFY-EMBEDGRAD-001: Train 350M model (24 layers) for 5 steps.
  Assert embedding weights contain zero NaN values after each optimizer step.
```

### C-HYPERPARAMS-001: Optimizer Hyperparameter Propagation

Every optimizer hyperparameter in the YAML config must reach the actual
optimizer constructor. No implicit defaults.

**Status: VERIFIED** — 350M CUDA test uses explicit `AdamW::new()` with
YAML config values (beta2=0.95, wd=0.1). Fix in `entrenar@86eec38`.

```yaml
obligation: |
  For every optimizer in the training loop (GPU AdamW, CPU AdamW, LM head AdamW):
    assert optimizer.lr == config.lr (adjusted for warmup)
    assert optimizer.beta1 == config.beta1
    assert optimizer.beta2 == config.beta2
    assert optimizer.weight_decay == config.weight_decay
    assert optimizer.epsilon == 1e-8 (or config.epsilon if specified)
falsification: |
  FALSIFY-HYPERPARAMS-001: Construct CudaTransformerTrainer with non-default
  YAML config (beta2=0.95, wd=0.1). Verify CPU embed_optimizer.beta2 == 0.95
  and embed_optimizer.weight_decay == 0.1 (not 0.999 and 0.01).
anti_pattern: |
  NEVER: AdamW::default_params(lr)  — hides beta2, wd, epsilon
  ALWAYS: AdamW::new(lr, beta1, beta2, epsilon, wd)  — explicit from config
```

### C-BUFSIZE-001: CUDA Kernel Buffer Size Invariants

Every GPU buffer passed to a CUDA kernel must have algebraically verifiable
size matching the kernel's expected dimensions.

```yaml
obligation: |
  For gemm_forward(A, B, C, M, K, N):
    assert A.len() >= M * K
    assert B.len() >= K * N
    assert C.len() >= M * N
  For silu_backward(input, grad_output, output):
    assert output.len() >= input.len()
  For rms_norm_backward(input, weight, grad_output, grad_input, grad_weight, S, H):
    assert grad_input.len() >= S * H
    assert grad_weight.len() >= H
falsification: |
  FALSIFY-BUFSIZE-001: Run compute-sanitizer on 10-step 50M training.
  Assert zero illegal address errors.
anti_pattern: |
  NEVER: Reuse a buffer sized for hidden_size as temp for intermediate_size
  ALWAYS: Use dedicated buffers or verify size >= required before kernel call
```

### C-GEMMARGS-001: GEMM Kernel Constructor Argument Ordering

Every GEMM kernel constructor call must pass arguments in the exact order
documented by the kernel's API. Compile-time stride constants baked into PTX
are determined by constructor args — wrong order produces wrong strides, not
wrong results at the kernel boundary (bounds check passes but data lands in
wrong memory).

**Status: VERIFIED** — 350M CUDA test (50 steps) produces correct backward
gradients. Fix in `entrenar@846ae0c`.

```yaml
motivation: |
  GemmBackwardAKernel::tiled_unrolled(m, n, k, tile_size) bakes self.n and
  self.k as immediate PTX constants for row/col strides. When called as
  tiled_unrolled(m, k, n, tile) with k and n swapped, the output stride
  becomes vocab_size (32768) instead of hidden_size (512) — writing output
  rows 64× too far apart and overflowing into adjacent GPU allocations.
obligation: |
  For every kernel constructor call:
    assert arg_order matches constructor signature exactly
  Specifically for GEMM backward:
    GemmBackwardAKernel::tiled_unrolled(m, n, k, tile)  # NOT (m, k, n, tile)
    GemmBackwardBKernel::tiled_unrolled(m, n, k, tile)  # NOT (m, k, n, tile)
falsification: |
  FALSIFY-GEMMARGS-001: Train 350M model for 5 steps. Download v_w_k[block0]
  after backward. Assert zero corruption (all values ≥ 0 after optimizer init,
  no values from adjacent buffers).
anti_pattern: |
  NEVER: Guess argument order from variable names (m/n/k are ambiguous)
  ALWAYS: Check constructor signature in trueno-gpu kernel source
```

### C-GPUINIT-001: GPU Buffer Zero Initialization

All optimizer state buffers (m and v for AdamW) must be zero-initialized.
`GpuBuffer::new()` uses `cuMemAlloc` which returns uninitialized VRAM —
the contents are whatever was previously in that memory region.

**Status: VERIFIED** — All 34 optimizer buffers (18 per-block + 12 LoRA + 4 LM head/norm)
zero-initialized via `GpuBuffer::from_host(&ctx, &vec![0.0f32; n])`. Fix in `entrenar@846ae0c`.

```yaml
obligation: |
  For every GpuBuffer used as optimizer state (m, v):
    assert buffer is zero-initialized after allocation
    Use GpuBuffer::from_host(&ctx, &vec![0.0f32; n])
    NOT GpuBuffer::new(&ctx, n)  -- returns uninitialized VRAM
falsification: |
  FALSIFY-GPUINIT-001: Allocate optimizer state, download immediately.
  Assert all values == 0.0.
```

### C-GRADFLOW-001: Gradient Flow Verification

Every trainable parameter must receive a non-zero gradient after one
forward+backward step on a non-trivial batch.

```yaml
obligation: |
  After one forward+backward step on a batch with non-constant inputs:
    for param in model.trainable_parameters():
      assert param.grad().abs().max() > 0
falsification: |
  FALSIFY-GRADFLOW-001: Train 1 step on 50M model with random init.
  Verify all 110 parameter tensors have max(|grad|) > 0.
anti_pattern: |
  NEVER: Create tensors with requires_grad=false in the forward path
  NEVER: Use ops that don't register backward (e.g., manual array copies)
  ALWAYS: Verify gradient flow when adding new layers or ops
```

### C-TRAINCFG-001: Training Configuration Algebraic Consistency

Every training configuration must be algebraically validated BEFORE GPU time is
consumed. The epoch/step/data/LR relationship must be provably sufficient.

**Status: OPEN** — ALB-060. The 350M training ran only 43/5000 steps because
`epochs: 1` exhausted data before `max_steps`. Contract written, config fixed
(`epochs: 117`), awaiting re-training verification.

```yaml
motivation: |
  ALB-060: pretrain-350m.yaml had epochs=1 with 22K sequences and grad_accum=128.
  steps_per_epoch = floor(22079 / 4 / 128) = 43. max_steps=5000 unreachable.
  warmup_steps=2000 never completed. LR peaked at 6.45e-6 (target 3e-4).
  Loss flat at ~10.39 for all 43 steps. Checkpoint contains untrained weights.
  Total wasted: ~12 seconds GPU + debugging time. Contract prevents recurrence.
equations:
  - "steps_per_epoch = floor(num_sequences / batch_size / grad_accum)"
  - "total_achievable_steps = num_epochs × steps_per_epoch"
  - "total_achievable_steps >= max_steps  (HARD REQUIREMENT)"
  - "warmup_steps < total_achievable_steps  (warmup must complete)"
  - "warmup_fraction = warmup_steps / actual_total_steps <= 0.10"
  - "min_epochs = ceil(max_steps / steps_per_epoch)"
  - "total_tokens = actual_steps × batch_size × grad_accum × seq_len"
obligations:
  - "Epoch count sufficient: num_epochs >= ceil(max_steps / steps_per_epoch)"
  - "Warmup completes: warmup_steps < actual_total_steps"
  - "Peak LR reached: exists step t where lr(t) = lr_peak"
  - "Training tokens sufficient: total_tokens >= 10 × num_params"
falsification: |
  FALSIFY-CFG-001: Compute steps_per_epoch for pretrain-350m.yaml.
  With 22079 seqs, batch=4, accum=128: steps_per_epoch=43.
  Assert 1 × 43 < 5000 (proves epochs=1 is insufficient).
  FALSIFY-CFG-002: Assert warmup_steps (2000) > total_steps (43)
  (proves warmup never completes with epochs=1).
```

Full contract: `contracts/training-config-kernel-v1.yaml` — 7 equations,
8 proof obligations, 5 falsification tests, 2 Kani harnesses.

### 12.7.1 Observability Discipline

**All training observability MUST use the renacer tracing infrastructure.**

entrenar integrates renacer in `src/run.rs` (span lifecycle: `create_span`,
`emit_metric_event`, `end_span`). The `src/monitor/drift.rs` module provides
anomaly detection (`DriftStatus`, `AnomalySeverity`) that can automatically
flag NaN, gradient explosion, and loss divergence.

```yaml
obligation: |
  NEVER: eprintln!(), println!(), dbg!() for training diagnostics
  ALWAYS: tracing::debug!(), tracing::warn!() with structured fields
  ALWAYS: emit_metric_event() for training metrics (loss, grad_norm, lr)
motivation: |
  Ad-hoc eprintln! creates cleanup debt, is invisible to tracing infra,
  loses brick profiling boundary isolation, and cannot be filtered at runtime.
  renacer BrickTracer provides structured, filterable, permanent observability.
```

---

---


# 13. pmat Compliance & Quality Gates

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


# 14. Batuta Falsification Checklist

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


# 15. Implementation Phases

### Phase 0: Pipeline Manifest, Contracts & Quality Baseline (Week 1)
- [ ] Write `configs/pipeline/albor.yaml` — full pipeline manifest (infra + data + train + eval + publish)
- [ ] `apr pipeline plan` — validate entire DAG, estimate resources
- [ ] `apr pipeline apply --target cuda-driver --target vulkan-driver --target data-dir` — provision infra
- [ ] Verify `trueno` wgpu on W5700X via Vulkan (not Metal — Linux)
- [ ] Verify `trueno` CUDA on 4090
- [ ] Download Qwen3-Coder-Next to intel box, verify it loads in realizar
- [ ] `pmat tdg baseline create` on all stack components
- [ ] `pv coverage contracts/ --binding` — establish contract coverage baseline
- [ ] `batuta falsify . --critical-only` — initial falsification assessment

### Phase 1: Data Pipeline + Tokenizer Contract (Week 1-2)
- [ ] Ingest local ground truth corpora via `alimentar import local` (fix ALB-019 if needed)
  - [ ] depyler: examples/ + tdd-book/tests/ (~1,845 files, ~219K lines)
  - [ ] hf-ground-truth-corpus (~11,928 files)
  - [ ] jax-ground-truth-corpus (~2,697 files)
  - [ ] vllm-ground-truth-corpus (~1,118 files)
- [ ] Ingest local ML framework code (Tier 2, ~53K files)
- [ ] Download external datasets via `alimentar import hf` (StarCoder Python, FineWeb-Edu)
- [ ] Quality validation via `alimentar quality check` on all sources
- [ ] Build weighted training mix with 10x upsampling on Tier 1 (fix ALB-020 if needed)
- [ ] Write `bpe-tokenizer-kernel-v1.yaml` contract (ALB-014)
- [ ] `pv probar` + `pv kani` on tokenizer contract
- [ ] Train BPE tokenizer on mixed corpus (fix ALB-001 if needed)
- [ ] Verify FALSIFY roundtrip: `decode(encode(text)) = text` for all test data
- [ ] Tokenize all data into sharded Parquet
- [ ] Apply FIM transforms to code sequences (fix ALB-018 if needed)
- [ ] Create train/val/test splits via `alimentar`
- [ ] Record SHA-256 hashes + provenance manifest for all data artifacts
- [ ] `pmat comply check --strict` on alimentar changes

### Phase 2: Pipeline Validation — 50M Model (Week 2) -- COMPLETE
- [x] Write `gradient-accumulation-kernel-v1.yaml` contract (ALB-017)
- [x] Write `configs/train/pretrain-50m.yaml` (model arch + training + monitoring)
- [x] Train albor-50M on 4090 — 500 rows, 31 steps, 110.7s, loss 10.3→4.42
- [x] Validate `apr monitor` — ALB-025 FIXED (presentar widget migration complete)
- [ ] Validate Andon alerts during full training run
- [x] ~~Fix ALB-009~~ FIXED
- [x] Verify FALSIFY-ALBOR-001 (loss decreases) — CORROBORATED
- [x] Verify FALSIFY-ALBOR-002 (gradient bounds) — per-step logging now available (~~ALB-035~~ FIXED)
- [x] `pv audit` — PASS: 7/7 contracts, 0 findings
- [x] **Milestone**: Training loop converges ✓, contracts pass ✓

### Phase 3: Base Model — 350M Pre-Training (Week 2-4) -- IN PROGRESS
- [x] Write `configs/train/pretrain-350m.yaml` — pre-tokenized ByteLevel BPE v2, 22K×2048 tokens
- [x] Train albor-base-350m on 4090 — STARTED (2760 batches, ~20h est.)
- [x] Build evaluation infrastructure — eval-code.py, eval-perplexity.py, 35 benchmark problems
- [x] ~~Fix ALB-038~~ FIXED — RMSNorm + attention backward ops, all 20 params receive gradients
- [x] ~~Fix ALB-041~~ FIXED — D2D buffer size mismatch in backward_attention (`entrenar@a48e3d2`)
- [x] ~~Fix ALB-043~~ FIXED — backward_ffn buffer overflow + SwiGLU gradients (`entrenar@f7805f1`)
- [x] ~~Fix ALB-044~~ FIXED — activation gradient clipping at GPU-CPU boundary + CPU optimizer hyperparams (`entrenar@86eec38`)
- [x] ~~Fix ALB-059~~ FIXED — GEMM backward constructor args n/k swapped, buffer overflow into optimizer states + zero-init optimizer m/v (`entrenar@846ae0c`)
- [x] Write `training-memory-kernel-v1.yaml` contract (ALB-039) — VRAM budget estimation
- [x] Write `training-gpu-kernel-v1.yaml` contract (ALB-040) — GPU-resident training invariants
- [x] Implement `CudaTransformerTrainer` (ALB-040) — 3 PCIe transfers/step vs ~16K
- [x] Dogfood CUDA training — 50M test: 3 steps, loss 10.4→11.7, GPU forward+backward working
- [x] ~~ALB-037~~ FIXED — realizar loads trained SafeTensors checkpoint, generates tokens (e2e verified)
- [x] 350M CUDA test training — 50 steps, loss 10.39→5.92 (best 5.53), checkpoint valid
- [x] realizar inference verified — 218 tensors loaded, generates from trained weights
- [x] Checkpoint validation: PASS (weights trained, not initialization)
- [x] Perplexity eval: 31,926 (finite, consistent with 50-step model — random baseline ~32,768)
- [x] ~~Fix ALB-060~~ FIXED — epochs=1 only ran 43/5000 steps. C-TRAINCFG-001 contract written. Config fixed (v1: epochs=117, v2: epochs=38)
- [x] Expand training data: Tier 1 10x + 8 Tier 2 repos → v2 dataset (67,977 seqs, 139M tokens)
- [ ] Full 350M training — **FAIL (ALB-060)**: retrain with v2 config pending
- [ ] Monitor training via `apr monitor` (ALB-025 FIXED)
- [ ] Validate loss curve, perplexity convergence
- [ ] Tune hyperparameters (LR, batch size, warmup)
- [ ] Verify FALSIFY-ALBOR-003 (checkpoint determinism)
- [ ] `pmat tdg check-regression` on all touched components
- [ ] **Milestone**: Perplexity < 30, TDG grade A maintained

### Phase 4: Teacher Setup & Logit Pre-Computation (Week 3-5)
- [ ] Fix ALB-010: Add Qwen3-Coder-Next support to realizar (stretch — 3-4 week blocker)
- [x] Download Qwen2.5-Coder-3B interim teacher (5.75 GiB, Apache 2.0) — unblocks distillation without ALB-010
- [x] Validate 3B teacher: `apr distill --stage precompute` works, RosettaStone handles sharded SafeTensors
- [x] Create distillation config: `configs/train/distill-qwen3b.yaml` (T=4.0, α=0.5, LoRA r=16)
- [ ] Validate teacher inference on intel (CPU, fp16, 300GB RAM) — for 80B stretch goal
- [x] Write `knowledge-distillation-kernel-v1.yaml` contract (ALB-013) — DOGFOODING
- [ ] `pv kani` on KD loss contract (KL non-negativity, temperature scaling)
- [x] ~~Fix ALB-011~~ FIXED — `apr distill --config --stage precompute|train` works
- [ ] Pre-compute 3B teacher logits on v2 dataset (background, 4-8h CPU)
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
- [x] Write `model-merging-kernel-v1.yaml` contract (ALB-015) — DOGFOODING
- [x] Write `pruning-kernel-v1.yaml` contract (ALB-016) — DOGFOODING
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


# 16. Reproducibility Protocol

Every artifact in the albor pipeline is reproducible from source. This chapter
documents the exact commands, seeds, and checksums needed to reproduce the
full training pipeline from raw code corpora to trained model.

## 16.1 Artifact Tracking

| Artifact | How Recorded |
|----------|----------|
| Random seed | 42 (global), per-component seeds derived |
| Data versions | HuggingFace dataset commit SHAs + local repo git SHAs |
| Data provenance | `docs/PROVENANCE.md` (source path, git SHA, file count, token count per source) |
| Data checksums | SHA-256 of every Parquet shard (recorded in PROVENANCE.md) |
| Tokenizer v1 | `models/albor-tokenizer/` (vocab.json + merges.txt + tokenizer.json) |
| Tokenizer v2 | `models/albor-tokenizer-v2/tokenizer.json` (ByteLevel BPE) |
| Training config | YAML checked into git (`configs/train/*.yaml`) |
| Checkpoint hashes | SHA-256 of model.safetensors |
| Software versions | `apr --version`, `alimentar --version`, `pv --version` |
| Hardware | nvidia-smi + free -h captured in training logs |
| Training logs | `checkpoints/*/training.log` + `final_model.json` |
| Eval results | `configs/eval/*.jsonl` (benchmarks) + eval scripts |

## 16.2 Full Reproduction Commands

### Step 1: Corpus Preparation

**v1 pipeline** (Tier 1 only, 17K rows):

```bash
# Import Tier 1 ground truth corpora
alimentar import local /path/to/depyler -o data/raw/depyler.parquet
alimentar import local /path/to/hf-ground-truth-corpus -o data/raw/hf.parquet
alimentar import local /path/to/jax-ground-truth-corpus -o data/raw/jax.parquet
alimentar import local /path/to/vllm-ground-truth-corpus -o data/raw/vllm.parquet

# Mix training split (weighted sampling)
alimentar mix \
    data/raw/depyler.parquet:0.4 \
    data/raw/hf.parquet:0.3 \
    data/raw/jax.parquet:0.15 \
    data/raw/vllm.parquet:0.15 \
    -o data/tokenized/train/mixed.parquet \
    --seed 42
```

**v2 pipeline** (Tier 1 10x + 8 Tier 2 repos, 45K rows → 68K sequences):

```bash
# Convert Tier 2 source repos to Parquet (alimentar can't read source dirs)
for repo in pytorch hf-repos mlflow vllm-full tgi algo-corpus cuda-python llms-with-hf; do
    python3 scripts/source-to-parquet.py ~/src/$repo $repo data/parquet/tier2/$repo.parquet
done

# Mix Tier 1 (10x upsampled) + Tier 2 (1x)
alimentar mix \
    data/parquet/depyler/shard_0000.parquet:10.0 \
    data/parquet/hf-ground-truth/shard_0000.parquet:10.0 \
    data/parquet/jax/shard_0000.parquet:10.0 \
    data/parquet/vllm/shard_0000.parquet:10.0 \
    data/parquet/tier2/pytorch.parquet:1.0 \
    data/parquet/tier2/hf-repos.parquet:1.0 \
    data/parquet/tier2/mlflow.parquet:1.0 \
    data/parquet/tier2/vllm-full.parquet:1.0 \
    data/parquet/tier2/tgi.parquet:1.0 \
    data/parquet/tier2/algo-corpus.parquet:1.0 \
    data/parquet/tier2/cuda-python.parquet:1.0 \
    data/parquet/tier2/llms-with-hf.parquet:1.0 \
    -o data/staging/mixed-expanded.parquet --seed 42

# Apply FIM (50% PSM)
alimentar fim data/staging/mixed-expanded.parquet \
    -o data/staging/mixed-expanded-fim.parquet --rate 0.5 --format psm --seed 42
```

### Step 2: Tokenizer Training

```bash
# v1 tokenizer (whitespace-split BPE — has ALB-036 limitation)
apr tokenize apply \
    --data data/staging/corpus-raw.txt \
    --vocab-size 32768 \
    --algorithm bpe \
    -o models/albor-tokenizer/ \
    --max-lines 100000

# v2 tokenizer (ByteLevel BPE — preserves whitespace)
python scripts/train-tokenizer-v2.py \
    --corpus data/staging/corpus-raw.txt \
    --vocab-size 32768 \
    --output models/albor-tokenizer-v2/
```

### Step 3: Pre-Tokenization

```bash
# Pre-tokenize full training data (v2 tokenizer, 2048-token chunks)
python scripts/pretokenize.py \
    --input data/tokenized/train/mixed.parquet \
    --tokenizer models/albor-tokenizer-v2/tokenizer.json \
    --seq-len 2048 \
    --output data/pretokenized-2048/train/train.parquet

# Pre-tokenize validation data
python scripts/pretokenize.py \
    --input data/tokenized/val/val.parquet \
    --tokenizer models/albor-tokenizer-v2/tokenizer.json \
    --seq-len 2048 \
    --output data/pretokenized-2048/val/val.parquet
```

### Step 4: Model Training

```bash
# 50M pipeline validation (< 2 minutes)
make train-50m
# Equivalent to:
# apr train apply --task pretrain --config configs/train/pretrain-50m.yaml

# 350M base model, v2 data (~20 hours on RTX 4090)
apr train apply --task pretrain --config configs/train/pretrain-350m-v2.yaml
# v2 config: epochs=38, warmup=500, 67977 seqs, 5000 max_steps
# C-TRAINCFG-001 verified: steps_per_epoch=132, 38×132=5016 >= 5000

# Legacy v1 (22K seqs, fixed epochs=117 post ALB-060)
# apr train apply --task pretrain --config configs/train/pretrain-350m.yaml
```

### Step 5: Checkpoint Conversion (for evaluation)

```bash
# Convert entrenar 1D-flat SafeTensors to realizar 2D format
python scripts/convert-checkpoint.py checkpoints/albor-base-350m/ \
    --config configs/train/pretrain-350m.yaml
```

### Step 6: Evaluation

```bash
# Validate all benchmarks (no model needed)
make eval-validate

# Perplexity evaluation (needs trained model)
make eval-perplexity-350m

# Monitor active training
make training-status
```

## 16.3 Key SHA-256 Checksums

See `docs/PROVENANCE.md` for complete checksums. Key artifacts:

| Artifact | SHA-256 (first 8 hex) |
|----------|----------------------|
| Training data (mixed.parquet) | `bdfe8742` |
| Val data (val.parquet) | `6be03768` |
| v1 tokenizer (vocab.json) | `aca6fa72` |
| v2 tokenizer (tokenizer.json) | `d999cc9e` |
| Pre-tokenized train (2048) | `4f54e422` |
| Pre-tokenized val (2048) | `c9c1d093` |

## 16.4 Verification

```bash
# Verify data checksums
sha256sum data/tokenized/train/mixed.parquet
sha256sum data/pretokenized-2048/train/train.parquet
sha256sum models/albor-tokenizer-v2/tokenizer.json

# Verify training config reproducibility
apr train plan --task pretrain --config configs/train/pretrain-350m.yaml

# Verify contract integrity
pv validate contracts/*.yaml
pv coverage contracts
pv audit contracts/*.yaml
```

---


# 17. Success Criteria

### Minimum Viable (Phase 3 complete)
- [ ] 350M base model trained on 4090 to convergence (target: ~10B tokens; current: 139M v2 dataset)
- [x] FIM (fill-in-the-middle) training implemented and validated (~~ALB-018~~ FIXED — `alimentar fim` verified)
- [ ] **HumanEval pass@1 > 8%** (baseline Python capability, beat random)
- [ ] **HumanEval-FIM working** (model can infill Python code)
- [ ] Entire pipeline uses only sovereign stack components
- [ ] All training artifacts reproducible from spec
- [ ] All existing kernel contracts pass `pv audit` (Level 2+)
- [ ] `pmat comply check` passes on all modified components

**Current blockers for Phase 3 completion:**
- ~~ALB-038 (Critical): entrenar saves initialization weights, not trained weights~~ **FIXED** (`entrenar@91ba9da`, `@1ede409`)
- ~~ALB-035: No per-step loss logging during training~~ **FIXED** (`entrenar@5d41a96`)
- ~~ALB-041: D2D buffer mismatch in backward_attention~~ **FIXED** (`entrenar@a48e3d2`)
- ~~ALB-037: realizar ignores loaded weights~~ **FIXED** (e2e verified: `realizar run` loads 350M trained checkpoint, generates tokens from 218 tensors)
- ~~ALB-043 (Critical): backward_ffn buffer overflow + missing SwiGLU gradients~~ **FIXED** (`entrenar@f7805f1`)
- ~~ALB-044 (Critical): activation gradient clipping + CPU optimizer hyperparams~~ **FIXED** (`entrenar@86eec38`)
- ~~ALB-059 (Critical): GEMM backward constructor n/k swapped — buffer overflow into optimizer states~~ **FIXED** (`entrenar@846ae0c`)
- ~~ALB-040: GPU-resident pretraining~~ **VERIFIED** — 350M CUDA test: 50 steps, loss 10.39→5.92, checkpoint valid, realizar inference works
- ALB-042: CUDA runtime errors produce silent loss=0.0 — **OPEN** (workaround: `CUDA_VISIBLE_DEVICES=""`)
- **ALB-060 (Critical)**: Training ran only 43/5000 steps (epochs=1). Fixed: C-TRAINCFG-001 contract + v2 config (epochs=38, warmup=500). Awaiting retrain.

**350M CUDA test results (50 steps, post ALB-059 fix):**
- Loss: 10.39 → 5.92 (best: 5.53) — clear convergence with correct GEMM backward
- Training time: ~400s (~8s/step)
- Checkpoint: 1.59 GB SafeTensors, 218 tensors, config.json saved
- Checkpoint validation: PASS (weights trained, layers distinct)
- realizar inference: loads model, generates tokens (gibberish at 50 steps — expected)
- Perplexity: 31,926 (finite; random baseline ~32,768 for vocab 32K)

### Good (Phase 5 complete)
- [ ] Distillation from Qwen2.5-Coder-3B demonstrated (interim); Qwen3-Coder-Next 80B (stretch, ALB-010)
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
- [x] Critical path gaps (ALB-001, 006, 009, 011, 018) closed with upstream fixes; ALB-010 (Qwen3-Coder-Next) remains OPEN
- [ ] Models published on HuggingFace as `paiml/albor-python-*`
- [ ] Q4 quantized model < 100MB, runs on consumer hardware
- [ ] **All 8 kernel contracts written and verified** (ALB-013–017, ALB-039–040, ALB-060)
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


# 18. Reference Commands

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

# Unified pipeline (apr pipeline wraps forjar + batuta)
apr pipeline plan configs/pipeline/albor.yaml
apr pipeline apply configs/pipeline/albor.yaml
apr pipeline status

# ═══════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════

# Import local codebases
alimentar import local /path/to/codebase -o data/raw/corpus.parquet

# Weighted mix with upsampling
alimentar mix a.parquet:0.4 b.parquet:0.3 c.parquet:0.15 d.parquet:0.15 \
    -o data/tokenized/train/mixed.parquet --seed 42

# FIM transform
alimentar fim data.parquet -o data-fim.parquet --rate 0.5 --format psm

# Quality profiles
alimentar quality profiles

# ═══════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════

# v1: BPE with apr (whitespace-split — ALB-036 limitation)
apr tokenize plan --data corpus.txt --vocab-size 32768
apr tokenize apply --data corpus.txt --vocab-size 32768 --algorithm bpe -o tokenizer/

# v2: ByteLevel BPE with Python (recommended — preserves whitespace)
python scripts/train-tokenizer-v2.py --corpus corpus.txt --vocab-size 32768 \
    --output models/albor-tokenizer-v2/

# Pre-tokenize for training (bypasses tokenizer format gap ALB-033)
python scripts/pretokenize.py --input data.parquet \
    --tokenizer models/albor-tokenizer-v2/tokenizer.json \
    --seq-len 2048 --output data/pretokenized-2048/train/train.parquet

# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

# Plan (dry-run, validate config)
apr train plan --task pretrain --config configs/train/pretrain-350m.yaml

# Train (execute)
apr train apply --task pretrain --config configs/train/pretrain-350m.yaml

# Makefile shortcuts
make train-50m        # ~2 min on RTX 4090
make train-350m       # ~20 hours on RTX 4090
make training-status  # Check running training

# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

# apr eval (perplexity — ALB-037 FIXED, realizar loads checkpoints)
apr eval checkpoints/albor-base-350m/model.safetensors \
    --dataset custom --text "def foo():" --threshold 30

# Python eval scripts (supplement)
python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --validate-only
python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --api http://localhost:8080
python scripts/eval-perplexity.py checkpoints/albor-base-350m/ \
    --data data/pretokenized-2048/val/val.parquet --seq-len 2048 --threshold 30

# Convert entrenar checkpoint for realizar
python scripts/convert-checkpoint.py checkpoints/albor-base-350m/ \
    --config configs/train/pretrain-350m.yaml

# Makefile shortcuts
make eval-validate           # Validate all benchmark canonical solutions
make eval-perplexity-350m    # Run perplexity eval

# ═══════════════════════════════════════════════════════════
# MONITORING (run in a separate terminal during training)
# ═══════════════════════════════════════════════════════════

bash scripts/monitor-training.sh                     # Training process + GPU + log
apr monitor ./checkpoints/albor-base-350m/           # Live training TUI (ALB-025 FIXED)
apr experiment view --db .entrenar/experiments.db     # Browse past experiments

# ═══════════════════════════════════════════════════════════
# POST-TRAINING (Phases 4-6)
# ═══════════════════════════════════════════════════════════

# Distillation
apr distill --config configs/train/distill.yaml --plan
apr distill --config configs/train/distill.yaml --stage precompute
apr distill --config configs/train/distill.yaml --stage train

# Fine-tuning
apr finetune --plan --model-size 350M --vram 24 --method lora --rank 16

# Model operations
apr merge a.safetensors b.safetensors --strategy slerp -o merged.safetensors
apr prune model.safetensors --method wanda --sparsity 0.5 -o pruned.safetensors
apr quantize model.safetensors --method q4_k -o model.gguf
apr export model.safetensors --format gguf -o model.gguf
apr publish checkpoints/albor-350m/ paiml/albor-base-350m

# ═══════════════════════════════════════════════════════════
# QUALITY (bashrs is KING of linting)
# ═══════════════════════════════════════════════════════════

# bashrs — sovereign linter for all shell artifacts
bashrs make lint Makefile                          # Makefile quality
bashrs classify Makefile                           # Safety classification
bashrs make purify Makefile                        # Deterministic output

# provable-contracts — kernel correctness
pv validate contracts/*.yaml                       # Contract schemas
pv coverage contracts                              # Obligation coverage
pv generate contracts/*.yaml                       # Scaffold + tests + harnesses
pv book contracts/                                 # mdBook pages
pv audit contracts/*.yaml                          # Audit for issues
pv graph contracts/ --format mermaid               # Verification DAG
pv lean contracts/*.yaml                           # Lean 4 theorem stubs

# batuta — falsification
batuta falsify . --format markdown                 # 108-item checklist
batuta oracle --list                               # Stack components
batuta oracle --local                              # Local workspace status

# pmat — code quality (upstream repos)
pmat tdg baseline create                           # TDG baseline
pmat comply check --strict ../aprender

# ═══════════════════════════════════════════════════════════
# VALIDATION (Makefile)
# ═══════════════════════════════════════════════════════════

make validate          # All validation (YAML + contracts + forjar + Makefile)
make lint              # Lint with bashrs
make eval-validate     # Validate benchmark canonical solutions
make dogfood           # Full 12-section dogfooding suite
make book              # Build mdBook
make help              # Show all targets
```

---


# Appendices


# Appendix A: Batuta Oracle Consultation

**Query**: "distributed LLM training across heterogeneous GPUs using sovereign AI stack"

**Response** (2026-03-01):
- Primary: `repartir` (95% confidence) — distributed computing primitives
- Supporting: `entrenar` (70%) — distributed_training pattern
- Supporting: `trueno` (80%) — SIMD/GPU backend for compute acceleration

---


# Appendix B: Stack Version Matrix

*Last verified: 2026-03-02*

| Component | Version | Role in Albor |
|-----------|---------|---------------|
| aprender (`apr`) | 0.4.10 (7c27c2b3) | Unified CLI: train, tokenize, eval, distill, merge, export, publish, pipeline |
| entrenar | 0.7.5 (with local patches: ALB-038/041/043/044 fixes) | Training engine, autograd, CudaTransformerTrainer, optimizers, LoRA |
| trueno | 0.16.1 | SIMD/GPU tensor backend |
| realizar | 0.8.0 | Inference engine (SafeTensors loading, teacher model, eval, serving) |
| alimentar | 0.2.6 | Data pipeline, Parquet I/O, HF Hub import, FIM transforms, mixing |
| repartir | 2.0.3 | Distributed compute (future: gradient sync) |
| forjar | 1.0.0 | Pipeline orchestration (DAG engine, infra + task resources) |
| presentar | 0.3.2 | Training visualization (TUI dashboards, WASM, experiment browser) |
| bashrs (Rash) | 6.65.0 | Makefile lint/purify/classify, shell safety, pipeline command validation |
| batuta | 0.7.2 | Stack orchestration, oracle, falsification (108 checks), playbook DAG engine |
| provable-contracts (`pv`) | 0.1.0 | Design-by-contract YAML specs, Kani proofs, falsification tests |
| pmat | 3.6.1 | TDG scoring, comply check, fault patterns, coverage gaps |
| certeza | latest | Three-tier test effectiveness (unit → property → formal) |
| renacer | latest | Tracing infrastructure (BrickTracer, spans, metric events) |

**Note**: `apr` uses `[patch.crates-io]` to override entrenar/realizar with
local paths. The installed entrenar 0.7.5 includes unpublished fixes for
ALB-038, ALB-041, ALB-043, ALB-044 (gradient flow, buffer sizes, activation
clipping, optimizer hyperparams).


---


# Appendix C: Qwen3-Coder-Next Architecture Details

| Layer Pattern | Count | Description |
|---------------|-------|-------------|
| Gated DeltaNet → MoE | 36 (3 per block × 12 blocks) | Linear attention with gating, routed to 10/512 experts |
| Gated Attention → MoE | 12 (1 per block × 12 blocks) | Standard GQA with gating, routed to 10/512 experts |
| **Total layers** | **48** | |

This hybrid architecture means realizar needs to support:
- DeltaNet (linear attention variant) — likely a new gap
- MoE routing (top-k expert selection) — may partially exist
- Gated variants of both attention types

---


# Appendix D: W5700X Vulkan Validation

The W5700X has been validated with trueno's wgpu backend on **Metal** (macOS)
with documented performance numbers (trueno book, 2026-01-03). The intel box
runs **Linux**, so the backend will be **Vulkan** (not Metal). Vulkan support
for RDNA 1 on Linux via Mesa RADV is mature and well-tested.

**Action item**: Run trueno GPU tests on intel via Vulkan to confirm parity
with Metal benchmarks before relying on W5700X for compute tasks.

---


# Appendix E: Leaderboard Strategy

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

---


# Appendix F: Dogfooding Log

> Living record of tool validation against the Albor repo.
> Updated as gaps are discovered and resolved.

## Summary (2026-03-03)

| Tool | Command | Result | Gap |
|------|---------|--------|-----|
| `pv validate` | `pv validate contracts/*.yaml` | **PASS** (all 7 contracts) | — |
| `pv coverage` | `pv coverage contracts` | **PASS** (100% obligation coverage) | — |
| `pv graph` | `pv graph contracts` | **PASS** (8 nodes, correct deps) | — |
| `pv probar` | `pv probar contracts/*.yaml` | **PASS** (generates property tests) | — |
| `pv kani` | `pv kani contracts/*.yaml` | **PASS** (generates Kani harnesses) | — |
| `pv generate` | `pv generate contracts/*.yaml` | **PASS** (20 files: scaffold, kani, probar, book) | — |
| `pv scaffold` | `pv scaffold contracts/*.yaml` | **PASS** (Rust trait + test stubs) | — |
| `pv status` | `pv status contracts/*.yaml` | **PASS** (equation/obligation counts) | — |
| `pv audit` | `pv audit contracts/*.yaml` | **PASS** (no findings) | — |
| `pv equations` | `pv equations contracts/*.yaml` | **PASS** (formatted equations) | — |
| `pv book` | `pv book contracts/` | **PASS** (7 mdBook pages) | — |
| `pv lean` | `pv lean contracts/*.yaml` | **INFO** (needs `lean:` metadata blocks) | — |
| `forjar validate` | `forjar validate -f infra-only.yaml` | **PASS** (2 machines, 6 resources) | — |
| `forjar validate` | `forjar validate -f albor.yaml` | **PASS** (2 machines, 22 resources) | ~~ALB-027~~ FIXED |
| `forjar graph` | `forjar graph -f infra-only.yaml` | **PASS** (Mermaid output) | — |
| `apr finetune --plan` | `apr finetune --plan --model-size 350M --vram 24` | **PASS** (VRAM estimate correct) | — |
| `apr train plan --task pretrain` | `apr train plan --task pretrain --config pretrain-350m.yaml` | **PASS** (validates config, shows arch/params) | ~~ALB-009~~ FIXED |
| `apr distill --plan` | `apr distill --plan` | **PASS** (file-based mode) | — |
| `apr distill --config --plan` | `apr distill --config distill-entrenar.yaml --plan` | **PASS** (validates config, shows two-stage workflow) | ~~ALB-011~~ FIXED |
| `apr distill --config --plan --json` | `apr distill --config distill-entrenar.yaml --plan --json` | **PASS** (structured JSON with verdict) | ~~ALB-011~~ FIXED |
| `apr distill --config --stage precompute` | `apr distill --config distill-entrenar.yaml --stage precompute` | **PASS** (inspects teacher, 290 tensors, writes manifest) | ~~ALB-011~~ FIXED |
| `apr distill --config --stage train` | `apr distill --config distill-entrenar.yaml --stage train` | **PASS** (reads manifest, validates, sets up KD) | ~~ALB-011~~ FIXED |
| `apr train apply --parquet` | `apr train apply --task pretrain --config pretrain-parquet.yaml` | **PASS** (8 rows from Parquet, 4 batches, CUDA training) | ~~ALB-007~~ FIXED |
| `apr quantize --plan` | `apr quantize --plan <file>` | **PASS** (plan mode works) | — |
| `apr prune --plan` | `apr prune --plan <file>` | **PASS** (plan mode exists) | — |
| `alimentar quality profiles` | `alimentar quality profiles` | **PASS** (ml-training profile exists) | — |
| `alimentar import` | `alimentar import local <in> -o <out>` | **PASS** (local import works) | ~~ALB-019~~ FIXED |
| `alimentar mix` | `alimentar mix a.parquet:0.8 b.parquet:0.2 -o out.parquet` | **PASS** (weighted sampling + upsampling) | ~~ALB-020~~ FIXED |
| `apr tokenize plan` | `apr tokenize plan --data corpus.txt --vocab-size 32000` | **PASS** (validates corpus, estimates time) | ~~ALB-001~~ FIXED |
| `apr tokenize apply` | `apr tokenize apply --data corpus.txt --vocab-size 100` | **PASS** (trains BPE, writes vocab.json + merges.txt) | ~~ALB-001~~ FIXED |
| `alimentar fim` | `alimentar fim data.parquet -o fim.parquet --rate 0.5` | **PASS** (PSM/SPM FIM transform) | ~~ALB-018~~ FIXED |
| `batuta falsify` | `batuta falsify . --format markdown` | **PASS** (108 checks, 73.1% score) | ~~ALB-029~~ FIXED |
| `batuta falsify --critical-only` | `batuta falsify . --critical-only` | **PARTIAL** (3/5 pass, 1 fail) | ~~ALB-029~~ FIXED |
| `batuta stack status` | `batuta stack status --simple` | **PASS** (11 tools detected, 5 healthy) | ~~ALB-030~~ FIXED |
| `batuta oracle --list` | `batuta oracle --list` | **PASS** (lists all 40+ stack components) | — |
| `batuta oracle --recommend` | `batuta oracle --recommend --problem "train 350M LLM"` | **PASS** (recommends aprender) | — |
| `batuta oracle --local` | `batuta oracle --local` | **PASS** (47 PAIML projects discovered) | — |
| `batuta oracle --capabilities` | `batuta oracle --capabilities entrenar` | **PASS** (autograd, lora, qlora, quantization, model_merge, distillation) | — |
| `batuta playbook validate` | `batuta playbook validate albor-playbook.yaml` | **PASS** (19 stages, 14 params, acyclic DAG) | — |
| `batuta hf search` | `batuta hf search model "code completion"` | **PARTIAL** (returns placeholder/mock data) | — |
| `bashrs make lint` | `bashrs make lint Makefile` | **PASS** (2 warnings, 0 errors) | — |
| `bashrs make parse` | `bashrs make parse Makefile` | **PASS** (full AST) | — |
| `bashrs make purify` | `bashrs make purify Makefile` | **PASS** (purified output) | — |
| `bashrs classify` | `bashrs classify Makefile` | **PASS** (safe: 85%) | — |
| `apr pipeline validate` | `apr pipeline validate albor.yaml` | **PASS** (2 machines, 22 resources) | ~~ALB-028~~ FIXED |
| `apr pipeline plan` | `apr pipeline plan albor.yaml` | **PASS** (23 resources, full DAG) | ~~ALB-028~~ FIXED |
| `apr pipeline plan --json` | `apr pipeline plan albor.yaml --json` | **PASS** (structured JSON with deps) | ~~ALB-028~~ FIXED |
| `apr pipeline status` | `apr pipeline status albor.yaml` | **EXPECTED FAIL** (no state dir yet) | — |
| `pmat query` | `pmat query "training"` | **PASS** (0 functions, 5 document matches) | — |
| `pmat analyze makefile` | `pmat analyze makefile Makefile` | **PASS** (64% quality score) | — |
| `pv lean` | `pv lean contracts/kd-v1.yaml` | **PASS** (6 Lean 4 theorem stubs generated) | — |
| `pv lean-status` | `pv lean-status contracts/` | **PASS** (0% L4 coverage, 4 sorry debt) | — |
| `apr train plan --task classify` | `apr train plan --data <JSONL>` | **PASS** (classification fine-tuning) | — |
| `apr merge` | `apr merge --strategy slerp` | **PASS** (SLERP, TIES, DARE supported) | — |
| `apr export --list-formats` | `apr export --list-formats` | **PASS** (SafeTensors, GGUF, MLX) | — |
| `apr publish` | `apr publish <dir> <repo>` | **PASS** (HF Hub publish exists) | — |
| `apr eval` | `apr eval <model>` | **PASS** (perplexity eval) | — |
| `apr eval --task code` | `apr eval model --task code --data bench.jsonl` | **PASS** (pass@1 scoring, 10/10 on basic set) | ~~ALB-006~~ FIXED |
| `apr eval --task plan` | `apr eval model --task plan --data bench.jsonl` | **PASS** (dry-run validation) | ~~ALB-006~~ FIXED |
| `alimentar mix` (test) | `alimentar mix ...parquet:0.25 -o test.parquet -n 200 --seed 456` | **PASS** (200 rows, 50 per corpus) | — |
| `alimentar fim` (prod) | `alimentar fim mixed.parquet -o mixed-fim.parquet --rate 0.5 --format psm` | **PASS** (17,070 rows, PSM FIM 50%) | — |
| `apr tokenize apply` (prod) | `apr tokenize apply --data corpus-raw.txt --vocab-size 32768 --algorithm bpe -o tokenizer/ --max-lines 100000` | **PASS** (32,768 vocab, 2022.5s, 8/8 Python patterns) | ~~ALB-001~~ FIXED |
| `alimentar quality` | `alimentar quality profiles` | **PASS** (ml-training profile) | — |
| `alimentar convert` | `alimentar convert` | **PASS** (format conversion) | — |
| `bashrs score` | `bashrs score Makefile` | **PASS** (D grade, 5.2/10) | — |
| `bashrs audit` | `bashrs audit Makefile` | **PASS** (comprehensive audit) | — |
| `entrenar train` (50M) | `entrenar train pretrain-50m-test.yaml` | **PASS** (demo batches, 465ms, loss 10.34→9.67) | ALB-033 (tokenizer format) |
| `apr train apply` (50M) | `apr train apply --task pretrain --config pretrain-50m-test.yaml` | **PASS** (10-row micro, 5 batches, 2.1s CUDA) | ~~ALB-034~~ FIXED |
| `apr train apply` (50M full) | `apr train apply --task pretrain --config pretrain-50m.yaml` | **PASS** (500 rows, 125 batches, 31 steps, 110.7s CUDA, loss 10.3→4.42) | ~~ALB-034~~ FIXED |
| `apr train apply` (50M v2) | `apr train apply --task pretrain --config pretrain-50m-v2.yaml` | **PASS** (pre-tokenized ByteLevel BPE, 108.5s CUDA, loss→5.51) | — |
| `apr train plan` (350M) | `apr train plan --task pretrain --config pretrain-350m.yaml` | **PASS** (config validated, ready for apply) | — |
| `entrenar validate` | `entrenar validate pretrain-350m-manifest.yaml` | **PASS** (architecture overrides bridge through) | ~~ALB-021~~ FIXED |
| `entrenar shorthand` | `vocab_size: "32K"` in YAML manifest | **PASS** (parses to 32768) | ~~ALB-022~~ FIXED |
| `apr merge --plan` | `apr merge a.apr b.apr --plan --strategy slerp -o merged.apr` | **PASS** (validates inputs, shows strategy, sizes) | ~~ALB-023~~ FIXED |
| `apr export --plan` | `apr export model.apr --plan --format gguf -o model.gguf` | **PASS** (validates format, shows plan) | ~~ALB-023~~ FIXED |
| `apr publish --plan` | `apr publish dir repo --plan` | **PASS** (alias for --dry-run) | ~~ALB-023~~ FIXED |
| `apr train apply` (350M full) | `apr train apply --task pretrain --config pretrain-350m.yaml` | **FAIL** (ALB-060: epochs=1 exhausted data at step 43/5000, loss flat ~10.39, LR still in warmup at 6.45e-6) | ALB-060 |
| `apr train apply` (350M v2) | `apr train apply --task pretrain --config pretrain-350m-v2.yaml` | **PASS** (ALB-065 fixed: `stream.synchronize()` before D2H gradient transfers. Training stable without `CUDA_LAUNCH_BLOCKING=1`, 441 tok/s) | ~~ALB-064~~ ~~ALB-065~~ FIXED |
| `train-guard.sh` | `bash scripts/train-guard.sh configs/train/pretrain-350m-v2.yaml` | **PASS** (crash-resilient supervisor with auto-diagnostic CUDA blocking mode, exit code classification, GPU state capture, JSON crash reports, backoff restart, heartbeat monitoring) | ~~ALB-064~~ FIXED |
| `pv validate` (memory) | `pv validate contracts/training-memory-kernel-v1.yaml` | **PASS** (0 errors, 0 warnings) | ALB-039 |
| `pv validate` (GPU) | `pv validate contracts/training-gpu-kernel-v1.yaml` | **PASS** (0 errors, 0 warnings) | ALB-040 |
| `apr train apply` (50M CUDA) | `apr train apply --config pretrain-50m-v2-test.yaml` | **PASS** (3 steps, loss 10.4→11.7, GPU forward+backward) | ~~ALB-041~~ FIXED |
| `apr eval` (50M safetensors) | `apr eval checkpoints/albor-base-50m/model.safetensors --dataset custom` | **FAIL** (PPL 679,614 — weights ignored) | ~~ALB-037~~ FIXED |
| `apr train apply` (350M CUDA test) | `apr train apply --config pretrain-350m-cuda-test.yaml` | **PASS** (50 steps, ~400s, loss 10.39→5.92, best 5.53, checkpoint saved) | ~~ALB-043~~ ~~ALB-044~~ ~~ALB-059~~ FIXED |
| `realizar run` (350M) | `realizar run checkpoints/albor-350m-cuda-test/model.safetensors "def fibonacci(" --raw` | **PASS** (218 tensors loaded, 50 tokens generated, 1.0 tok/s) | ~~ALB-037~~ FIXED |
| `eval-perplexity.py` (350M validate) | `python scripts/eval-perplexity.py checkpoints/albor-350m-cuda-test/ --validate-checkpoint` | **PASS** (weights trained, layers distinct) | — |
| `eval-perplexity.py` (350M perplexity) | `python scripts/eval-perplexity.py checkpoints/albor-350m-cuda-test/ --data val.parquet --max-sequences 3 --seq-len 64` | **PASS** (PPL 31,926 — finite, consistent with 50-step model) | — |
| `eval-code.py` (validate) | `python scripts/eval-code.py configs/eval/python-intermediate.jsonl --validate-only` | **PASS** (15/15 canonical solutions) | — |
| `eval-code.py` (HumanEval) | `python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --validate-only` | **PASS** (20/20 canonical solutions) | — |
| `convert-checkpoint.py` (50M) | `python scripts/convert-checkpoint.py checkpoints/albor-base-50m/` | **PASS** (110→111 tensors, 85 reshaped, lm_head created) | ALB-037 |
| `eval-perplexity.py --validate` | `python scripts/eval-perplexity.py checkpoints/albor-base-50m/ --validate-checkpoint` | **FAIL** → **FIXED** (ALB-038 root cause in autograd) | ~~ALB-038~~ FIXED |
| checkpoint analysis | byte-compare layers 0-11 q_proj, gate_proj | **FAIL** → **FIXED** (all parameters now receive gradients) | ~~ALB-038~~ FIXED |
| `apr monitor` (TUI) | `apr monitor checkpoints/albor-base-350m/` | **PASS** (presentar TUI, live GPU telemetry, loss curve, tok/s) | ~~ALB-045~~ ~~ALB-046~~ ~~ALB-047~~ ~~ALB-048~~ FIXED |
| `apr monitor --json` | `apr monitor --json checkpoints/albor-base-350m/` | **PASS** (headless JSON with full TUI parity) | ~~ALB-053~~ ~~ALB-058~~ FIXED |
| `apr monitor` (discover) | `apr monitor` (no args) | **PASS** (discovers active runs from global SQLite registry) | ~~ALB-054~~ FIXED |
| `apr train apply` (SQLite) | `apr train apply --config pretrain-50m-quick.yaml` | **PASS** (creates both local + global experiments.db, logs params + metrics) | ~~ALB-055~~ ~~ALB-056~~ FIXED |
| `apr runs ls --global` | `apr runs ls --global` | **PASS** (table output: experiment, run ID, status, loss, tok/s, duration) | ~~ALB-050~~ FIXED |
| `apr runs ls --global --json` | `apr runs ls --global --json` | **PASS** (JSON array with all run metadata) | ~~ALB-050~~ FIXED |
| `apr runs show` | `apr runs show <id> --global` | **PASS** (params, loss, tok/s, lr, duration) | ~~ALB-050~~ FIXED |
| `apr runs show --json` | `apr runs show <id> --global --json` | **PASS** (clean JSON with native param values) | ~~ALB-050~~ FIXED |
| `realizar run` (350M v2) | `realizar run checkpoints/albor-350m-cuda-test/model.safetensors "def fibonacci("` | **PASS** (24 layers, 32768 vocab, 50 tokens, 1.9 tok/s, garbage output expected from 5-step model) | — |
| `pv audit` (all) | `pv audit contracts/*.yaml` (7 contracts) | **PASS** (0 findings, 22 equations, 43 obligations, 26 falsification tests) | — |
| `batuta falsify --critical-only` | `batuta falsify . --critical-only` | **PARTIAL** (3/5 pass, 80.0% score, AI-01/AI-05 partial) | — |
| `apr runs diff` | `apr runs diff <a> <b> --global` | **PASS** (side-by-side sparklines, config diff, loss comparison, verdict) | ~~ALB-051~~ FIXED |
| `apr runs diff --json` | `apr runs diff <a> <b> --global --json` | **PASS** (structured JSON: summaries, config_diff, verdict for LLM agents) | ~~ALB-051~~ FIXED |
| `apr monitor` (widget composition) | `TrainingDashboard` composes `Layout`, `Border`, `Meter`, `GpuPanel`, `Sparkline`, `Text` | **PASS** (builds clean, widget tree rebuilt each frame, panel verification wired) | ~~ALB-057~~ FIXED |
| `apr experiment view --global --json` | `apr experiment view --global --json` | **PASS** (JSON output with experiments, run_ids, loss_values, params from SQLite) | ~~ALB-024~~ FIXED |
| `apr experiment view --global` | `apr experiment view --global` | **PASS** (ratatui TUI: run table, sparkline, braille loss chart, j/k navigation) | ~~ALB-024~~ FIXED |
| `pv validate` (training-config) | `pv validate contracts/training-config-kernel-v1.yaml` | **PASS** (0 errors, 8 obligations, 5 falsification tests, 2 Kani harnesses) | ALB-060 |
| `pv coverage` (all 8 contracts) | `pv coverage contracts/` | **PASS** (8 contracts, 31 equations, 51 obligations, 34 falsification tests, 100% coverage) | — |
| `apr train apply` (50M post-fix) | `apr train apply --config pretrain-50m-quick.yaml` | **PASS** (5 steps, loss 10.42→9.45, GEMM backward now correct) | ~~ALB-059~~ FIXED |
| `apr train apply` (350M post-fix) | `apr train apply --config pretrain-350m-cuda-test.yaml` | **PASS** (50 steps, loss 10.39→5.92, best 5.53, zero NaN, correct backward gradients) | ~~ALB-059~~ FIXED |
| `realizar run` (350M post-fix) | `realizar run checkpoints/albor-350m-cuda-test/model.safetensors "def fibonacci("` | **PASS** (218 tensors, generates tokens from correctly-trained weights) | ~~ALB-059~~ FIXED |
| `apr quantize` (50M int4) | `apr quantize model.safetensors -s int4` | **PASS** (238 MiB → 30 MiB, 87.5% reduction, 7.99x) | — |
| `apr quantize` (50M q4k) | `apr quantize model.safetensors -s q4k` | **PASS** (238 MiB → 238 MiB, 0% reduction — q4k no-op on 1D tensors) | — |
| `apr quantize` (350M int4) | `apr quantize model.safetensors -s int4` | **PASS** (1.48 GiB → 191 MiB, 87.5% reduction, 7.99x) | — |
| `apr quantize` (350M q4k) | `apr quantize model.safetensors -s q4k` | **PASS** (1.48 GiB → 1.48 GiB, 0% reduction — q4k no-op on 1D tensors) | — |
| `apr prune` (50M magnitude) | `apr prune model.safetensors --method magnitude --sparsity 0.5` | **PASS** (50.0% zeros, 31.2M/62.4M params zeroed) | — |
| `apr prune` (50M depth) | `apr prune model.safetensors --method depth --remove-layers "8-11"` | **PASS** (110→74 tensors, 238→180 MiB, layers 8-11 removed) | — |
| `apr prune` (350M magnitude) | `apr prune model.safetensors --method magnitude --sparsity 0.3` | **PASS** (50.0% zeros — sparsity param may be ignored) | — |
| `source-to-parquet.py` (Tier 2) | `python scripts/source-to-parquet.py ~/src/pytorch pytorch data/parquet/tier2/pytorch.parquet` | **PASS** (8 repos → 28,553 Python files imported) | — |
| `alimentar mix` (expanded) | `alimentar mix ...T1:10.0 ...T2:1.0 -o mixed.parquet --seed 42` | **PASS** (12 datasets → 45,420 rows, proportional weighted sampling) | — |
| `alimentar fim` (expanded) | `alimentar fim mixed.parquet -o mixed-fim.parquet --rate 0.5 --format psm` | **PASS** (45,420 rows, 50% PSM FIM) | — |
| `pretokenize.py` (v2) | `python scripts/pretokenize.py --input mixed-fim.parquet --seq-len 2048` | **PASS** (67,977 sequences, 139M tokens, 191 MiB) | — |
| `realizar run` (0.5B teacher) | `realizar run qwen2.5-coder-0.5b/model.safetensors "def fibonacci("` | **PASS** (24 layers, 151936 vocab, 2.8 tok/s, generates tokens) | — |
| `apr distill --stage precompute` (0.5B) | `apr distill --config distill-entrenar.yaml --stage precompute` | **PASS** (290 tensors, 942 MiB, manifest written) | — |
| `apr distill --stage precompute` (3B) | `apr distill --config distill-qwen3b.yaml --stage precompute` | **PASS** (434 tensors, 5.75 GiB, sharded SafeTensors loaded) | — |
| `realizar run` (3B sharded) | `realizar run qwen2.5-coder-3b/model-00001-of-00002.safetensors` | **FAIL** (sharded SafeTensors not supported — model.norm.weight in shard 2) | — |
| C-TRAINCFG-001 pre-flight (v2) | `python3 -c "..."` (algebraic check) | **PASS** (67977 seqs, 132 steps/epoch, 38 epochs, warmup=500=10%) | ALB-060 |

## ALB-060: Training Config Epoch/Step Mismatch (Critical)

**Discovery**: The 350M "full training" run completed in 11.8 seconds instead of
the expected 12+ hours, producing an effectively untrained model.

**Five Whys (per CLAUDE.md Rule 7)**:

1. **Why did loss stay flat at ~10.39?** The learning rate never reached a meaningful
   value — max LR achieved was 6.45e-6 vs target 3e-4.
2. **Why was LR so low?** The warmup schedule is linear over 2000 steps, but training
   only ran 43 steps. At step 43: lr = 3e-4 × (43/2000) = 6.45e-6.
3. **Why only 43 steps?** `steps_per_epoch = floor(22079 / 4 / 128) = 43`. With
   `epochs: 1`, total achievable steps = 43. `max_steps: 5000` is unreachable.
4. **Why only 1 epoch?** The config comment says "Pre-training uses max_steps, not epochs"
   but entrenar's training loop respects `epochs` as a hard cap — it does NOT loop
   data to fill `max_steps`.
5. **Why no validation?** No pre-flight check computes `steps_per_epoch` and compares
   against `max_steps` + `warmup_steps`. The algebraic inconsistency is invisible.

**Algebraic proof (from C-TRAINCFG-001 contract)**:
```
num_sequences       = 22,079
micro_batch_size    = 4
grad_accum_steps    = 128
steps_per_epoch     = floor(22079 / 4 / 128) = 43
total_achievable    = 1 × 43 = 43
max_steps           = 5,000       ← UNREACHABLE
warmup_steps        = 2,000       ← NEVER COMPLETES
tokens_trained      = 43 × 4 × 128 × 1024 = 22.5M
chinchilla_min      = 10 × 370M = 3.7B   ← undertrained by 164×
```

**Fix required (two options)**:
1. Set `epochs: 117` (ceil(5000/43)) to cycle data 117 times → reaches 5031 steps
2. Add epoch-looping to entrenar: when `max_steps` is set and epochs exhausted,
   reshuffle data and continue (treats `max_steps` as authoritative, `epochs` as informational)

**Contract**: `contracts/training-config-kernel-v1.yaml` (C-TRAINCFG-001) with
7 equations, 8 proof obligations, 5 falsification tests, 2 Kani harnesses.
FALSIFY-CFG-001 and FALSIFY-CFG-002 algebraically prove this config is invalid.

**Training state.json analysis**: The `loss_history` array (55 entries, all ~10.39-10.40)
and `learning_rate: 0.0` confirm the model never learned. The `status: "Running"` field
is stale (training completed but status was not updated to "Completed" — minor bug).

**Secondary bug**: The training log displays `loss=0.0000` for every step despite
training_state.json recording real loss values ~10.39. This is the known ALB-042
display bug (loss=0.0 reporting).

## Contract Validation Detail

All 8 contracts pass `pv validate` with 0 errors. The original 5 were rewritten from
a custom schema to match `pv`'s schema (`metadata:`, `formula:`, `proof_obligations:`,
`falsification_tests:`). The two training kernel contracts (ALB-039, ALB-040) and the
training config contract (ALB-060) were written directly in the correct schema.

```
pv coverage contracts
---------------------
Contracts:            8
Equations:            31
Obligations:          51
Falsification tests:  34
Kani harnesses:       10
Overall coverage:     100.0%
```

## pv generate Detail

`pv generate` produces 4 files per contract (28 total):

| Type | Content | Example |
|------|---------|---------|
| `*_scaffold.rs` | Rust trait with documented invariants | `knowledge-distillation-kernel-v1_scaffold.rs` |
| `*_probar.rs` | Property tests derived from proof obligations | 6 property tests + 5 falsification test stubs |
| `*_kani.rs` | Kani verification harnesses | 2 harnesses with `stub_float` strategy |
| `*_book.md` | mdBook page with equations, deps, obligations | Mermaid dependency graph, LaTeX equations |

`pv book contracts/` generates 7 contract pages directly into mdBook format.
These have been integrated into the albor mdBook under "Kernel Contracts".

## Pipeline Manifest Validation Detail

The full pipeline manifest (`configs/pipeline/albor.yaml`) now passes `forjar validate`
after the ALB-027 fix added the `task` resource type:

```
forjar validate -f configs/pipeline/albor.yaml
OK: albor-training-pipeline (2 machines, 22 resources)
```

Forjar supports all 13 resource types: `package`, `file`, `service`, `mount`, `user`,
`docker`, `pepita`, `network`, `cron`, `recipe`, `model`, `gpu`, `task`.

The `task` resource type is the key piece that turns forjar from an infrastructure
tool into a pipeline orchestrator — it runs arbitrary commands with idempotency
tracking via output artifact hashing.

### Spec Correction: `names` to `packages`

Dogfooding revealed that the spec used `names:` for forjar package resources, but
forjar expects `packages:`. Also requires `provider: apt` (not implicit). Both the
spec and configs were corrected.

## Batuta Playbook Detail

Created `configs/pipeline/albor-playbook.yaml` -- a batuta playbook that expresses
the full albor ML pipeline as a 19-stage deterministic DAG with BLAKE3 caching:

```
batuta playbook validate configs/pipeline/albor-playbook.yaml
Playbook 'albor-training-pipeline' is valid
  Stages: 19
  Params: 14
```

Stages: validate-contracts, validate-configs, data-download, data-tokenize, data-mix,
pretrain, eval-base, teacher-logits, distill, eval-distill, finetune, eval-sft,
merge, eval-merged, prune, eval-pruned, quantize, eval-q4, publish.

This playbook is the **actual executable pipeline** (once upstream gaps are resolved).
The forjar manifest handles infrastructure; the batuta playbook handles ML orchestration.

## Batuta Falsification Detail (Full Report)

`batuta falsify . --format markdown` runs 108 checks across 10 categories:

| Category | Passed | Failed | Partial | Total |
|----------|--------|--------|---------|-------|
| Numerical Reproducibility | 13 | 0 | 2 | 15 |
| Jidoka Automated Gates | 4 | 5 | 1 | 10 |
| Architectural Invariants | 1 | 3 | 1 | 5 |
| Performance & Waste Elimination | 7 | 0 | 8 | 15 |
| ML Technical Debt Prevention | 2 | 1 | 7 | 10 |
| Hypothesis-Driven Development | 5 | 0 | 8 | 13 |
| Sovereign Data Governance | 12 | 0 | 3 | 15 |
| Cross-Platform & API | 2 | 0 | 3 | 5 |
| Safety & Formal Verification | 5 | 1 | 4 | 10 |
| Model Cards & Auditability | 3 | 0 | 7 | 10 |

**Before ALB-029 fix:** Score 72.2% (58 pass, 10 fail, 40 partial).

**After ALB-029 fix:** Score 73.1% (55 pass, 5 fail, 48 partial).

Upstream fixes resolved AI-01 (configs/ glob), AI-04 (book-output/ exclusion),
and AI-05 (non-Rust schema detection via pv/forjar).
Full report saved to `docs/falsification-report.md`.

## bashrs Makefile Linting Detail

bashrs `make lint` is the **sovereign Makefile linter** -- it validates
Makefile quality, safety, and best practices:

```
bashrs make lint Makefile
  MAKE010: Command 'rm' missing error handling
  MAKE015: Missing .DELETE_ON_ERROR
bashrs classify Makefile
  safe: 85.0%
```

Both warnings were addressed. bashrs also provides:
- `bashrs make parse` -- full Makefile AST
- `bashrs make purify` -- deterministic + idempotent Makefile output
- `bashrs classify` -- safety classification with multi-label support

## apr train plan/apply Detail

`apr train plan/apply` exists but is currently scoped to **classification fine-tuning
with HPO** (Tree-of-Parzen Estimators):

```
Current:  apr train plan --data <JSONL> --model-size 0.5B --task classify
Target:   apr train plan configs/train/pretrain-350m.yaml
```

The plan/apply infrastructure is solid -- `apr train plan` generates structured
summaries with resource estimates. The gap (ALB-009) is in scope: extending from
classification to causal LM pre-training, and from flag-driven to config-file-driven.

## Upstream Fixes Implemented

Dogfooding cycle 2 identified gaps that were **fixed upstream** and verified:

### ALB-029: batuta falsify false positives (FIXED)

Three fixes in `batuta/src/falsification/`:

1. **AI-01**: Added `configs/**` glob pattern (plural) alongside `config/**` in `invariants.rs`
2. **AI-04**: Added `book-output/` to JS exclusion list in `is_excluded_js_path()`
3. **AI-05**: Extended `detect_schema_deps()` to detect non-Rust validation:
   - pv/forjar validation commands in Makefile and CI configs
   - Python validation libs (pydantic, marshmallow, cerberus)
   - pv contracts (YAML with `proof_obligations:` key)

Commit: `batuta@905a862` → Score improved from 72.2% to 73.1%.

### ALB-030: batuta stack status without Cargo.toml (FIXED)

`DependencyGraph::from_workspace()` now falls back to binary detection
when no Cargo.toml exists. Discovers installed PAIML binaries via `which`,
extracts versions from `--version` output.

Commit: `batuta@371557a` → `batuta stack status` works in albor.

### ALB-019: alimentar import subcommand (FIXED)

Made `Import` command always available (not feature-gated behind `hf-hub`).
Added `alimentar import local <input> -o <output>` for local file import
with format conversion (CSV, JSON, JSONL, Parquet).

Commit: `alimentar@265541b` → `alimentar import local` works.

### ALB-020: alimentar mix subcommand (FIXED)

Added `alimentar mix` with weighted sampling and upsampling. Supports
`file:weight` syntax for weighted input, deterministic seeding, and
efficient Arrow batch processing with `arrow::compute::take`.

Commit: `alimentar@64b1e92` → `alimentar mix` works.

### ALB-001: apr tokenize plan/apply (FIXED)

Added `apr tokenize plan/apply` subcommands for BPE vocabulary training:
- `plan` validates corpus (lines, bytes, unique chars), estimates training time
- `apply` trains BPE/WordPiece/Unigram tokenizer, writes `vocab.json` + `merges.txt`
- Supports text, JSON, and YAML output formats for plan

Commit: `aprender@90427205` → `apr tokenize plan/apply` works.

### ALB-018: Fill-in-the-Middle (FIM) data transform (FIXED)

Added `alimentar fim` subcommand and `Fim` transform implementing PSM/SPM
FIM formats (Bavarian et al. 2022). Features:
- Configurable FIM rate (probability per row)
- PSM and SPM format variants
- Custom sentinel tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`)
- Deterministic with seed, respects char boundaries
- Rows below `min_chars` threshold left unchanged
- 10 unit tests

Commit: `alimentar@290582d` → `alimentar fim` works.

### ALB-021: Custom model architecture params in YAML (FIXED)

Added `ArchitectureOverrides` to `ModelRef` in entrenar's config schema.
The bridge converter (`manifest_to_spec`) now maps YAML manifest
`architecture:` fields to overrides that are applied on top of the
resolved `TransformerConfig` (from `config.json` or demo defaults).

Supported override fields: `hidden_size`, `num_hidden_layers`,
`num_attention_heads`, `num_kv_heads`, `intermediate_size`, `vocab_size`,
`max_position_embeddings`, `rms_norm_eps`, `rope_theta`, `use_bias`.

The YAML manifest `ArchitectureConfig` also gained serde aliases
(`num_hidden_layers` → `num_layers`, `num_attention_heads` → `num_heads`,
`num_key_value_heads` → `num_kv_heads`, `max_position_embeddings` → `max_seq_length`)
for compatibility with HuggingFace config.json field names.

Commit: `entrenar@a414861` → Architecture overrides work end-to-end.

### ALB-022: Human-readable value shorthand in YAML configs (FIXED)

Added `shorthand` module with `parse_human_usize()` and
`deserialize_human_usize_opt` custom serde deserializer. Supports:

- **SI suffixes (binary)**: `32K` (32×1024), `1M` (1×1024²), `1G` (1×1024³)
- **SI suffixes (decimal)**: `10B` (10×10⁹), `1T` (1×10¹²)
- **Scientific notation**: `1e6`, `3.2e4`
- **Fractional suffixes**: `1.5K` (1536)
- **Plain numbers**: `1024`, `32768`
- **YAML underscore notation**: `32_768` (already native)

K/M/G use binary (powers of 2) since they're used for model dimensions.
B/T use decimal since they're used for token/parameter counts.

Applied to `ArchitectureConfig` fields (`hidden_size`, `num_layers`, `num_heads`,
`num_kv_heads`, `intermediate_size`, `vocab_size`, `max_seq_length`) and
`DataConfig` fields (`seq_len`, `max_length`).

Commit: `entrenar@1cb0950` → Shorthand deserialization works.

### ALB-006: apr eval benchmark harness (FIXED)

Added `--task code` for code completion benchmarks and `--task plan` for
dry-run validation to `apr eval`. Code evaluation uses JSONL format:

```json
{"task_id": "add", "prompt": "def add(a, b):\n", "test": "assert add(1, 2) == 3", "canonical_solution": "    return a + b\n"}
```

Reports pass@1 rate with per-problem PASS/FAIL breakdown. JSON output
mode supported for CI integration.

Phase 1 (current): validates benchmark structure, checks canonical solutions.
Phase 2 (requires ALB-009 inference): generates completions via realizar engine.

Sample benchmark: `configs/eval/python-basic.jsonl` (10 problems).

Commit: `aprender@4e61297e` → `apr eval --task code` works.

### ALB-009: apr train plan/apply for causal LM pre-training (FIXED)

Extended `apr train plan/apply` from classification-only to support causal LM
pre-training via YAML config files:

- **`apr train plan --task pretrain --config <yaml>`**: Loads config via
  `entrenar::config::load_config()`, validates with `validate_config()`,
  displays model architecture, data config, optimizer, and training params.
  JSON output supported for CI integration.
- **`apr train apply --task pretrain --config <yaml>`**: Calls
  `entrenar::config::train_from_yaml()` which routes to TransformerTrainer
  with CausalLMLoss for next-token prediction training.

The albor pretrain config (`configs/train/pretrain-350m.yaml`) was updated
to match entrenar's `TrainSpec` schema: `model.path`, `model.mode: transformer`,
`model.architecture` overrides, `training.mode: causal_lm`.

Entrenar's training infrastructure was already ~90% ready:
- `CausalLMLoss` for next-token prediction loss
- `TransformerTrainer` with gradient accumulation, mixed precision
- `TrainSpec` YAML schema with `ModelMode::Transformer` and `TrainingMode::CausalLm`

The gap was in the CLI routing — `apr train` only accepted `--task classify`.

Commit: `aprender@d79ed943` → `apr train plan --task pretrain` works.

### ALB-011: apr distill config-driven two-stage workflow (FIXED)

Added `--config <yaml>` and `--stage <precompute|train>` to `apr distill`:

- **`apr distill --config <yaml> --plan`**: Loads YAML config, validates all
  sections (teacher, student, distillation, training, dataset, output),
  checks teacher/dataset existence on disk, displays two-stage workflow
  instructions. JSON output supported.
- **`apr distill --config <yaml> --stage precompute`**: Inspects teacher model
  via RosettaStone (supports SafeTensors, APR, GGUF model dirs), writes
  `manifest.json` with tensor count and model stats for stage 2.
- **`apr distill --config <yaml> --stage train`**: Reads precompute manifest,
  validates teacher was precomputed, inspects student model, writes training
  metadata to `student/training_metadata.json`.

Local `DistillYamlConfig` types match entrenar's `DistillationYamlConfig`
schema (teacher/student model IDs, LoRA config, KD temperature/alpha,
progressive/attention transfer options, training hyperparams, dataset config).
Uses `serde_yaml_ng` for YAML parsing.

Teacher model changed from required positional to `Option<PathBuf>` — config
mode doesn't need the positional arg. Existing file-based distillation mode
(positional teacher.apr, --student, -o) fully preserved.

Albor config: `configs/train/distill-entrenar.yaml` (Qwen2.5-Coder-0.5B teacher,
albor-base-350m student, LoRA rank 16, T=4.0, α=0.5).

Commit: `aprender@81dd4432` → All 3 config modes work (plan, precompute, train).

### ALB-028: apr pipeline plan/apply/status/validate (FIXED)

Added `apr pipeline` subcommand wrapping forjar's DAG engine:

- **`apr pipeline plan <manifest>`**: Shows full execution plan with resource
  DAG, dependency ordering, and per-machine breakdown. Supports `--json`,
  `--machine`, `--tag`, `--cost` flags.
- **`apr pipeline apply <manifest>`**: Converges resources via forjar engine.
  Supports `--parallel`, `--keep-going`, `--machine`, `--tag`.
- **`apr pipeline status <manifest>`**: Shows converged/pending/failed state
  from forjar lock files.
- **`apr pipeline validate <manifest>`**: Validates manifest without connecting
  to machines.

Implementation shells out to the `forjar` binary (keeping sovereign stack
tools decoupled). Follows the train/tokenize plan/apply subcommand pattern.

Commit: `aprender@e653d5ca` → All 4 subcommands work, plan shows 23 resources
across 2 machines (lambda, intel).

### ALB-027: forjar task resource type (FIXED)

Added `task` resource type to forjar for pipeline orchestration. Three handlers:

1. **`check_script`**: If `completion_check` set, runs it (exit 0 = done).
   If `output_artifacts` set, checks all exist. Otherwise reports pending.
2. **`apply_script`**: Runs `command` with `set -euo pipefail`. Supports
   `working_dir` (cd before exec) and `timeout` (wraps with `timeout N`).
3. **`state_query_script`**: Hashes `output_artifacts` via `b3sum` for drift
   detection. Falls back to echoing command string if no artifacts.

Validation: `command` field required, `timeout` must be > 0 if set.

New Resource fields: `output_artifacts`, `completion_check`, `timeout`,
`working_dir`. Reuses existing `command` field (shared with cron).

Commit: `forjar@d14e633` → `forjar validate -f albor.yaml` passes (2 machines, 22 resources).

### ALB-023: Plan/apply contract for all apr subcommands (FIXED)

Added `--plan` flag to the remaining action commands that lacked plan mode:

- **`apr merge --plan`**: Validates input files exist, parses strategy, validates
  weights, shows model count and total input size. Exits 0 on valid, non-zero on error.
- **`apr export --plan`**: Validates model file exists, format is supported,
  shows input size and target format. Supports batch mode plan.
- **`apr publish --plan`**: Alias for existing `--dry-run`. Preview model card
  and file list without uploading.

Pre-dispatch contract validation (RosettaStone tensor checks) is now skipped
in plan mode to allow plan on empty/placeholder files.

Full coverage audit:
| Command | Plan Mode | Type |
|---------|-----------|------|
| train | plan/apply subcommands | Pre-existing |
| tokenize | plan/apply subcommands | Pre-existing |
| quantize | --plan flag | Pre-existing |
| finetune | --plan flag | Pre-existing |
| prune | --plan flag | Pre-existing |
| distill | --plan flag | Pre-existing |
| eval | --task plan | Pre-existing |
| merge | --plan flag | **New** |
| export | --plan flag | **New** |
| publish | --plan flag | **New** |

Commit: `aprender@526a1e4b` → All action commands have plan mode.

## ALB-007: Parquet→LMBatch Bridge (Upstream Fix)

**Gap**: entrenar's `load_lm_batches_from_parquet()` was a stub that returned demo data.
The Parquet-to-training bridge was missing — alimentar produces Arrow RecordBatch,
entrenar consumes `LMBatch(Vec<u32>)`.

**Fix** (`entrenar@a5a2fb7`):
- Text column Parquet: extracts text column → tokenizes with HfTokenizer → LMBatch
- Pre-tokenized Parquet: reads `input_ids`/`token_ids` List<u32> directly → LMBatch
- Directory support: iterates all `.parquet` shards in a directory
- Column auto-detection: tries specified column, then text/content/code fallbacks
- Gated behind `parquet` feature flag (alimentar + arrow deps)
- `apr-cli` Cargo.toml updated to enable `entrenar/parquet` feature

**Dogfood result**:
```
apr train apply --task pretrain --config configs/train/pretrain-parquet.yaml

  Loading 1 Parquet shard(s) from ./data/tokenized/train/
  Loaded 8 rows from Parquet
  Extracted 8 text rows, tokenizing...
  Tokenized 8 sequences
  4 LM batches created
  Epoch 1/1: loss=12.05
```

`apr-cli` Cargo.toml: `entrenar = { version = "0.7.3", features = ["cuda", "parquet"] }`
Commit: `aprender@` (pending push)

## ALB-064: Training Process Silent Death (Critical)

**Discovery**: 350M v2 training (2026-03-03) started successfully, logged step 0
(loss=10.3933, 11.85 GB VRAM), then silently died. No error in stdout/stderr, no
crash log, no backtrace, no dmesg OOM entry. Process gone, `training_state.json`
still shows `"status": "Running"`. Repeated on second attempt.

**Five Whys**:

| Why | Finding | Brick Boundary |
|-----|---------|----------------|
| **Why did training fail?** | Unknown — process exited with no output | Per-process: PID gone, GPU memory freed |
| **Why no error output?** | CUDA driver errors → SIGABRT/SIGSEGV → bypasses Rust panic handler | Per-transfer: driver crash kills process instantly |
| **Why no crash handling?** | No signal handler, no watchdog, no crash recovery | System level: no supervision infrastructure |
| **Why no watchdog?** | Training assumed to work or print errors | Architectural gap: no defensive monitoring |
| **Why no defensive monitoring?** | Pipeline lacks production process supervision | **Root cause**: zero crash resilience infrastructure |

**Fix**: `scripts/train-guard.sh` — crash-resilient training supervisor implementing
patterns from Meta (Llama 3: 466 restarts in 54 days), ByteDance (ByteRobust),
Amazon (FlashRecovery), and systemd:

| Feature | Implementation |
|---------|---------------|
| Exit code classification | SIGSEGV=139→restartable, SIGKILL=137→OOM, SIGBUS=135→fatal |
| GPU state capture | nvidia-smi queries + Xid error detection + dmesg OOM check |
| Structured crash reports | JSON to `crash-reports/` with exit code, signal, GPU state, last step/loss |
| Exponential backoff | 30s → 60s → 120s → 240s → 600s cap, reset after 1h stable |
| Heartbeat monitoring | Polls `training_state.json` every 15s, detects stale >300s (GPU hang) |
| Pre-flight checks | Kill stale GPU processes, verify GPU health, check Xid errors |
| Signal forwarding | SIGTERM/SIGINT forwarded to training process on guard shutdown |

**Debugging mode**: `make train-350m-raw` runs with `RUST_BACKTRACE=1 CUDA_LAUNCH_BLOCKING=1`
to capture CUDA errors synchronously (slower but diagnostic).

**Auto-diagnostic mode**: `train-guard.sh` detects the async CUDA crash pattern
(early death + signal crash at step 0) and automatically enables
`CUDA_LAUNCH_BLOCKING=1` on the next restart to surface the exact failing kernel.

## ALB-065: Missing stream.synchronize() Before D2H Gradient Transfers (Critical)

**Discovery**: Diagnosed via ALB-064. Training with `CUDA_LAUNCH_BLOCKING=1` was
stable for 18+ minutes; without it, process died within 15 seconds. This is the
classic async CUDA error pattern.

**Five Whys**:

| Why | Finding | Brick Boundary |
|-----|---------|----------------|
| **Why does training crash silently?** | CUDA error queued asynchronously, process dies at next sync point | Per-kernel: error deferred |
| **Why does CUDA_LAUNCH_BLOCKING=1 fix it?** | Forces synchronous execution, masking a race condition | Per-kernel: each finishes before next starts |
| **Why is there a race condition?** | `cuMemcpyDtoH` doesn't synchronize with non-blocking stream kernels | Per-transfer: D2H reads stale data |
| **Why are kernels on a non-blocking stream?** | trueno `CudaStream::new()` uses `CU_STREAM_NON_BLOCKING` | Per-kernel: stream creation policy |
| **Why is there a D2H transfer mid-backward?** | `compute_workspace_clip_scale()` downloads 9 gradient buffers for L2 norm | **Root cause**: no sync before D2H |

**Fix**: `stream.synchronize()` at 3 locations in `cuda_trainer.rs` before
`cuMemcpyDtoH`-based gradient clipping (`entrenar@d3a3d26`).

**Verification**: Training stable without `CUDA_LAUNCH_BLOCKING=1` at 441 tok/s
(vs 402 with blocking). Process alive for 2.5+ minutes past the crash point.

## Post-Training Pipeline Validation Detail

### Quantization (2026-03-03)

| Model | Scheme | Original | Quantized | Reduction | Notes |
|-------|--------|----------|-----------|-----------|-------|
| 50M | Int4 | 238 MiB | 30 MiB | 87.5% (8.0x) | Working as expected |
| 50M | Q4K | 238 MiB | 238 MiB | 0% (1.0x) | **No-op** — entrenar saves 1D flat tensors; Q4K requires 2D |
| 350M | Int4 | 1.48 GiB | 191 MiB | 87.5% (8.0x) | Working as expected |
| 350M | Q4K | 1.48 GiB | 1.48 GiB | 0% (1.0x) | **No-op** — same 1D tensor issue |

**Finding**: `apr quantize -s q4k` is a no-op on entrenar checkpoints because
entrenar stores weights as 1D flat tensors, and Q4K quantization requires 2D
weight matrices to compute per-block statistics. Int4 (simple bit-width reduction)
works correctly. Fix: either (a) reshape before quantize, or (b) run
`convert-checkpoint.py` first to produce HF-format 2D tensors.

### Pruning (2026-03-03)

| Model | Method | Params | Zeros | Output Size | Notes |
|-------|--------|--------|-------|-------------|-------|
| 50M | Magnitude (0.5) | 62.4M | 31.2M (50.0%) | 238 MiB | Working — 50% sparsity |
| 50M | Depth (layers 8-11) | 62.4M→47.2M | 1 | 180 MiB | Working — 4 layers removed |
| 350M | Magnitude (0.3) | 398.5M | 199.2M (50.0%) | 1.48 GiB | **Bug**: sparsity=0.3 produced 50% — param may be ignored |

**Finding**: `apr prune --method magnitude --sparsity 0.3` on 350M checkpoint
produced 50.0% zeros instead of 30.0%. The `--sparsity` parameter may not be
correctly wired through to the pruning implementation for magnitude pruning.
Depth pruning works correctly.

### Distillation Setup (2026-03-03)

| Teacher | Size | Tensors | Precompute | Notes |
|---------|------|---------|------------|-------|
| Qwen2.5-Coder-0.5B | 942 MiB | 290 | **PASS** | Single-file SafeTensors, loads in realizar |
| Qwen2.5-Coder-3B | 5.75 GiB | 434 | **PASS** | Sharded SafeTensors (2 files), loads in apr distill |

**Finding**: realizar doesn't support sharded SafeTensors (multiple `.safetensors`
files). `apr distill` uses RosettaStone which handles sharding. For inference with
realizar, the 3B model would need to be merged into a single file.

### Data Expansion (2026-03-03)

| Source | Type | Files | Parquet Size |
|--------|------|-------|-------------|
| depyler | Tier 1 | 1,843 | 5.8 MiB |
| hf-ground-truth | Tier 1 | 11,493 | 188 MiB |
| jax | Tier 1 | 2,637 | 47 MiB |
| vllm (original) | Tier 1 | 1,100 | 17 MiB |
| **pytorch** | **Tier 2** | **3,801** | **15.6 MiB** |
| **hf-repos** | **Tier 2** | **19,781** | **73.8 MiB** |
| **mlflow** | **Tier 2** | **1,780** | **4.6 MiB** |
| **vllm-full** | **Tier 2** | **2,239** | **7.7 MiB** |
| **tgi** | **Tier 2** | **372** | **1.0 MiB** |
| **algo-corpus** | **Tier 2** | **186** | **0.2 MiB** |
| **cuda-python** | **Tier 2** | **157** | **0.4 MiB** |
| **llms-with-hf** | **Tier 2** | **37** | **35 KiB** |

Pipeline: 45,420 mixed rows → 45,420 FIM (50% PSM) → **67,977 pretokenized sequences** (2048 tokens each)

**Token count**: 139M tokens (up from 45M — 3.1× expansion)

C-TRAINCFG-001 pre-flight for pretrain-350m-v2.yaml:
- steps_per_epoch: 132
- min_epochs: 38 (38 × 132 = 5016 ≥ 5000)
- warmup_steps: 500 (10% of 5000)
- total_tokens: 2.6B

## Tool Availability

All sovereign stack tools are installed and reachable:

| Tool | Path | Version |
|------|------|---------|
| `apr` | `/home/noah/.local/bin/apr` | aprender |
| `pv` | `/home/noah/.cargo/bin/pv` | provable-contracts |
| `forjar` | `/home/noah/.cargo/bin/forjar` | forjar |
| `alimentar` | `/home/noah/.cargo/bin/alimentar` | alimentar |
| `batuta` | `/home/noah/.cargo/bin/batuta` | batuta |
| `pmat` | `/home/noah/.cargo/bin/pmat` | pmat |
| `bashrs` | `/home/noah/.cargo/bin/bashrs` | bashrs v6.65.0 |

---


# Appendix G: Data Pipeline

> Documents the Phase 1 data ingestion, tokenization, and augmentation pipeline.

## Source Corpora

| Source | Repository | Files | Rows | Parquet Size |
|--------|-----------|-------|------|-------------|
| depyler | depyler examples + TDD book | 1,843 | 1,843 | 6MB |
| hf-ground-truth | HuggingFace ground truth corpus | 11,928 | 11,493 | 197MB |
| jax-ground-truth | JAX ground truth corpus | 2,697 | 2,637 | 50MB |
| vllm-ground-truth | vLLM ground truth corpus | 1,118 | 1,100 | 18MB |

All sources are Python code, collected via `alimentar import local`.

## Training Mix

Weighted sampling with Tier 1 (depyler) upsampled:

```
alimentar mix \
  depyler.parquet:0.4 \
  hf.parquet:0.3 \
  jax.parquet:0.15 \
  vllm.parquet:0.15 \
  --output mixed.parquet \
  --seed 42
```

Result: **17,070 rows** (depyler upsampled 3.7x from 1,843 to ~6,829).

## Data Splits

| Split | Rows | Size | Seed | Weights |
|-------|------|------|------|---------|
| train | 17,070 | 201MB | 42 | depyler:0.4 hf:0.3 jax:0.15 vllm:0.15 |
| val | 500 | 7MB | 123 | equal 0.25 each |
| test | 200 | 2.4MB | 456 | equal 0.25 each |

## FIM Augmentation

Fill-in-the-Middle transforms applied via `alimentar fim`:

```
alimentar fim mixed.parquet \
  --output mixed-fim.parquet \
  --column text \
  --rate 0.5 \
  --format psm \
  --seed 42
```

- Format: PSM (Prefix-Suffix-Middle)
- Rate: 50% of rows receive FIM transform
- Sentinel tokens: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`

## BPE Tokenizer

Trained via `apr tokenize apply`:

```
apr tokenize apply \
  --data corpus-raw.txt \
  --vocab-size 32768 \
  --algorithm bpe \
  --max-lines 100000 \
  -o tokenizer/
```

Results:
- Final vocab size: 32,768
- Merges: 32,518
- Training time: 2022.5s (~33.7 min)
- Training data: 100K lines of Python code
- Special tokens: `<unk>`, `<s>`, `</s>`, `<pad>`
- Python pattern coverage: 8/8 (`def`, `return`, `self`, `import`, `class`, `for`, `if`, `in`)
- Output: `tokenizer/vocab.json` + `tokenizer/merges.txt`

## HuggingFace tokenizer.json Conversion

Entrenar requires HuggingFace `tokenizer.json` format, but `apr tokenize apply`
produces raw `vocab.json` + `merges.txt`. A Python conversion step bridges the gap
([ALB-033](https://github.com/paiml/albor/issues/31)):

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
bpe = models.BPE(vocab=vocab, merges=merges, end_of_word_suffix='</w>')
tokenizer = Tokenizer(bpe)
tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=' ', behavior='removed')
tokenizer.decoder = decoders.BPEDecoder(suffix='</w>')
tokenizer.save('models/albor-tokenizer/tokenizer.json')
```

Key details:
- Merges must be string format (`"i n"`) not array format (`["i", "n"]`)
- Pre-tokenizer matches aprender's `split_whitespace()` behavior
- `</w>` end-of-word suffix matches aprender's BPE encoding
- Regular vocab: 32,768 tokens (IDs 0-32767)
- FIM special tokens: 3 additional (IDs 32768-32770)

## Parquet Schema

All data files use a consistent schema:

```
{
  text: Utf8,    -- Python source code
  source: Utf8,  -- Corpus name (depyler, hf, jax, vllm)
  file: Utf8     -- Original file path
}
```

## Provenance

SHA-256 hashes for all data artifacts are recorded in `docs/PROVENANCE.md`.
Each split uses a different random seed for reproducibility.

## ByteLevel BPE Tokenizer (v2)

The v1 tokenizer (from `apr tokenize apply`) normalizes whitespace, which loses
Python indentation. The v2 tokenizer uses ByteLevel BPE (like GPT-2/CodeLlama):

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=32768, special_tokens=[...])
tokenizer.train(["corpus-raw.txt"], trainer)
tokenizer.save("models/albor-tokenizer-v2/tokenizer.json")
```

- Vocab: 32,768 (same size, different encoding)
- Roundtrip: 6/6 PASS (preserves newlines, indentation, blank lines)
- Merges: 32,557

## Pre-Tokenized Data

Training data pre-tokenized and chunked for efficient training:

| Dataset | Sequences | Seq Length | Total Tokens | Format |
|---------|-----------|-----------|--------------|--------|
| pretokenized-2048/train (v1) | 22,079 | 2048 | 45.2M | Parquet (input_ids: List&lt;u32&gt;) |
| pretokenized-2048/val | 814 | 2048 | 1.7M | Parquet (input_ids: List&lt;u32&gt;) |
| pretokenized-2048-v2/train | 67,977 | 2048 | 139M | Parquet (input_ids: List&lt;u32&gt;) |
| pretokenized-2048-v2/val | 814 | 2048 | 1.7M | Parquet (reused from v1) |

Pre-tokenization avoids the entrenar↔aprender BPE compatibility issue (ALB-033)
and enables direct `input_ids` column loading.

## v2 Data Expansion (2026-03-03)

The v2 dataset expands from Tier 1 only to Tier 1 (10x upsampled) + 8 Tier 2 repos:

| Source | Type | Files | Weight |
|--------|------|-------|--------|
| depyler | Tier 1 | 1,843 | 10x |
| hf-ground-truth | Tier 1 | 11,493 | 10x |
| jax-ground-truth | Tier 1 | 2,637 | 10x |
| vllm-ground-truth | Tier 1 | 1,100 | 10x |
| pytorch | Tier 2 | 3,801 | 1x |
| hf-repos | Tier 2 | 19,781 | 1x |
| mlflow | Tier 2 | 1,780 | 1x |
| vllm-full | Tier 2 | 2,239 | 1x |
| tgi | Tier 2 | 372 | 1x |
| algo-corpus | Tier 2 | 186 | 1x |
| cuda-python | Tier 2 | 157 | 1x |
| llms-with-hf | Tier 2 | 37 | 1x |

**Pipeline**: source-to-parquet.py → alimentar mix → alimentar fim (50% PSM) → pretokenize.py

**Key finding**: `alimentar import local` expects data files (CSV/JSON/Parquet),
not source code directories. The workaround script `scripts/source-to-parquet.py`
converts Python repos to Parquet with the Tier 1 schema (file, source, text columns).

**Result**: 45,420 mixed rows → 67,977 pretokenized sequences × 2048 = 139M tokens (191 MiB).

## Tools Used

- `alimentar import local` — JSONL to Parquet conversion
- `alimentar mix` — weighted sampling with upsampling
- `alimentar fim` — Fill-in-the-Middle augmentation
- `apr tokenize plan/apply` — BPE vocabulary training (v1, whitespace-split)
- Python `tokenizers` — ByteLevel BPE training (v2, whitespace-preserving)
- `scripts/source-to-parquet.py` — Python source code to Parquet (for Tier 2 repos)
- `entrenar` (parquet feature) — Parquet-to-LMBatch bridge for training

---

