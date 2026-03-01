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
