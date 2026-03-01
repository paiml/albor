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

| ID | Component | Gap | Severity | Status | Acceptance Criterion |
|----|-----------|-----|----------|--------|---------------------|
| ALB-001 | apr (aprender) | `apr tokenize plan/apply` subcommand | Medium | OPEN | `apr tokenize plan` validates inputs + estimates time; `apr tokenize apply` trains BPE tokenizer |
| ALB-006 | apr (aprender) | `apr eval plan/apply` benchmark harness | High | OPEN | `apr eval plan` validates model + tasks exist; `apr eval apply` produces scores comparable to lm-evaluation-harness |
| ALB-007 | alimentar | Streaming tokenized data to entrenar | Medium | OPEN | `alimentar` streams pre-tokenized Parquet shards directly to training loop |
| ALB-009 | apr (entrenar) | `apr train plan/apply` for pre-training from scratch | Critical | OPEN | `apr train plan` validates YAML + estimates VRAM/time; `apr train apply` runs full pre-training with cosine LR, gradient checkpointing, mixed precision |
| ALB-010 | realizar | Qwen3-Coder-Next / DeltaNet / MoE architecture support | Critical | OPEN | `realizar` loads and runs inference on Qwen3-Coder-Next (80B MoE with DeltaNet layers) |
| ALB-011 | apr (entrenar + realizar) | `apr distill plan/apply` (precompute + train stages) | Critical | OPEN | `apr distill plan` checks teacher RAM fit + disk; `apply --stage precompute` extracts logits; `apply --stage train` trains student with KD loss |
| ALB-018 | entrenar/alimentar | Fill-in-the-Middle (FIM) data transform (PSM/SPM) | High | OPEN | Code sequences randomly split into prefix/suffix/middle format during training |
| ALB-019 | alimentar | `alimentar import local` for local Python files | Medium | OPEN | `alimentar import local ../path/ --lang python` recursively ingests .py files into Parquet |
| ALB-020 | alimentar | `alimentar mix` with weighted upsampling | Medium | OPEN | `alimentar mix --input a.parquet --weight 0.4 --upsample 10` produces training-ready shards |
| ALB-021 | entrenar | Custom model architecture params in YAML | High | OPEN | `model:` section with `hidden_size`, `num_layers`, `num_kv_heads`, etc. accepted (not just preset strings) |
| ALB-022 | entrenar | Human-readable value shorthand in YAML configs | Low | OPEN | Config parser accepts `10B`, `512K`, `3e-4` shorthand alongside raw numbers. YAML underscore notation (`32_768`) works natively. |
| ALB-023 | apr (aprender) | Plan/apply contract for all subcommands | High | OPEN | Every `apr <cmd>` exposes `plan` (dry-run, exit 0/1) and `apply` (execute). `plan` validates inputs, estimates resources, prints structured summary. No subcommand may skip the contract. |
| ALB-024 | presentar + apr | `apr experiment view` — SQLite experiment browser (TUI + WASM) | Medium | OPEN | `apr experiment view --db .entrenar/experiments.db` renders presentar `LossCurve`, metric comparison, run diff. Terminal via `presentar-terminal`, browser via `presentar serve`. Both read same SQLite tables. |
| ALB-025 | presentar + apr | `apr monitor` upgrade — presentar widgets for live training TUI | Medium | OPEN | `apr monitor` uses presentar `LossCurve`, `Gauge`, `Sparkline`, `BrailleGraph` instead of entrenar's built-in `TuiMonitor` renderer. Same widget tree compiles to terminal and WASM. |
| ALB-026 | presentar | WASM training dashboard — `albor-dashboard.yaml` | Medium | OPEN | Declarative YAML dashboard config that renders training metrics, experiment comparison, and model card via `presentar serve`. Embeddable in HuggingFace model card as static WASM artifact. |
| ALB-027 | forjar | `task` resource type for pipeline orchestration | Critical | OPEN | New forjar resource type: runs arbitrary command, tracks exit code, hashes `output_artifacts` for idempotency, supports `completion_check` and `timeout`. bashrs validates `command:` at plan time. |
| ALB-028 | apr (aprender) | `apr pipeline plan/apply` wrapping forjar DAG engine | Critical | OPEN | `apr pipeline plan` parses manifest, shows full DAG with estimates. `apr pipeline apply` executes via forjar engine. `apr pipeline status` shows converged/pending/failed. Resumable, multi-machine. |

### 11.2 Distributed Training Gaps (Stretch / Future)

| ID | Component | Gap | Severity | Status | Acceptance Criterion |
|----|-----------|-----|----------|--------|---------------------|
| ALB-002 | repartir | Ring all-reduce implementation | High | OPEN | Gradient tensors synchronized across 2+ workers with <5% overhead |
| ALB-003 | entrenar | repartir integration for distributed training | High | OPEN | Training loop calls `repartir::GradientSync` for multi-worker training |
| ALB-004 | entrenar | Unified CUDA + wgpu backend dispatch | Medium | OPEN | Same training config runs on CUDA (4090) and wgpu (W5700X) |
| ALB-005 | trueno | wgpu backward pass (gradient WGSL shaders) | High | OPEN | Compute shaders for matmul_backward, gelu_backward, rmsnorm_backward, attention_backward |
| ALB-008 | repartir | Heterogeneous worker throughput balancing | Medium | OPEN | Workers with different GPU speeds get proportional workload |

### 11.3 Quality & Verification Gaps

| ID | Component | Gap | Severity | Status | Acceptance Criterion |
|----|-----------|-----|----------|--------|---------------------|
| ALB-013 | provable-contracts | Knowledge distillation contract | High | OPEN | `knowledge-distillation-kernel-v1.yaml` with KL divergence falsification tests |
| ALB-014 | provable-contracts | BPE tokenizer contract | Medium | OPEN | `bpe-tokenizer-kernel-v1.yaml` with roundtrip invariant: `decode(encode(x)) = x` |
| ALB-015 | provable-contracts | Model merging contract (SLERP, TIES, DARE) | Medium | OPEN | `model-merging-kernel-v1.yaml` with interpolation bound proofs |
| ALB-016 | provable-contracts | Pruning contract (WANDA, magnitude) | Medium | OPEN | `pruning-kernel-v1.yaml` with sparsity invariants |
| ALB-017 | provable-contracts | Gradient accumulation contract | High | OPEN | `gradient-accumulation-kernel-v1.yaml` proving numerical equivalence within tolerance |

*Gaps will be added as they are discovered during implementation.*
