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
| ALB-006 | [#7](https://github.com/paiml/albor/issues/7) | apr (aprender) | `apr eval plan/apply` benchmark harness | High | OPEN | `apr eval plan` validates model + tasks exist; `apr eval apply` produces scores comparable to lm-evaluation-harness. Note: `apr eval` exists for perplexity/classification but needs task-specific benchmark support. |
| ALB-007 | [#8](https://github.com/paiml/albor/issues/8) | alimentar | Streaming tokenized data to entrenar | Medium | OPEN | `alimentar` streams pre-tokenized Parquet shards directly to training loop |
| ALB-009 | [#1](https://github.com/paiml/albor/issues/1) | apr (entrenar) | `apr train plan/apply` for pre-training from scratch | Critical | OPEN | `apr train plan` validates YAML + estimates VRAM/time; `apr train apply` runs full pre-training. Note: `apr train plan/apply` exists but currently scoped to classification fine-tuning with HPO — needs causal LM pre-training mode. |
| ALB-010 | [#2](https://github.com/paiml/albor/issues/2) | realizar | Qwen3-Coder-Next / DeltaNet / MoE architecture support | Critical | OPEN | `realizar` loads and runs inference on Qwen3-Coder-Next (80B MoE with DeltaNet layers) |
| ALB-011 | [#3](https://github.com/paiml/albor/issues/3) | apr (entrenar + realizar) | `apr distill plan/apply` (precompute + train stages) | Critical | OPEN | `apr distill plan` checks teacher RAM fit + disk; `apply --stage precompute` extracts logits; `apply --stage train` trains student with KD loss. Note: `apr distill` exists with `--plan` flag but needs config-file-driven two-stage workflow. |
| ALB-018 | [#19](https://github.com/paiml/albor/issues/19) | entrenar/alimentar | Fill-in-the-Middle (FIM) data transform (PSM/SPM) | High | OPEN | Code sequences randomly split into prefix/suffix/middle format during training |
| ALB-019 | [#20](https://github.com/paiml/albor/issues/20) | alimentar | `alimentar import local` for local Python files | Medium | **FIXED** | `alimentar import local` subcommand now available (`alimentar@265541b`). Supports CSV/JSON/JSONL/Parquet format conversion. |
| ALB-020 | [#21](https://github.com/paiml/albor/issues/21) | alimentar | `alimentar mix` with weighted upsampling | Medium | **FIXED** | `alimentar mix` with weighted sampling and upsampling now available (`alimentar@64b1e92`). Syntax: `alimentar mix a.parquet:0.8 b.parquet:0.2 -o out.parquet`. |
| ALB-021 | [#22](https://github.com/paiml/albor/issues/22) | entrenar | Custom model architecture params in YAML | High | OPEN | `model:` section with `hidden_size`, `num_layers`, `num_kv_heads`, etc. accepted (not just preset strings) |
| ALB-022 | [#23](https://github.com/paiml/albor/issues/23) | entrenar | Human-readable value shorthand in YAML configs | Low | OPEN | Config parser accepts `10B`, `512K`, `3e-4` shorthand alongside raw numbers. YAML underscore notation (`32_768`) works natively. |
| ALB-023 | [#24](https://github.com/paiml/albor/issues/24) | apr (aprender) | Plan/apply contract for all subcommands | High | OPEN | Every `apr <cmd>` exposes `plan` (dry-run, exit 0/1) and `apply` (execute). Note: `--plan` flag exists on quantize, finetune, prune, distill — but not all subcommands, and `plan` subcommand (not flag) needed for consistency. |
| ALB-024 | [#25](https://github.com/paiml/albor/issues/25) | presentar + apr | `apr experiment view` — SQLite experiment browser (TUI + WASM) | Medium | OPEN | `apr experiment view --db .entrenar/experiments.db` renders presentar `LossCurve`, metric comparison, run diff. Terminal via `presentar-terminal`, browser via `presentar serve`. Both read same SQLite tables. |
| ALB-025 | [#26](https://github.com/paiml/albor/issues/26) | presentar + apr | `apr monitor` upgrade — presentar widgets for live training TUI | Medium | OPEN | `apr monitor` uses presentar `LossCurve`, `Gauge`, `Sparkline`, `BrailleGraph` instead of entrenar's built-in `TuiMonitor` renderer. Same widget tree compiles to terminal and WASM. |
| ALB-026 | [#27](https://github.com/paiml/albor/issues/27) | presentar | WASM training dashboard — `albor-dashboard.yaml` | Medium | OPEN | Declarative YAML dashboard config that renders training metrics, experiment comparison, and model card via `presentar serve`. Embeddable in HuggingFace model card as static WASM artifact. |
| ALB-027 | [#4](https://github.com/paiml/albor/issues/4) | forjar | `task` resource type for pipeline orchestration | Critical | OPEN | New forjar resource type: runs arbitrary command, tracks exit code, hashes `output_artifacts` for idempotency, supports `completion_check` and `timeout`. bashrs validates `command:` at plan time. Dogfooding: `forjar validate` rejects `type: task` — only supports package, file, service, mount, user, docker, pepita, network, cron, recipe, model, gpu. |
| ALB-028 | [#5](https://github.com/paiml/albor/issues/5) | apr (aprender) | `apr pipeline plan/apply` wrapping forjar DAG engine | Critical | OPEN | `apr pipeline plan` parses manifest, shows full DAG with estimates. `apr pipeline apply` executes via forjar engine. `apr pipeline status` shows converged/pending/failed. Resumable, multi-machine. |

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

