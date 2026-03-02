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
| ALB-024 | [#25](https://github.com/paiml/albor/issues/25) | presentar + apr | `apr experiment view` — SQLite experiment browser (TUI + WASM) | Medium | OPEN | `apr experiment view --db .entrenar/experiments.db` renders presentar `LossCurve`, metric comparison, run diff. Terminal via `presentar-terminal`, browser via `presentar serve`. Both read same SQLite tables. |
| ALB-025 | [#26](https://github.com/paiml/albor/issues/26) | presentar + apr | `apr monitor` upgrade — presentar widgets for live training TUI | Medium | OPEN | `apr monitor` uses presentar `LossCurve`, `Gauge`, `Sparkline`, `BrailleGraph` instead of entrenar's built-in `TuiMonitor` renderer. Same widget tree compiles to terminal and WASM. |
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

**Contract coverage report** (`pv coverage contracts`): 5 contracts, 13 equations, 29 obligations, 19 falsification tests, 7 Kani harnesses, **100% obligation coverage**. All contracts at impl=0/N — waiting for upstream bindings.

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
| ALB-037 | [#35](https://github.com/paiml/albor/issues/35) | realizar | SafeTensors inference ignores loaded weights | High | DOGFOODING | Root cause: ALB-038 (training didn't modify weights). Secondary: entrenar didn't save config.json for realizar. Fixed: `entrenar@6097780` saves HuggingFace config.json. Pending end-to-end verification with trained model. |
| ALB-038 | [#36](https://github.com/paiml/albor/issues/36) | entrenar | Saves initialization weights, not trained weights | Critical | **FIXED** | Root cause: `RMSNorm::forward_batched()` created tensors with no backward op, blocking all gradient flow. Attention `forward()` also broke Q/K/V gradients. Fixed in `entrenar@91ba9da` (norm backward) and `entrenar@1ede409` (attention backward). All 20 model parameters now receive gradients. |
| ALB-040 | [#38](https://github.com/paiml/albor/issues/38) | entrenar | GPU-resident pretraining — wire CudaTransformerBlock into TransformerTrainer | Critical | DOGFOODING | `CudaTransformerTrainer` in `cuda_trainer.rs` follows classify_pipeline.rs pattern. 3 PCIe transfers/step vs 16K. Auto-detect CUDA with graceful CPU fallback. Contract: `training-gpu-kernel-v1.yaml`. Pending 350M training verification. |

*Gaps are added as they are discovered during implementation and dogfooding.*

