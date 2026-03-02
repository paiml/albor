# Appendix F: Dogfooding Log

> Living record of tool validation against the Albor repo.
> Updated as gaps are discovered and resolved.

## Summary (2026-03-02)

| Tool | Command | Result | Gap |
|------|---------|--------|-----|
| `pv validate` | `pv validate contracts/*.yaml` | **PASS** (all 5 contracts) | — |
| `pv coverage` | `pv coverage contracts` | **PASS** (100% obligation coverage) | — |
| `pv graph` | `pv graph contracts` | **PASS** (8 nodes, correct deps) | — |
| `pv probar` | `pv probar contracts/*.yaml` | **PASS** (generates property tests) | — |
| `pv kani` | `pv kani contracts/*.yaml` | **PASS** (generates Kani harnesses) | — |
| `pv generate` | `pv generate contracts/*.yaml` | **PASS** (20 files: scaffold, kani, probar, book) | — |
| `pv scaffold` | `pv scaffold contracts/*.yaml` | **PASS** (Rust trait + test stubs) | — |
| `pv status` | `pv status contracts/*.yaml` | **PASS** (equation/obligation counts) | — |
| `pv audit` | `pv audit contracts/*.yaml` | **PASS** (no findings) | — |
| `pv equations` | `pv equations contracts/*.yaml` | **PASS** (formatted equations) | — |
| `pv book` | `pv book contracts/` | **PASS** (5 mdBook pages) | — |
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
| `apr train apply` (50M) | `apr train apply --task pretrain --config pretrain-50m-test.yaml` | **PASS** (10-row micro, 5 batches, 2.1s CUDA) | ALB-034 (max_steps ignored) |
| `apr train apply` (50M full) | `apr train apply --task pretrain --config pretrain-50m.yaml` | **PASS** (500 rows, 125 batches, 31 steps, 110.7s CUDA, loss 10.3→4.42) | ALB-034 (max_steps) |
| `apr train apply` (50M v2) | `apr train apply --task pretrain --config pretrain-50m-v2.yaml` | **PASS** (pre-tokenized ByteLevel BPE, 108.5s CUDA, loss→5.51) | — |
| `apr train plan` (350M) | `apr train plan --task pretrain --config pretrain-350m.yaml` | **PASS** (config validated, ready for apply) | — |
| `entrenar validate` | `entrenar validate pretrain-350m-manifest.yaml` | **PASS** (architecture overrides bridge through) | ~~ALB-021~~ FIXED |
| `entrenar shorthand` | `vocab_size: "32K"` in YAML manifest | **PASS** (parses to 32768) | ~~ALB-022~~ FIXED |
| `apr merge --plan` | `apr merge a.apr b.apr --plan --strategy slerp -o merged.apr` | **PASS** (validates inputs, shows strategy, sizes) | ~~ALB-023~~ FIXED |
| `apr export --plan` | `apr export model.apr --plan --format gguf -o model.gguf` | **PASS** (validates format, shows plan) | ~~ALB-023~~ FIXED |
| `apr publish --plan` | `apr publish dir repo --plan` | **PASS** (alias for --dry-run) | ~~ALB-023~~ FIXED |
| `apr train apply` (350M) | `apr train apply --task pretrain --config pretrain-350m.yaml` | **IN PROGRESS** (2760 batches, 398.5M params, 6.4GB VRAM, ~20h est.) | — |
| `apr eval` (50M safetensors) | `apr eval checkpoints/albor-base-50m/model.safetensors --dataset custom` | **FAIL** (PPL 679,614 — weights ignored) | ALB-037 |
| `eval-code.py` (validate) | `python scripts/eval-code.py configs/eval/python-intermediate.jsonl --validate-only` | **PASS** (15/15 canonical solutions) | — |
| `eval-code.py` (HumanEval) | `python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --validate-only` | **PASS** (20/20 canonical solutions) | — |
| `convert-checkpoint.py` (50M) | `python scripts/convert-checkpoint.py checkpoints/albor-base-50m/` | **PASS** (110→111 tensors, 85 reshaped, lm_head created) | ALB-037 |
| `eval-perplexity.py --validate` | `python scripts/eval-perplexity.py checkpoints/albor-base-50m/ --validate-checkpoint` | **FAIL** → **FIXED** (ALB-038 root cause in autograd) | ~~ALB-038~~ FIXED |
| checkpoint analysis | byte-compare layers 0-11 q_proj, gate_proj | **FAIL** → **FIXED** (all parameters now receive gradients) | ~~ALB-038~~ FIXED |

## Contract Validation Detail

All 5 contracts initially **failed** `pv validate` because they used a custom schema
(`contract:` top-level key, `latex:` field, `obligations:` list). After rewriting to
match the actual `pv` schema (`metadata:`, `formula:`, `proof_obligations:`, `falsification_tests:`),
all 5 pass with 0 errors.

```
pv coverage contracts
---------------------
Contracts:            5
Equations:            13
Obligations:          29
Falsification tests:  19
Kani harnesses:       7
Overall coverage:     100.0%
```

## pv generate Detail

`pv generate` produces 4 files per contract (20 total):

| Type | Content | Example |
|------|---------|---------|
| `*_scaffold.rs` | Rust trait with documented invariants | `knowledge-distillation-kernel-v1_scaffold.rs` |
| `*_probar.rs` | Property tests derived from proof obligations | 6 property tests + 5 falsification test stubs |
| `*_kani.rs` | Kani verification harnesses | 2 harnesses with `stub_float` strategy |
| `*_book.md` | mdBook page with equations, deps, obligations | Mermaid dependency graph, LaTeX equations |

`pv book contracts/` generates 5 contract pages directly into mdBook format.
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
