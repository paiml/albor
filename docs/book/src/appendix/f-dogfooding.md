# Appendix F: Dogfooding Log

> Living record of tool validation against the Albor repo.
> Updated as gaps are discovered and resolved.

## Summary (2026-03-01)

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
| `forjar validate` | `forjar validate -f albor.yaml` | **FAIL** (`task` type unknown) | ALB-027 |
| `forjar graph` | `forjar graph -f infra-only.yaml` | **PASS** (Mermaid output) | — |
| `apr finetune --plan` | `apr finetune --plan --model-size 350M --vram 24` | **PASS** (VRAM estimate correct) | — |
| `apr train plan` | `apr train plan configs/train/pretrain-350m.yaml` | **FAIL** (expects `--data` flag, not config file) | ALB-009 |
| `apr distill --plan` | `apr distill --plan` | **PASS** (exists, needs config-driven workflow) | ALB-011 |
| `apr quantize --plan` | `apr quantize --plan <file>` | **PASS** (plan mode works) | — |
| `apr prune --plan` | `apr prune --plan <file>` | **PASS** (plan mode exists) | — |
| `alimentar quality profiles` | `alimentar quality profiles` | **PASS** (ml-training profile exists) | — |
| `alimentar import` | `alimentar import local <in> -o <out>` | **PASS** (local import works) | ~~ALB-019~~ FIXED |
| `alimentar mix` | N/A | **MISSING** (no `mix` subcommand) | ALB-020 |
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
| `pmat query` | `pmat query "training"` | **PASS** (0 functions, 5 document matches) | — |
| `pmat analyze makefile` | `pmat analyze makefile Makefile` | **PASS** (64% quality score) | — |
| `pv lean` | `pv lean contracts/kd-v1.yaml` | **PASS** (6 Lean 4 theorem stubs generated) | — |
| `pv lean-status` | `pv lean-status contracts/` | **PASS** (0% L4 coverage, 4 sorry debt) | — |
| `apr train plan` | `apr train plan --data <JSONL>` | **PASS** (exists, classification only) | ALB-009 |
| `apr merge` | `apr merge --strategy slerp` | **PASS** (SLERP, TIES, DARE supported) | — |
| `apr export --list-formats` | `apr export --list-formats` | **PASS** (SafeTensors, GGUF, MLX) | — |
| `apr publish` | `apr publish <dir> <repo>` | **PASS** (HF Hub publish exists) | — |
| `apr eval` | `apr eval <model>` | **PASS** (perplexity eval, needs benchmark tasks) | ALB-006 |
| `alimentar quality` | `alimentar quality profiles` | **PASS** (ml-training profile) | — |
| `alimentar convert` | `alimentar convert` | **PASS** (format conversion) | — |
| `bashrs score` | `bashrs score Makefile` | **PASS** (D grade, 5.2/10) | — |
| `bashrs audit` | `bashrs audit Makefile` | **PASS** (comprehensive audit) | — |

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

The full pipeline manifest (`configs/pipeline/albor.yaml`) fails `forjar validate`
because it uses `type: task` for ML pipeline steps. Forjar currently supports:
`package`, `file`, `service`, `mount`, `user`, `docker`, `pepita`, `network`, `cron`,
`recipe`, `model`, `gpu`.

The `task` resource type (ALB-027, [#4](https://github.com/paiml/albor/issues/4))
is the key missing piece that turns forjar from an infrastructure tool into a
pipeline orchestrator.

A separate `infra-only.yaml` manifest validates successfully -- this allows provisioning
infrastructure (GPU drivers, directories, NFS mounts, teacher model) independently
while waiting for the `task` type to be implemented.

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
