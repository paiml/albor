# Appendix F: Dogfooding Log

> Living record of tool validation against the Albor repo.
> Updated as gaps are discovered and resolved.

## Summary (2026-03-01)

| Tool | Command | Result | Gap |
|------|---------|--------|-----|
| `pv validate` | `pv validate contracts/*.yaml` | **PASS** (all 5 contracts) | — |
| `pv coverage` | `pv coverage contracts` | **PASS** (100% obligation coverage) | — |
| `pv graph` | `pv graph contracts` | **PASS** (8 nodes, correct deps) | — |
| `forjar validate` | `forjar validate -f infra-only.yaml` | **PASS** (2 machines, 6 resources) | — |
| `forjar validate` | `forjar validate -f albor.yaml` | **FAIL** (`task` type unknown) | ALB-027 |
| `forjar graph` | `forjar graph -f infra-only.yaml` | **PASS** (Mermaid output) | — |
| `apr finetune --plan` | `apr finetune --plan --model-size 350M --vram 24` | **PASS** (VRAM estimate correct) | — |
| `apr train plan` | `apr train plan configs/train/pretrain-350m.yaml` | **FAIL** (expects `--data` flag, not config file) | ALB-009 |
| `apr distill --plan` | `apr distill --plan` | **PASS** (exists, needs config-driven workflow) | ALB-011 |
| `apr quantize --plan` | `apr quantize --plan <file>` | **PASS** (plan mode works) | — |
| `apr prune --plan` | `apr prune --plan <file>` | **PASS** (plan mode exists) | — |
| `alimentar quality profiles` | `alimentar quality profiles` | **PASS** (ml-training profile exists) | — |
| `alimentar import` | N/A | **MISSING** (no `import` subcommand) | ALB-019 |
| `alimentar mix` | N/A | **MISSING** (no `mix` subcommand) | ALB-020 |
| `batuta falsify` | `batuta falsify . --critical-only` | **PARTIAL** (2/5 pass, 3 false positives) | ALB-029 |

## Contract Validation Detail

All 5 contracts initially **failed** `pv validate` because they used a custom schema
(`contract:` top-level key, `latex:` field, `obligations:` list). After rewriting to
match the actual `pv` schema (`metadata:`, `formula:`, `proof_obligations:`, `falsification_tests:`),
all 5 pass with 0 errors.

```
pv coverage contracts
─────────────────────
Contracts:            5
Equations:            13
Obligations:          29
Falsification tests:  19
Kani harnesses:       7
Overall coverage:     100.0%
```

## Pipeline Manifest Validation Detail

The full pipeline manifest (`configs/pipeline/albor.yaml`) fails `forjar validate`
because it uses `type: task` for ML pipeline steps. Forjar currently supports:
`package`, `file`, `service`, `mount`, `user`, `docker`, `pepita`, `network`, `cron`,
`recipe`, `model`, `gpu`.

The `task` resource type (ALB-027, [#4](https://github.com/paiml/albor/issues/4))
is the key missing piece that turns forjar from an infrastructure tool into a
pipeline orchestrator.

A separate `infra-only.yaml` manifest validates successfully — this allows provisioning
infrastructure (GPU drivers, directories, NFS mounts, teacher model) independently
while waiting for the `task` type to be implemented.

### Spec Correction: `names` → `packages`

Dogfooding revealed that the spec used `names:` for forjar package resources, but
forjar expects `packages:`. Also requires `provider: apt` (not implicit). Both the
spec and configs were corrected.

## apr train plan/apply Detail

`apr train plan/apply` exists but is currently scoped to **classification fine-tuning
with HPO** (Tree-of-Parzen Estimators):

```
Current:  apr train plan --data <JSONL> --model-size 0.5B --task classify
Target:   apr train plan configs/train/pretrain-350m.yaml
```

The plan/apply infrastructure is solid — `apr train plan` generates structured
summaries with resource estimates. The gap (ALB-009) is in scope: extending from
classification to causal LM pre-training, and from flag-driven to config-file-driven.

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

## Batuta Falsification Detail

`batuta falsify . --critical-only` reports 2/5 critical items passing (40% score,
"STOP THE LINE" grade). However, 3 failures are **false positives** because batuta's
checklist is designed for Rust library projects, not project repos:

| Check | Result | Analysis |
|-------|--------|----------|
| AI-01 Declarative YAML | FAIL | False positive: batuta looks in `src/` but albor has 9 YAML configs in `configs/` |
| AI-02 Zero Scripting | PASS | Correct: no Python/JS in production |
| AI-03 Pure Rust Testing | PASS | Correct: no non-Rust test artifacts |
| AI-04 WASM-First Browser | FAIL | False positive: 8 JS files are from mdBook in `book-output/`, not project code |
| AI-05 Schema Validation | FAIL | False positive: no `Cargo.toml` because albor is not a Rust project; configs ARE validated by pv and forjar |

Filed as ALB-029 ([#28](https://github.com/paiml/albor/issues/28)) — batuta needs
to handle non-Rust project repos that contain only configs, contracts, and documentation.
