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
# Current score: 100.0% (108/108 PASS) — achieved 2026-03-04
```
