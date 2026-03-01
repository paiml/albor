# 18. Reference Commands

```bash
# ═══════════════════════════════════════════════════════════
# THE PIPELINE (this is all you need)
# ═══════════════════════════════════════════════════════════

# Plan: show full DAG, validate all configs, estimate all resources
apr pipeline plan configs/pipeline/albor.yaml

# Apply: execute everything (resumable — skips converged steps)
apr pipeline apply configs/pipeline/albor.yaml

# Status: what's done, what's pending, what failed
apr pipeline status

# Targeted: run just one step (+ its dependencies)
apr pipeline apply configs/pipeline/albor.yaml --target train-350m
apr pipeline apply configs/pipeline/albor.yaml --target eval-code
apr pipeline apply configs/pipeline/albor.yaml --target publish

# Force re-run a step (ignore converged state)
apr pipeline apply configs/pipeline/albor.yaml --target distill --force

# ═══════════════════════════════════════════════════════════
# MONITORING (run in a separate terminal during training)
# ═══════════════════════════════════════════════════════════

apr monitor ./checkpoints/albor-base-350m/        # Live training TUI
apr experiment view --db .entrenar/experiments.db  # Browse past experiments
apr cbtop ./checkpoints/albor-base-350m/           # GPU profiler

# ═══════════════════════════════════════════════════════════
# QUALITY (upstream repos — run independently of pipeline)
# ═══════════════════════════════════════════════════════════

pmat tdg baseline create                           # TDG baseline
pmat comply check --strict ../aprender
pmat comply check --strict ../entrenar
pmat comply check --strict ../alimentar
pmat comply check --strict ../realizar
pv validate contracts/*.yaml                       # Contract schemas
pv status contracts/                               # Contract completeness
pv graph contracts/ --format mermaid               # Verification DAG
batuta falsify . --min-grade toyota-standard       # 108-item checklist
cargo mutants --no-times                           # Mutation score ≥ 85%
cargo llvm-cov --summary-only                      # Coverage ≥ 95%

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
