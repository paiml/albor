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

# Future: unified pipeline (apr pipeline wraps forjar + batuta)
apr pipeline plan configs/pipeline/albor.yaml      # (blocked: ALB-028)
apr pipeline apply configs/pipeline/albor.yaml
apr pipeline status

# ═══════════════════════════════════════════════════════════
# MONITORING (run in a separate terminal during training)
# ═══════════════════════════════════════════════════════════

apr monitor ./checkpoints/albor-base-350m/        # Live training TUI
apr experiment view --db .entrenar/experiments.db  # Browse past experiments
apr cbtop ./checkpoints/albor-base-350m/           # GPU profiler

# ═══════════════════════════════════════════════════════════
# QUALITY (bashrs is KING of linting)
# ═══════════════════════════════════════════════════════════

# bashrs — sovereign linter for all shell artifacts
bashrs make lint Makefile                          # Makefile quality
bashrs classify Makefile                           # Safety classification
bashrs make purify Makefile                        # Deterministic output
bashrs lint scripts/*.sh                           # Shell script safety

# provable-contracts — kernel correctness
pv validate contracts/*.yaml                       # Contract schemas
pv coverage contracts                              # Obligation coverage
pv generate contracts/*.yaml                       # Scaffold + tests + harnesses
pv book contracts/                                 # mdBook pages
pv status contracts/                               # Contract completeness
pv graph contracts/ --format mermaid               # Verification DAG

# batuta — falsification
batuta falsify . --format markdown                 # 108-item checklist
batuta oracle --list                               # Stack components
batuta oracle --local                              # Local workspace status

# pmat — code quality (upstream repos)
pmat tdg baseline create                           # TDG baseline
pmat comply check --strict ../aprender
pmat comply check --strict ../entrenar

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

