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

# Unified pipeline (apr pipeline wraps forjar + batuta)
apr pipeline plan configs/pipeline/albor.yaml
apr pipeline apply configs/pipeline/albor.yaml
apr pipeline status

# ═══════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════

# Import local codebases
alimentar import local /path/to/codebase -o data/raw/corpus.parquet

# Weighted mix with upsampling
alimentar mix a.parquet:0.4 b.parquet:0.3 c.parquet:0.15 d.parquet:0.15 \
    -o data/tokenized/train/mixed.parquet --seed 42

# FIM transform
alimentar fim data.parquet -o data-fim.parquet --rate 0.5 --format psm

# Quality profiles
alimentar quality profiles

# ═══════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════

# v1: BPE with apr (whitespace-split — ALB-036 limitation)
apr tokenize plan --data corpus.txt --vocab-size 32768
apr tokenize apply --data corpus.txt --vocab-size 32768 --algorithm bpe -o tokenizer/

# v2: ByteLevel BPE with Python (recommended — preserves whitespace)
python scripts/train-tokenizer-v2.py --corpus corpus.txt --vocab-size 32768 \
    --output models/albor-tokenizer-v2/

# Pre-tokenize for training (bypasses tokenizer format gap ALB-033)
python scripts/pretokenize.py --input data.parquet \
    --tokenizer models/albor-tokenizer-v2/tokenizer.json \
    --seq-len 2048 --output data/pretokenized-2048/train/train.parquet

# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

# Plan (dry-run, validate config)
apr train plan --task pretrain --config configs/train/pretrain-350m.yaml

# Train (execute)
apr train apply --task pretrain --config configs/train/pretrain-350m.yaml

# Makefile shortcuts
make train-50m        # ~2 min on RTX 4090
make train-350m       # ~20 hours on RTX 4090
make training-status  # Check running training

# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

# apr eval (perplexity — ALB-037 FIXED, realizar loads checkpoints)
apr eval checkpoints/albor-base-350m/model.safetensors \
    --dataset custom --text "def foo():" --threshold 30

# Python eval scripts (supplement)
python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --validate-only
python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --api http://localhost:8080
python scripts/eval-perplexity.py checkpoints/albor-base-350m/ \
    --data data/pretokenized-2048/val/val.parquet --seq-len 2048 --threshold 30

# Convert entrenar checkpoint for realizar
python scripts/convert-checkpoint.py checkpoints/albor-base-350m/ \
    --config configs/train/pretrain-350m.yaml

# Makefile shortcuts
make eval-validate           # Validate all benchmark canonical solutions
make eval-perplexity-350m    # Run perplexity eval

# ═══════════════════════════════════════════════════════════
# MONITORING (run in a separate terminal during training)
# ═══════════════════════════════════════════════════════════

bash scripts/monitor-training.sh                     # Training process + GPU + log
apr monitor ./checkpoints/albor-base-350m/           # Live training TUI (ALB-025 FIXED)
apr experiment view --db .entrenar/experiments.db     # Browse past experiments

# ═══════════════════════════════════════════════════════════
# POST-TRAINING (Phases 4-6)
# ═══════════════════════════════════════════════════════════

# Distillation
apr distill --config configs/train/distill.yaml --plan
apr distill --config configs/train/distill.yaml --stage precompute
apr distill --config configs/train/distill.yaml --stage train

# Fine-tuning
apr finetune --plan --model-size 350M --vram 24 --method lora --rank 16

# Model operations
apr merge a.safetensors b.safetensors --strategy slerp -o merged.safetensors
apr prune model.safetensors --method wanda --sparsity 0.5 -o pruned.safetensors
apr quantize model.safetensors --method q4_k -o model.gguf
apr export model.safetensors --format gguf -o model.gguf
apr publish checkpoints/albor-350m/ paiml/albor-base-350m

# ═══════════════════════════════════════════════════════════
# QUALITY (bashrs is KING of linting)
# ═══════════════════════════════════════════════════════════

# bashrs — sovereign linter for all shell artifacts
bashrs make lint Makefile                          # Makefile quality
bashrs classify Makefile                           # Safety classification
bashrs make purify Makefile                        # Deterministic output

# provable-contracts — kernel correctness
pv validate contracts/*.yaml                       # Contract schemas
pv coverage contracts                              # Obligation coverage
pv generate contracts/*.yaml                       # Scaffold + tests + harnesses
pv book contracts/                                 # mdBook pages
pv audit contracts/*.yaml                          # Audit for issues
pv graph contracts/ --format mermaid               # Verification DAG
pv lean contracts/*.yaml                           # Lean 4 theorem stubs

# batuta — falsification
batuta falsify . --format markdown                 # 108-item checklist
batuta oracle --list                               # Stack components
batuta oracle --local                              # Local workspace status

# pmat — code quality (upstream repos)
pmat tdg baseline create                           # TDG baseline
pmat comply check --strict ../aprender

# ═══════════════════════════════════════════════════════════
# VALIDATION (Makefile)
# ═══════════════════════════════════════════════════════════

make validate          # All validation (YAML + contracts + forjar + Makefile)
make lint              # Lint with bashrs
make eval-validate     # Validate benchmark canonical solutions
make dogfood           # Full 12-section dogfooding suite
make book              # Build mdBook
make help              # Show all targets
```
