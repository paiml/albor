# Albor Makefile — Development and Dogfooding Commands
#
# This Makefile exercises the sovereign stack tools against albor's
# configs, contracts, and pipeline manifest. It documents what works
# and what's blocked by upstream gaps.
#
# NOTE: This is a development convenience, NOT the pipeline orchestrator.
# The pipeline is `batuta playbook run configs/pipeline/albor-playbook.yaml`.
# Infrastructure provisioning is `forjar apply -f configs/pipeline/infra-only.yaml`.

.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: validate validate-contracts validate-forjar validate-yaml validate-makefile \
        plan-finetune plan-finetune-lora plan-pretrain-50m plan-pretrain-350m \
        train-50m train-350m train-350m-raw \
        eval-validate eval-perplexity-50m eval-perplexity-350m \
        eval-suite-50m eval-suite-350m \
        validate-checkpoint-50m validate-checkpoint-350m validate-convergence \
        training-status \
        test test-rust clippy bench security-audit reproduce \
        book book-serve spec dogfood dogfood-batuta dogfood-playbook \
        lint clean help

# ═══════════════════════════════════════════════════════════
# VALIDATION (all tools, no side effects)
# ═══════════════════════════════════════════════════════════

validate: validate-yaml validate-contracts validate-forjar validate-makefile ## Run all validation
	@echo ""
	@echo "All validation passed"

validate-yaml: ## Validate YAML syntax for all configs and contracts
	@echo "--- YAML Syntax ---"
	@for f in configs/**/*.yaml configs/*.yaml contracts/*.yaml; do \
		python3 -c "import yaml; yaml.safe_load(open('$$f'))" && echo "  PASS $$f" || exit 1; \
	done

validate-contracts: ## Validate all contracts with pv
	@echo "--- pv validate ---"
	@for f in contracts/*.yaml; do \
		pv validate "$$f" 2>&1 | tail -1; \
	done
	@echo ""
	@echo "--- pv coverage ---"
	@pv coverage contracts
	@echo ""
	@echo "--- pv graph ---"
	@pv graph contracts

validate-forjar: ## Validate infra manifest with forjar
	@echo "--- forjar validate (infra-only) ---"
	@forjar validate -f configs/pipeline/infra-only.yaml
	@echo ""
	@echo "--- forjar graph ---"
	@forjar graph -f configs/pipeline/infra-only.yaml

validate-makefile: ## Lint Makefile with bashrs (sovereign Makefile linter)
	@echo "--- bashrs make lint ---"
	@bashrs make lint Makefile
	@echo ""
	@echo "--- bashrs classify ---"
	@bashrs classify Makefile

# ═══════════════════════════════════════════════════════════
# LINT (bashrs is KING of linting)
# ═══════════════════════════════════════════════════════════

lint: ## Lint all shell artifacts with bashrs + clippy
	@echo "--- bashrs make lint ---"
	@bashrs make lint Makefile
	@echo ""
	@echo "--- bashrs classify Makefile ---"
	@bashrs classify Makefile --multi-label
	@echo ""
	@echo "--- cargo clippy (sovereignty linting) ---"
	@cargo clippy --all-targets -- -D warnings

# ═══════════════════════════════════════════════════════════
# TEST (Rust integration tests)
# ═══════════════════════════════════════════════════════════

test: test-rust ## Run all tests

test-rust: ## Run Rust integration tests (sovereign stack validation)
	@echo "--- cargo test (sovereign stack integration) ---"
	@cargo test --verbose

# ═══════════════════════════════════════════════════════════
# SECURITY (supply chain + advisory audit)
# ═══════════════════════════════════════════════════════════

clippy: ## Run clippy sovereignty linting
	cargo clippy --all-targets -- -D warnings

bench: ## Run benchmarks (performance regression gate)
	@echo "--- cargo bench (sovereign tools) ---"
	cargo bench

security-audit: ## Run cargo-deny supply chain audit
	@echo "--- cargo deny check (license + advisories) ---"
	@cargo deny check

reproduce: ## Reproduce training (gold standard reproducibility)
	@echo "--- Reproduce 50M validation model ---"
	@echo "Config: configs/train/pretrain-50m-quick.yaml"
	@echo "Expected: loss 10.3 → 4.4 in 31 steps"
	apr train apply --task pretrain --config configs/train/pretrain-50m-quick.yaml

# ═══════════════════════════════════════════════════════════
# PLAN (dry-run, estimate resources)
# ═══════════════════════════════════════════════════════════

plan-finetune: ## apr finetune plan for 350M model
	@echo "--- apr finetune plan (350M, 24GB VRAM) ---"
	apr finetune --plan --model-size 350M --vram 24

plan-finetune-lora: ## apr finetune plan with LoRA
	@echo "--- apr finetune plan (350M, LoRA, 24GB VRAM) ---"
	apr finetune --plan --model-size 350M --vram 24 --method lora --rank 16

plan-pretrain-50m: ## Validate 50M pre-training config
	apr train plan --task pretrain --config configs/train/pretrain-50m.yaml

plan-pretrain-350m: ## Validate 350M pre-training config
	apr train plan --task pretrain --config configs/train/pretrain-350m.yaml

# ═══════════════════════════════════════════════════════════
# TRAIN (execute training runs)
# ═══════════════════════════════════════════════════════════

train-50m: ## Train 50M validation model (~2 min)
	@mkdir -p checkpoints/albor-base-50m
	apr train apply --task pretrain --config configs/train/pretrain-50m.yaml

train-350m: ## Train 350M v2 with crash-resilient guard (~12 hours)
	@mkdir -p checkpoints/albor-base-350m-v2 crash-reports
	bash scripts/train-guard.sh configs/train/pretrain-350m-v2.yaml

train-350m-raw: ## Train 350M v2 without guard (debugging)
	@mkdir -p checkpoints/albor-base-350m-v2
	RUST_BACKTRACE=1 CUDA_LAUNCH_BLOCKING=1 apr train apply --task pretrain --config configs/train/pretrain-350m-v2.yaml

# ═══════════════════════════════════════════════════════════
# EVAL (benchmarks and model evaluation)
# ═══════════════════════════════════════════════════════════

eval-validate: ## Validate all canonical solutions in benchmarks
	@echo "--- Validating python-intermediate (15 problems) ---"
	@.venv/bin/python scripts/eval-code.py configs/eval/python-intermediate.jsonl --validate-only
	@echo ""
	@echo "--- Validating humaneval-subset (20 problems) ---"
	@.venv/bin/python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --validate-only

eval-perplexity-50m: ## Evaluate 50M model perplexity (needs checkpoint)
	.venv/bin/python scripts/eval-perplexity.py checkpoints/albor-base-50m/ \
		--data data/pretokenized-128/train/train.parquet \
		--max-sequences 50 --seq-len 128 --threshold 200

eval-suite-350m: ## Run full evaluation suite on 350M model
	bash scripts/run-eval-suite.sh checkpoints/albor-base-350m/

eval-suite-50m: ## Run full evaluation suite on 50M model (quick mode)
	bash scripts/run-eval-suite.sh checkpoints/albor-base-50m/ --quick

eval-perplexity-350m: ## Evaluate 350M model perplexity (needs checkpoint)
	.venv/bin/python scripts/eval-perplexity.py checkpoints/albor-base-350m/ \
		--data data/pretokenized-2048/val/val.parquet \
		--max-sequences 100 --seq-len 2048 --threshold 30

validate-checkpoint-50m: ## Validate 50M checkpoint integrity (ALB-038 detection)
	.venv/bin/python scripts/eval-perplexity.py checkpoints/albor-base-50m/ --validate-checkpoint

validate-checkpoint-350m: ## Validate 350M checkpoint integrity (ALB-038 detection)
	.venv/bin/python scripts/eval-perplexity.py checkpoints/albor-base-350m/ --validate-checkpoint

validate-convergence: ## Validate training convergence (FALSIFY-ALBOR-001)
	.venv/bin/python scripts/validate-training-convergence.py \
		checkpoints/albor-base-350m/training.log

training-status: ## Check 350M training status
	@echo "--- Training Process ---"
	@ps aux | grep "apr train" | grep -v grep || echo "  No training process found"
	@echo ""
	@echo "--- GPU Status ---"
	@nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
	@echo ""
	@echo "--- Training Log (last 10 lines) ---"
	@tail -10 checkpoints/albor-base-350m/training.log 2>/dev/null || echo "  No training log found"

# ═══════════════════════════════════════════════════════════
# BOOK (mdBook build and serve)
# ═══════════════════════════════════════════════════════════

book: ## Build mdBook
	cd docs/book && mdbook build

book-serve: ## Serve mdBook locally
	cd docs/book && mdbook serve --open

spec: ## Regenerate monolithic spec from mdBook chapters
	bash scripts/generate-spec.sh > docs/specifications/albor-llm-spec.md
	@echo "Generated docs/specifications/albor-llm-spec.md ($$(wc -l < docs/specifications/albor-llm-spec.md) lines)"

# ═══════════════════════════════════════════════════════════
# DOGFOOD (exercise tools, report what works and what's blocked)
# ═══════════════════════════════════════════════════════════

dogfood: ## Run full dogfooding suite — exercise all tools
	@echo "==========================================================="
	@echo " Albor Dogfooding Suite"
	@echo "==========================================================="
	@echo ""
	@echo "--- 1. pv validate (contracts) ---"
	@for f in contracts/*.yaml; do \
		echo -n "  $$f: "; \
		pv validate "$$f" 2>&1 | grep -c "is valid" | xargs -I{} sh -c '[ {} -eq 1 ] && echo "PASS" || echo "FAIL"'; \
	done
	@echo ""
	@echo "--- 2. pv coverage ---"
	@pv coverage contracts 2>&1 | tail -6
	@echo ""
	@echo "--- 3. forjar validate (infra-only) ---"
	@forjar validate -f configs/pipeline/infra-only.yaml 2>&1 && echo "  PASS infra-only.yaml" || echo "  FAIL infra-only.yaml"
	@echo ""
	@echo "--- 4. forjar validate (full pipeline — expects failure: ALB-027) ---"
	@forjar validate -f configs/pipeline/albor.yaml 2>&1 && echo "  PASS albor.yaml" || echo "  FAIL albor.yaml (expected: task resource type not yet supported)"
	@echo ""
	@echo "--- 5. apr finetune plan ---"
	@apr finetune --plan --model-size 350M --vram 24 2>&1 | head -15
	@echo ""
	@echo "--- 6. alimentar quality profiles ---"
	@alimentar quality profiles 2>&1 | head -5
	@echo ""
	@echo "--- 7. batuta playbook validate ---"
	@batuta playbook validate configs/pipeline/albor-playbook.yaml 2>&1
	@echo ""
	@echo "--- 8. pv generate (all contracts) ---"
	@for f in contracts/*.yaml; do \
		pv generate "$$f" 2>&1; \
	done
	@echo ""
	@echo "--- 9. bashrs make lint (sovereign Makefile linter) ---"
	@bashrs make lint Makefile 2>&1 | tail -3
	@echo ""
	@echo "--- 10. apr train plan (50M) ---"
	@apr train plan --task pretrain --config configs/train/pretrain-50m.yaml 2>&1 | tail -5
	@echo ""
	@echo "--- 11. apr train plan (350M) ---"
	@apr train plan --task pretrain --config configs/train/pretrain-350m.yaml 2>&1 | tail -5
	@echo ""
	@echo "--- 12. pv audit (all contracts) ---"
	@for f in contracts/*.yaml; do \
		echo -n "  $$f: "; \
		pv audit "$$f" 2>&1 | grep -c "No audit findings" | xargs -I{} sh -c '[ {} -ge 1 ] && echo "PASS" || echo "FINDINGS"'; \
	done
	@echo ""
	@echo "==========================================================="
	@echo " Dogfooding complete. See gap register for blocked items."
	@echo "==========================================================="

dogfood-batuta: ## Run batuta falsification (full 108-check report)
	@batuta falsify . --format markdown 2>&1 | tee docs/falsification-report.md || true

dogfood-playbook: ## Validate batuta playbook
	@batuta playbook validate configs/pipeline/albor-playbook.yaml

# ═══════════════════════════════════════════════════════════
# CLEAN
# ═══════════════════════════════════════════════════════════

clean: ## Clean build artifacts
	rm -rf book-output/ generated/ book/ .pmat/ || exit 1

# ═══════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
