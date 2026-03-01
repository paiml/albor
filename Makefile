# Albor Makefile — Development and Dogfooding Commands
#
# This Makefile exercises the sovereign stack tools against albor's
# configs, contracts, and pipeline manifest. It documents what works
# and what's blocked by upstream gaps.
#
# NOTE: This is a development convenience, NOT the pipeline orchestrator.
# The pipeline is `apr pipeline plan/apply configs/pipeline/albor.yaml`.

.PHONY: validate validate-contracts validate-forjar validate-yaml \
        plan book book-serve dogfood clean help

# ═══════════════════════════════════════════════════════════
# VALIDATION (all tools, no side effects)
# ═══════════════════════════════════════════════════════════

validate: validate-yaml validate-contracts validate-forjar ## Run all validation
	@echo ""
	@echo "✅ All validation passed"

validate-yaml: ## Validate YAML syntax for all configs and contracts
	@echo "─── YAML Syntax ───"
	@for f in configs/**/*.yaml configs/*.yaml contracts/*.yaml; do \
		python3 -c "import yaml; yaml.safe_load(open('$$f'))" && echo "  ✅ $$f" || exit 1; \
	done

validate-contracts: ## Validate all contracts with pv
	@echo "─── pv validate ───"
	@for f in contracts/*.yaml; do \
		pv validate "$$f" 2>&1 | tail -1; \
	done
	@echo ""
	@echo "─── pv coverage ───"
	@pv coverage contracts
	@echo ""
	@echo "─── pv graph ───"
	@pv graph contracts

validate-forjar: ## Validate infra manifest with forjar
	@echo "─── forjar validate (infra-only) ───"
	@forjar validate -f configs/pipeline/infra-only.yaml
	@echo ""
	@echo "─── forjar graph ───"
	@forjar graph -f configs/pipeline/infra-only.yaml

# ═══════════════════════════════════════════════════════════
# PLAN (dry-run, estimate resources)
# ═══════════════════════════════════════════════════════════

plan-finetune: ## apr finetune plan for 350M model
	@echo "─── apr finetune plan (350M, 24GB VRAM) ───"
	apr finetune --plan --model-size 350M --vram 24

plan-finetune-lora: ## apr finetune plan with LoRA
	@echo "─── apr finetune plan (350M, LoRA, 24GB VRAM) ───"
	apr finetune --plan --model-size 350M --vram 24 --method lora --rank 16

# ═══════════════════════════════════════════════════════════
# BOOK (mdBook build and serve)
# ═══════════════════════════════════════════════════════════

book: ## Build mdBook
	cd docs/book && mdbook build

book-serve: ## Serve mdBook locally
	cd docs/book && mdbook serve --open

# ═══════════════════════════════════════════════════════════
# DOGFOOD (exercise tools, report what works and what's blocked)
# ═══════════════════════════════════════════════════════════

dogfood: ## Run full dogfooding suite — exercise all tools
	@echo "═══════════════════════════════════════════════════════"
	@echo " Albor Dogfooding Suite"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@echo "─── 1. pv validate (contracts) ───"
	@for f in contracts/*.yaml; do \
		echo -n "  $$f: "; \
		pv validate "$$f" 2>&1 | grep -c "is valid" | xargs -I{} sh -c '[ {} -eq 1 ] && echo "✅" || echo "❌"'; \
	done
	@echo ""
	@echo "─── 2. pv coverage ───"
	@pv coverage contracts 2>&1 | tail -6
	@echo ""
	@echo "─── 3. forjar validate (infra-only) ───"
	@forjar validate -f configs/pipeline/infra-only.yaml 2>&1 && echo "  ✅ infra-only.yaml" || echo "  ❌ infra-only.yaml"
	@echo ""
	@echo "─── 4. forjar validate (full pipeline — expects failure: ALB-027) ───"
	@forjar validate -f configs/pipeline/albor.yaml 2>&1 && echo "  ✅ albor.yaml" || echo "  ❌ albor.yaml (expected: task resource type not yet supported)"
	@echo ""
	@echo "─── 5. apr finetune plan ───"
	@apr finetune --plan --model-size 350M --vram 24 2>&1 | head -15
	@echo ""
	@echo "─── 6. alimentar quality profiles ───"
	@alimentar quality profiles 2>&1 | head -5
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo " Dogfooding complete. See gap register for blocked items."
	@echo "═══════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════
# CLEAN
# ═══════════════════════════════════════════════════════════

clean: ## Clean build artifacts
	rm -rf book-output/

# ═══════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
