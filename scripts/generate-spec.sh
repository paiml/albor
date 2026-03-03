#!/usr/bin/env bash
# Generate monolithic spec from mdBook chapters.
# Usage: bash scripts/generate-spec.sh > docs/specifications/albor-llm-spec.md
#
# This script concatenates the mdBook chapters in order, stripping the
# per-file H1 headers and replacing them with H2 to create a single
# coherent document. The introduction serves as the document header.

set -euo pipefail

BOOK_SRC="docs/book/src"
OUT="docs/specifications/albor-llm-spec.md"

# Header from introduction
cat "$BOOK_SRC/introduction.md"
echo ""
echo "---"
echo ""

# Spec chapters (in order from SUMMARY.md)
spec_chapters=(
    "spec/01-objectives.md"
    "spec/02-hardware.md"
    "spec/03-architecture.md"
    "spec/04-distillation.md"
    "spec/05-data.md"
    "spec/06-training.md"
    "spec/07-improvement-ladder.md"
    "spec/08-evaluation.md"
    "spec/09-distributed.md"
    "spec/10-pipeline.md"
    "spec/11-gaps.md"
    "spec/12-quality-contracts.md"
    "spec/13-pmat.md"
    "spec/14-batuta.md"
    "spec/15-phases.md"
    "spec/16-reproducibility.md"
    "spec/17-success.md"
    "spec/18-commands.md"
)

for chapter in "${spec_chapters[@]}"; do
    echo ""
    cat "$BOOK_SRC/$chapter"
    echo ""
    echo "---"
    echo ""
done

# Appendices
echo ""
echo "# Appendices"
echo ""

appendices=(
    "appendix/a-oracle.md"
    "appendix/b-versions.md"
    "appendix/c-qwen3.md"
    "appendix/d-vulkan.md"
    "appendix/e-leaderboard.md"
    "appendix/f-dogfooding.md"
    "appendix/g-data-pipeline.md"
)

for appendix in "${appendices[@]}"; do
    echo ""
    cat "$BOOK_SRC/$appendix"
    echo ""
    echo "---"
    echo ""
done
