# 14. Batuta Falsification Checklist

### 14.1 108-Item Popperian Assessment

The Albor project itself is subject to batuta's 108-item falsification checklist:

```bash
# Full assessment
batuta falsify . --verbose --format markdown --output docs/falsification-report.md

# Critical-only (blocks release)
batuta falsify . --critical-only

# CI-friendly output
batuta falsify . --format github-actions --min-grade kaizen-required
```

### 14.2 Key Sections Applied to Albor

**Section 1: Sovereign Data Governance (SDG)**
- All training data has documented provenance (HuggingFace commit SHAs)
- No PII in training corpus (alimentar quality check)
- Data residency: all data stored on owned hardware (lambda + intel)
- Teacher model license verified (Apache 2.0)

**Section 3: Hypothesis-Driven Development (HDD)**
- Each improvement stage has a falsifiable hypothesis:
  - "Distillation improves avg benchmark by >5%" (FALSIFY-ALBOR-005)
  - "Pruning at 50% sparsity degrades benchmarks by <2%" (FALSIFY-ALBOR-008)
  - "Q4 quantization degrades perplexity by <5%" (FALSIFY-ALBOR-009)
- Reproducibility standard: **Gold** (deterministic seeds, versioned data,
  BLAKE3 checkpoint hashes, Cargo.lock pinning)

**Section 4: Numerical Reproducibility (NR)**
- Float determinism enforced via fixed seeds and operator ordering
- Cross-platform consistency: checkpoint trained on lambda loads on intel
- SIMD parity: all kernels have provable-contracts SIMD equivalence obligations

**Section 5: Performance & Waste Elimination (PW)**
- Seven Wastes (Muda) applied to training pipeline:
  - No redundant data copies (alimentar streaming)
  - No idle GPU time (pre-computed teacher logits)
  - No over-processing (progressive model sizing: 50M → 125M → 350M)

**Section 6: Safety & Formal Verification (SF)**
- Critical kernels have Kani proofs (softmax, attention, cross-entropy)
- New kernels (KD loss, gradient accumulation) get Kani harnesses

**Section 10: Architectural Invariants (AI) — CRITICAL**
- AI-01: All model operations use apr (no manual weight manipulation)
- AI-02: Every checkpoint is BLAKE3-hashed and version-tracked
- AI-03: Training config is immutable once committed (no runtime overrides)
- AI-04: Eval results are reproducible (fixed seed, deterministic batching)
- AI-05: No undeclared dependencies (Cargo.lock enforced)

### 14.3 Current Grade

**Perfect Score: 100.0% (108/108 PASS)** — achieved 2026-03-04.

This exceeds the Toyota Standard (90-100%) target:
- All 5 Critical items pass (Section 10)
- All Major items pass
- All Minor items pass
- Zero PARTIAL, zero FAIL

Score progression across 14 MLOps survey batches: 34% → 100%
(see `entrenar/docs/specifications/world-class-mlops-survey.md`).
