# Albor LLM Specification

**Version**: 0.6.0
**Date**: 2026-03-03
**Status**: Phase 3 — 350M Base Model Retraining (ALB-060 fix, v2 data)
**Author**: Noah Gift / Pragmatic AI Labs

> *Albor* (Spanish: "dawn") — A sovereign Python code completion model trained
> from first principles using only the Sovereign AI stack. Python-only following
> the phi-1 playbook: maximum concentration on one language, distilled from
> Qwen3-Coder-Next (80B), then optimized through fine-tuning, merging, pruning,
> and quantization into a fast, local, zero-dependency code completion engine.
> The goal is twofold: produce a **usable Python code assist model** that runs
> anywhere Rust compiles, **and** identify + fix every gap in the stack that
> blocks end-to-end LLM development.

**Latest milestone**: 350M CUDA test training verified — 50 steps, loss
10.39→5.92 (best 5.53), checkpoint loads in realizar, all training stability
contracts pass. First full training run failed (ALB-060: epochs=1 only ran
43/5000 steps). Fixed with C-TRAINCFG-001 contract + v2 config (67,977
sequences, 139M tokens, epochs=38). Qwen2.5-Coder-3B interim teacher validated
for distillation. 24+ upstream gaps fixed across 8 sovereign stack components.

---
