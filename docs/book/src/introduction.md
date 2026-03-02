# Albor LLM Specification

**Version**: 0.5.0
**Date**: 2026-03-02
**Status**: Phase 3 — 350M Base Model Training (GPU-Resident, In Progress)
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
10.39→6.07, checkpoint loads in realizar, all training stability contracts
pass. Full training running on RTX 4090 (~17h ETA). 22 upstream gaps fixed
across 8 sovereign stack components.

---
