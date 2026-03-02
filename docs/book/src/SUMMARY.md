# Summary

[Introduction](./introduction.md)

# Specification

- [Objectives](./spec/01-objectives.md)
- [Hardware Inventory](./spec/02-hardware.md)
- [Model Architecture](./spec/03-architecture.md)
- [Distillation Teacher](./spec/04-distillation.md)
- [Training Data](./spec/05-data.md)
- [Training Configuration](./spec/06-training.md)
- [Post-Training Improvement Ladder](./spec/07-improvement-ladder.md)
- [Evaluation & Benchmarks](./spec/08-evaluation.md)
- [Distributed Training](./spec/09-distributed.md)
- [Pipeline Orchestration](./spec/10-pipeline.md)
- [Gap Register](./spec/11-gaps.md)
- [Provable Quality](./spec/12-quality-contracts.md)
- [pmat Compliance](./spec/13-pmat.md)
- [Batuta Falsification](./spec/14-batuta.md)
- [Implementation Phases](./spec/15-phases.md)
- [Reproducibility Protocol](./spec/16-reproducibility.md)
- [Success Criteria](./spec/17-success.md)
- [Reference Commands](./spec/18-commands.md)

# Kernel Contracts (pv generated)

- [Knowledge Distillation](./contracts/knowledge-distillation-kernel-v1.md)
- [BPE Tokenizer](./contracts/bpe-tokenizer-kernel-v1.md)
- [Gradient Accumulation](./contracts/gradient-accumulation-kernel-v1.md)
- [Model Merging](./contracts/model-merging-kernel-v1.md)
- [Pruning](./contracts/pruning-kernel-v1.md)

# Model Cards

- [albor-base-50m](./model-cards/albor-base-50m.md)

# Appendices

- [Batuta Oracle](./appendix/a-oracle.md)
- [Stack Version Matrix](./appendix/b-versions.md)
- [Qwen3-Coder-Next Architecture](./appendix/c-qwen3.md)
- [W5700X Vulkan Validation](./appendix/d-vulkan.md)
- [Leaderboard Strategy](./appendix/e-leaderboard.md)
- [Dogfooding Log](./appendix/f-dogfooding.md)
- [Data Pipeline](./appendix/g-data-pipeline.md)
