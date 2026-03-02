# Appendix B: Stack Version Matrix

*Last verified: 2026-03-02*

| Component | Version | Role in Albor |
|-----------|---------|---------------|
| aprender (`apr`) | 0.4.10 (7c27c2b3) | Unified CLI: train, tokenize, eval, distill, merge, export, publish, pipeline |
| entrenar | 0.7.5 (with local patches: ALB-038/041/043/044 fixes) | Training engine, autograd, CudaTransformerTrainer, optimizers, LoRA |
| trueno | 0.16.1 | SIMD/GPU tensor backend |
| realizar | 0.8.0 | Inference engine (SafeTensors loading, teacher model, eval, serving) |
| alimentar | 0.2.6 | Data pipeline, Parquet I/O, HF Hub import, FIM transforms, mixing |
| repartir | 2.0.3 | Distributed compute (future: gradient sync) |
| forjar | 1.0.0 | Pipeline orchestration (DAG engine, infra + task resources) |
| presentar | 0.3.2 | Training visualization (TUI dashboards, WASM, experiment browser) |
| bashrs (Rash) | 6.65.0 | Makefile lint/purify/classify, shell safety, pipeline command validation |
| batuta | 0.7.2 | Stack orchestration, oracle, falsification (108 checks), playbook DAG engine |
| provable-contracts (`pv`) | 0.1.0 | Design-by-contract YAML specs, Kani proofs, falsification tests |
| pmat | 3.6.1 | TDG scoring, comply check, fault patterns, coverage gaps |
| certeza | latest | Three-tier test effectiveness (unit → property → formal) |
| renacer | latest | Tracing infrastructure (BrickTracer, spans, metric events) |

**Note**: `apr` uses `[patch.crates-io]` to override entrenar/realizar with
local paths. The installed entrenar 0.7.5 includes unpublished fixes for
ALB-038, ALB-041, ALB-043, ALB-044 (gradient flow, buffer sizes, activation
clipping, optimizer hyperparams).

