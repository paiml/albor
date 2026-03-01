# Appendix B: Stack Version Matrix

| Component | Version | Role in Albor |
|-----------|---------|---------------|
| aprender | 0.27.2 | ML library, BPE tokenizer, transformer layers |
| entrenar | 0.7.5 | Training engine, autograd, optimizers, LoRA |
| trueno | 0.16.1 | SIMD/GPU tensor backend |
| realizar | 0.8.0 | Inference engine (teacher model, eval, serving) |
| alimentar | 0.2.6 | Data pipeline, Parquet I/O, HF Hub import |
| repartir | 2.0.3 | Distributed compute (future: gradient sync) |
| forjar | 1.0.0 | Pipeline orchestration (DAG engine, infra + task resources) |
| presentar | 0.3.2 | Training visualization (TUI dashboards, WASM, experiment browser) |
| bashrs (Rash) | 6.65.0 | Shell fragment validation for pipeline task resources |
| batuta | 0.6.6 | Stack orchestration, oracle, falsification checklist |
| provable-contracts | latest | Design-by-contract YAML specs, Kani proofs, falsification tests |
| pmat | latest | TDG scoring, comply check, fault patterns, coverage gaps |
| certeza | latest | Three-tier test effectiveness (unit → property → formal) |
