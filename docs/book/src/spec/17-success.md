# 17. Success Criteria

### Minimum Viable (Phase 3 complete)
- [ ] 350M base model trained on 4090 to convergence (~10B tokens, 80% Python)
- [ ] FIM (fill-in-the-middle) training implemented and validated (ALB-018)
- [ ] **HumanEval pass@1 > 8%** (baseline Python capability, beat random)
- [ ] **HumanEval-FIM working** (model can infill Python code)
- [ ] Entire pipeline uses only sovereign stack components
- [ ] All training artifacts reproducible from spec
- [ ] All existing kernel contracts pass `pv audit` (Level 2+)
- [ ] `pmat comply check` passes on all modified components

**Current blockers for Phase 3 completion:**
- ~~ALB-038 (Critical): entrenar saves initialization weights, not trained weights~~ **FIXED** (`entrenar@91ba9da`, `@1ede409`)
- ~~ALB-035: No per-step loss logging during training~~ **FIXED** (`entrenar@5d41a96`)
- ~~ALB-041: D2D buffer mismatch in backward_attention~~ **FIXED** (`entrenar@a48e3d2`)
- ~~ALB-037: realizar ignores loaded weights~~ **FIXED** (e2e verified: `realizar run` loads 350M trained checkpoint, generates tokens from 218 tensors)
- ~~ALB-043 (Critical): backward_ffn buffer overflow + missing SwiGLU gradients~~ **FIXED** (`entrenar@f7805f1`)
- ~~ALB-044 (Critical): activation gradient clipping + CPU optimizer hyperparams~~ **FIXED** (`entrenar@86eec38`)
- ~~ALB-059 (Critical): GEMM backward constructor n/k swapped — buffer overflow into optimizer states~~ **FIXED** (`entrenar@846ae0c`)
- ~~ALB-040: GPU-resident pretraining~~ **VERIFIED** — 350M CUDA test: 50 steps, loss 10.39→5.92, checkpoint valid, realizar inference works
- ALB-042: CUDA runtime errors produce silent loss=0.0 — **OPEN** (workaround: `CUDA_VISIBLE_DEVICES=""`)

**350M CUDA test results (50 steps, post ALB-059 fix):**
- Loss: 10.39 → 5.92 (best: 5.53) — clear convergence with correct GEMM backward
- Training time: ~400s (~8s/step)
- Checkpoint: 1.59 GB SafeTensors, 218 tensors, config.json saved
- Checkpoint validation: PASS (weights trained, layers distinct)
- realizar inference: loads model, generates tokens (gibberish at 50 steps — expected)
- Perplexity: 31,926 (finite; random baseline ~32,768 for vocab 32K)

### Good (Phase 5 complete)
- [ ] Distillation from Qwen3-Coder-Next demonstrated
- [ ] albor-distill-350m outperforms albor-base-350m on all code benchmarks
- [ ] **HumanEval pass@1 > 15%** (beat CodeGen-350M-mono's 12.8% via distillation)
- [ ] **MBPP pass@1 > 12%**
- [ ] **FIM infill working** (qualitatively: model can complete Python between prefix and suffix)
- [ ] KD contract at Level 4 (Kani-proved KL non-negativity)
- [ ] All FALSIFY-ALBOR tests pass (001-006)

### Full Success (Phase 8 complete)
- [ ] All 6 model variants benchmarked (base → distill → instruct → merged → pruned → q4)
- [ ] Benchmark trajectory published showing improvement at each stage
- [ ] **Submitted to Big Code Models Leaderboard** — first sub-1B model on the board
- [ ] **Q4 model: <50ms/token on CPU, <10ms/token on GPU** (code completion latency)
- [x] Critical path gaps (ALB-001, 006, 009, 011, 018) closed with upstream fixes; ALB-010 (Qwen3-Coder-Next) remains OPEN
- [ ] Models published on HuggingFace as `paiml/albor-python-*`
- [ ] Q4 quantized model < 100MB, runs on consumer hardware
- [ ] **All 7 kernel contracts written and verified** (ALB-013–017, ALB-039–040)
- [ ] **batuta falsify: Toyota Standard grade (≥90/108)**
- [ ] **pmat TDG: Grade A on all touched components**
- [ ] **Test coverage ≥ 95%, mutation score ≥ 85% on all new code**
- [ ] **All 9 FALSIFY-ALBOR tests pass**
- [ ] **Verification DAG published via `pv graph`**

### Stretch Goals
- [ ] **HumanEval pass@1 > 20%** (strong distillation result at 350M)
- [ ] **DS-1000 pass@1 > 10%** (data science code generation)
- [ ] Editor integration: VS Code / Neovim / Helix extension using realizar as backend
- [ ] Distributed gradient-parallel training across 4090 + W5700X demonstrated
- [ ] `apr pipeline apply` reproduces entire ladder from bare metal to published model
- [ ] BabyLM 2026 submission using constrained data variant
- [ ] All critical kernels at Level 4 (Kani formal proofs)
- [ ] Lean 4 theorem stubs generated for core training loop invariants
