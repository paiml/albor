# 17. Success Criteria

### Minimum Viable (Phase 3 complete)
- [ ] 350M base model trained on 4090 to convergence (target: ~10B tokens; current: 139M v2 dataset)
- [x] FIM (fill-in-the-middle) training implemented and validated (~~ALB-018~~ FIXED — `alimentar fim` verified)
- [ ] **HumanEval pass@1 > 8%** (baseline Python capability, beat random)
- [ ] **HumanEval-FIM working** (model can infill Python code)
- [ ] Entire pipeline uses only sovereign stack components
- [ ] All training artifacts reproducible from spec
- [ ] All existing kernel contracts pass `pv audit` (Level 2+)
- [ ] `pmat comply check` passes on all modified components

**Current blocker for Phase 3 completion:**
- ALB-042: CUDA runtime errors produce silent loss=0.0 — **OPEN** (workaround: `CUDA_VISIBLE_DEVICES=""`)

**All critical training bugs FIXED (94+ gaps closed):**
- ~~ALB-038–044, 059–060, 063, 065, 069, 071–074, 079–080, 083, 092, 096–097, 099–119~~ — see §11 gap register

**Training run history:**

| Run | Steps | Best val_ppl | tok/s | MFU | Outcome |
|-----|-------|-------------|-------|-----|---------|
| v2 | 1,183 | 1,008 | — | — | Crashed: ALB-073 (PTX selp) + ALB-074 (stale binary) |
| v3 | 28,000 | 1,018 | 6,700 | 19.3% | **STOPPED**: plateau — ALB-079 no cosine decay + ALB-080 batch too small |
| v4 | ~850 | 918 | 3,564 | 10.3% | **STOPPED**: HumanEval 0/164, cosine decay barely engaging |
| v5 | — | — | — | — | **FAILED**: ALB-092 (grad_gamma + uninitialized accum) |
| v6 | 2,000 | 776 | 6,500 | — | **KILLED** for distillation pivot |
| v7 | 550 | ~780 | 6,900 | — | **KILLED** for ALB-097 (checkpoint resume bug) |
| v8 | 5,337 | — | 7,800 | 24.6% | **KILLED**: trained without RoPE (ALB-106) |
| v9 | 14,950 | **129** | 8,200 | 23.8% | **STOPPED** (patience=10): 490M tokens, 7% Chinchilla |
| v10 | 5,058 | 660 | — | — | **KILLED**: ALB-118 — fresh GPU optimizer + low LR |
| v11 | 8,150 | 750 | — | — | **KILLED**: ALB-118 — fresh GPU optimizer |
| v12 | 37 | 5,639 | — | — | **KILLED**: ALB-118 — only CPU embed optimizer restored |
| **v13** | **25,000+** | **286** | **8,308** | **24.1%** | **RUNNING** — best val_ppl=286 (step 25K, NEW BEST). Broke 9K-step plateau (was 308 at step 16K). Cosine decay engaging. Target: 155K steps (5.08B tokens), ETA March 24 |

**v13 training (ACTIVE):** From scratch with RoPE forward+backward (ALB-119), full epoch. 16.1% complete (step 25K/155K). **Best val_ppl=286 at step 25K** — NEW BEST, breaking 9K-step plateau (was 308 at step 16K). 7.1% improvement coincides with early cosine decay (LR=94.7% of peak). Oscillation pattern resolved: spikes (up to 829) are cosmetic — model always recovers within 1-2 evals. B_noise definitively uncorrelated with spikes. Log-linear best-envelope fit (R²=0.67) predicts v9-match (ppl≈129) around step 42K. Projected val_ppl 80-120 at step 155K.

### Good (Phase 5 complete)
- [x] Distillation from Qwen3-Coder-30B demonstrated (ALB-010); text-based synthetic data pipeline
- [ ] albor-distill-350m outperforms albor-base-350m on all code benchmarks
- [ ] **HumanEval pass@1 > 15%** (beat CodeGen-350M-mono's 12.8% via distillation from 30B MoE teacher)
- [ ] **MBPP pass@1 > 12%**
- [ ] **FIM infill working** (qualitatively: model can complete Python between prefix and suffix)
- [ ] KD contract at Level 4 (Kani-proved KL non-negativity)
- [ ] All FALSIFY-ALBOR tests pass (001-006)

### Full Success (Phase 8 complete)
- [ ] All 6 model variants benchmarked (base → distill → instruct → merged → pruned → q4)
- [ ] Benchmark trajectory published showing improvement at each stage
- [ ] **Submitted to Big Code Models Leaderboard** — first sub-1B model on the board
- [ ] **Q4 model: <50ms/token on CPU, <10ms/token on GPU** (code completion latency)
- [x] Critical path gaps (ALB-001, 006, 009, 011, 018) closed with upstream fixes; ALB-010 (Qwen3-Coder-30B MoE inference) FIXED — Q4K GPU, 15 tok/s, synthetic data pipeline working
- [ ] Models published on HuggingFace as `paiml/albor-python-*`
- [ ] Q4 quantized model < 100MB, runs on consumer hardware
- [ ] **All 8 kernel contracts written and verified** (ALB-013–017, ALB-039–040, ALB-060)
- [x] **batuta falsify: Toyota Standard grade (≥90/108)** — ACHIEVED: 100% (108/108 PASS)
- [ ] **pmat TDG: Grade A on all touched components**
- [ ] **Test coverage ≥ 95%, mutation score ≥ 85% on all new code**
- [ ] **All 9 FALSIFY-ALBOR tests pass**
- [ ] **Verification DAG published via `pv graph`**

### Stretch Goals
- [ ] **HumanEval pass@1 > 20%** (strong distillation result at 350M)
- [ ] **DS-1000 pass@1 > 10%** (data science code generation)
- [ ] Editor integration: VS Code / Neovim / Helix extension using realizar as backend
- [ ] Distributed gradient-parallel training across 4090 + W5700X demonstrated (entrenar DDP #133 infra in place)
- [ ] `apr pipeline apply` reproduces entire ladder from bare metal to published model
- [ ] BabyLM 2026 submission using constrained data variant
- [ ] All critical kernels at Level 4 (Kani formal proofs)
- [ ] Lean 4 theorem stubs generated for core training loop invariants
