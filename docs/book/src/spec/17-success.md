# 17. Success Criteria

### Minimum Viable (Phase 3 complete)
- [ ] 350M base model trained on 4090 to convergence (val_ppl < 100; target: 5.08B tokens)
- [x] FIM (fill-in-the-middle) training implemented and validated (~~ALB-018~~ FIXED — `alimentar fim` verified)
- [ ] **HumanEval pass@1 > 5%** (baseline Python capability — any valid code generation from sovereign stack)
- [ ] **val_ppl < 50** on codeparrot-clean validation set (syntactic structure captured)
- [ ] Entire pipeline uses only sovereign stack components
- [ ] All training artifacts reproducible from spec
- [ ] All existing kernel contracts pass `pv audit` (Level 2+)
- [ ] `pmat comply check` passes on all modified components

**Current blocker for Phase 3 completion:**
- ALB-042: CUDA runtime errors produce silent loss=0.0 — **OPEN** (workaround: `CUDA_VISIBLE_DEVICES=""`)

**All critical training bugs FIXED (97 gaps closed, 10 open):**
- ~~ALB-038–044, 059–060, 063, 065, 069, 071–074, 079–080, 083, 092, 096–097, 099–120~~ — see §11 gap register
- ALB-121 (shard shuffling) — OPEN, nice-to-have

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
| v13 | 62,000 | 239 (inflated) | 8,400 | 26.4% | **STOPPED** (patience=30) — 2x data overlap from reboot. See §6 post-mortem. |
| v14 | 20,000 | 571 | 8,190 | 23.7% | **KILLED** (plateau) — val_ppl stuck at ~782 for 19K steps. Degenerate init (seed=42). |
| v15 | 47,000 | 309 (pre-outage) | 11,000 | 34.6% | **KILLED** — power outage at step 11K. Post-resume stuck at ~400. Trajectory analysis: -0.017 val_loss/10K — too slow. |
| **v16** | **2,600+** | **—** | **8,269** | **23.9%** | **RUNNING** — seed=456. Canary-validated (PyTorch loss 10.65→9.02). Phase change expected step 3-5K. |

**v16 training (ACTIVE):** From scratch with seed=456. PyTorch canary validated config is sane (loss drops 10.65→9.02 in 50 steps). Same architecture/hyperparameters as v13-v15. ALB-120 + ALB-122 fixes active. v9 remains best genuine result (val_ppl=129). Phase change expected step 3-5K; if absent by step 10K, kill and try seed=789.

### Good (Phase 5 complete)
- [x] Distillation from Qwen3-Coder-30B demonstrated (ALB-010); text-based synthetic data pipeline
- [ ] albor-distill-350m outperforms albor-base-350m on all code benchmarks
- [ ] **HumanEval pass@1 > 10%** (beat CodeGen-350M-mono's 10.2% via distillation from 30B MoE teacher)
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
