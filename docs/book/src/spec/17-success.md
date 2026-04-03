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

**Phase 3 progress:**
- v28 training at step 8.3K/38K, val_ppl=38.53 (best), predicted ~29 at completion
- val_ppl < 50 target **MET** (38.53 at step 6K)
- val_ppl < 100 target **MET** (98.56 at step 1K)
- HumanEval > 5% — **PENDING** (0% on v4 baseline, awaiting v28 checkpoint eval)

**129+ gaps filed, 52 contracts validated:**
- All critical training bugs fixed (ALB-038–132) — see §11 gap register
- Key fixes: ALB-078 (fused grad clip), ALB-128 (HPO), ALB-129 (cosine horizon)

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
| v16 | 2,600+ | — | 8,269 | 23.9% | **KILLED** — seed=456, superseded by HPO-era runs |
| v22 | 7K | 9.44 (memorized) | — | — | **STOPPED**: Overfitting without shuffle |
| v23 | 10K | 50.38 | 8,200 | 25.7% | **STOPPED**: HPO baseline |
| v27 | 10.2K | 9.39→82 | 14,700 | 46.1% | **STOPPED**: ALB-129 cosine horizon broken |
| v28 (orig) | 5.4K | **5.88** | 14,700 | 46.1% | **KILLED** by cargo-killer, best val_ppl ever |
| **v28 (fresh)** | **8.3K** | **38.53** | **11,000** | **36.3%** | **RUNNING** — HPO-validated, ALB-078/129 fixed. Plateau 38-42, predicted ~29 at 38K. |

**v28 training (ACTIVE):** Fresh start with all fixes (ALB-078 fused grad clip, ALB-129 cosine horizon, ALB-130-132 checkpoint fixes). HPO-validated hyperparameters (C-HPO-001): lr=7.35e-5, ga=32, wd=0.012. Step 8.3K/38K (~22%), ETA ~3 days. After v28: v29 on filtered data (2.04B tokens), then distillation.

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
