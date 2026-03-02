# 15. Implementation Phases

### Phase 0: Pipeline Manifest, Contracts & Quality Baseline (Week 1)
- [ ] Write `configs/pipeline/albor.yaml` — full pipeline manifest (infra + data + train + eval + publish)
- [ ] `apr pipeline plan` — validate entire DAG, estimate resources
- [ ] `apr pipeline apply --target cuda-driver --target vulkan-driver --target data-dir` — provision infra
- [ ] Verify `trueno` wgpu on W5700X via Vulkan (not Metal — Linux)
- [ ] Verify `trueno` CUDA on 4090
- [ ] Download Qwen3-Coder-Next to intel box, verify it loads in realizar
- [ ] `pmat tdg baseline create` on all stack components
- [ ] `pv coverage contracts/ --binding` — establish contract coverage baseline
- [ ] `batuta falsify . --critical-only` — initial falsification assessment

### Phase 1: Data Pipeline + Tokenizer Contract (Week 1-2)
- [ ] Ingest local ground truth corpora via `alimentar import local` (fix ALB-019 if needed)
  - [ ] depyler: examples/ + tdd-book/tests/ (~1,845 files, ~219K lines)
  - [ ] hf-ground-truth-corpus (~11,928 files)
  - [ ] jax-ground-truth-corpus (~2,697 files)
  - [ ] vllm-ground-truth-corpus (~1,118 files)
- [ ] Ingest local ML framework code (Tier 2, ~53K files)
- [ ] Download external datasets via `alimentar import hf` (StarCoder Python, FineWeb-Edu)
- [ ] Quality validation via `alimentar quality check` on all sources
- [ ] Build weighted training mix with 10x upsampling on Tier 1 (fix ALB-020 if needed)
- [ ] Write `bpe-tokenizer-kernel-v1.yaml` contract (ALB-014)
- [ ] `pv probar` + `pv kani` on tokenizer contract
- [ ] Train BPE tokenizer on mixed corpus (fix ALB-001 if needed)
- [ ] Verify FALSIFY roundtrip: `decode(encode(text)) = text` for all test data
- [ ] Tokenize all data into sharded Parquet
- [ ] Apply FIM transforms to code sequences (fix ALB-018 if needed)
- [ ] Create train/val/test splits via `alimentar`
- [ ] Record SHA-256 hashes + provenance manifest for all data artifacts
- [ ] `pmat comply check --strict` on alimentar changes

### Phase 2: Pipeline Validation — 50M Model (Week 2) -- COMPLETE
- [x] Write `gradient-accumulation-kernel-v1.yaml` contract (ALB-017)
- [x] Write `configs/train/pretrain-50m.yaml` (model arch + training + monitoring)
- [x] Train albor-50M on 4090 — 500 rows, 31 steps, 110.7s, loss 10.3→4.42
- [ ] Validate `apr monitor` — BLOCKED (ALB-025: presentar widget migration)
- [ ] Validate Andon alerts — BLOCKED (ALB-025)
- [x] ~~Fix ALB-009~~ FIXED
- [x] Verify FALSIFY-ALBOR-001 (loss decreases) — CORROBORATED
- [x] Verify FALSIFY-ALBOR-002 (gradient bounds) — per-step logging now available (~~ALB-035~~ FIXED)
- [x] `pv audit` — PASS: 7/7 contracts, 0 findings
- [x] **Milestone**: Training loop converges ✓, contracts pass ✓

### Phase 3: Base Model — 350M Pre-Training (Week 2-4) -- IN PROGRESS
- [x] Write `configs/train/pretrain-350m.yaml` — pre-tokenized ByteLevel BPE v2, 22K×2048 tokens
- [x] Train albor-base-350m on 4090 — STARTED (2760 batches, ~20h est.)
- [x] Build evaluation infrastructure — eval-code.py, eval-perplexity.py, 35 benchmark problems
- [x] ~~Fix ALB-038~~ FIXED — RMSNorm + attention backward ops, all 20 params receive gradients
- [x] ~~Fix ALB-041~~ FIXED — D2D buffer size mismatch in backward_attention (`entrenar@a48e3d2`)
- [x] Write `training-memory-kernel-v1.yaml` contract (ALB-039) — VRAM budget estimation
- [x] Write `training-gpu-kernel-v1.yaml` contract (ALB-040) — GPU-resident training invariants
- [x] Implement `CudaTransformerTrainer` (ALB-040) — 3 PCIe transfers/step vs ~16K
- [x] Dogfood CUDA training — 50M test: 3 steps, loss 10.4→11.7, GPU forward+backward working
- [ ] ALB-037: realizar ignores SafeTensors weights — DOGFOODING (config.json save fixed, pending e2e)
- [ ] Restart 350M training with CUDA trainer (killed due to GPU contention with 4B finetune)
- [ ] Monitor training via `apr monitor` — BLOCKED (ALB-025)
- [ ] Run eval on intel concurrently — BLOCKED (ALB-037)
- [ ] Validate loss curve, perplexity convergence
- [ ] Tune hyperparameters (LR, batch size, warmup)
- [ ] Verify FALSIFY-ALBOR-003 (checkpoint determinism)
- [ ] `pmat tdg check-regression` on all touched components
- [ ] **Milestone**: Perplexity < 30, TDG grade A maintained

### Phase 4: Teacher Setup & Logit Pre-Computation (Week 3-5)
- [ ] Fix ALB-010: Add Qwen3-Coder-Next support to realizar
- [ ] Validate teacher inference on intel (CPU, fp16, 300GB RAM)
- [x] Write `knowledge-distillation-kernel-v1.yaml` contract (ALB-013) — DOGFOODING
- [ ] `pv kani` on KD loss contract (KL non-negativity, temperature scaling)
- [x] ~~Fix ALB-011~~ FIXED — `apr distill --config --stage precompute|train` works
- [ ] Pre-compute teacher logits on curated subset (~500M-2B tokens)
- [ ] Verify FALSIFY-ALBOR-006 (teacher logit integrity)
- [ ] Store as sharded Parquet via alimentar
- [ ] `pmat comply check --strict` on realizar changes
- [ ] **Milestone**: Teacher logits verified, KD contract at Level 4

### Phase 5: Knowledge Distillation (Week 5-6)
- [ ] Implement `apr distill apply` with KD loss
- [ ] Distill albor-base-350m → albor-distill-350m
- [ ] Verify FALSIFY-ALBOR-004 (KL non-negativity in production)
- [ ] Verify FALSIFY-ALBOR-005 (distillation improves benchmarks)
- [ ] Benchmark: measure improvement over base
- [ ] `pv probar --binding` on KD contract with actual training data
- [ ] **Milestone**: >5% avg benchmark improvement, KD contract fully wired

### Phase 6: Post-Training Optimization (Week 6-8)
- [x] Write `model-merging-kernel-v1.yaml` contract (ALB-015) — DOGFOODING
- [x] Write `pruning-kernel-v1.yaml` contract (ALB-016) — DOGFOODING
- [ ] Fine-tune with LoRA: `apr finetune` → albor-instruct
- [ ] Merge variants: `apr merge --method slerp` → albor-merged
- [ ] Verify FALSIFY-ALBOR-007 (SLERP interpolation bound)
- [ ] Prune: `apr prune --method wanda` → albor-pruned
- [ ] Verify FALSIFY-ALBOR-008 (sparsity guarantee)
- [ ] Quantize: `apr quantize --method q4_k` → albor-q4
- [ ] Verify FALSIFY-ALBOR-009 (quantization fidelity)
- [ ] Benchmark every variant
- [ ] `pv coverage contracts/ --binding` — final contract coverage report
- [ ] **Milestone**: Full ladder complete, all post-training contracts pass

### Phase 7: Quality Assurance & Falsification Sweep (Week 8)
- [ ] `batuta falsify . --min-grade toyota-standard --verbose` — full 108-item assessment
- [ ] `pmat rust-project-score --full` on all touched components
- [ ] `pmat tdg check-regression --baseline` — no quality regressions
- [ ] `pv graph contracts/ --format mermaid` — publish verification DAG
- [ ] `pv status contracts/` — all contracts at Level 3+, critical at Level 4
- [ ] `cargo mutants --no-times` on all new code — mutation score ≥ 85%
- [ ] `cargo llvm-cov` — coverage ≥ 95% on all new code
- [ ] Address any falsification failures or contract violations
- [ ] **Milestone**: Toyota Standard grade, all quality gates green

### Phase 8: Evaluation, Leaderboard Submission & Publication (Week 8-9)
- [ ] Final eval on all benchmark tasks (all 6 model variants)
- [ ] Run `bigcode-evaluation-harness` with leaderboard-standard params on best model
- [ ] Submit PR to Big Code Models Leaderboard (`community_results/` folder)
- [ ] Export all models: SafeTensors + GGUF
- [ ] `apr publish` to HuggingFace Hub as `paiml/albor-*`
- [ ] Write model card with full reproducibility details + leaderboard results
- [ ] Publish training logs, loss curves, eval trajectories
- [ ] Publish verification report (contract status, falsification results)
- [ ] `batuta falsify . --format markdown --output docs/falsification-report.md`
- [ ] **Milestone**: Models on HuggingFace, leaderboard submission live, quality evidence published

### Phase 9: Distributed Training — Stretch (Week 9+)
- [ ] Implement ring all-reduce in repartir (ALB-002)
- [ ] Wire into apr training loop (ALB-003)
- [ ] wgpu backward pass in trueno (ALB-005)
- [ ] Full distributed training: 4090 + W5700X x2
- [ ] **Milestone**: Multi-GPU training demonstrated
