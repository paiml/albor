# 15. Implementation Phases

### Phase 0: Pipeline Manifest, Contracts & Quality Baseline (Week 1)
- [ ] Write `configs/pipeline/albor.yaml` — full pipeline manifest (infra + data + train + eval + publish)
- [ ] `apr pipeline plan` — validate entire DAG, estimate resources
- [ ] `apr pipeline apply --target cuda-driver --target vulkan-driver --target data-dir` — provision infra
- [ ] Verify `trueno` wgpu on W5700X via Vulkan (not Metal — Linux)
- [ ] Verify `trueno` CUDA on 4090
- [x] ~~Download Qwen3-Coder-30B~~ DONE — Q4K APR on lambda (17 GB), 15 tok/s GPU inference
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
- [x] Validate `apr monitor` — ALB-025 FIXED (presentar widget migration complete)
- [ ] Validate Andon alerts during full training run
- [x] ~~Fix ALB-009~~ FIXED
- [x] Verify FALSIFY-ALBOR-001 (loss decreases) — CORROBORATED
- [x] Verify FALSIFY-ALBOR-002 (gradient bounds) — per-step logging now available (~~ALB-035~~ FIXED)
- [x] `pv audit` — PASS: 7/7 contracts, 0 findings
- [x] **Milestone**: Training loop converges ✓, contracts pass ✓

### Phase 3: Base Model — 350M Pre-Training (Week 2-4) — v13 RUNNING
- [x] Write `configs/train/pretrain-350m.yaml` — pre-tokenized ByteLevel BPE v2, 22K×2048 tokens
- [x] Train albor-base-350m on 4090 — STARTED (2760 batches, ~20h est.)
- [x] Build evaluation infrastructure — eval-code.py, eval-perplexity.py, 35 benchmark problems
- [x] ~~Fix ALB-038~~ FIXED — RMSNorm + attention backward ops, all 20 params receive gradients
- [x] ~~Fix ALB-041~~ FIXED — D2D buffer size mismatch in backward_attention (`entrenar@a48e3d2`)
- [x] ~~Fix ALB-043~~ FIXED — backward_ffn buffer overflow + SwiGLU gradients (`entrenar@f7805f1`)
- [x] ~~Fix ALB-044~~ FIXED — activation gradient clipping at GPU-CPU boundary + CPU optimizer hyperparams (`entrenar@86eec38`)
- [x] ~~Fix ALB-059~~ FIXED — GEMM backward constructor args n/k swapped, buffer overflow into optimizer states + zero-init optimizer m/v (`entrenar@846ae0c`)
- [x] Write `training-memory-kernel-v1.yaml` contract (ALB-039) — VRAM budget estimation
- [x] Write `training-gpu-kernel-v1.yaml` contract (ALB-040) — GPU-resident training invariants
- [x] Implement `CudaTransformerTrainer` (ALB-040) — 3 PCIe transfers/step vs ~16K
- [x] Dogfood CUDA training — 50M test: 3 steps, loss 10.4→11.7, GPU forward+backward working
- [x] ~~ALB-037~~ FIXED — realizar loads trained SafeTensors checkpoint, generates tokens (e2e verified)
- [x] 350M CUDA test training — 50 steps, loss 10.39→5.92 (best 5.53), checkpoint valid
- [x] realizar inference verified — 218 tensors loaded, generates from trained weights
- [x] Checkpoint validation: PASS (weights trained, not initialization)
- [x] Perplexity eval: 31,926 (finite, consistent with 50-step model — random baseline ~32,768)
- [x] ~~Fix ALB-060~~ CONFIG FIXED — epochs=1 only ran 43/5000 steps. C-TRAINCFG-001 contract written. Config fixed (v1: epochs=117, v2: epochs=1 with 68K seqs)
- [x] Expand training data: Tier 1 10x + 8 Tier 2 repos → v2 dataset (67,977 seqs, 139M tokens)
- [x] ~~Fix ALB-071~~ FIXED — embed gradient clipping decoupled from weight grad_clip (`entrenar@d07d67d`)
- [x] ~~Fix ALB-072~~ FIXED — fp16 loss scaling (65536x) removed from fused CE kernel; all backward uses f32, no underflow risk (`entrenar@44d3e74`)
- [x] Full 350M v2 training — reached step 1183/5000, loss 10.40→6.85, val_ppl=1008. Crashed: ALB-073 (PTX selp) + ALB-074 (buffer overflow from stale binary). Step 1000 checkpoint saved (1520 MB).
- [x] ~~Fix ALB-073~~ FIXED — fused_cross_entropy selp arg order, same class as ALB-069 (`trueno@10bec89`)
- [x] ~~Fix ALB-074~~ FIXED — stale binary missed eval truncation fix. Rebuilt with `entrenar@5c4c2d8`.
- [x] Monitor training via `apr monitor` (ALB-025 FIXED)
- [x] **Data scaling**: codeparrot-clean downloaded (2M files) → pretokenized at 1024 → 19 shards, ~4.9M seqs, 5.0B tokens
- [x] v3 training — 28K steps, loss 6.43, val_ppl=1018, 6.7K tok/s, 19.3% MFU. **STOPPED**: plateau (ALB-079 no cosine decay + ALB-080 batch too small)
- [x] ~~Fix ALB-079~~ FIXED — cosine LR schedule (`entrenar` PR #241)
- [x] ~~Fix ALB-080~~ FIXED — batch size scaling via gradient_accumulation (`contracts/batch-size-scaling-v1.yaml`)
- [x] v4 training — 500+ steps with cosine decay + ga=32. val_ppl=918. **STOPPED**: HumanEval 0/164
- [x] ~~Fix ALB-092~~ FIXED — RMSNorm grad_gamma + GPU accum uninitialized (`trueno` PR #178, `entrenar` PR #257)
- [x] v5 training — **FAILED**: two bugs above. Fixed in ALB-092
- [x] v6 training — 2000 steps, val_ppl=776, 6.5K tok/s. **KILLED** for distillation pivot
- [x] ~~Fix ALB-096~~ FIXED — APR checkpoint format for training (`entrenar@604f32f`)
- [x] ~~Fix ALB-097~~ FIXED — tied LM head not saved in checkpoint (`entrenar@604f32f`)
- [x] v7 training — 550 steps, loss ~6.62, 6.9K tok/s. **KILLED** for checkpoint resume bug fix (ALB-097: checkpoint cannot resume)
- [x] ~~Fix ALB-099~~ FIXED — dhat-rs profiling: 15 memory issues across 5 repos
- [x] ~~Fix ALB-100~~ FIXED — LMBatch dedup (`entrenar@8e62668`)
- [x] ~~Fix ALB-101~~ FIXED — Streaming Parquet loader (`entrenar@86e0faa`)
- [x] ~~Fix ALB-102~~ FIXED — KV cache GQA sizing (`realizar@747d921`)
- [x] ~~Fix ALB-103~~ FIXED — sample_topk zero-alloc (`realizar@747d921`)
- [x] ~~Fix ALB-104~~ FIXED — APR reader cached offset (`aprender@d4c5a4c6`)
- [x] ~~Fix ALB-105~~ FIXED — APR streaming writer (`aprender@d4c5a4c6`)
- [x] v8 training — 5,337 steps, 7.8K tok/s, 24.6% MFU. **KILLED**: trained without RoPE (ALB-106)
- [x] ~~Fix ALB-106~~ FALSIFIED — RoPE IS applied in CUDA training (in-place batched kernel). Eval confirmed consistency.
- [x] v9 training — 14,950 steps, val_ppl=**129**, 8.2K tok/s, 23.8% MFU. **STOPPED** (patience=10): 490M tokens, only 7% Chinchilla.
- [x] v10/v11/v12 continuation failures — all KILLED. Root cause: ALB-118 (GPU optimizer state not checkpointed).
- [x] ~~Fix ALB-118~~ VERIFIED — GPU optimizer state (438 tensors, 2.3 GB) now saved in APR checkpoints.
- [x] ~~Fix ALB-119~~ FIXED — RoPE backward (inverse rotation) added to both F32 and NF4 paths.
- [x] v13 training — **RUNNING** (155K steps target, 5.08B tokens). Phase change at step 4000: val_ppl 812→**499** (outperforming v9 by 25%). 8.3K tok/s, 23.9% MFU.
- [ ] HumanEval pass@1 evaluation (target >8%) — eval pipeline ready, awaiting checkpoint
- [ ] Verify FALSIFY-ALBOR-003 (checkpoint determinism)
- [ ] `pmat tdg check-regression` on all touched components
- [ ] **Milestone**: HumanEval pass@1 > 8%, Perplexity < 30, TDG grade A maintained

### Phase 4: Teacher Setup & Distillation Data (Week 3-5) — COMPLETE
- [x] ~~Fix ALB-010~~ FIXED — Qwen3-Coder-30B MoE inference in realizar (Q4K, 15 tok/s GPU)
- [x] Download Qwen3-Coder-30B-A3B-Instruct teacher (17 GB Q4K APR)
- [x] Validate teacher: FIM-capable, SWE-bench 50.3%, Apache 2.0
- [x] Create distillation config: `configs/train/distill-30b.yaml` (synthetic data generation)
- [x] Write `knowledge-distillation-kernel-v1.yaml` contract (ALB-013) — DOGFOODING
- [x] ~~Fix ALB-011~~ FIXED — `apr distill --config --stage precompute|train` works
- [x] Generate synthetic data: 10,043 completions, 5.8M tokens from Qwen3-Coder-30B teacher
- [x] Create mixed training data: 56,660 seqs (5,666 synthetic + 50,994 codeparrot), 58M tokens
- [x] distill-v3 training: 2,400 steps, val_ppl=658. HumanEval 0% — insufficient tokens + raw format
- [x] **Milestone**: Teacher inference working, synthetic data pipeline demonstrated
- [ ] `pv kani` on KD loss contract (KL non-negativity, temperature scaling)
- [ ] **Future**: Re-run distillation after v13 base model converges

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
- [x] entrenar native DDP infrastructure (TCP wire protocol v2, GradientServer, WorkerClient, PerBlockGradientAccumulator, RingAllReduce) — entrenar#133
- [x] Wire DDP train_batch() into DistributedCudaTrainer — COMPLETE (train_loop_cuda_distributed, allreduce_impl, spawn_coordinator_thread)
- [x] Multi-process launcher — COMPLETE (rank 0 auto-spawns GradientServer, all ranks connect as WorkerClient via `--distributed` CLI flags)
- [ ] wgpu backward pass in trueno (ALB-005) — for cross-vendor GPU support
- [ ] Full distributed training: 4090 + W5700X x2
- [ ] **Milestone**: Multi-GPU training demonstrated
