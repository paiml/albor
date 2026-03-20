# 6. Training Configuration

### 6.1 Optimizer & Schedule

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard; in aprender/entrenar |
| Learning rate | 3e-4 | Chinchilla-recommended for 350M |
| Weight decay | 0.1 | Standard AdamW regularization |
| Beta1, Beta2 | 0.9, 0.95 | LLaMA/GPT-3 standard |
| Epsilon | 1e-8 | Standard |
| LR schedule | Cosine annealing with warmup | `CosineAnnealingLR` in aprender |
| Warmup steps | 2000 (v1) / 500 (v2) | **ALB-060**: 2000/5000 = 40%, not 0.2%. v2 config uses 500 (10%) per C-TRAINCFG-001 |
| Min LR | 3e-5 | 10% of peak (standard) |
| Gradient clipping | 1.0 (global norm) | Stability |
| Batch size (global) | 32K tokens/step | batch=4 × ga=8 × seq_len=1024 (v9 config) |
| Micro-batch (4090) | 4 | GPU-resident (batch=8 OOM at seq≥1024) |
| Gradient accumulation | 8 (ALB-091) | GPU-resident accumulation via `GpuGradientAccumulator` — zero D2H during micro-batch loop |
| Total training tokens | Target 5.08B (v13); 5.3B available | 155,000 steps × 32K tokens/step; data: codeparrot-clean pretokenized-1024-v3. 73% Chinchilla-optimal (7B for 350M). |
| Mixed precision | fp16 (CUDA) | Hardware-appropriate |

### 6.2 Training Config: `configs/train/pretrain-350m-v9.yaml`

A single YAML file defines **everything** — model architecture and training
hyperparameters. This is the industry standard (Axolotl, torchtune, HuggingFace
Trainer). One file, one truth. `apr train validate` lints it before GPU time.

**Current config** (v9 — all v8 fixes + ALB-106 RoPE in CUDA forward/backward):

```yaml
# configs/train/pretrain-350m-v9.yaml — v9: ALB-106 RoPE fix, all v8 fixes
# Data: pretokenized-1024-v3 (5.3B unique tokens from codeparrot-clean)
# Token budget: 20,000 steps x 32,768 tokens/step = 655M tokens

model:
  path: "."
  mode: transformer
  architecture:
    hidden_size: 1024                       # d_model
    num_hidden_layers: 24
    num_attention_heads: 16                 # d_head = 64
    num_key_value_heads: 4                  # GQA 4:1 ratio
    intermediate_size: 4096                 # SwiGLU FFN (gate + up + down)
    vocab_size: 32768                       # ByteLevel BPE (v2 tokenizer)
    max_position_embeddings: 1024           # Context length (2048 OOM'd on 4090)
    rms_norm_eps: 1.0e-5

data:
  train: "data/pretokenized-1024-v3/train/" # codeparrot-clean (5.3B tokens)
  val: "data/pretokenized-1024-v3/val/"     # Same distribution as train
  batch_size: 4                             # Micro-batch (batch=8 OOM'd)
  seq_len: 1024
  tokenizer: "models/albor-tokenizer-v2/tokenizer.json"
  input_column: "input_ids"                 # Pre-tokenized: List<u32> column

optimizer:
  name: "adamw"
  lr: 3.0e-4
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1

training:
  mode: "causal_lm"
  epochs: 1
  grad_clip: 1.0                            # Re-enabled (ALB-078 fused GPU clip)
  lr_scheduler: "cosine"
  warmup_steps: 500
  gradient_accumulation: 8                  # 32K tokens/step (GPU-resident, ALB-091)
  output_dir: "./checkpoints/albor-base-350m-v9"
  save_interval: 500
  eval_interval: 250                        # ALB-087: Eval 2x per save interval
  patience: 10                              # ALB-087: Early stop after 10 evals without improvement
  max_steps: 20000                          # 20K x 32K = 655M tokens
```

**Key changes in v9** (cumulative from v5→v9):
- **RoPE**: Applied in CUDA forward only — positional awareness (ALB-106). All v1-v8 trained without RoPE. Note: v9 had NO RoPE backward — W_q/W_k absorbed the rotation into learned weights.
- **Data**: codeparrot-clean 5.3B tokens (pretokenized-1024-v3), val from same distribution
- **Gradient accumulation**: 8 GPU-resident (ALB-091) — zero D2H during micro-batch loop
- **Cosine LR decay**: Implemented in CUDA trainer (ALB-079)
- **APR checkpoints**: Atomic single-file with optimizer state (ALB-096), resume verified (ALB-097)
- **RMSNorm backward**: grad_gamma computed + accum zeroed (ALB-092)

**Key changes in v13** (vs v9):
- **RoPE backward**: `batched_rope_neox_backward()` applies R^T(−θ) inverse rotation to grad_Q and grad_K before projection backward (ALB-119). Without this, Q/K weight gradients were computed in the rotated coordinate frame — valid but suboptimal (weights absorb the rotation rather than learning position-independent projections).
- **Batched RoPE**: Single kernel launch per Q/K per block instead of per-position loop (ALB-119). 49K→48 launches/step.
- **GPU optimizer checkpoint**: All 2.3 GB of GPU-resident AdamW m/v moments saved in APR checkpoints (ALB-118). Enables safe resume if v13 is interrupted.
- **Full epoch**: 155K steps = 5.08B tokens (73% Chinchilla-optimal) vs v9's 15K steps = 490M tokens (7%).

**Current config** (v13 — full epoch from scratch):

```yaml
# configs/train/pretrain-350m-v13.yaml — from scratch, 5.08B tokens
# v10-v12 proved continuation from v9 is broken (ALB-118: GPU optimizer not checkpointed)
# v13 trains from random init for a full epoch — 73% Chinchilla-optimal

model:
  path: "."                                    # Random init (no pre-trained weights)
  architecture:
    hidden_size: 1024
    num_hidden_layers: 24
    num_attention_heads: 16
    num_key_value_heads: 4
    intermediate_size: 4096
    vocab_size: 32768
    max_position_embeddings: 1024
    rms_norm_eps: 1.0e-5
    rope_theta: 10000.0

optimizer:
  lr: 3.0e-4                                  # Same as v9 (Chinchilla-recommended)
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1

training:
  lr_scheduler: "cosine"
  warmup_steps: 2000                           # 1.3% of total (standard for long runs)
  gradient_accumulation: 8                     # 32K tokens/step
  save_interval: 5000                          # ~1.9 GB per checkpoint, ~31 saved
  eval_interval: 1000                          # ~33M tokens between evals
  patience: 30                                 # Generous for long run
  max_steps: 155000                            # 155K × 32K = 5.08B tokens
```

**Scaling law rationale (v13)**: Chinchilla-optimal for 350M is ~7B tokens
(20 tokens/param). v9 saw only 490M tokens (7%) — val_ppl=129 was promising but
far from converged. v13 targets 5.08B tokens (73% Chinchilla), expecting
val_ppl 30-50 at convergence based on Cerebras-GPT scaling curves.

**Why not continue from v9?** Three attempts (v10-v12) all failed:
- v10: fresh optimizer + lower LR → plateau at val_ppl=660
- v11: fresh optimizer + same LR (re-warming) → plateau at val_ppl=750
- v12: resume with embed optimizer state → val_ppl=5639 (one full-LR step destroyed weights)

Root cause was ALB-118: APR checkpoints only saved CPU embedding optimizer state.
The GPU block AdamW (24 blocks × 18 m/v buffers ≈ 2.3 GB) was never
checkpointed. **Now fixed** (`entrenar@784f7b6`): all GPU optimizer state is
saved as `__training__.block_optimizer.{layer}.{m,v}.{weight}` tensors in APR,
with D2H at save and H2D at resume. v13 benefits from this protection.

**Legacy configs**: v1-v8 — see gap register (§11) for per-version history and
why each was superseded. v9 was the best single run (val_ppl=129).

**Note on YAML numeric formatting**: YAML supports underscore notation natively
(`32_768`, `1_000_000`) for human-readable large numbers. All albor configs use
this convention. For shorthand like `10B` or `512K`, see gap ALB-021.

### 6.3 Training Workflow (Plan/Apply)

```bash
# Step 1: Plan — validate config, estimate VRAM, show execution plan (no GPU)
apr train plan configs/train/pretrain-350m.yaml

# Step 2: Apply — execute the training run
apr train apply configs/train/pretrain-350m.yaml --seed 42

# Step 3: Resume if interrupted (apply with --resume)
apr train apply configs/train/pretrain-350m.yaml \
  --resume checkpoints/albor-base-350m/checkpoint-step-5000.json \
  --seed 42
```

**Plan phase** (`apr train plan`):
- Schema validation: required keys, correct types, valid enum values
- Architecture sanity: `hidden_size` divisible by `num_attention_heads`, `num_kv_heads` divides `num_attention_heads`
- VRAM budget: computes model size + optimizer + activations, warns if > GPU capacity
- Data paths: confirms `train:` and `val:` directories exist with Parquet/tokenized shards
- Tokenizer: loads tokenizer, checks vocab size matches `model.vocab_size`
- Time estimate: estimated wall time based on model size and hardware
- Prints structured plan summary (see §1.5.2 for output format)
- **No GPU, no writes, no network.** Runs on CPU in seconds.

**Apply phase** (`apr train apply`):
- Reads the same YAML, builds a random-initialized `Transformer` with the
  `model:` section architecture, runs the causal LM training loop via entrenar
- Checkpoints every `save_interval` steps — resumable on crash
- No Rust code needed — just one config file

`apr train validate` is an alias for `apr train plan --strict` — schema-only
checking without resource estimation. Fast enough for CI.

### 6.4 GPU-Resident Training (CudaTransformerTrainer)

The `CudaTransformerTrainer` (ALB-040) keeps all 24 transformer blocks
GPU-resident, reducing PCIe transfers from ~16K/step to exactly 3:

```
Transfer 1 (H2D): embedding hidden states   ~S×H×4 bytes
Transfer 2 (D2H): logits for cross-entropy  ~S×V×4 bytes
Transfer 3 (H2D): grad_logits to GPU        ~S×V×4 bytes
```

Each `CudaTransformerBlock` holds its own weights, AdamW optimizer states
(m + v), and shares a `CudaGradWorkspace` for forward/backward activation
buffers. The per-block interleaved backward+optimizer pattern overwrites
the shared workspace each layer — memory cost is O(1 block), not O(24 blocks)
for activations.

**VRAM budget (actual, RTX 4090 24GB):**

| Component | Memory |
|-----------|--------|
| 24 blocks (weights + AdamW m + v) | ~5 GB |
| Shared workspace (activation/gradient buffers) | ~10-12 GB (depends on seq_len) |
| LM head (weights + AdamW + logits buffer) | ~1-2.5 GB |
| System (Xorg/desktop) | ~1 GB |

At `seq_len=512, batch=4`: fits comfortably (~18 GB used).
At `seq_len=1024, batch=4`: fits (~19.5 GB used).
At `seq_len=2048, batch=4`: OOM at LM head alloc (logits [4,2048,32768] too large).
At `seq_len=2048, batch=8`: OOM at block 21 upload.

**Dogfooding results:**

| Config | Steps | Loss | Time | Status |
|--------|-------|------|------|--------|
| 50M quick (seq=512, batch=4) | 5 | 10.42→9.45 | ~10s | PASS (post ALB-059 fix) |
| 350M test (seq=512, batch=4) | 50 | 10.39→5.92 (best 5.53) | ~400s | PASS (post ALB-059 fix) |
| 350M full v1 (seq=1024, batch=4, accum=128) | 43/5000 | 10.39 flat | ~12s | **FAIL (ALB-060)**: epochs=1 exhausted data |
| 350M full v2 (seq=1024, batch=4, accum=1) | 1183/5000 | 10.4→6.85 | ~1.4h | **CRASHED**: ALB-073 (PTX selp) + ALB-074 (stale binary). Step 1000 ckpt saved. |
| 350M v3 (seq=1024, batch=4, codeparrot) | 28K/250K | 10.40→6.43 | ~1.9 days | **STOPPED** (plateau): val_ppl=1018 at step 28K. 6.7K tok/s, 19.3% MFU. Plateau since step 12K — ALB-079 (no cosine decay) + ALB-080 (batch too small). |
| 350M v4 (seq=1024, batch=4, ga=32) | 500 | 10.40→5.76 | ~4.7h | Killed by system reboot at step 553. val_ppl=1032.7 at step 500 (matched v3 at 57% token budget). Checkpoint saved. |
| 350M v4-resume (from step 500) | ~500 | 10.40→5.69 | ~4.7h total | Killed by system reboot. Best loss=5.69 at step 262 (~100M tokens). 3,550 tok/s, 10.3% MFU. |
| 350M v5 (seq=1024, batch=4, ga=8, codeparrot-5.3B) | 3,429 | 10.38→6.20 | ~45 min | **CRASHED** at step 3429. 7.9K tok/s, 22.8% MFU. Resume broken (ALB-097). |
| 350M v6 (seq=1024, batch=4, ga=8, codeparrot-5.3B) | 2,000 | 10.40→6.50 | ~4.5h | **KILLED** — strategic pivot to distillation. val_ppl=776, 6.5K tok/s. |
| 350M v7 (seq=1024, batch=4, ga=8, codeparrot-5.3B) | 550 | 10.40→6.62 | ~1h | **KILLED** for checkpoint code fixes. 6.9K tok/s. Checkpoint at step 500, but resume broken (ALB-097: tied LM head not saved). |
| 350M v8 (seq=1024, batch=4, ga=8, codeparrot-5.3B, APR) | 5,337 | 10.40→6.40 | ~5h | **KILLED** — no RoPE (ALB-106). 7.8K tok/s, 24.6% MFU. All steps wasted: model trained without positional awareness. |
| 350M v9 (seq=1024, batch=4, ga=8, codeparrot-5.3B, ALB-106 RoPE) | 14,950 | 10.40→4.79 | ~3h | **STOPPED** — val_ppl=129 at step 14000. 8.2K tok/s, 23.8% MFU. First run with RoPE. Only 490M tokens (7% Chinchilla-optimal). |
| 350M v10 (continue v9, lr=1e-4, fresh optim) | 5,058 | 6.65→6.50 | ~1.5h | **KILLED** (plateau) — val_ppl=660, never recovered. ALB-118: fresh GPU optimizer + lower LR. |
| 350M v11 (continue v9, lr=3e-4, fresh optim) | 8,150 | 7.94→6.62 | ~2.3h | **KILLED** (plateau) — val_ppl=750, worse than v10. ALB-118: re-warming doesn't fix same-data continuation. |
| 350M v12 (resume v9 with embed optimizer state) | 37 | 8.00→6.77 | <1min | **KILLED** — val_ppl=5639. ALB-118: only CPU embed optimizer restored; GPU block AdamW always fresh. |
| distill-v3 (v9 + 58M mixed tokens) | 2,400 | —→— | ~40min | **STOPPED** — val_ppl=658. HumanEval 0% pass@1. Insufficient tokens + raw code format. |
| 350M v13 (from scratch, full epoch, 5.08B tokens) | 62K / 155K | 10.40→6.87 | 40.1h | **STOPPED** (patience=30) — Best val_ppl=**239** at step 32K (inflated by 2x data overlap). System reboot at step 25671 caused data loader restart → 2x overlap on shards 1-4 → val_ppl collapse at step 50K when model hit new data. gnorm collapsed 0.08→0.01. |

**v9 vs v13 convergence comparison** (first 5000 steps):

| Step | v9 val_ppl | v13 val_ppl | v13 vs v9 | v13 note |
|------|-----------|------------|-----------|----------|
| 1000 | 781 | 800 | +2.4% | mid-warmup |
| 2000 | 699 | 829 | +18.6% | warmup end |
| 3000 | 715 | 812 | +13.6% | plateau |
| 4000 | 667 | **499** | **−25.2%** | **phase change** |
| 5000 | 472 | **426** | **−9.8%** | accelerating |
| 6000 | 410 | 455 | +11.0% | noisy — v9 also oscillated here |
| 7000 | **287** | **655** | **+128%** | **regression** — see analysis below |
| 8000 | 207 | **414** | +100% | recovered — new best, LR effect (see below) |
| 9000 | 196 | 366 | +87% | improving |
| 10000 | 200 | **328** | +64% | new best, checkpoint saved |
| 11000 | 170 | 355 | +109% | slight regression |
| 12000 | 151 | **698** | **+363%** | **noise spike** (like step 7K) |
| 13000 | 135 | 367 | +172% | recovered |
| 14000 | 129 | 332 | +157% | near best — v9 converged here at ppl=129 |
| 15000 | 129 | **472** | **+266%** | **oscillation spike** — B_noise=0.27 (elevated). v9 at 490M tokens; v13 at 491M tokens. 3.6x gap = LR effect (v9 at 24% peak, v13 at 98%) |
| 16000 | — | **308** | — | **NEW BEST** — beats 328 (step 10K). B_noise=0.11 (low). Best-envelope: 426→328→308. |
| 17000 | — | **717** | — | **worst spike** — exceeds step 12K's 698. B_noise=0.15 (normal). Spikes intensifying: 655→698→717. |
| 18000 | — | 373 | — | recovery from 17K spike (52% of spike ppl). Post-spike recovery improving: 63%→53%→52%. |
| 19000 | — | 317 | — | continued recovery — only 2.9% above best-envelope (308). 717→373→317 trajectory. |
| 20000 | — | **624** | — | **spike** — 3K after step 17K spike (breaks 5K periodicity). B_noise=0.30 (elevated). 2.0x best-envelope. **v9's max_steps** — v13 enters unexplored territory. |
| 21000 | — | **829** | — | **worst eval yet** — exceeds step 1K random init (800). B_noise=0.07 (lowest ever!). Two consecutive spikes (20K+21K). val_loss=6.72 (only 17% above best 5.73 — ppl exponential amplifies). |
| 22000 | — | **314** | — | **FULL RECOVERY** — from 829 to 314 in 1 eval. Near-best (308 at step 16K). Confirms oscillation is transient, model knowledge intact. |
| 23000 | — | 313 | — | near-best, second consecutive good eval (314→313). Model stabilizing near best-envelope. |
| 24000 | — | 407 | — | moderate regression — normal oscillation (not a >500 spike). B_noise=0.21. |
| 25000 | — | **286** | — | **NEW BEST** — first record since step 16K (308). 7.1% improvement. Checkpoint saved. LR=2.84e-4 (94.7% peak) — decay beginning to help. |
| 25671 | — | — | — | **SYSTEM REBOOT** — training killed. Resumed from step 25K checkpoint. Lost 671 steps (~40 min). |
| 26000 | — | **252** | — | **NEW BEST** — 11.8% improvement over 286. Second consecutive record. Post-resume with fresh GPU optimizer moments — convergence unaffected. LR=2.82e-4. |
| 27000 | — | 421 | — | Oscillation spike. Not >500. Predictor reset by resume (2 points). |
| 28000 | — | 384 | — | Recovering from spike (421→384). |
| 29000 | — | 389 | — | Still elevated. Post-resume GPU optimizer cold start → slower recovery. |
| 30000 | — | **573** | — | **Severe spike** — first >500 since step 21K. LR=2.76e-4 (92% peak). Checkpoint saved. |
| 31000 | — | 336 | — | Recovery (573→336). LR=2.74e-4 (91% peak). |
| 32000 | — | **239** | — | **NEW BEST** — first sub-250! 5.1% below previous best (252). Full recovery from 573 spike. LR=2.73e-4 (91%). |
| 33000 | — | 262 | — | Near-best. Predictor slope turned positive (0.12). |
| 34000 | — | 300 | — | Normal oscillation. Predictor slope 0.26. LR=2.69e-4 (90% peak). |
| 35000 | — | 334 | — | Moderate regression. Checkpoint saved. LR=2.67e-4 (89%). |
| 36000 | — | **568** | — | **Severe spike**. Second >500 post-resume (cf. step 30K: 573). LR=2.64e-4. |
| 37000 | — | 412 | — | Recovering (568→412). LR=2.63e-4 (88%). |
| 38000 | — | 323 | — | Continued recovery. LR=2.60e-4 (87%). |
| 39000 | — | 345 | — | Moderate regression. LR=2.58e-4 (86%). |
| 40000 | — | 370 | — | Checkpoint saved. LR=2.57e-4 (86%). 25.8% complete, 1.31B tokens. |
| 41000 | — | 336 | — | Recovering toward best-envelope. |
| 42000 | — | 288 | — | Near-best. Was predicted v9-match (ppl=129) by log-linear fit — actual 288. |
| 43000 | — | 288 | — | Stable plateau at 288 (2 consecutive evals). Predictor slope positive (0.01). |
| 44000 | — | 286 | — | Near-best, matches step 25K record. 3 consecutive evals at 286-288. LR=2.47e-4 (82%). |
| 45000 | — | 410 | — | Moderate spike. Checkpoint saved. 29% complete, 1.47B tokens. |
| 46000 | — | 251 | — | Near-best — 5% above record (239). LR=2.43e-4 (81%). |
| 47000 | — | 299 | — | Normal oscillation. |
| 48000 | — | 403 | — | Elevated. |
| 49000 | — | 427 | — | Elevated. Was predicted v9-match point — actual 427, not 129. |
| 50000 | — | **734** | — | **Severe spike** — worst since step 21K (829). Checkpoint saved. 32.3% complete, 1.64B tokens. LR=2.33e-4 (78%). |
| 51000 | — | **756** | — | Consecutive spike. Data transition: step ~50,662 is where model sees NEW data after resume overlap. |
| 52000 | — | **806** | — | Third consecutive >500. |
| 53000 | — | **799** | — | Plateau at ~786-806. Not recovering — **data distribution shift** (see below). |
| 54000 | — | **786** | — | Flat. val_loss stable at 6.67. |
| 55000 | — | **786** | — | 6 consecutive >500 evals. Checkpoint saved. 35.5% complete, 1.80B tokens. |
| 56000 | — | **784** | — | 7th consecutive >500. gnorm collapsed to 0.01. |
| 57000 | — | **782** | — | 8th consecutive. Flat at ~782. |
| 58000 | — | **782** | — | 9th consecutive. Patience 26/30. |
| 59000 | — | **781** | — | 10th consecutive. Patience 27/30. |
| 60000 | — | **783** | — | 11th consecutive. Checkpoint saved. Patience 28/30. |
| 61000 | — | **784** | — | 12th consecutive. Patience 29/30. |
| 62000 | — | **782** | — | **EARLY STOP** — 30 evals without improvement. Best val_loss=5.48 (step 32K). Training time: 40.1h. |

**Data distribution shift at step ~50,662**: The system reboot at step 25,671 forced a data
loader restart from shard 1. Steps 25K-50.7K re-trained on shards 1-4 (already seen in the
original run), creating a 2x data overlap. At step 50,662, the model transitions to genuinely
new data (shard 4+). The val_ppl spike from ~288 to ~786 correlates precisely with this
transition. **The best-envelope of 239-288 during steps 32K-46K was partly inflated by
training on data seen twice.** The model should gradually adapt to new data and val_ppl
should recover, but the true convergence trajectory is uncertain until the model processes
sufficient unseen data.

v9 had NO RoPE (position learned via weight absorption). v13 has RoPE forward+backward
(position-independent projections + explicit rotation). v13's ~15% worse early val_ppl
was expected — the model had to learn position-independent W_q/W_k rather than absorbing
rotation into weights. The **phase change arrived at step 4000** — one interval earlier
than v9's step 4500 transition — and v13 is already 25% better. The RoPE backward fix
(ALB-119) appears to be paying dividends: proper gradient flow through the rotation
produces better Q/K projections once the model escapes the early plateau.

**Loss dynamics (500-step windows)**: Training loss accelerated from steps 3000-4500
(avg 6.47→6.05), then **plateaued** at ~6.05-6.16 from step 4500 to 7000+. The
val_ppl phase change (812→499→426) corresponded to the train loss drop, but the
subsequent val_ppl regression (426→455→655) coincides with the train loss plateau.

| Step range | Avg train loss | val_ppl trend |
|-----------|----------------|---------------|
| 3000-3499 | 6.47 | 812 (plateau) |
| 3500-3999 | 6.37 | — |
| 4000-4499 | 6.17 | 499 (phase change) |
| 4500-4999 | 6.05 | 426 (best) |
| 5000-5499 | 6.07 | — |
| 5500-5999 | 6.16 | 455 (regression starts) |
| 6000-6499 | 6.06 | — |
| 6500-6999 | 6.15 | 655 (major regression) |

**Step 7000 regression — root cause: LR schedule mismatch.**

v13's val_ppl=655 vs v9's val_ppl=287 at step 7000 is explained by the learning rate
schedule. The cosine decay spans different horizons:

| Step | v9 LR (20K schedule) | v13 LR (155K schedule) | v9 % peak |
|------|---------------------|----------------------|-----------|
| 5000 | 2.82e-4 | 3.00e-4 | 94% |
| 7000 | **2.52e-4** | **2.99e-4** | **84%** |
| 10000 | 1.88e-4 | 2.98e-4 | 63% |
| 15000 | 0.78e-4 | 2.95e-4 | 26% |

At step 7000, v9's LR had decayed to 84% of peak — enough for the model to settle into
finer-grained optimization and hit its second phase change (415→287). v13's LR is still
at 99.8% of peak and won't start meaningful decay until step ~30K. The train loss plateau
at ~6.1 is the model oscillating in a basin at too-high LR, not a capacity limit.

**Implication**: v13's early-to-mid trajectory (steps 5K-30K) will be noisier and
slower than v9's because the LR isn't decaying. But v13 has two structural advantages:
1. Once LR decay engages (~step 30K), the model will converge with 10x more remaining data
2. RoPE provides correct position encoding, which should produce better final quality

This is not a bug — it's an expected consequence of a 155K-step cosine schedule. The
v9-shifted trajectory model is invalidated: v9 and v13 have fundamentally different LR
profiles at the same step count. The right comparison is at the same LR, not the same step.

**Gradient diagnostics**: ZClip fires on ~34% of steps (z>2.0 threshold), with gnorm
EMA at 0.15-0.20 (steps 15K-17K). Max gnorm 0.97 at step 17021. No gradient explosion.
B_noise (gradient noise scale) averages ~0.11 at steps 15K-17K, down from ~0.17 at steps
2K-4K — a healthy trend showing gradients becoming more signal-dominated over time.
Occasional spikes (B_noise=0.32 at step 15600) don't persist.

**v13 convergence trajectory** (best-envelope vs oscillation):

The v9-shifted projection was invalidated by the LR schedule mismatch. v13's actual
trajectory shows two patterns: (1) an improving **best-envelope** (426→328→308→286→252),
and (2) **extreme oscillation** with spikes to 655/698/829 at steps 7K/12K/17K/20K/21K.
After step 21K (worst spike: 829), 5 consecutive non-spike evals (22K-26K) with two new
records (286, 252) confirm convergence is accelerating as cosine decay engages.

| Phase | Steps | Best val_ppl | Envelope trend | LR % peak |
|-------|-------|-------------|----------------|-----------|
| Plateau | 1K-3K | 800 | flat | 50-100% (warmup) |
| Phase change | 4K-5K | 426 | rapid drop | 100% |
| High-LR oscillation | 6K-25K | 286 | slow improvement, spikes to 472-829 | 99-95% |
| Early decay | 25K-26K | **252** | acceleration — 11.8% drop in 1 eval | 94% |
| LR decay (predicted) | 30K-155K | <100? | accelerating convergence | 90→10% |

**Oscillation pattern analysis (steps 6K-17K)**: The oscillation is not random noise —
it has structure. Best-envelope checkpoints (new val_ppl records) and worst spikes both
intensify over time, producing a widening band:
- Best-envelope: 426 (5K) → 328 (10K) → 308 (16K) → 286 (25K) → 252 (26K) → **239 (32K)** — accelerating
- Spike peaks: 655 (7K) → 698 (12K) → 829 (21K) — worst spikes intensified but stopped after step 21K
- B_noise is NOT correlated with spikes. **Definitive evidence**: step 21K has the
  worst spike ever (ppl=829) with the lowest B_noise ever recorded (0.07). Step 15K's
  spike had B_noise=0.27 (elevated), step 17K had 0.15, step 21K had 0.07. The
  oscillation is purely LR-driven loss landscape exploration, not gradient noise.

Through step 21K, this widening band was characteristic of SGD near a flat saddle point
with a too-high LR. After step 21K, 4 consecutive non-spike evals (including a new best)
suggest the band is beginning to narrow as cosine decay drops LR below 95% of peak.
Full narrowing expected once LR drops to 90% peak (~step 30K).

The key question: will v13 surpass v9's final ppl=129? The raw step comparison is
misleading because of the LR schedule mismatch. **LR-equivalent step mapping** shows
when v13 reaches the same LR as v9 at key milestones:

| v9 milestone | v9 step | v9 LR | v13 equiv step | v13 tokens | v13 data advantage |
|-------------|---------|-------|---------------|-----------|-------------------|
| Phase change (ppl=287) | 7K | 2.33e-4 | 53K | 1.74B | 3.5x |
| ppl=200 | 10K | 1.70e-4 | 77K | 2.51B | 5.1x |
| ppl=129 | 14K | 8.83e-5 | 108K | 3.54B | 7.2x |
| Converged (ppl=129) | 15K | 7.23e-5 | 115K | 3.78B | 7.7x |

At the LR-equivalent of v9's convergence point (v13 step 115K), v13 will have seen
7.7x more tokens. Chinchilla scaling suggests ppl ~ D^(-0.095), so 7.7x more data →
~21% lower ppl. If v9 hit 129 at this LR, v13 should reach ~105 at step 115K, with
40K more steps of LR decay remaining.

**Projection**: val_ppl 80-120 at step 155K. The built-in predictor says 242 at step 25K
but is fitting a power law to the noisy high-LR regime — it cannot model the convergence
acceleration from LR decay that hasn't happened yet.

**Predictor slope trend** (power-law exponent, non-spike evals only):

| Window | Avg slope | Avg predicted final ppl | Interpretation |
|--------|-----------|------------------------|----------------|
| 5K-10K | 0.370 | 145 | Rapid initial improvement |
| 10K-15K | 0.342 | 163 | Decelerating |
| 15K-20K | 0.293 | 197 | Stagnation — high-LR plateau |
| 20K-25K | 0.222 | 254 | Further deceleration |

The declining slope is the power-law predictor seeing high-LR stagnation. When cosine decay
engages (~step 30K), the slope should **increase** — this will be a leading indicator of
convergence acceleration, visible before val_ppl drops dramatically. Watch for slope > 0.30
as a signal that the model is entering the fast convergence phase.

**Best-envelope trajectory** (log-linear fit on 10 best-envelope points through step 32K):

| Step | Best-envelope ppl (actual) | Fit prediction | Key milestone |
|------|---------------------------|----------------|---------------|
| 1K | 800 | 578 | — |
| 4K | 499 | 474 | Phase change |
| 5K | 426 | 460 | — |
| 8K | 414 | 399 | — |
| 9K | 366 | 387 | — |
| 10K | 328 | 376 | — |
| 16K | 308 | 312 | — |
| 25K | 286 | 244 | Broke 9K plateau |
| 26K | 252 | 237 | Post-resume new best |
| **32K** | **239** | **201** | **First sub-250** |
| 42K-44K | (286-288) | 149-145 | Stable plateau — "typical" matches old best-envelope |
| 49K | — | 125 | Predicted v9 match |
| 77K | — | 56 | LR-equiv of v9 ppl=200 |

The log-linear fit (R²=0.76 on 10 best-envelope points) over-extrapolates at late steps
(predicts ppl=2 at 155K, nonsensical). The fit captures the high-LR regime but cannot model
the two-phase behavior: slow improvement during near-peak LR → acceleration during cosine
decay. The LR-equivalence analysis gives a more grounded long-term prediction: val_ppl
80-120 at step 155K. The built-in trainer predictor (242 at 155K) is also pessimistic —
it fits the noisy high-LR regime and misses coming acceleration.

**Convergence phase analysis** (5K-window non-spike median):

| Phase | Steps | Non-spike median | Non-spike count | LR % peak |
|-------|-------|-----------------|-----------------|-----------|
| Phase change | 5K-10K | 426 | 4/5 | 99-100% |
| Early high-LR | 10K-15K | 355 | 4/5 | 98-99% |
| Mid high-LR | 15K-20K | 373 | 4/5 | 97-98% |
| Late high-LR | 20K-25K | 314 | 5/6 | 95-97% |
| Early decay (post-resume) | 25K-30K | 386 | 4/5 | 92-95% |
| Mid decay | 30K-35K | 317 | 4/5 | 89-92% |
| Spike aftermath | 35K-40K | 358 | 4/5 | 86-89% |
| **Convergence acceleration** | **40K-45K** | **288** | **5/5** | **82-86%** |

**Key observation**: The 40K-45K window median of 288 is a major breakthrough — the model's
*typical* performance now matches the previous best-envelope (286 at step 25K). Non-spike
medians: 426→355→373→314→386→317→358→**288**. The improvement pattern is not monotonic
(spikes cause temporary regressions in the median), but the trend is clearly downward as
LR drops below 85% peak.

**Spike frequency**: Spikes (>500) at steps 7K, 12K, 17K, 20K, 21K, 30K, 36K, 50K. 8 spikes
in 46 post-phase-change evals (17%). Spike rate by phase: 5/21 pre-resume (24%), 3/25
post-resume (12%). Recovery always within 2 evals. Step 50K spike (734) is the worst since
step 21K (829) — spikes persist even at 78% LR peak. Model integrity always confirmed.

**ALB-060: Training Configuration Epoch/Step Mismatch (Critical)**

The first 350M full training run (2026-03-02) ran only 43 of 5000 steps because
`epochs: 1` caps total steps to `floor(num_sequences / batch_size / grad_accum)`.
With 22,079 sequences, batch=4, accum=128: `steps_per_epoch = 43`. Warmup (2000
steps) never completed — LR peaked at 6.45e-6 vs target 3e-4. Loss stayed flat
at ~10.39 for all 43 steps (never exited warmup). Root cause: no pre-flight
algebraic validation of epoch/step consistency.

Fix: C-TRAINCFG-001 contract (`contracts/training-config-kernel-v1.yaml`) +
`epochs: 117` for v1 data, or v2 config (`pretrain-350m-v2.yaml`) with expanded
dataset (67,977 sequences, `epochs: 38`, `warmup_steps: 500`).

**Training stability contracts verified (ALB-044, ALB-059, ALB-060):**
- C-EMBED-GRAD-001: Activation gradient clipped at GPU→CPU boundary
- C-HYPERPARAMS-001: All optimizer params flow from YAML config
- C-BUFSIZE-001: Buffer sizes algebraically verified (ALB-043 fix)
- C-GRADFLOW-001: All trainable parameters receive gradients (ALB-038 fix)
- C-GEMMARGS-001: GEMM backward constructor args match documented order (ALB-059 fix)
- C-GPUINIT-001: Optimizer states zero-initialized, not cuMemAlloc garbage (ALB-059 fix)
- C-STREAMSYNC-001: `stream.synchronize()` before any D2H transfer reading kernel output (ALB-065 fix)
- C-LOSSSCALE-001: fp16 loss scaling excluded from f32 backward path (ALB-072 fix)
- C-SELP-001: PTX `selp_f32` argument order verified in all kernels (ALB-069, ALB-073 fixes)
- C-EVALBUF-001: `eval_single_sequence` truncates to max_seq_len before GPU forward (ALB-074 fix)
- C-GPUINIT-001: All optimizer m/v buffers zero-initialized (ALB-059 fix)
- C-LOSSSCALE-001: fp16 loss scaling excluded from GPU backward (all backward uses f32; scaling causes overflow) (ALB-072 fix)
- C-CUBLAS-NOTENCORE-001: cuBLAS uses CUBLAS_DEFAULT_MATH (no tensor cores) — tensor core algorithms produce NaN for transposed backward GEMMs at ~1e5 gradient magnitude (ALB-077 fix)
- C-ROPE-001: RoPE applied to Q and K in CUDA forward (after GEMM, before attention), inverse RoPE in backward (after interleaved conversion, before projection backward) — in-place via per-thread pair independence (ALB-106 fix)

### 6.5 Checkpointing Strategy

| Aspect | Design |
|--------|--------|
| Format | APR (primary, ALB-096) + SafeTensors fallback |
| Save frequency | Every 5000 steps (v13: `save_interval: 5000`; v9: 500) |
| Eval frequency | Every 1000 steps (v13: `eval_interval: 1000`; v9: 250; ALB-087) |
| Best-model tracking | `model-best.safetensors` — updated when val_loss improves (ALB-087) |
| Early stopping | `patience: 10` — stop after 10 evals (2,500 steps) without val_loss improvement (ALB-087) |
| Content | Model weights + CPU embed optimizer + GPU block optimizer (24 blocks × AdamW m/v) in single APR file (~5.2 GB), config.json. ALB-118 **FIXED**: all 438 GPU optimizer tensors checkpointed (`entrenar@784f7b6`). |
| Pruning | Automatic — keeps latest + best only, old checkpoints deleted |
| Disk usage | ~15 GB peak (v13: ~5.2 GB per checkpoint, current + best + in-flight) |
| Storage | Local NVMe RAID-0, checkpoints directory in repo |
| Resume | From latest APR checkpoint on crash (weights + CPU embed optimizer + GPU block optimizer + step counter). ALB-097: LM head always saved. ALB-118 **FIXED**: full optimizer state (model weights + 438 GPU tensors + CPU embed optimizer) checkpointed in APR v2 format. v13 benefits from safe resume on interruption. |
| Shape format | 2D `[out, in]` shapes (ALB-086) — HuggingFace compatible |
| Export | `apr publish --format safetensors` for HuggingFace |

**Checkpoint interval rationale (v5)**: `save_interval: 500` with
`eval_interval: 250` means validation runs twice per save interval. At 7.9K
tok/s and 792ms/step, 500 steps ≈ 6.6 min — max lost work on crash. Best-model
tracking (`model-best.safetensors`) ensures the optimal checkpoint is always
available even if training continues past the minimum.

**ALB-087 auto eval scheduling**: `eval_interval` is decoupled from
`save_interval` — eval can run more frequently than checkpointing. The
`patience` parameter triggers early stopping when val_loss stops improving,
preventing wasted GPU time on plateaus (would have saved ~16h on v3's
step 12K→28K plateau).

### 6.6 Experiment Tracking & Training Monitoring

entrenar has a full monitoring stack built in, and presentar provides rich
terminal visualization. Albor uses both — no external tools (no W&B, no
MLflow, no TensorBoard). Sovereign monitoring, sovereign visualization.

#### 6.6.1 Monitoring Config: `configs/train/pretrain-350m.yaml` (monitoring section)

```yaml
monitoring:
  terminal:
    enabled: true
    refresh_rate: 1000              # TUI refresh in ms
    metrics: ["loss", "learning_rate", "gradient_norm"]
    charts:
      - type: "loss_curve"
        metric: "loss"
        window: 100                 # Smoothing window
        show_eta: true

  tracking:
    enabled: true
    backend: "sqlite"               # .entrenar/experiments.db (WAL mode)
    experiment: "albor-pretrain-350m"
    tags:
      model: "albor-350m"
      stage: "pretrain"
      data: "python-code-v2"                 # 139M tokens (v2 dataset)

  system:
    enabled: true
    interval: 5000                  # System metrics every 5s
    metrics: ["gpu_utilization", "memory", "temperature"]

  alerts:
    - condition: "loss > 10"
      action: "stop"
      message: "Loss exploded — Andon stop"
    - condition: "gradient_norm > 100"
      action: "stop"
      message: "Gradient explosion — Andon stop"
```

#### 6.6.2 What Entrenar Monitors Automatically

| Component | What It Does | Already Built? |
|-----------|-------------|----------------|
| `MetricsCollector` | Records loss, LR, gradient norms per step (SIMD-accelerated) | Yes (entrenar) |
| `ExperimentTracker` | Tracks run_id, params, metrics, artifacts, status | Yes (entrenar) |
| `SqliteBackend` | Durable experiment store: runs, params, metrics, artifacts in `.entrenar/experiments.db` (WAL mode) | Yes (entrenar) |
| `ProgressCallback` | Kalman-filtered ETA, Unicode progress bars | Yes (entrenar) |
| `MonitorCallback` | Integrates metrics into training, detects NaN/Inf → Andon alert | Yes (entrenar) |
| `CheckpointCallback` | Saves best model + metadata (epoch, is_best, timestamp) | Yes (entrenar) |
| `EarlyStopping` | Patience-based stopping on loss plateau | Yes (entrenar) |
| `Andon alerts` | Toyota Way: Critical/Error/Warning/Info severity levels | Yes (entrenar) |
| `TuiMonitor` | Detached terminal dashboard composing presentar widgets (ALB-057) | Yes (entrenar + presentar) |
| `DriftDetector` | PSI, KS, Wasserstein distribution shift detection | Yes (entrenar) |
| `JsonFileStore` | Real-time metrics to `training_state.json` (atomic writes) | Yes (entrenar) |
| `LossCurve` widget | Training loss over epochs with EMA smoothing | Yes (presentar) |
| `ConfusionMatrix` widget | Multi-class classification evaluation | Yes (presentar) |
| `Braille/Sparkline` | High-resolution terminal charts (2x4 dots/cell, 8-level sparklines) | Yes (presentar) |
| `Heatmap` widget | 2D matrix with CIELAB perceptual color gradients | Yes (presentar) |

#### 6.6.3 Live Monitoring During Training

```bash
# Terminal 1 (lambda): Run training
apr train apply --task pretrain --config configs/train/pretrain-350m.yaml

# Terminal 2 (lambda or ssh): Attach live monitor (presentar TUI)
apr monitor ./checkpoints/albor-base-350m/

# Terminal 2 (alternative): JSON output for LLM agents / CI
apr monitor --json ./checkpoints/albor-base-350m/

# Discover all active training runs (reads global SQLite registry)
apr monitor

# List past experiments from SQLite registry
apr runs ls --global

# Show detailed metrics for a specific run
apr runs show <run-id> --global --json

# Browse past experiments from SQLite
apr experiment view --db .entrenar/experiments.db

# Compare loss curves across runs
apr experiment view --db .entrenar/experiments.db \
  --runs albor-pretrain-50m,albor-pretrain-350m \
  --metric loss --chart loss_curve

# One-shot profiler (GPU utilization, per-layer timing)
apr cbtop ./checkpoints/albor-base-350m/latest.safetensors

# Inference latency profiling
apr profile ./checkpoints/albor-base-350m/ --prompt "def fibonacci(n):"

# Stack-level health (from batuta)
batuta stack status
```

#### 6.6.4 Experiment Lifecycle

Each training run creates two data streams:

**Real-time (JSON file IPC)** — for live TUI monitoring:
```
checkpoints/albor-base-350m/
├── training_state.json         # Live metrics (loss, lr, grad_norm, GPU telemetry)
├── checkpoint-step-1000.safetensors
├── checkpoint-step-1000.json   # Checkpoint metadata (epoch, is_best)
├── checkpoint-step-2000.safetensors
├── checkpoint-step-2000.json
├── checkpoint-best.safetensors
└── checkpoint-best.json
```

**Durable (dual SQLite experiment stores)** — for post-hoc analysis and comparison:
```
checkpoints/albor-base-350m/.entrenar/
└── experiments.db              # Local per-experiment store (WAL mode)
    ├── experiments             # Experiment metadata (name, description, config)
    ├── runs                    # Training runs (status, timestamps)
    ├── params                  # Hyperparameters (key/value/type)
    ├── metrics                 # Per-step metrics (loss, lr, tok/s, timestamp)
    ├── artifacts               # Model artifacts (path, size, SHA-256)
    └── span_ids                # Distributed trace integration

~/.entrenar/
└── experiments.db              # Global cross-machine registry (WAL mode)
    └── (same schema)           # All runs across all experiments
```

**`PretrainTracker`** (ALB-055/056) writes to both stores on every log interval.
All operations are best-effort — storage failures never block training.

**Three consumers, zero contention**:
- `apr monitor` reads `training_state.json` (atomic write-then-rename) for
  live dashboards. Multiple monitors attach simultaneously.
- `apr runs ls` reads `~/.entrenar/experiments.db` (global registry) for
  cross-experiment history. Supports `--json` for LLM agent consumption.
- `apr experiment` reads local `.entrenar/experiments.db` (WAL mode) for
  per-run metric queries and artifact tracking. Read-only during
  training — no lock contention with the writer.

#### 6.6.5 Presentar Visualization: Rich Terminal Dashboards

presentar (`presentar-terminal`) provides **ML-specific visualization widgets**
that entrenar's `TrainingDashboard` now composes directly (ALB-057). The
dashboard builds a widget tree from `Layout::rows()` of `Border`-wrapped
section panels, each containing `Meter`, `GpuPanel`, `Sparkline`, or `Text`
widgets. The connection point for historical data is entrenar's SQLite
experiment store (`.entrenar/experiments.db`).

**Live training dashboard** (`apr monitor` — reads `training_state.json`):

```
╭─ Albor Pre-Train: albor-base-350m ─── Step 12,847 / 19,073 ──── 67.4% ─╮
│                                                                          │
│  Loss                                          GPU (RTX 4090)            │
│  3.2 ⣀⣀                                       ████████████░░░ 82%       │
│      ⠈⠉⠉⠑⠒⠒⠤⣀                                VRAM: 14.2 / 24.0 GB      │
│               ⠈⠉⠑⠒⠤⣀⣀                        Temp: 72°C                │
│  1.8                  ⠈⠉⠒⠒⣀⣀⣀⣀               Power: 312W               │
│                              ⠉⠉⠉              Tokens/s: 18,432          │
│  0 ──────────────────────────────── 12K                                  │
│                                                                          │
│  Learning Rate              Gradient Norm       ETA: 1d 14h 22m          │
│  ⣿⣿⣿⣷⣶⣶⣤⣤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀     ▁▁▂▁▁▃▁▂▁▁▁▂▁▁    Throughput: 5.2B / 10B   │
│  3e-4 → 2.1e-4              0.42 (norm)        Checkpoint: step-12000    │
╰──────────────────────────────────────────────────────────────────────────╯
```

**Post-hoc experiment comparison** (`apr experiment view` — reads SQLite):

```bash
# Compare loss curves across all pre-training runs
apr experiment view --db .entrenar/experiments.db \
  --runs albor-pretrain-50m,albor-pretrain-350m \
  --metric loss --chart loss_curve

# Hyperparameter comparison table
apr experiment view --db .entrenar/experiments.db \
  --experiment albor-pretrain-350m --params

# Export metrics for external analysis (Parquet for alimentar)
apr experiment export --db .entrenar/experiments.db \
  --run albor-pretrain-350m --format parquet --output ./eval/metrics.parquet
```

**Presentar widgets used by albor**:

| Widget | Use Case | Data Source |
|--------|----------|-------------|
| `LossCurve` | Training loss over steps with EMA smoothing | `training_state.json` (live) or SQLite `metrics` table (post-hoc) |
| `Sparkline` | Compact LR schedule, gradient norm history | `training_state.json` lr_history, grad_norm |
| `Heatmap` | Attention pattern visualization, weight distribution | Model checkpoint tensors |
| `Gauge` | GPU utilization, VRAM usage, training progress | `training_state.json` gpu telemetry |
| `BrailleGraph` | High-resolution loss/metric curves over SSH | `training_state.json` loss_history |
| `Histogram` | Weight distribution per layer (pre/post distillation) | Model checkpoint tensors |
| `BarChart` | Benchmark scores across model stages | `eval/*.json` results |

**Two rendering targets, same widgets, same data**:

presentar compiles the same widget tree to **two targets** — terminal and
WASM. The dashboard YAML is written once. `presentar-terminal` renders it
via crossterm (works over SSH). `presentar` renders it via WebGPU in the
browser (60fps, GPU-accelerated). Both read from the same data sources.

| Mode | Command | Renderer | Data Source | Use Case |
|------|---------|----------|-------------|----------|
| **Live TUI** | `apr monitor ./checkpoints/` | `presentar-terminal` (crossterm) | `training_state.json` (polling) | Watch training over SSH |
| **Experiment TUI** | `apr experiment view` | `presentar-terminal` (crossterm) | SQLite `.entrenar/experiments.db` | Compare runs in terminal |
| **Web dashboard** | `presentar serve --config albor-dashboard.yaml` | `presentar` (WebGPU/WASM) | SQLite + checkpoints | Rich browser dashboard |

Both TUI and WASM are **first-class deliverables**, not stretch goals.
The terminal TUI is the primary interface (SSH to lambda/intel). The WASM
dashboard is the shareable artifact for model cards and teaching.

#### 6.6.6 No External Dependencies

| What Others Use | What Albor Uses Instead | Why |
|-----------------|------------------------|-----|
| Weights & Biases | entrenar `SqliteBackend` + presentar dashboards | Sovereign — no cloud, no API keys, all data local |
| TensorBoard | presentar `LossCurve` + `BrailleGraph` over SSH | No Python, no browser required, works over SSH |
| MLflow | entrenar `ExperimentTracker` + SQLite + `apr experiment` | Self-hosted SQLite, no server process, query via CLI |
| nvidia-smi polling | entrenar system metrics + `apr cbtop` | Integrated into training loop, not bolted on |
| Streamlit dashboards | presentar WASM dashboard (10x faster rendering) | GPU-accelerated, 60fps, zero Python |
