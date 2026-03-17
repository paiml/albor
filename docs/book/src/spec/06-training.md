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
- **RoPE**: Applied in CUDA forward/backward — positional awareness (ALB-106). All v1-v8 trained without RoPE.
- **Data**: codeparrot-clean 5.3B tokens (pretokenized-1024-v3), val from same distribution
- **Gradient accumulation**: 8 GPU-resident (ALB-091) — zero D2H during micro-batch loop
- **Cosine LR decay**: Implemented in CUDA trainer (ALB-079)
- **APR checkpoints**: Atomic single-file with optimizer state (ALB-096), resume verified (ALB-097)
- **RMSNorm backward**: grad_gamma computed + accum zeroed (ALB-092)

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

Root cause: APR checkpoints only save CPU embedding optimizer state. The GPU
block AdamW (24 blocks × m/v moments = 99%+ of parameters) is never
checkpointed (ALB-118). Fresh GPU optimizer moments make continuation training
impossible — the optimizer can't navigate the loss landscape from v9's learned
weight configuration without its curvature estimates.

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
| 350M v13 (from scratch, full epoch, 5.08B tokens) | 155K target | 10.40→6.54 | ~12 days | **RUNNING** — val_ppl=813 at step 2000. 5.1K tok/s, 14.9% MFU. 73% Chinchilla-optimal. |

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
| Save frequency | Every 500 steps (`save_interval: 500`) |
| Eval frequency | Every 250 steps (`eval_interval: 250`, ALB-087) |
| Best-model tracking | `model-best.safetensors` — updated when val_loss improves (ALB-087) |
| Early stopping | `patience: 10` — stop after 10 evals (2,500 steps) without val_loss improvement (ALB-087) |
| Content | Model weights + CPU embed optimizer state in single APR file (~1.9 GB), config.json. **ALB-118**: GPU block optimizer (24 blocks × AdamW m/v) NOT checkpointed — continuation training broken. |
| Pruning | Automatic — keeps latest + best only, old checkpoints deleted |
| Disk usage | ~8.4 GB peak (3 checkpoints: current + best + in-flight) |
| Storage | Local NVMe RAID-0, checkpoints directory in repo |
| Resume | From latest APR checkpoint on crash (weights + CPU embed optimizer + step counter). ALB-097: LM head always saved. **Limitation (ALB-118)**: GPU block AdamW m/v not checkpointed; resume restores weights but not GPU optimizer moments — continuation from pre-trained weights degrades model (v10-v12 post-mortem). |
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
