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
| Batch size (global) | 512K tokens | ~512 sequences x 1024 tokens |
| Micro-batch (4090) | 4 | GPU-resident (batch=8 OOM at seq≥1024) |
| Gradient accumulation | 1 (ALB-066) | Per-block CPU accumulation now works (PerBlockGradientAccumulator); kept at 1 for v2 config |
| Total training tokens | Target 10B; current 139M (v2 dataset) | ~5000 steps × 4 seqs × 1024 tokens = 20M tokens/run (v2: 68K seqs) |
| Mixed precision | fp16 (CUDA) | Hardware-appropriate |

### 6.2 Training Config: `configs/train/pretrain-350m-v2.yaml`

A single YAML file defines **everything** — model architecture and training
hyperparameters. This is the industry standard (Axolotl, torchtune, HuggingFace
Trainer). One file, one truth. `apr train validate` lints it before GPU time.

**Current config** (v2 — expanded dataset, ALB-066 gradient_accumulation=1):

```yaml
# configs/train/pretrain-350m-v2.yaml — Albor 350M with expanded dataset
# C-TRAINCFG-001: steps_per_epoch=16994 >= max_steps=5000

model:
  path: "."                                  # From scratch (random init)
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
  train: "data/pretokenized-2048-v2/train/" # Expanded v2 dataset (68K sequences)
  val: "data/pretokenized-2048/val/"
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
  epochs: 1                                 # C-TRAINCFG-001: steps_per_epoch=16994 >= 5000
  # grad_clip: 1.0                           # ALB-067: disabled (CPU-side L2 norm bottleneck)
  lr_scheduler: "cosine"
  warmup_steps: 500                         # 10% of max_steps (C-TRAINCFG-001)
  gradient_accumulation: 1                  # ALB-066: per-sequence optimizer (no true accum in CUDA)
  mixed_precision: "fp16"
  output_dir: "./checkpoints/albor-base-350m-v2"
  save_interval: 25
  max_steps: 5000
```

**Legacy v1 config** (`pretrain-350m.yaml`) used 22K sequences with
`gradient_accumulation: 128` and `epochs: 117` — see ALB-060 for why
`epochs: 1` was fatal with the original data size.

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
| 350M v3 (seq=1024, batch=4, codeparrot) | 26K/250K | 10.40→6.61 | ~1.7 days | **RUNNING** (PID 1975811): val_loss=6.91, val_ppl=1000 at step 26K. 6.7K tok/s, 19.3% MFU. **Plateau since step 12K** — ALB-079 (no cosine decay) + ALB-080 (batch too small). |
| 350M v4 (seq=1024, batch=4, ga=32) | planned | — | ~1.6 days | **PENDING**: Fixes ALB-079 (cosine decay) + ALB-080 (131K tokens/step). Target val_ppl < 100. |

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

### 6.5 Checkpointing Strategy

| Aspect | Design |
|--------|--------|
| Format | SafeTensors (primary) + JSON metadata |
| Frequency | Every 1,000 steps (~1.2h at 4.2s/step, ~4M tokens) |
| Content | Model weights (~1.5 GB), optimizer state (~1.3 GB), config.json |
| Pruning | Automatic — keeps latest + best only, old checkpoints deleted |
| Disk usage | ~8.4 GB peak (3 checkpoints: current + best + in-flight) |
| Storage | Local NVMe RAID-0, checkpoints directory in repo |
| Resume | From latest checkpoint on crash (weights + optimizer state) |
| Export | `apr publish --format safetensors` for HuggingFace |

**Checkpoint interval rationale (v3)**: `save_interval: 1000` balances crash
recovery (~8.7min max lost work at 525ms/step) against I/O overhead (~3s per
checkpoint write vs ~525s between checkpoints = 0.6% overhead). With automatic
pruning, disk usage stays constant regardless of training length. For the
250K-step v3 run (~1.5 days at 7,579 tok/s), this yields 250 checkpoint events
with ~8.4 GB steady-state disk.

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
