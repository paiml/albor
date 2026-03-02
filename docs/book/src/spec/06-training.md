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
| Warmup steps | 2000 | ~0.2% of total steps |
| Min LR | 3e-5 | 10% of peak (standard) |
| Gradient clipping | 1.0 (global norm) | Stability |
| Batch size (global) | 512K tokens | ~512 sequences x 1024 tokens |
| Micro-batch (4090) | 4 | GPU-resident (batch=8 OOM at seq≥1024) |
| Gradient accumulation | 128 steps | Reach global batch size |
| Total training tokens | 10B | ~19,531 steps at 512K tokens/step |
| Mixed precision | fp16 (CUDA) | Hardware-appropriate |

### 6.2 Training Config: `configs/train/pretrain-350m.yaml`

A single YAML file defines **everything** — model architecture and training
hyperparameters. This is the industry standard (Axolotl, torchtune, HuggingFace
Trainer). One file, one truth. `apr train validate` lints it before GPU time.

```yaml
# configs/train/pretrain-350m.yaml — Albor 350M pre-training config

model:
  path: "."                                  # From scratch (random init)
  mode: transformer                         # LLM transformer mode
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
  train: "data/pretokenized-2048/train/"    # Pre-tokenized ByteLevel BPE v2
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
  epochs: 1                                # Pre-training uses max_steps, not epochs
  grad_clip: 1.0
  lr_scheduler: "cosine"
  warmup_steps: 2000
  gradient_accumulation: 128               # Global batch = 4 * 128 * 1024 = 512K tokens
  mixed_precision: "fp16"
  output_dir: "./checkpoints/albor-base-350m"
  save_interval: 25
  max_steps: 5000
```

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
| 350M full (seq=1024, batch=4, accum=128) | 5000 | TBD | ~20h | PENDING |

**Training stability contracts verified (ALB-044, ALB-059):**
- C-EMBED-GRAD-001: Activation gradient clipped at GPU→CPU boundary
- C-HYPERPARAMS-001: All optimizer params flow from YAML config
- C-BUFSIZE-001: Buffer sizes algebraically verified (ALB-043 fix)
- C-GRADFLOW-001: All trainable parameters receive gradients (ALB-038 fix)
- C-GEMMARGS-001: GEMM backward constructor args match documented order (ALB-059 fix)
- C-GPUINIT-001: All optimizer m/v buffers zero-initialized (ALB-059 fix)

### 6.5 Checkpointing Strategy

| Aspect | Design |
|--------|--------|
| Format | SafeTensors (primary) + JSON metadata |
| Frequency | Every 1000 steps (~512M tokens) |
| Content | Model weights, optimizer state, LR scheduler state, RNG state, step count |
| Storage | Local on lambda, rsync to intel (300GB RAM box) for backup |
| Resume | `--resume checkpoint-step-5000.json` |
| Export | `apr publish --format safetensors` for HuggingFace |

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
      data: "10B-python-80pct"

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
