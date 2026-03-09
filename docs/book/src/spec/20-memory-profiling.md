# 20. Memory Profiling with dhat-rs

## 0. Motivation

The sovereign stack (realizar, aprender, trueno, entrenar) has zero heap
profiling instrumentation. All four repos use criterion for latency benchmarks
but have no memory tracking beyond entrenar's VRAM ledger. This creates blind
spots:

- **Inference**: realizar loads 17 GB Q4K models — host-side allocations during
  tokenization, KV cache management, and HTTP serving are unmeasured.
- **Training**: entrenar allocates gradient buffers, optimizer state (2× model
  for AdamW m/v), and checkpoint snapshots — no peak RSS tracking.
- **CLI**: aprender's `apr import` streams 67 GB SafeTensors — ALB-081 OOM was
  only caught by swap storm, not by profiling.
- **Tensors**: trueno SIMD ops allocate temporaries for tiling, reduction, and
  transposition — no per-op allocation tracking.

## 1. Tool Selection: dhat-rs

**dhat-rs** (crate: `dhat`, version `0.3`) is a Rust heap profiler that:

- Tracks every allocation (count, size, lifetime)
- Reports peak heap usage, total bytes allocated, allocation hot spots
- Produces DHAT-compatible JSON for visualization in `dh_view.html`
- Has ~2× runtime overhead (acceptable for profiling builds, not production)
- Zero overhead when disabled (feature-gated, compiles to nothing)

**Why not alternatives:**

| Tool | Rejection Reason |
|------|-----------------|
| Valgrind DHAT | External tool, not composable with `cargo test` |
| jemalloc + `MALLOC_CONF` | Non-portable, Linux-only, coarse-grained |
| `peak_alloc` | Only tracks peak RSS, no per-allocation detail |
| `tikv-jemallocator` + stats | Heavy dependency, overkill for profiling |

## 2. Integration Architecture

### 2.1 Feature Flag Convention

Every repo uses the same feature flag: `dhat-heap`.

```toml
[features]
dhat-heap = ["dep:dhat"]

[dependencies]
dhat = { version = "0.3", optional = true }
```

The flag name `dhat-heap` follows dhat-rs convention and is already recognized
by the crate's `#[cfg(feature = "dhat-heap")]` guards.

### 2.2 Global Allocator Pattern

In each binary crate's `main.rs` (or `lib.rs` for libraries with integration
tests):

```rust
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
```

The profiler is activated at program start:

```rust
fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    // ... existing main logic ...
}
```

When the `_profiler` drops (at program exit), dhat-rs writes:
- Summary to stderr (peak heap, total bytes, allocation count)
- `dhat-heap.json` to CWD (viewable in `dh_view.html`)

### 2.3 Per-Repo Integration Points

| Repo | Crate | Entry Point | Purpose |
|------|-------|-------------|---------|
| realizar | `realizar` (bin) | `src/main.rs` | Inference memory: model load, KV cache, serve |
| aprender | `apr-cli` (bin) | `crates/apr-cli/src/main.rs` | CLI memory: import, quantize, eval |
| trueno | `trueno` (lib) | `src/lib.rs` + integration tests | Tensor op allocations |
| entrenar | `entrenar` (bin) | `src/main.rs` | Training memory: buffers, optimizer, checkpoints |

### 2.4 Integration Test Pattern

For library crates (trueno), use dhat's assertion API in tests:

```rust
#[cfg(test)]
#[cfg(feature = "dhat-heap")]
mod dhat_tests {
    use super::*;

    #[test]
    fn test_vector_add_allocations() {
        let _profiler = dhat::Profiler::builder().testing().build();
        // ... exercise code ...
        let stats = dhat::HeapStats::get();
        // Assert allocation bounds
        dhat::assert!(stats.total_bytes < 1_000_000);
    }
}
```

## 3. Usage

### 3.1 Profile a Binary

```bash
# Profile realizar inference
cd ~/src/realizar
CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/realizar \
  cargo run --release --features dhat-heap -- \
  run /mnt/nvme-raid0/models/qwen3-coder-30b-q4k.apr "def fib(n):" --gpu --max-tokens 20

# Profile entrenar training
cd ~/src/entrenar
CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/entrenar \
  cargo run --release --features "dhat-heap,cuda,parquet" -- \
  train --config configs/pretrain-50m-quick.yaml

# Profile apr CLI
cd ~/src/aprender
CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/aprender \
  cargo run --release --features dhat-heap -p apr-cli -- \
  tensors info /mnt/nvme-raid0/models/qwen3-coder-30b-q4k.apr
```

### 3.2 View Results

```bash
# dhat-rs writes dhat-heap.json on exit
# Open in browser: https://nnethercote.github.io/dh_view/dh_view.html
# Load dhat-heap.json → tree view of allocation sites
```

### 3.3 Run dhat-Gated Tests

```bash
# Run tests with allocation assertions
cargo test --features dhat-heap
```

## 4. Contracts

Contract: `contracts/memory-profiling-v1.yaml` (ALB-099)

### 4.1 Obligations

| ID | Description | Falsification |
|----|-------------|---------------|
| C-DHAT-001 | All 4 repos compile with `--features dhat-heap` | `cargo check --features dhat-heap` exits 0 |
| C-DHAT-002 | Binary crates produce `dhat-heap.json` on exit | Run binary, assert file exists |
| C-DHAT-003 | Zero overhead when feature disabled | `cargo build --release` size unchanged ±1% |
| C-DHAT-004 | Feature flag is `dhat-heap` in all repos | `grep 'dhat-heap' Cargo.toml` in all 4 repos |

## 5. Immediate Profiling Targets

Once dhat-rs is integrated, profile these known hotspots:

| Repo | Scenario | What to Measure |
|------|----------|-----------------|
| realizar | Load 17 GB Q4K APR | Host-side allocations during tensor scan + upload |
| realizar | Serve 100 requests | Per-request allocation pattern, memory leaks |
| aprender | `apr import` 67 GB SafeTensors | Peak RSS during streaming import |
| entrenar | 100-step 350M training | Gradient buffer + optimizer state peak |
| trueno | Matrix multiply 4096×4096 | Temporary allocation count |
