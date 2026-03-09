# 20. Memory Profiling with dhat-rs

> **Status: COMPLETE (ALB-099 FIXED)**. dhat-rs integrated in all 5 repos.
> 21 memory issues found and fixed across 4 profiling rounds.

## 0. Motivation & Results

The sovereign stack originally had zero heap profiling instrumentation.
dhat-rs profiling (ALB-099) uncovered 21 issues across 5 repos:

**Round 1 (binary workloads):**
1. realizar: `std::fs::read` on 17 GB model → peak 19.6 GB. Fixed: early-out for GPU Q4K → **1.3 GB (93% reduction)**
2. realizar: PMAT-045 variable scoping bug (compile error)
3. aprender: `validate_gguf` dequantized every tensor to f32 → 8.2 GB. Fixed: metadata-only → **6.0 GB**
4. entrenar: `TRACER` accumulated unbounded Vec → ~2.8 GB at 28K steps. Fixed: aggregated HashMap → **O(1) memory**
5. alimentar: `cmd_info` loaded entire dataset → 73 MB. Fixed: Parquet footer → **0.2 MB (348x)**
6. trueno: `TunerFeatures::to_vector()` → 140K tiny Vec allocs. Fixed: `to_array()` → **zero alloc**

**Round 2 (deeper workloads):**
7. trueno: BLIS `gemm_blis()` allocated 4.3 MB workspace per call. Fixed: thread-local high-water-mark → **zero alloc after first call**
8. alimentar: `Unique::apply()` O(rows×cols) String allocs. Fixed: u64 hash keys → **zero String allocs**
9. aprender+entrenar: APR checkpoint save copied 450 MB weights twice. Fixed: consuming pipeline → **900 MB eliminated**

**Round 3 (code analysis):**
10. aprender: `resolve_f32_tied_embeddings` cloned BTreeMap (~1.4 GB). Fixed: Cow → **only clones when needed**
11. aprender: AprV2Writer output Vec no capacity hint. Fixed: `with_capacity(total_size)`
12. entrenar: ArrowDataset triple materialization (~1 GB/shard). Fixed: scoped drop
13. entrenar: 3 Vec allocations without `with_capacity` in hot path. Fixed: capacity hints
14. realizar: `detect_model()` read entire file for 8-byte magic. Fixed: `File::open` + `read(8)` → **~0 bytes**
15. realizar: `run_inference()` read entire file for format detection. Fixed: `read_exact(8)` → **~0 bytes**

**Round 4 (architectural issues — ALB-100 through ALB-105):**
16. entrenar: LMBatch stored input+target separately → 50% waste. Fixed: shared stride-based tokens (ALB-100)
17. entrenar: Eager dataset loading → 37 GB for 19 shards. Fixed: StreamingParquetLoader (ALB-101)
18. realizar: KV cache used `hidden_dim` not `kv_dim` → 4-8x over-allocation. Fixed: GQA-aware sizing (ALB-102)
19. realizar: `sample_topk` allocated 2.4 MB per token. Fixed: in-place masking + O(n) partial sort (ALB-103)
20. aprender: APR reader re-parsed header per tensor read. Fixed: cached `data_offset` (ALB-104)
21. aprender: APR `write_into()` buffered entire file in RAM. Fixed: AprV2StreamingWriter (ALB-105)

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

## 5. Profiling Results

All originally-planned hotspots have been profiled and fixed:

| Repo | Scenario | Before | After | Issue |
|------|----------|--------|-------|-------|
| realizar | Load 17 GB Q4K APR | 19.6 GB peak host RAM | 1.3 GB | ALB-099 #1 |
| realizar | Format detection | Read entire file | 8 bytes | ALB-099 #14-15 |
| realizar | KV cache (GQA) | 4-8x over-allocated | Correct kv_dim | ALB-102 |
| realizar | Top-k sampling | 2.4 MB/token | In-place, O(n) | ALB-103 |
| aprender | APR validation | 8.2 GB (dequant all) | Metadata-only | ALB-099 #3 |
| aprender | APR read | Re-parse per tensor | Cached offset | ALB-104 |
| aprender | APR write | Buffer in RAM | Streaming writer | ALB-105 |
| aprender | Tied embeddings | Clone 1.4 GB BTreeMap | Cow (lazy clone) | ALB-099 #10 |
| entrenar | Dataset loading | 37 GB eager | Streaming per-shard | ALB-101 |
| entrenar | LMBatch storage | 2x tokens (input+target) | Shared stride | ALB-100 |
| entrenar | Tracer memory | O(steps) unbounded | O(1) aggregated | ALB-099 #4 |
| entrenar | Checkpoint save | 900 MB redundant copies | Consuming pipeline | ALB-099 #9 |
| trueno | BLIS workspace | 4.3 MB/call | Thread-local reuse | ALB-099 #7 |
| trueno | TunerFeatures | 140K Vec allocs | Stack array | ALB-099 #6 |
| alimentar | cmd_info | 73 MB | 0.2 MB | ALB-099 #5 |
| alimentar | Unique::apply | O(rows×cols) Strings | u64 hash keys | ALB-099 #8 |

### Contract

`contracts/memory-profiling-v1.yaml` (ALB-099) — 4 obligations, all verified.
