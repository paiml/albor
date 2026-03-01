# 16. Reproducibility Protocol

| Artifact | Recorded |
|----------|----------|
| Random seed | 42 (global), per-component seeds derived |
| Data versions | HuggingFace dataset commit SHAs + local repo git SHAs |
| Data provenance | `alimentar provenance` manifest (source path, git SHA, file count, token count per source) |
| Data checksums | SHA-256 of every Parquet shard |
| Tokenizer | Vocabulary + merge rules saved in APR format |
| Training config | YAML checked into git |
| Checkpoint hashes | BLAKE3 of every checkpoint (forjar tripwire) |
| Software versions | Cargo.lock pinning all crate versions |
| Hardware | nvidia-smi + vulkaninfo + free -h captured |
| Training logs | Stdout/stderr + structured JSON logs |
| Eval results | JSON per checkpoint per stage, diffable |
| Teacher logits | BLAKE3 hashes of all pre-computed Parquet shards |
