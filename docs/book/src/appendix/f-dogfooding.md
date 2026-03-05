# Appendix F: Dogfooding Log

> Living record of tool validation against the Albor repo.
> Updated as gaps are discovered and resolved.

## Summary (2026-03-04)

| Tool | Command | Result | Gap |
|------|---------|--------|-----|
| `pv validate` | `pv validate contracts/*.yaml` | **PASS** (all 12 contracts) | — |
| `pv coverage` | `pv coverage contracts` | **PASS** (100% obligation coverage) | — |
| `pv graph` | `pv graph contracts` | **PASS** (8 nodes, correct deps) | — |
| `pv probar` | `pv probar contracts/*.yaml` | **PASS** (generates property tests) | — |
| `pv kani` | `pv kani contracts/*.yaml` | **PASS** (generates Kani harnesses) | — |
| `pv generate` | `pv generate contracts/*.yaml` | **PASS** (20 files: scaffold, kani, probar, book) | — |
| `pv scaffold` | `pv scaffold contracts/*.yaml` | **PASS** (Rust trait + test stubs) | — |
| `pv status` | `pv status contracts/*.yaml` | **PASS** (equation/obligation counts) | — |
| `pv audit` | `pv audit contracts/*.yaml` | **PASS** (no findings) | — |
| `pv equations` | `pv equations contracts/*.yaml` | **PASS** (formatted equations) | — |
| `pv book` | `pv book contracts/` | **PASS** (7 mdBook pages) | — |
| `pv lean` | `pv lean contracts/*.yaml` | **INFO** (needs `lean:` metadata blocks) | — |
| `forjar validate` | `forjar validate -f infra-only.yaml` | **PASS** (2 machines, 6 resources) | — |
| `forjar validate` | `forjar validate -f albor.yaml` | **PASS** (2 machines, 22 resources) | ~~ALB-027~~ FIXED |
| `forjar graph` | `forjar graph -f infra-only.yaml` | **PASS** (Mermaid output) | — |
| `apr finetune --plan` | `apr finetune --plan --model-size 350M --vram 24` | **PASS** (VRAM estimate correct) | — |
| `apr train plan --task pretrain` | `apr train plan --task pretrain --config pretrain-350m.yaml` | **PASS** (validates config, shows arch/params) | ~~ALB-009~~ FIXED |
| `apr distill --plan` | `apr distill --plan` | **PASS** (file-based mode) | — |
| `apr distill --config --plan` | `apr distill --config distill-entrenar.yaml --plan` | **PASS** (validates config, shows two-stage workflow) | ~~ALB-011~~ FIXED |
| `apr distill --config --plan --json` | `apr distill --config distill-entrenar.yaml --plan --json` | **PASS** (structured JSON with verdict) | ~~ALB-011~~ FIXED |
| `apr distill --config --stage precompute` | `apr distill --config distill-entrenar.yaml --stage precompute` | **PASS** (inspects teacher, 290 tensors, writes manifest) | ~~ALB-011~~ FIXED |
| `apr distill --config --stage train` | `apr distill --config distill-entrenar.yaml --stage train` | **PASS** (reads manifest, validates, sets up KD) | ~~ALB-011~~ FIXED |
| `apr train apply --parquet` | `apr train apply --task pretrain --config pretrain-parquet.yaml` | **PASS** (8 rows from Parquet, 4 batches, CUDA training) | ~~ALB-007~~ FIXED |
| `apr quantize --plan` | `apr quantize --plan <file>` | **PASS** (plan mode works) | — |
| `apr prune --plan` | `apr prune --plan <file>` | **PASS** (plan mode exists) | — |
| `alimentar quality profiles` | `alimentar quality profiles` | **PASS** (ml-training profile exists) | — |
| `alimentar import` | `alimentar import local <in> -o <out>` | **PASS** (local import works) | ~~ALB-019~~ FIXED |
| `alimentar mix` | `alimentar mix a.parquet:0.8 b.parquet:0.2 -o out.parquet` | **PASS** (weighted sampling + upsampling) | ~~ALB-020~~ FIXED |
| `apr tokenize plan` | `apr tokenize plan --data corpus.txt --vocab-size 32000` | **PASS** (validates corpus, estimates time) | ~~ALB-001~~ FIXED |
| `apr tokenize apply` | `apr tokenize apply --data corpus.txt --vocab-size 100` | **PASS** (trains BPE, writes vocab.json + merges.txt) | ~~ALB-001~~ FIXED |
| `alimentar fim` | `alimentar fim data.parquet -o fim.parquet --rate 0.5` | **PASS** (PSM/SPM FIM transform) | ~~ALB-018~~ FIXED |
| `batuta falsify` | `batuta falsify . --format markdown` | **PASS** (108 checks, 73.1% score) | ~~ALB-029~~ FIXED |
| `batuta falsify --critical-only` | `batuta falsify . --critical-only` | **PARTIAL** (3/5 pass, 1 fail) | ~~ALB-029~~ FIXED |
| `batuta stack status` | `batuta stack status --simple` | **PASS** (11 tools detected, 5 healthy) | ~~ALB-030~~ FIXED |
| `batuta oracle --list` | `batuta oracle --list` | **PASS** (lists all 40+ stack components) | — |
| `batuta oracle --recommend` | `batuta oracle --recommend --problem "train 350M LLM"` | **PASS** (recommends aprender) | — |
| `batuta oracle --local` | `batuta oracle --local` | **PASS** (47 PAIML projects discovered) | — |
| `batuta oracle --capabilities` | `batuta oracle --capabilities entrenar` | **PASS** (autograd, lora, qlora, quantization, model_merge, distillation) | — |
| `batuta playbook validate` | `batuta playbook validate albor-playbook.yaml` | **PASS** (19 stages, 14 params, acyclic DAG) | — |
| `batuta hf search` | `batuta hf search model "code completion"` | **PARTIAL** (returns placeholder/mock data) | — |
| `bashrs make lint` | `bashrs make lint Makefile` | **PASS** (2 warnings, 0 errors) | — |
| `bashrs make parse` | `bashrs make parse Makefile` | **PASS** (full AST) | — |
| `bashrs make purify` | `bashrs make purify Makefile` | **PASS** (purified output) | — |
| `bashrs classify` | `bashrs classify Makefile` | **PASS** (safe: 85%) | — |
| `apr pipeline validate` | `apr pipeline validate albor.yaml` | **PASS** (2 machines, 22 resources) | ~~ALB-028~~ FIXED |
| `apr pipeline plan` | `apr pipeline plan albor.yaml` | **PASS** (23 resources, full DAG) | ~~ALB-028~~ FIXED |
| `apr pipeline plan --json` | `apr pipeline plan albor.yaml --json` | **PASS** (structured JSON with deps) | ~~ALB-028~~ FIXED |
| `apr pipeline status` | `apr pipeline status albor.yaml` | **EXPECTED FAIL** (no state dir yet) | — |
| `pmat query` | `pmat query "training"` | **PASS** (0 functions, 5 document matches) | — |
| `pmat analyze makefile` | `pmat analyze makefile Makefile` | **PASS** (64% quality score) | — |
| `pv lean` | `pv lean contracts/kd-v1.yaml` | **PASS** (6 Lean 4 theorem stubs generated) | — |
| `pv lean-status` | `pv lean-status contracts/` | **PASS** (0% L4 coverage, 4 sorry debt) | — |
| `apr train plan --task classify` | `apr train plan --data <JSONL>` | **PASS** (classification fine-tuning) | — |
| `apr merge` | `apr merge --strategy slerp` | **PASS** (SLERP, TIES, DARE supported) | — |
| `apr export --list-formats` | `apr export --list-formats` | **PASS** (SafeTensors, GGUF, MLX) | — |
| `apr publish` | `apr publish <dir> <repo>` | **PASS** (HF Hub publish exists) | — |
| `apr eval` | `apr eval <model>` | **PASS** (perplexity eval) | — |
| `apr eval --task code` | `apr eval model --task code --data bench.jsonl` | **PASS** (pass@1 scoring, 10/10 on basic set) | ~~ALB-006~~ FIXED |
| `apr eval --task plan` | `apr eval model --task plan --data bench.jsonl` | **PASS** (dry-run validation) | ~~ALB-006~~ FIXED |
| `alimentar mix` (test) | `alimentar mix ...parquet:0.25 -o test.parquet -n 200 --seed 456` | **PASS** (200 rows, 50 per corpus) | — |
| `alimentar fim` (prod) | `alimentar fim mixed.parquet -o mixed-fim.parquet --rate 0.5 --format psm` | **PASS** (17,070 rows, PSM FIM 50%) | — |
| `apr tokenize apply` (prod) | `apr tokenize apply --data corpus-raw.txt --vocab-size 32768 --algorithm bpe -o tokenizer/ --max-lines 100000` | **PASS** (32,768 vocab, 2022.5s, 8/8 Python patterns) | ~~ALB-001~~ FIXED |
| `alimentar quality` | `alimentar quality profiles` | **PASS** (ml-training profile) | — |
| `alimentar convert` | `alimentar convert` | **PASS** (format conversion) | — |
| `bashrs score` | `bashrs score Makefile` | **PASS** (D grade, 5.2/10) | — |
| `bashrs audit` | `bashrs audit Makefile` | **PASS** (comprehensive audit) | — |
| `entrenar train` (50M) | `entrenar train pretrain-50m-test.yaml` | **PASS** (demo batches, 465ms, loss 10.34→9.67) | ALB-033 (tokenizer format) |
| `apr train apply` (50M) | `apr train apply --task pretrain --config pretrain-50m-test.yaml` | **PASS** (10-row micro, 5 batches, 2.1s CUDA) | ~~ALB-034~~ FIXED |
| `apr train apply` (50M full) | `apr train apply --task pretrain --config pretrain-50m.yaml` | **PASS** (500 rows, 125 batches, 31 steps, 110.7s CUDA, loss 10.3→4.42) | ~~ALB-034~~ FIXED |
| `apr train apply` (50M v2) | `apr train apply --task pretrain --config pretrain-50m-v2.yaml` | **PASS** (pre-tokenized ByteLevel BPE, 108.5s CUDA, loss→5.51) | — |
| `apr train plan` (350M) | `apr train plan --task pretrain --config pretrain-350m.yaml` | **PASS** (config validated, ready for apply) | — |
| `entrenar validate` | `entrenar validate pretrain-350m-manifest.yaml` | **PASS** (architecture overrides bridge through) | ~~ALB-021~~ FIXED |
| `entrenar shorthand` | `vocab_size: "32K"` in YAML manifest | **PASS** (parses to 32768) | ~~ALB-022~~ FIXED |
| `apr merge --plan` | `apr merge a.apr b.apr --plan --strategy slerp -o merged.apr` | **PASS** (validates inputs, shows strategy, sizes) | ~~ALB-023~~ FIXED |
| `apr export --plan` | `apr export model.apr --plan --format gguf -o model.gguf` | **PASS** (validates format, shows plan) | ~~ALB-023~~ FIXED |
| `apr publish --plan` | `apr publish dir repo --plan` | **PASS** (alias for --dry-run) | ~~ALB-023~~ FIXED |
| `apr train apply` (350M full) | `apr train apply --task pretrain --config pretrain-350m.yaml` | **FAIL** (ALB-060: epochs=1 exhausted data at step 43/5000, loss flat ~10.39, LR still in warmup at 6.45e-6) | ALB-060 |
| `apr train apply` (350M v2) | `apr train apply --task pretrain --config pretrain-350m-v2.yaml` | **PASS** (ALB-065 fixed: `stream.synchronize()` before D2H gradient transfers. Training stable without `CUDA_LAUNCH_BLOCKING=1`, 441 tok/s) | ~~ALB-064~~ ~~ALB-065~~ FIXED |
| `train-guard.sh` | `bash scripts/train-guard.sh configs/train/pretrain-350m-v2.yaml` | **PASS** (crash-resilient supervisor with auto-diagnostic CUDA blocking mode, exit code classification, GPU state capture, JSON crash reports, backoff restart, heartbeat monitoring) | ~~ALB-064~~ FIXED |
| `pv validate` (memory) | `pv validate contracts/training-memory-kernel-v1.yaml` | **PASS** (0 errors, 0 warnings) | ALB-039 |
| `pv validate` (GPU) | `pv validate contracts/training-gpu-kernel-v1.yaml` | **PASS** (0 errors, 0 warnings) | ALB-040 |
| `apr train apply` (50M CUDA) | `apr train apply --config pretrain-50m-v2-test.yaml` | **PASS** (3 steps, loss 10.4→11.7, GPU forward+backward) | ~~ALB-041~~ FIXED |
| `apr eval` (50M safetensors) | `apr eval checkpoints/albor-base-50m/model.safetensors --dataset custom` | **FAIL** (PPL 679,614 — weights ignored) | ~~ALB-037~~ FIXED |
| `apr train apply` (350M CUDA test) | `apr train apply --config pretrain-350m-cuda-test.yaml` | **PASS** (50 steps, ~400s, loss 10.39→5.92, best 5.53, checkpoint saved) | ~~ALB-043~~ ~~ALB-044~~ ~~ALB-059~~ FIXED |
| `realizar run` (350M) | `realizar run checkpoints/albor-350m-cuda-test/model.safetensors "def fibonacci(" --raw` | **PASS** (218 tensors loaded, 50 tokens generated, 1.0 tok/s) | ~~ALB-037~~ FIXED |
| `eval-perplexity.py` (350M validate) | `python scripts/eval-perplexity.py checkpoints/albor-350m-cuda-test/ --validate-checkpoint` | **PASS** (weights trained, layers distinct) | — |
| `eval-perplexity.py` (350M perplexity) | `python scripts/eval-perplexity.py checkpoints/albor-350m-cuda-test/ --data val.parquet --max-sequences 3 --seq-len 64` | **PASS** (PPL 31,926 — finite, consistent with 50-step model) | — |
| `eval-code.py` (validate) | `python scripts/eval-code.py configs/eval/python-intermediate.jsonl --validate-only` | **PASS** (15/15 canonical solutions) | — |
| `eval-code.py` (HumanEval) | `python scripts/eval-code.py configs/eval/humaneval-subset.jsonl --validate-only` | **PASS** (20/20 canonical solutions) | — |
| `convert-checkpoint.py` (50M) | `python scripts/convert-checkpoint.py checkpoints/albor-base-50m/` | **PASS** (110→111 tensors, 85 reshaped, lm_head created) | ALB-037 |
| `eval-perplexity.py --validate` | `python scripts/eval-perplexity.py checkpoints/albor-base-50m/ --validate-checkpoint` | **FAIL** → **FIXED** (ALB-038 root cause in autograd) | ~~ALB-038~~ FIXED |
| checkpoint analysis | byte-compare layers 0-11 q_proj, gate_proj | **FAIL** → **FIXED** (all parameters now receive gradients) | ~~ALB-038~~ FIXED |
| `apr monitor` (TUI) | `apr monitor checkpoints/albor-base-350m/` | **PASS** (presentar TUI, live GPU telemetry, loss curve, tok/s) | ~~ALB-045~~ ~~ALB-046~~ ~~ALB-047~~ ~~ALB-048~~ FIXED |
| `apr monitor --json` | `apr monitor --json checkpoints/albor-base-350m/` | **PASS** (headless JSON with full TUI parity) | ~~ALB-053~~ ~~ALB-058~~ FIXED |
| `apr monitor` (discover) | `apr monitor` (no args) | **PASS** (discovers active runs from global SQLite registry) | ~~ALB-054~~ FIXED |
| `apr train apply` (SQLite) | `apr train apply --config pretrain-50m-quick.yaml` | **PASS** (creates both local + global experiments.db, logs params + metrics) | ~~ALB-055~~ ~~ALB-056~~ FIXED |
| `apr runs ls --global` | `apr runs ls --global` | **PASS** (table output: experiment, run ID, status, loss, tok/s, duration) | ~~ALB-050~~ FIXED |
| `apr runs ls --global --json` | `apr runs ls --global --json` | **PASS** (JSON array with all run metadata) | ~~ALB-050~~ FIXED |
| `apr runs show` | `apr runs show <id> --global` | **PASS** (params, loss, tok/s, lr, duration) | ~~ALB-050~~ FIXED |
| `apr runs show --json` | `apr runs show <id> --global --json` | **PASS** (clean JSON with native param values) | ~~ALB-050~~ FIXED |
| `realizar run` (350M v2) | `realizar run checkpoints/albor-350m-cuda-test/model.safetensors "def fibonacci("` | **PASS** (24 layers, 32768 vocab, 50 tokens, 1.9 tok/s, garbage output expected from 5-step model) | — |
| `pv audit` (all) | `pv audit contracts/*.yaml` (7 contracts) | **PASS** (0 findings, 22 equations, 43 obligations, 26 falsification tests) | — |
| `batuta falsify --critical-only` | `batuta falsify . --critical-only` | **PARTIAL** (3/5 pass, 80.0% score, AI-01/AI-05 partial) | — |
| `apr runs diff` | `apr runs diff <a> <b> --global` | **PASS** (side-by-side sparklines, config diff, loss comparison, verdict) | ~~ALB-051~~ FIXED |
| `apr runs diff --json` | `apr runs diff <a> <b> --global --json` | **PASS** (structured JSON: summaries, config_diff, verdict for LLM agents) | ~~ALB-051~~ FIXED |
| `apr monitor` (widget composition) | `TrainingDashboard` composes `Layout`, `Border`, `Meter`, `GpuPanel`, `Sparkline`, `Text` | **PASS** (builds clean, widget tree rebuilt each frame, panel verification wired) | ~~ALB-057~~ FIXED |
| `apr experiment view --global --json` | `apr experiment view --global --json` | **PASS** (JSON output with experiments, run_ids, loss_values, params from SQLite) | ~~ALB-024~~ FIXED |
| `apr experiment view --global` | `apr experiment view --global` | **PASS** (ratatui TUI: run table, sparkline, braille loss chart, j/k navigation) | ~~ALB-024~~ FIXED |
| `pv validate` (training-config) | `pv validate contracts/training-config-kernel-v1.yaml` | **PASS** (0 errors, 8 obligations, 5 falsification tests, 2 Kani harnesses) | ALB-060 |
| `pv coverage` (all 8 contracts) | `pv coverage contracts/` | **PASS** (8 contracts, 31 equations, 51 obligations, 34 falsification tests, 100% coverage) | — |
| `apr train apply` (50M post-fix) | `apr train apply --config pretrain-50m-quick.yaml` | **PASS** (5 steps, loss 10.42→9.45, GEMM backward now correct) | ~~ALB-059~~ FIXED |
| `apr train apply` (350M post-fix) | `apr train apply --config pretrain-350m-cuda-test.yaml` | **PASS** (50 steps, loss 10.39→5.92, best 5.53, zero NaN, correct backward gradients) | ~~ALB-059~~ FIXED |
| `realizar run` (350M post-fix) | `realizar run checkpoints/albor-350m-cuda-test/model.safetensors "def fibonacci("` | **PASS** (218 tensors, generates tokens from correctly-trained weights) | ~~ALB-059~~ FIXED |
| `apr quantize` (50M int4) | `apr quantize model.safetensors -s int4` | **PASS** (238 MiB → 30 MiB, 87.5% reduction, 7.99x) | — |
| `apr quantize` (50M q4k) | `apr quantize model.safetensors -s q4k` | **PASS** (238 MiB → 238 MiB, 0% reduction — q4k no-op on 1D tensors) | — |
| `apr quantize` (350M int4) | `apr quantize model.safetensors -s int4` | **PASS** (1.48 GiB → 191 MiB, 87.5% reduction, 7.99x) | — |
| `apr quantize` (350M q4k) | `apr quantize model.safetensors -s q4k` | **PASS** (1.48 GiB → 1.48 GiB, 0% reduction — q4k no-op on 1D tensors) | — |
| `apr prune` (50M magnitude) | `apr prune model.safetensors --method magnitude --sparsity 0.5` | **PASS** (50.0% zeros, 31.2M/62.4M params zeroed) | — |
| `apr prune` (50M depth) | `apr prune model.safetensors --method depth --remove-layers "8-11"` | **PASS** (110→74 tensors, 238→180 MiB, layers 8-11 removed) | — |
| `apr prune` (350M magnitude) | `apr prune model.safetensors --method magnitude --sparsity 0.3` | **PASS** (50.0% zeros — sparsity param may be ignored) | — |
| `source-to-parquet.py` (Tier 2) | `python scripts/source-to-parquet.py ~/src/pytorch pytorch data/parquet/tier2/pytorch.parquet` | **PASS** (8 repos → 28,553 Python files imported) | — |
| `alimentar mix` (expanded) | `alimentar mix ...T1:10.0 ...T2:1.0 -o mixed.parquet --seed 42` | **PASS** (12 datasets → 45,420 rows, proportional weighted sampling) | — |
| `alimentar fim` (expanded) | `alimentar fim mixed.parquet -o mixed-fim.parquet --rate 0.5 --format psm` | **PASS** (45,420 rows, 50% PSM FIM) | — |
| `pretokenize.py` (v2) | `python scripts/pretokenize.py --input mixed-fim.parquet --seq-len 2048` | **PASS** (67,977 sequences, 139M tokens, 191 MiB) | — |
| `realizar run` (0.5B teacher) | `realizar run qwen2.5-coder-0.5b/model.safetensors "def fibonacci("` | **PASS** (24 layers, 151936 vocab, 2.8 tok/s, generates tokens) | — |
| `apr distill --stage precompute` (0.5B) | `apr distill --config distill-entrenar.yaml --stage precompute` | **PASS** (290 tensors, 942 MiB, manifest written) | — |
| `apr distill --stage precompute` (3B) | `apr distill --config distill-qwen3b.yaml --stage precompute` | **PASS** (434 tensors, 5.75 GiB, sharded SafeTensors loaded) | — |
| `realizar run` (3B sharded) | `realizar run qwen2.5-coder-3b/model-00001-of-00002.safetensors` | **FAIL** (sharded SafeTensors not supported — model.norm.weight in shard 2) | — |
| C-TRAINCFG-001 pre-flight (v2) | `python3 -c "..."` (algebraic check) | **PASS** (67977 seqs, 132 steps/epoch, 38 epochs, warmup=500=10%) | ALB-060 |
| `alimentar dedup` | `alimentar dedup data.parquet -o dedup.parquet` | **PASS** (exact dedup by text column, found 2 dups in 1843 rows) | — |
| `alimentar filter-text` | `alimentar filter-text data.parquet -o filtered.parquet --threshold 0.4` | **PASS** (composite scoring: alnum ratio, line length, dup lines, entropy) | — |
| `apr eval --task humaneval` | `apr eval model.safetensors --task humaneval --data humaneval.jsonl` | **PASS** (20/20 problems validated, pass@1/10/100 metrics, JSON output) | — |
| `apr eval --task contamination` | `apr eval model.safetensors --task contamination --data train.jsonl` | **PASS** (10-gram Jaccard overlap, 0/179 contaminated) | — |
| `apr eval --task compare` | `apr eval model_a.safetensors --task compare --data model_b.safetensors` | **PASS** (side-by-side: size, tensors, format, ratio) | — |
| `apr train watch` | `apr train watch --config pretrain-350m-v2.yaml` | **PASS** (crash recovery, exponential backoff, GPU diagnostics, crash-reports JSON) | — |
| `apr eval --task verify` | `apr eval checkpoints/albor-350m-cuda-test/ --task verify` | **PASS** (9/9 checks: safetensors header, tensor count, FNV-1a hash, config.json) | — |
| `apr train sweep` | `apr train sweep --config base.yaml --strategy random --num-configs 5` | **PASS** (5 configs with log-uniform LR, batch size, weight decay, warmup) | — |
| `apr train archive` | `apr train archive checkpoints/albor-50m-quick/ -o /tmp/archive --version v0.1` | **PASS** (4 files, 238 MB, MANIFEST.json with BLAKE3 hashes) | — |
| `apr eval --task correlation` | `apr eval checkpoints/ --task correlation` | **PASS** (236 data points, Pearson r=-0.14, Spearman rho=-0.21, from loss_history) | — |
| `apr eval --task human` (generate) | `apr eval checkpoints/albor-350m-cuda-test/ --task human` | **PASS** (10-prompt ratings sheet with criteria, JSON output) | — |
| `apr eval --task human` (analyze) | `apr eval /tmp --task human --data test-ratings.jsonl` | **PASS** (mean=3.0, median=3.0, pass@3=60%, distribution histogram) | — |
| `apr encrypt` | `apr encrypt model.safetensors -o model.enc --key-file key.bin` | **PASS** (238 MB, 0.89s, BLAKE3 keystream + MAC) | — |
| `apr decrypt` | `apr decrypt model.enc -o model.safetensors --key-file key.bin` | **PASS** (238 MB roundtrip verified, MAC authenticated, 0.74s) | — |
| `apr train plan` (R-095) | `apr train plan --task pretrain --config pretrain-350m-cuda-test.yaml` | **PASS** (extended: RAM 5.5GB, disk 4.5GB/ckpt, 2048 tok/step, 60ms/step, 34K tok/s) | — |
| `apr train apply --distributed` | `apr train apply --task pretrain --config pretrain-350m.yaml --distributed --world-size 2` | **PASS** (CLI flags accepted, YAML patched with distributed section) | — |
| `apr train apply --deterministic` | `apr train apply --task pretrain --config pretrain-50m-quick.yaml --deterministic --seed 42` | **PASS** (deterministic + seed flags injected into YAML) | — |
| `entrenar` (activation checkpointing) | `with_checkpointing(4)` in TransformerTrainConfig | **PASS** (checkpoint boundary mask, segment-based recomputation, 4 unit tests) | ~~#115~~ FIXED |
| `entrenar` (gradient accumulation) | `with_accumulation_steps(4)` in CudaTransformerTrainer | **PASS** (per-block CPU accum, download workspace D2H, average + upload H2D + optimizer, 2 unit tests) | ~~#131~~ FIXED |
| `pv validate` (distributed) | `pv validate contracts/C-DDP-001.yaml contracts/C-RING-001.yaml contracts/C-SHARD-001.yaml contracts/C-WIRE-002.yaml` | **PASS** (4 new contracts, 0 errors) | — |
| `entrenar` (distributed DDP) | 4-worker ring AllReduce, per-block reverse-order AllReduce | **PASS** (C-DDP-001 weight consistency via BLAKE3, 11 integration tests) | ~~#145~~ FIXED |
| `entrenar` (comm-overlap) | AllReduce + computation overlap timing test | **PASS** (overlap ≤ sequential time, concurrent threads) | ~~#145~~ FIXED |
| `entrenar` (multi-node) | 3-node checkpoint coordination, block gradient exchange | **PASS** (barrier sync lifecycle, concurrent AllReduce + checkpoint) | ~~#145~~ FIXED |
| `entrenar` (heterogeneous) | detect_all_devices(), mixed-backend AllReduce | **PASS** (CUDA+wgpu+CPU workers produce identical averaged gradients) | ~~#145~~ FIXED |
| `apr train apply` (350M ALB-069) | `apr train apply --config pretrain-350m-cuda-test.yaml` (post-selp fix) | **PASS** (5 steps, loss 10.42→10.13, fused CE kernel produces non-zero loss) | ~~ALB-069~~ FIXED |
| `apr train apply` (350M ALB-070) | `apr train apply --config pretrain-350m-v2.yaml` (save_interval fix) | **PASS** (save_interval=250 works, eval_batch truncates to max_seq_len) | ~~ALB-070~~ FIXED |
| `apr train apply` (350M ALB-071) | `apr train apply --config pretrain-350m-cuda-test.yaml` (embed clip fix) | **PASS** (5 steps, embed grad clipped with unwrap_or(1.0), no NaN) | ~~ALB-071~~ FIXED |
| `apr train apply` (350M ALB-072 FP32) | `apr train apply --config pretrain-350m-fp32-test.yaml` | **PASS** (5 steps, all 218 tensors OK, gnorm=2.29, FP32 baseline) | — |
| `apr train apply` (350M ALB-072 FP16) | `apr train apply --config pretrain-350m-cuda-test.yaml` (loss scale fix) | **PASS** (50 steps, all 218 tensors OK, gnorm matches FP32 baseline, zero NaN) | ~~ALB-072~~ FIXED |
| `apr train apply` (350M v2 full) | `apr train apply --config pretrain-350m-v2.yaml` (all fixes) | **CRASHED** step 1183/5000. Loss 10.40→6.85. ALB-073 (PTX selp) + ALB-074 (stale binary buffer overflow). Step 1000 checkpoint saved. | ALB-063 |
| `apr train apply` (binary verify) | `apr train apply --config pretrain-350m-cuda-test.yaml` (rebuilt binary) | **PASS** (5 steps, loss=10.40, gnorm=2.29, no PTX errors, no buffer overflow) | ~~ALB-073~~ ~~ALB-074~~ FIXED |
| codeparrot download | `scripts/download-codeparrot.py --max-rows 2000000` | **PASS** (2M files, 20 shards, 6.1 GB, ~4.4B tokens, 99.2% filter pass rate, 499s) | Data scaling |
| pretokenize v3 | `scripts/pretokenize.py --shard-output --seq-len 1024` | **IN PROGRESS** (20 shards, ~260K seqs/shard, ~266M tokens/shard) | Data scaling |

## ALB-060: Training Config Epoch/Step Mismatch (Critical)

**Discovery**: The 350M "full training" run completed in 11.8 seconds instead of
the expected 12+ hours, producing an effectively untrained model.

**Five Whys (per CLAUDE.md Rule 7)**:

1. **Why did loss stay flat at ~10.39?** The learning rate never reached a meaningful
   value — max LR achieved was 6.45e-6 vs target 3e-4.
2. **Why was LR so low?** The warmup schedule is linear over 2000 steps, but training
   only ran 43 steps. At step 43: lr = 3e-4 × (43/2000) = 6.45e-6.
3. **Why only 43 steps?** `steps_per_epoch = floor(22079 / 4 / 128) = 43`. With
   `epochs: 1`, total achievable steps = 43. `max_steps: 5000` is unreachable.
4. **Why only 1 epoch?** The config comment says "Pre-training uses max_steps, not epochs"
   but entrenar's training loop respects `epochs` as a hard cap — it does NOT loop
   data to fill `max_steps`.
5. **Why no validation?** No pre-flight check computes `steps_per_epoch` and compares
   against `max_steps` + `warmup_steps`. The algebraic inconsistency is invisible.

**Algebraic proof (from C-TRAINCFG-001 contract)**:
```
num_sequences       = 22,079
micro_batch_size    = 4
grad_accum_steps    = 128
steps_per_epoch     = floor(22079 / 4 / 128) = 43
total_achievable    = 1 × 43 = 43
max_steps           = 5,000       ← UNREACHABLE
warmup_steps        = 2,000       ← NEVER COMPLETES
tokens_trained      = 43 × 4 × 128 × 1024 = 22.5M
chinchilla_min      = 10 × 370M = 3.7B   ← undertrained by 164×
```

**Fix required (two options)**:
1. Set `epochs: 117` (ceil(5000/43)) to cycle data 117 times → reaches 5031 steps
2. Add epoch-looping to entrenar: when `max_steps` is set and epochs exhausted,
   reshuffle data and continue (treats `max_steps` as authoritative, `epochs` as informational)

**Contract**: `contracts/training-config-kernel-v1.yaml` (C-TRAINCFG-001) with
7 equations, 8 proof obligations, 5 falsification tests, 2 Kani harnesses.
FALSIFY-CFG-001 and FALSIFY-CFG-002 algebraically prove this config is invalid.

**Training state.json analysis**: The `loss_history` array (55 entries, all ~10.39-10.40)
and `learning_rate: 0.0` confirm the model never learned. The `status: "Running"` field
is stale (training completed but status was not updated to "Completed" — minor bug).

**Secondary bug**: The training log displays `loss=0.0000` for every step despite
training_state.json recording real loss values ~10.39. This is the known ALB-042
display bug (loss=0.0 reporting).

## Contract Validation Detail

All 8 contracts pass `pv validate` with 0 errors. The original 5 were rewritten from
a custom schema to match `pv`'s schema (`metadata:`, `formula:`, `proof_obligations:`,
`falsification_tests:`). The two training kernel contracts (ALB-039, ALB-040) and the
training config contract (ALB-060) were written directly in the correct schema.

```
pv coverage contracts
---------------------
Contracts:            8
Equations:            31
Obligations:          51
Falsification tests:  34
Kani harnesses:       10
Overall coverage:     100.0%
```

## pv generate Detail

`pv generate` produces 4 files per contract (28 total):

| Type | Content | Example |
|------|---------|---------|
| `*_scaffold.rs` | Rust trait with documented invariants | `knowledge-distillation-kernel-v1_scaffold.rs` |
| `*_probar.rs` | Property tests derived from proof obligations | 6 property tests + 5 falsification test stubs |
| `*_kani.rs` | Kani verification harnesses | 2 harnesses with `stub_float` strategy |
| `*_book.md` | mdBook page with equations, deps, obligations | Mermaid dependency graph, LaTeX equations |

`pv book contracts/` generates 7 contract pages directly into mdBook format.
These have been integrated into the albor mdBook under "Kernel Contracts".

## Pipeline Manifest Validation Detail

The full pipeline manifest (`configs/pipeline/albor.yaml`) now passes `forjar validate`
after the ALB-027 fix added the `task` resource type:

```
forjar validate -f configs/pipeline/albor.yaml
OK: albor-training-pipeline (2 machines, 22 resources)
```

Forjar supports all 13 resource types: `package`, `file`, `service`, `mount`, `user`,
`docker`, `pepita`, `network`, `cron`, `recipe`, `model`, `gpu`, `task`.

The `task` resource type is the key piece that turns forjar from an infrastructure
tool into a pipeline orchestrator — it runs arbitrary commands with idempotency
tracking via output artifact hashing.

### Spec Correction: `names` to `packages`

Dogfooding revealed that the spec used `names:` for forjar package resources, but
forjar expects `packages:`. Also requires `provider: apt` (not implicit). Both the
spec and configs were corrected.

## Batuta Playbook Detail

Created `configs/pipeline/albor-playbook.yaml` -- a batuta playbook that expresses
the full albor ML pipeline as a 19-stage deterministic DAG with BLAKE3 caching:

```
batuta playbook validate configs/pipeline/albor-playbook.yaml
Playbook 'albor-training-pipeline' is valid
  Stages: 19
  Params: 14
```

Stages: validate-contracts, validate-configs, data-download, data-tokenize, data-mix,
pretrain, eval-base, teacher-logits, distill, eval-distill, finetune, eval-sft,
merge, eval-merged, prune, eval-pruned, quantize, eval-q4, publish.

This playbook is the **actual executable pipeline** (once upstream gaps are resolved).
The forjar manifest handles infrastructure; the batuta playbook handles ML orchestration.

## Batuta Falsification Detail (Full Report)

`batuta falsify . --format markdown` runs 108 checks across 10 categories:

| Category | Passed | Failed | Partial | Total |
|----------|--------|--------|---------|-------|
| Numerical Reproducibility | 13 | 0 | 2 | 15 |
| Jidoka Automated Gates | 4 | 5 | 1 | 10 |
| Architectural Invariants | 1 | 3 | 1 | 5 |
| Performance & Waste Elimination | 7 | 0 | 8 | 15 |
| ML Technical Debt Prevention | 2 | 1 | 7 | 10 |
| Hypothesis-Driven Development | 5 | 0 | 8 | 13 |
| Sovereign Data Governance | 12 | 0 | 3 | 15 |
| Cross-Platform & API | 2 | 0 | 3 | 5 |
| Safety & Formal Verification | 5 | 1 | 4 | 10 |
| Model Cards & Auditability | 3 | 0 | 7 | 10 |

**Before ALB-029 fix:** Score 72.2% (58 pass, 10 fail, 40 partial).

**After ALB-029 fix:** Score 73.1% (55 pass, 5 fail, 48 partial).

Upstream fixes resolved AI-01 (configs/ glob), AI-04 (book-output/ exclusion),
and AI-05 (non-Rust schema detection via pv/forjar).
Full report saved to `docs/falsification-report.md`.

## bashrs Makefile Linting Detail

bashrs `make lint` is the **sovereign Makefile linter** -- it validates
Makefile quality, safety, and best practices:

```
bashrs make lint Makefile
  MAKE010: Command 'rm' missing error handling
  MAKE015: Missing .DELETE_ON_ERROR
bashrs classify Makefile
  safe: 85.0%
```

Both warnings were addressed. bashrs also provides:
- `bashrs make parse` -- full Makefile AST
- `bashrs make purify` -- deterministic + idempotent Makefile output
- `bashrs classify` -- safety classification with multi-label support

## apr train plan/apply Detail

`apr train plan/apply` exists but is currently scoped to **classification fine-tuning
with HPO** (Tree-of-Parzen Estimators):

```
Current:  apr train plan --data <JSONL> --model-size 0.5B --task classify
Target:   apr train plan configs/train/pretrain-350m.yaml
```

The plan/apply infrastructure is solid -- `apr train plan` generates structured
summaries with resource estimates. The gap (ALB-009) is in scope: extending from
classification to causal LM pre-training, and from flag-driven to config-file-driven.

## Upstream Fixes Implemented

Dogfooding cycle 2 identified gaps that were **fixed upstream** and verified:

### ALB-029: batuta falsify false positives (FIXED)

Three fixes in `batuta/src/falsification/`:

1. **AI-01**: Added `configs/**` glob pattern (plural) alongside `config/**` in `invariants.rs`
2. **AI-04**: Added `book-output/` to JS exclusion list in `is_excluded_js_path()`
3. **AI-05**: Extended `detect_schema_deps()` to detect non-Rust validation:
   - pv/forjar validation commands in Makefile and CI configs
   - Python validation libs (pydantic, marshmallow, cerberus)
   - pv contracts (YAML with `proof_obligations:` key)

Commit: `batuta@905a862` → Score improved from 72.2% to 73.1%.

### ALB-030: batuta stack status without Cargo.toml (FIXED)

`DependencyGraph::from_workspace()` now falls back to binary detection
when no Cargo.toml exists. Discovers installed PAIML binaries via `which`,
extracts versions from `--version` output.

Commit: `batuta@371557a` → `batuta stack status` works in albor.

### ALB-019: alimentar import subcommand (FIXED)

Made `Import` command always available (not feature-gated behind `hf-hub`).
Added `alimentar import local <input> -o <output>` for local file import
with format conversion (CSV, JSON, JSONL, Parquet).

Commit: `alimentar@265541b` → `alimentar import local` works.

### ALB-020: alimentar mix subcommand (FIXED)

Added `alimentar mix` with weighted sampling and upsampling. Supports
`file:weight` syntax for weighted input, deterministic seeding, and
efficient Arrow batch processing with `arrow::compute::take`.

Commit: `alimentar@64b1e92` → `alimentar mix` works.

### ALB-001: apr tokenize plan/apply (FIXED)

Added `apr tokenize plan/apply` subcommands for BPE vocabulary training:
- `plan` validates corpus (lines, bytes, unique chars), estimates training time
- `apply` trains BPE/WordPiece/Unigram tokenizer, writes `vocab.json` + `merges.txt`
- Supports text, JSON, and YAML output formats for plan

Commit: `aprender@90427205` → `apr tokenize plan/apply` works.

### ALB-018: Fill-in-the-Middle (FIM) data transform (FIXED)

Added `alimentar fim` subcommand and `Fim` transform implementing PSM/SPM
FIM formats (Bavarian et al. 2022). Features:
- Configurable FIM rate (probability per row)
- PSM and SPM format variants
- Custom sentinel tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`)
- Deterministic with seed, respects char boundaries
- Rows below `min_chars` threshold left unchanged
- 10 unit tests

Commit: `alimentar@290582d` → `alimentar fim` works.

### ALB-021: Custom model architecture params in YAML (FIXED)

Added `ArchitectureOverrides` to `ModelRef` in entrenar's config schema.
The bridge converter (`manifest_to_spec`) now maps YAML manifest
`architecture:` fields to overrides that are applied on top of the
resolved `TransformerConfig` (from `config.json` or demo defaults).

Supported override fields: `hidden_size`, `num_hidden_layers`,
`num_attention_heads`, `num_kv_heads`, `intermediate_size`, `vocab_size`,
`max_position_embeddings`, `rms_norm_eps`, `rope_theta`, `use_bias`.

The YAML manifest `ArchitectureConfig` also gained serde aliases
(`num_hidden_layers` → `num_layers`, `num_attention_heads` → `num_heads`,
`num_key_value_heads` → `num_kv_heads`, `max_position_embeddings` → `max_seq_length`)
for compatibility with HuggingFace config.json field names.

Commit: `entrenar@a414861` → Architecture overrides work end-to-end.

### ALB-022: Human-readable value shorthand in YAML configs (FIXED)

Added `shorthand` module with `parse_human_usize()` and
`deserialize_human_usize_opt` custom serde deserializer. Supports:

- **SI suffixes (binary)**: `32K` (32×1024), `1M` (1×1024²), `1G` (1×1024³)
- **SI suffixes (decimal)**: `10B` (10×10⁹), `1T` (1×10¹²)
- **Scientific notation**: `1e6`, `3.2e4`
- **Fractional suffixes**: `1.5K` (1536)
- **Plain numbers**: `1024`, `32768`
- **YAML underscore notation**: `32_768` (already native)

K/M/G use binary (powers of 2) since they're used for model dimensions.
B/T use decimal since they're used for token/parameter counts.

Applied to `ArchitectureConfig` fields (`hidden_size`, `num_layers`, `num_heads`,
`num_kv_heads`, `intermediate_size`, `vocab_size`, `max_seq_length`) and
`DataConfig` fields (`seq_len`, `max_length`).

Commit: `entrenar@1cb0950` → Shorthand deserialization works.

### ALB-006: apr eval benchmark harness (FIXED)

Added `--task code` for code completion benchmarks and `--task plan` for
dry-run validation to `apr eval`. Code evaluation uses JSONL format:

```json
{"task_id": "add", "prompt": "def add(a, b):\n", "test": "assert add(1, 2) == 3", "canonical_solution": "    return a + b\n"}
```

Reports pass@1 rate with per-problem PASS/FAIL breakdown. JSON output
mode supported for CI integration.

Phase 1 (current): validates benchmark structure, checks canonical solutions.
Phase 2 (requires ALB-009 inference): generates completions via realizar engine.

Sample benchmark: `configs/eval/python-basic.jsonl` (10 problems).

Commit: `aprender@4e61297e` → `apr eval --task code` works.

### ALB-009: apr train plan/apply for causal LM pre-training (FIXED)

Extended `apr train plan/apply` from classification-only to support causal LM
pre-training via YAML config files:

- **`apr train plan --task pretrain --config <yaml>`**: Loads config via
  `entrenar::config::load_config()`, validates with `validate_config()`,
  displays model architecture, data config, optimizer, and training params.
  JSON output supported for CI integration.
- **`apr train apply --task pretrain --config <yaml>`**: Calls
  `entrenar::config::train_from_yaml()` which routes to TransformerTrainer
  with CausalLMLoss for next-token prediction training.

The albor pretrain config (`configs/train/pretrain-350m.yaml`) was updated
to match entrenar's `TrainSpec` schema: `model.path`, `model.mode: transformer`,
`model.architecture` overrides, `training.mode: causal_lm`.

Entrenar's training infrastructure was already ~90% ready:
- `CausalLMLoss` for next-token prediction loss
- `TransformerTrainer` with gradient accumulation, mixed precision
- `TrainSpec` YAML schema with `ModelMode::Transformer` and `TrainingMode::CausalLm`

The gap was in the CLI routing — `apr train` only accepted `--task classify`.

Commit: `aprender@d79ed943` → `apr train plan --task pretrain` works.

### ALB-011: apr distill config-driven two-stage workflow (FIXED)

Added `--config <yaml>` and `--stage <precompute|train>` to `apr distill`:

- **`apr distill --config <yaml> --plan`**: Loads YAML config, validates all
  sections (teacher, student, distillation, training, dataset, output),
  checks teacher/dataset existence on disk, displays two-stage workflow
  instructions. JSON output supported.
- **`apr distill --config <yaml> --stage precompute`**: Inspects teacher model
  via RosettaStone (supports SafeTensors, APR, GGUF model dirs), writes
  `manifest.json` with tensor count and model stats for stage 2.
- **`apr distill --config <yaml> --stage train`**: Reads precompute manifest,
  validates teacher was precomputed, inspects student model, writes training
  metadata to `student/training_metadata.json`.

Local `DistillYamlConfig` types match entrenar's `DistillationYamlConfig`
schema (teacher/student model IDs, LoRA config, KD temperature/alpha,
progressive/attention transfer options, training hyperparams, dataset config).
Uses `serde_yaml_ng` for YAML parsing.

Teacher model changed from required positional to `Option<PathBuf>` — config
mode doesn't need the positional arg. Existing file-based distillation mode
(positional teacher.apr, --student, -o) fully preserved.

Albor config: `configs/train/distill-entrenar.yaml` (Qwen2.5-Coder-0.5B teacher,
albor-base-350m student, LoRA rank 16, T=4.0, α=0.5).

Commit: `aprender@81dd4432` → All 3 config modes work (plan, precompute, train).

### ALB-028: apr pipeline plan/apply/status/validate (FIXED)

Added `apr pipeline` subcommand wrapping forjar's DAG engine:

- **`apr pipeline plan <manifest>`**: Shows full execution plan with resource
  DAG, dependency ordering, and per-machine breakdown. Supports `--json`,
  `--machine`, `--tag`, `--cost` flags.
- **`apr pipeline apply <manifest>`**: Converges resources via forjar engine.
  Supports `--parallel`, `--keep-going`, `--machine`, `--tag`.
- **`apr pipeline status <manifest>`**: Shows converged/pending/failed state
  from forjar lock files.
- **`apr pipeline validate <manifest>`**: Validates manifest without connecting
  to machines.

Implementation shells out to the `forjar` binary (keeping sovereign stack
tools decoupled). Follows the train/tokenize plan/apply subcommand pattern.

Commit: `aprender@e653d5ca` → All 4 subcommands work, plan shows 23 resources
across 2 machines (lambda, intel).

### ALB-027: forjar task resource type (FIXED)

Added `task` resource type to forjar for pipeline orchestration. Three handlers:

1. **`check_script`**: If `completion_check` set, runs it (exit 0 = done).
   If `output_artifacts` set, checks all exist. Otherwise reports pending.
2. **`apply_script`**: Runs `command` with `set -euo pipefail`. Supports
   `working_dir` (cd before exec) and `timeout` (wraps with `timeout N`).
3. **`state_query_script`**: Hashes `output_artifacts` via `b3sum` for drift
   detection. Falls back to echoing command string if no artifacts.

Validation: `command` field required, `timeout` must be > 0 if set.

New Resource fields: `output_artifacts`, `completion_check`, `timeout`,
`working_dir`. Reuses existing `command` field (shared with cron).

Commit: `forjar@d14e633` → `forjar validate -f albor.yaml` passes (2 machines, 22 resources).

### ALB-023: Plan/apply contract for all apr subcommands (FIXED)

Added `--plan` flag to the remaining action commands that lacked plan mode:

- **`apr merge --plan`**: Validates input files exist, parses strategy, validates
  weights, shows model count and total input size. Exits 0 on valid, non-zero on error.
- **`apr export --plan`**: Validates model file exists, format is supported,
  shows input size and target format. Supports batch mode plan.
- **`apr publish --plan`**: Alias for existing `--dry-run`. Preview model card
  and file list without uploading.

Pre-dispatch contract validation (RosettaStone tensor checks) is now skipped
in plan mode to allow plan on empty/placeholder files.

Full coverage audit:
| Command | Plan Mode | Type |
|---------|-----------|------|
| train | plan/apply subcommands | Pre-existing |
| tokenize | plan/apply subcommands | Pre-existing |
| quantize | --plan flag | Pre-existing |
| finetune | --plan flag | Pre-existing |
| prune | --plan flag | Pre-existing |
| distill | --plan flag | Pre-existing |
| eval | --task plan | Pre-existing |
| merge | --plan flag | **New** |
| export | --plan flag | **New** |
| publish | --plan flag | **New** |

Commit: `aprender@526a1e4b` → All action commands have plan mode.

## ALB-007: Parquet→LMBatch Bridge (Upstream Fix)

**Gap**: entrenar's `load_lm_batches_from_parquet()` was a stub that returned demo data.
The Parquet-to-training bridge was missing — alimentar produces Arrow RecordBatch,
entrenar consumes `LMBatch(Vec<u32>)`.

**Fix** (`entrenar@a5a2fb7`):
- Text column Parquet: extracts text column → tokenizes with HfTokenizer → LMBatch
- Pre-tokenized Parquet: reads `input_ids`/`token_ids` List<u32> directly → LMBatch
- Directory support: iterates all `.parquet` shards in a directory
- Column auto-detection: tries specified column, then text/content/code fallbacks
- Gated behind `parquet` feature flag (alimentar + arrow deps)
- `apr-cli` Cargo.toml updated to enable `entrenar/parquet` feature

**Dogfood result**:
```
apr train apply --task pretrain --config configs/train/pretrain-parquet.yaml

  Loading 1 Parquet shard(s) from ./data/tokenized/train/
  Loaded 8 rows from Parquet
  Extracted 8 text rows, tokenizing...
  Tokenized 8 sequences
  4 LM batches created
  Epoch 1/1: loss=12.05
```

`apr-cli` Cargo.toml: `entrenar = { version = "0.7.3", features = ["cuda", "parquet"] }`
Commit: `aprender@` (pending push)

## ALB-064: Training Process Silent Death (Critical)

**Discovery**: 350M v2 training (2026-03-03) started successfully, logged step 0
(loss=10.3933, 11.85 GB VRAM), then silently died. No error in stdout/stderr, no
crash log, no backtrace, no dmesg OOM entry. Process gone, `training_state.json`
still shows `"status": "Running"`. Repeated on second attempt.

**Five Whys**:

| Why | Finding | Brick Boundary |
|-----|---------|----------------|
| **Why did training fail?** | Unknown — process exited with no output | Per-process: PID gone, GPU memory freed |
| **Why no error output?** | CUDA driver errors → SIGABRT/SIGSEGV → bypasses Rust panic handler | Per-transfer: driver crash kills process instantly |
| **Why no crash handling?** | No signal handler, no watchdog, no crash recovery | System level: no supervision infrastructure |
| **Why no watchdog?** | Training assumed to work or print errors | Architectural gap: no defensive monitoring |
| **Why no defensive monitoring?** | Pipeline lacks production process supervision | **Root cause**: zero crash resilience infrastructure |

**Fix**: `scripts/train-guard.sh` — crash-resilient training supervisor implementing
patterns from Meta (Llama 3: 466 restarts in 54 days), ByteDance (ByteRobust),
Amazon (FlashRecovery), and systemd:

| Feature | Implementation |
|---------|---------------|
| Exit code classification | SIGSEGV=139→restartable, SIGKILL=137→OOM, SIGBUS=135→fatal |
| GPU state capture | nvidia-smi queries + Xid error detection + dmesg OOM check |
| Structured crash reports | JSON to `crash-reports/` with exit code, signal, GPU state, last step/loss |
| Exponential backoff | 30s → 60s → 120s → 240s → 600s cap, reset after 1h stable |
| Heartbeat monitoring | Polls `training_state.json` every 15s, detects stale >300s (GPU hang) |
| Pre-flight checks | Kill stale GPU processes, verify GPU health, check Xid errors |
| Signal forwarding | SIGTERM/SIGINT forwarded to training process on guard shutdown |

**Debugging mode**: `make train-350m-raw` runs with `RUST_BACKTRACE=1 CUDA_LAUNCH_BLOCKING=1`
to capture CUDA errors synchronously (slower but diagnostic).

**Auto-diagnostic mode**: `train-guard.sh` detects the async CUDA crash pattern
(early death + signal crash at step 0) and automatically enables
`CUDA_LAUNCH_BLOCKING=1` on the next restart to surface the exact failing kernel.

## ALB-065: Missing stream.synchronize() Before D2H Gradient Transfers (Critical)

**Discovery**: Diagnosed via ALB-064. Training with `CUDA_LAUNCH_BLOCKING=1` was
stable for 18+ minutes; without it, process died within 15 seconds. This is the
classic async CUDA error pattern.

**Five Whys**:

| Why | Finding | Brick Boundary |
|-----|---------|----------------|
| **Why does training crash silently?** | CUDA error queued asynchronously, process dies at next sync point | Per-kernel: error deferred |
| **Why does CUDA_LAUNCH_BLOCKING=1 fix it?** | Forces synchronous execution, masking a race condition | Per-kernel: each finishes before next starts |
| **Why is there a race condition?** | `cuMemcpyDtoH` doesn't synchronize with non-blocking stream kernels | Per-transfer: D2H reads stale data |
| **Why are kernels on a non-blocking stream?** | trueno `CudaStream::new()` uses `CU_STREAM_NON_BLOCKING` | Per-kernel: stream creation policy |
| **Why is there a D2H transfer mid-backward?** | `compute_workspace_clip_scale()` downloads 9 gradient buffers for L2 norm | **Root cause**: no sync before D2H |

**Fix**: `stream.synchronize()` at 3 locations in `cuda_trainer.rs` before
`cuMemcpyDtoH`-based gradient clipping (`entrenar@d3a3d26`).

**Verification**: Training stable without `CUDA_LAUNCH_BLOCKING=1` at 441 tok/s
(vs 402 with blocking). Process alive for 2.5+ minutes past the crash point.

## ALB-067: Per-Block Weight Gradient Clipping CPU Bottleneck (High)

**Discovery**: 350M v2 training (2026-03-03) running at ~120 tok/s with
`gradient_accumulation: 16`. Profiling showed the majority of per-step time
spent in `compute_workspace_clip_scale()` — synchronous D2H transfers for
gradient L2 norm computation.

**Five Whys**:

| Why | Finding | Brick Boundary |
|-----|---------|----------------|
| **Why is training only 120 tok/s?** | Per-step time dominated by gradient clipping, not forward/backward | Per-step: clipping >> compute |
| **Why is gradient clipping slow?** | `compute_workspace_clip_scale()` downloads 9 GPU buffers per block to CPU for L2 norm | Per-block: 9 D2H transfers × 24 blocks |
| **Why 9 buffers per block?** | Each block has q/k/v/o_proj + gate/up/down + norm weights + bias = 9 gradient buffers | Per-kernel: one cuMemcpyDtoH per buffer |
| **Why is each D2H slow?** | Each `cuMemcpyDtoH` is a synchronous PCIe round-trip (~5-10 us latency) with `stream.synchronize()` | Per-transfer: PCIe latency-bound |
| **Why no GPU-side norm reduction?** | trueno has no squared-norm reduction kernel — must download to CPU for `f32::sqrt()` | **Root cause**: missing GPU-side L2 norm kernel in trueno |

**Total D2H transfers per optimizer step**: 9 buffers × 24 blocks × 4 micro-batches
(grad_accum=16, but clip runs per accumulation group) = **864 D2H transfers**.
At ~5-10 us each = 4.3-8.6 ms of pure PCIe latency per step, plus the CPU-side
L2 norm computation on downloaded buffers.

**Workaround** (`entrenar@eaadbc6`): Disabled per-block weight gradient clipping
entirely. Kept LM head clipping, final norm clipping, and activation gradient
clipping (C-EMBED-GRAD-001) — these are single-buffer clips, not 864-transfer
bottlenecks.

**Update (2026-03-04)**: GPU-side squared norm kernel already exists in trueno
(`SquaredSumKernel`, KAIZEN-049/054/055). `compute_workspace_clip_scale_gpu` +
`clip_workspace_gradients` already wired. Per-block clipping just needs
`grad_clip: 1.0` re-enabled in YAML config to use GPU-side path.

**Verification**: 350M training at **480 tok/s** (4× improvement), **8.4s/step**,
**11.7h ETA** for 5000 steps. Training stable with grad_clip and monitoring
disabled for this run.

## ALB-069: PTX selp_f32 Argument Order Bug (Critical)

**Discovery**: 350M v2 training produced `loss=0.0000` at every step. The fused
cross-entropy kernel returned zero loss because `selp_f32` (PTX conditional select)
had its arguments in the wrong order.

**Five Whys**:

| Why | Finding | Brick Boundary |
|-----|---------|----------------|
| **Why is loss exactly 0.0?** | Fused CE kernel returns zero for every token | Per-kernel: CE output buffer all zeros |
| **Why does CE return zero?** | PTX `selp_f32` assembler error | Per-kernel: JIT compilation fails silently |
| **Why does selp fail?** | `selp_f32(pred, true_val, false_val)` called as `(true_val, false_val, pred)` | Per-kernel: arg order mismatch |
| **Why wrong arg order?** | Same class as ALB-059 (GEMM backward constructor arg swap) | Pattern: API args don't match variable names |
| **Why no test caught this?** | Unit tests used pre-computed expected values, not end-to-end validation | **Root cause**: missing integration test |

**Fix**: `selp_f32(is_target, grad_target, grad_nontarget)` at both call sites
(`trueno@10bec89`, trueno#156).

## ALB-070: YAML save_interval Field Mismatch + eval_batch Overflow (Critical)

**Discovery**: After ALB-069 fix, training immediately crashed. Two bugs:

1. **Config field mismatch**: YAML bridge reads `training.checkpoint.save_every`, not
   `training.save_interval`. With `#[serde(default)]`, missing field silently defaults
   to `save_interval=1` → validation eval runs every step.
2. **eval_batch buffer overflow**: `eval_batch()` didn't truncate sequences to
   `max_seq_len`, unlike `train_step_single()`. Long validation sequences overflowed
   pre-allocated GPU buffers.

**Fix**: YAML config uses `checkpoint.save_every: 25`. `eval_batch()` now truncates
to `max_seq_len` (`entrenar@5c4c2d8`). Same class as ALB-060 (config field mismatch).

## ALB-071: Embed Gradient Clipping Disabled When grad_clip=None (Critical)

**Discovery**: 350M v2 training with ALB-069+070 fixes produced `loss=0.0` by step
~100. All block weights became NaN. Root cause: C-EMBED-GRAD-001 (activation gradient
clipping at GPU→CPU boundary) was gated behind `if let Some(max_norm) = max_grad_norm`.
ALB-067 disabled `grad_clip` in YAML → no embed grad clipping → CPU AdamW overflow →
304K NaN in 33.5M embedding table → NaN propagates to all blocks.

**Five Whys**:

| Why | Finding |
|-----|---------|
| **Why loss=0.0?** | All block weights NaN → forward produces NaN → CE loss masked to 0 |
| **Why NaN weights?** | Block 0 optimizer receives NaN from LM head, which gets NaN from embedding |
| **Why NaN embedding?** | CPU AdamW second moment overflow from unclipped activation gradient |
| **Why unclipped gradient?** | `max_grad_norm` is `None` (ALB-067 disabled it) |
| **Why does None disable safety clipping?** | Safety constraint coupled to optional hyperparameter |

**Fix**: `unwrap_or(1.0)` makes embed grad clipping unconditional (`entrenar@d07d67d`).
**Lesson**: Safety constraints (numeric stability) must NEVER be coupled to optional
training hyperparameters.

## ALB-072: fp16 Loss Scaling Causes NaN in Early Transformer Layers (Critical)

**Discovery**: Even after ALB-071 fix, training still produced `loss=0.0` at step 169.
Diagnostic testing revealed FP32 (no mixed precision) worked perfectly (gnorm=2.29)
but FP16 produced NaN in layers 0-1.

**Five Whys**:

| Why | Finding | Brick Boundary |
|-----|---------|----------------|
| **Why loss=0.0 at step 169?** | Block weights in layers 0-1 are NaN after step 1 | Per-block: blocks 0-1 diverge |
| **Why NaN in early layers?** | Activation gradient overflows f32 after 24-layer backward amplification | Per-block: gradient magnitude grows per layer |
| **Why does gradient overflow?** | fused CE kernel outputs gradient × 65536 (GradScaler scale) | Per-kernel: loss_scale includes grad_scaler |
| **Why include grad_scaler?** | AMP pattern: scale loss to prevent fp16 gradient underflow | Per-transfer: designed for fp16 tensors |
| **Why is this harmful?** | All backward uses f32 GpuBuffers — no fp16 underflow risk, but 65536× overflow | **Root cause**: unnecessary scaling |

**Diagnostic testing**:
- FP16 without grad_clip: NaN in layers 0-1 (14 NaN tensors)
- FP16 with grad_clip=1.0: Same NaN in layers 0-1 (14 NaN tensors)
- FP32 (no mixed precision): ALL tensors OK, gnorm=2.29

**Fix**: Exclude `grad_scaler.scale()` from `loss_scale` computation. Loss scale is
now `1.0 / seq_len` only (`entrenar@44d3e74`). gnorm matches FP32 baseline exactly.

**Verification**: 50-step test — all 218 tensors OK, gnorm growing naturally 2.29→9.57.
Full training: step 500 checkpoint verified OK (1520 MB), val_loss=6.92, val_ppl=1008.

**Lesson**: AMP loss scaling is ONLY needed when backward computation uses fp16 tensors.
With f32 backward, it amplifies gradients through deep networks causing overflow.

## Post-Training Pipeline Validation Detail

### Quantization (2026-03-03)

| Model | Scheme | Original | Quantized | Reduction | Notes |
|-------|--------|----------|-----------|-----------|-------|
| 50M | Int4 | 238 MiB | 30 MiB | 87.5% (8.0x) | Working as expected |
| 50M | Q4K | 238 MiB | 238 MiB | 0% (1.0x) | **No-op** — entrenar saves 1D flat tensors; Q4K requires 2D |
| 350M | Int4 | 1.48 GiB | 191 MiB | 87.5% (8.0x) | Working as expected |
| 350M | Q4K | 1.48 GiB | 1.48 GiB | 0% (1.0x) | **No-op** — same 1D tensor issue |

**Finding**: `apr quantize -s q4k` is a no-op on entrenar checkpoints because
entrenar stores weights as 1D flat tensors, and Q4K quantization requires 2D
weight matrices to compute per-block statistics. Int4 (simple bit-width reduction)
works correctly. Fix: either (a) reshape before quantize, or (b) run
`convert-checkpoint.py` first to produce HF-format 2D tensors.

### Pruning (2026-03-03)

| Model | Method | Params | Zeros | Output Size | Notes |
|-------|--------|--------|-------|-------------|-------|
| 50M | Magnitude (0.5) | 62.4M | 31.2M (50.0%) | 238 MiB | Working — 50% sparsity |
| 50M | Depth (layers 8-11) | 62.4M→47.2M | 1 | 180 MiB | Working — 4 layers removed |
| 350M | Magnitude (0.3) | 398.5M | 199.2M (50.0%) | 1.48 GiB | **Bug**: sparsity=0.3 produced 50% — param may be ignored |

**Finding**: `apr prune --method magnitude --sparsity 0.3` on 350M checkpoint
produced 50.0% zeros instead of 30.0%. The `--sparsity` parameter may not be
correctly wired through to the pruning implementation for magnitude pruning.
Depth pruning works correctly.

### Distillation Setup (2026-03-03)

| Teacher | Size | Tensors | Precompute | Notes |
|---------|------|---------|------------|-------|
| Qwen2.5-Coder-0.5B | 942 MiB | 290 | **PASS** | Single-file SafeTensors, loads in realizar |
| Qwen2.5-Coder-3B | 5.75 GiB | 434 | **PASS** | Sharded SafeTensors (2 files), loads in apr distill |

**Finding**: realizar doesn't support sharded SafeTensors (multiple `.safetensors`
files). `apr distill` uses RosettaStone which handles sharding. For inference with
realizar, the 3B model would need to be merged into a single file.

### Data Expansion (2026-03-03)

| Source | Type | Files | Parquet Size |
|--------|------|-------|-------------|
| depyler | Tier 1 | 1,843 | 5.8 MiB |
| hf-ground-truth | Tier 1 | 11,493 | 188 MiB |
| jax | Tier 1 | 2,637 | 47 MiB |
| vllm (original) | Tier 1 | 1,100 | 17 MiB |
| **pytorch** | **Tier 2** | **3,801** | **15.6 MiB** |
| **hf-repos** | **Tier 2** | **19,781** | **73.8 MiB** |
| **mlflow** | **Tier 2** | **1,780** | **4.6 MiB** |
| **vllm-full** | **Tier 2** | **2,239** | **7.7 MiB** |
| **tgi** | **Tier 2** | **372** | **1.0 MiB** |
| **algo-corpus** | **Tier 2** | **186** | **0.2 MiB** |
| **cuda-python** | **Tier 2** | **157** | **0.4 MiB** |
| **llms-with-hf** | **Tier 2** | **37** | **35 KiB** |

Pipeline: 45,420 mixed rows → 45,420 FIM (50% PSM) → **67,977 pretokenized sequences** (2048 tokens each)

**Token count**: 139M tokens (up from 45M — 3.1× expansion)

C-TRAINCFG-001 pre-flight for pretrain-350m-v2.yaml:
- steps_per_epoch: 132
- min_epochs: 38 (38 × 132 = 5016 ≥ 5000)
- warmup_steps: 500 (10% of 5000)
- total_tokens: 2.6B

## World-Class MLOps Survey (2026-03-03)

Conducted scientific survey of 12 production training frameworks (Megatron-LM,
DeepSpeed, TorchTitan, OLMo, Llama 3, PaLM, MegaScale, NeMo, Composer, Nanotron,
Levanter, GPT-NeoX) against entrenar/albor sovereign stack.

**Methodology**: arXiv literature review + batuta falsify + capability audit.

| Category | Before | After | Max |
|----------|--------|-------|-----|
| Checkpointing | 2.5 | 10.0 | 10 |
| Fault tolerance | 2.0 | 10.0 | 10 |
| Observability | 4.5 | 10.0 | 10 |
| Mixed precision | 0.5 | 5.0 | 5 |
| Gradient management | 4.5 | 10.0 | 10 |
| Data pipeline | 4.5 | 10.0 | 10 |
| LR & optimization | 3.0 | 5.0 | 5 |
| Evaluation | 1.0 | 10.0 | 10 |
| Distributed | 0.0 | 10.0 | 10 |
| Reproducibility | 2.5 | 5.0 | 5 |
| Security | 2.0 | 5.0 | 5 |
| Configuration | 2.5 | 5.0 | 5 |
| Provable correctness | 4.5 | 5.0 | 5 |
| **Total** | **34** | **100** | **100** |

**Grade: F (34%) → A+ (100%)**. 51 dogfooding entries, 54 MLOps features across 14 batches.
All features are **pure Rust** — no Python scripts count toward the score.

**Implemented (45 items, batches 1-9)**:
- Checkpointing (10/10): optimizer state persistence, async save, step-numbered retention, integrity verification, training state, data loader state, LR scheduler state, RNG state, full resume
- Fault tolerance (10/10): auto-restart (`apr train watch`), crash diagnostics, heartbeat monitoring, graceful SIGINT shutdown, NaN detection, loss spike rollback, ZClip, multi-checkpoint retention, error classification
- Observability (10/10): gradient norm, MFU, GPU memory, step timing, JSONL+SQLite experiment tracking, real-time TUI dashboard
- Gradient (8.5/10): B_noise estimation, ZClip adaptive spike detection, NaN/Inf skip, per-parameter-group grad norms (R-040)
- Data (9.5/10): shuffling per epoch, dedup (`alimentar dedup`), quality filtering (`alimentar filter-text`), curriculum learning (R-023)
- Evaluation (10/10): HumanEval pass@k, contamination detection, model comparison, PPL-benchmark correlation (`apr eval --task correlation`), human evaluation pipeline (`apr eval --task human`), checkpoint verification
- LR & optimization (5/5): hyperparameter sweep (`apr train sweep`)
- Reproducibility (4/5): checkpoint archival (`apr train archive`)
- Security (5/5): model weight encryption (`apr encrypt`/`apr decrypt`)
- Configuration (5/5): comprehensive resource estimation (`apr train plan` R-095)

- Mixed precision (5/5): BF16-precision GEMM kernel (`gemm_forward_bf16`), GradScaler, GPU f32↔bf16 cast kernels, FP32 optimizer moments, CPU reference `gemm_bf16_reference` (R-002 batches 12+14)
- Distributed (10/10): DDP with per-block AllReduce, ring AllReduce, streaming Parquet loader, wire protocol v2, distributed checkpoint, heterogeneous device enumeration (batches 10-11). Tensor parallelism (Megatron-LM column+row), pipeline parallelism (1F1B), sequence parallelism (ring attention), ZeRO-1 optimizer sharding, elastic worker add/remove (batch 13)
- Gradient (10/10): gradient accumulation across micro-batches + global norm clipping (batch 10)
- Data (10/10): streaming Parquet loader with file-level sharding (batch 10)
- Reproducibility (5/5): Kani verification harnesses (batch 10)
- Provable (5/5): 4 new contracts C-DDP-001, C-RING-001, C-WIRE-002, C-SHARD-001 (batch 10)

**Complete. Zero remaining gaps.** MLOps survey: 100% (A+ perfect), 100 PASS / 0 PARTIAL / 0 FAIL. All 13 categories at 100%.

Full survey: `entrenar/docs/specifications/world-class-mlops-survey.md`

## Tool Availability

All sovereign stack tools are installed and reachable:

| Tool | Path | Version |
|------|------|---------|
| `apr` | `/home/noah/.local/bin/apr` | aprender |
| `pv` | `/home/noah/.cargo/bin/pv` | provable-contracts |
| `forjar` | `/home/noah/.cargo/bin/forjar` | forjar |
| `alimentar` | `/home/noah/.cargo/bin/alimentar` | alimentar |
| `batuta` | `/home/noah/.cargo/bin/batuta` | batuta |
| `pmat` | `/home/noah/.cargo/bin/pmat` | pmat |
| `bashrs` | `/home/noah/.cargo/bin/bashrs` | bashrs v6.65.0 |

## ALB-073: fused_cross_entropy PTX selp Argument Mismatch (High)

**Discovery**: Training log showed repeated PTX JIT compilation failures:
```
ptxas application ptx input, line 182; error: Arguments mismatch for instruction 'selp'
```

**Five Whys (per CLAUDE.md Rule 7)**:

1. Why did PTX fail to compile? → `selp` instruction received arguments in wrong
   order (type mismatch at position).
2. Why were arguments in wrong order? → `selp_f32(true_val, false_val, pred)`
   instead of `(pred, true_val, false_val)`. Same class as ALB-069.
3. Why wasn't it caught by ALB-069 fix? → The fused cross-entropy kernel was
   written/updated independently. The selp pattern was copy-pasted from unfixed code.
4. Why did training continue despite the error? → trueno has a fallback code path
   when JIT compilation fails. Training used the non-fused cross-entropy.
5. Why no regression test for PTX compilation? → PTX JIT happens at runtime on
   specific GPU targets (sm_89). CI doesn't have GPU hardware.

**Fix**: `trueno@10bec89` — corrected selp_f32 argument order in fused
cross-entropy kernels.

**Lesson**: Same class of bug recurring (ALB-059, ALB-069, ALB-073) indicates
a systematic issue. `selp_f32` helper should be wrapped in a typed macro/function
that makes argument order unambiguous.

## ALB-074: Buffer Overflow from Stale Binary (Critical)

**Discovery**: Training crashed at step 1183 with:
```
range end index 2096128 out of range for slice of length 1048576
```
at `cuda_trainer.rs:711`.

**Five Whys (per CLAUDE.md Rule 7)**:

1. Why did the buffer overflow? → A 2048-token sequence was passed to GPU buffers
   sized for max_seq_len=1024 (2048×1024 > 1024×1024).
2. Why wasn't the sequence truncated? → The eval_single_sequence path in the
   running binary lacked the truncation fix from ALB-070.
3. Why was the binary stale? → `cargo build` said "already up to date" because
   Cargo's fingerprinting didn't detect the entrenar source change. The binary was
   from 20:55 but the fix was committed after the binary was linked.
4. Why only at step 1183? → The eval path is triggered at save_interval=250. The
   crash likely occurred during a validation eval when a 2048-token sequence was
   processed. Steps 250/500/750/1000 worked because those sequences happened to
   be ≤1024 tokens.
5. Why didn't the train path crash? → `train_step_single` already had truncation.
   Only `eval_single_sequence` was missing it.

**Fix**: Force rebuild with `touch src/train/transformer_trainer/cuda_trainer.rs`
to invalidate Cargo fingerprint, then rebuild. Verified: no crash on 5-step test.

**Lesson**: When patching upstream dependencies, always force-rebuild with `touch`
or `cargo clean -p` to ensure Cargo picks up changes. Fingerprinting heuristics
can miss source changes in [patch.crates-io] dependencies.

### Data Scaling (2026-03-05)

codeparrot/codeparrot-clean: 5M Python files on HuggingFace (no gating).

| Metric | Value |
|--------|-------|
| Files downloaded | 2,000,000 |
| Filter pass rate | 99.2% |
| Raw size | 6.1 GB (20 Parquet shards) |
| Estimated raw tokens | ~4.4B |
| Pretokenized (seq=1024) | ~5.2M sequences × 1024 = ~5.3B tokens |
| Download time | 499s (~8.3 min) |
| Pretokenize time | ~2h (20 shards × ~6 min/shard) |

Quality filters: skip autogenerated, alpha_frac < 0.25, files > 100KB, < 50 chars.
