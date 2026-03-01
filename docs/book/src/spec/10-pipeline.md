# 10. Pipeline Orchestration (`apr pipeline` + forjar DAG)

### 10.1 Architecture: One Manifest, One DAG

The entire albor pipeline — from bare metal to published model — lives in a
single YAML manifest: `configs/pipeline/albor.yaml`. Forjar's DAG engine
resolves dependencies, tracks state, and dispatches steps across machines.
`apr pipeline` wraps forjar, so the user never calls forjar directly.

```
apr pipeline plan configs/pipeline/albor.yaml    # Show full DAG, estimate everything
apr pipeline apply configs/pipeline/albor.yaml   # Execute (resumable)
apr pipeline status                              # Show what's converged/pending/failed
apr pipeline drift                               # Detect unauthorized state changes
```

**How it works**:

```
                     configs/pipeline/albor.yaml
                              │
                    apr pipeline plan/apply
                              │
                     forjar DAG engine
                    (Kahn's toposort)
                              │
         ┌────────────┬───────┴───────┬────────────┐
         │            │               │            │
    infra resources   │          task resources    │
    (package, gpu,    │          (run apr cmds,    │
     file, mount,     │           track output)    │
     model)           │               │            │
         │            │               │            │
    forjar native     │     apr train apply        │
    convergence       │     apr distill apply      │
                      │     apr eval apply         │
                      │     apr publish apply      │
                      │               │            │
                 state/lambda/     state/intel/
                 state.lock.yaml   state.lock.yaml
```

**Key properties**:
- **Resumable**: BLAKE3 hashes per resource. Re-run skips converged steps.
- **Multi-machine**: Infra + tasks dispatch to lambda or intel via SSH.
- **Plan/apply**: `apr pipeline plan` shows the full DAG with estimates before
  committing any resources. Exit 0 if valid, exit 1 with diagnostics.
- **Idempotent**: Same manifest, same state → zero changes (all NoOp).
- **bashrs linted**: All shell fragments in task `command:` fields are validated
  by bashrs (Rash v6.65) at plan time. No unvalidated shell reaches execution.
  bashrs is KING of linting — `bashrs make lint` validates Makefiles, `bashrs lint`
  validates shell scripts, `bashrs classify` classifies safety.

**Dual orchestration**:
- **forjar manifest** (`configs/pipeline/albor.yaml`): Infrastructure provisioning
  (GPU drivers, packages, directories, mounts, teacher model download). Blocked on
  `type: task` (ALB-027) for ML steps.
- **batuta playbook** (`configs/pipeline/albor-playbook.yaml`): ML pipeline orchestration
  (data prep, train, distill, finetune, merge, prune, quantize, eval, publish).
  19-stage deterministic DAG with BLAKE3 caching. Validates successfully.

### 10.2 Pipeline Manifest: `configs/pipeline/albor.yaml`

```yaml
version: "1.0"
name: albor-training-pipeline
description: "Sovereign Python code completion model — full pipeline"

machines:
  lambda:
    hostname: lambda
    addr: 127.0.0.1
    user: noah
    arch: x86_64
    roles: [gpu-train, student]

  intel:
    hostname: intel
    addr: intel
    user: noah
    ssh_key: ~/.ssh/id_ed25519
    arch: x86_64
    roles: [teacher-inference, data-pipeline, eval, checkpoint-backup]

resources:
  # ═══════════════════════════════════════════════════════════
  # INFRASTRUCTURE (forjar native resources)
  # ═══════════════════════════════════════════════════════════

  cuda-driver:
    type: gpu
    machine: lambda
    gpu_backend: nvidia
    driver_version: "550"
    cuda_version: "12.4"
    persistence_mode: true
    compute_mode: exclusive_process

  vulkan-driver:
    type: package
    machine: intel
    provider: apt
    state: present
    packages: [mesa-vulkan-drivers, vulkan-tools, libvulkan-dev]

  data-dir:
    type: file
    machine: [lambda, intel]
    path: /data/albor
    state: directory
    mode: "0755"

  teacher-model:
    type: model
    machine: intel
    name: Qwen/Qwen3-Coder-Next
    state: present
    cache_dir: /data/albor/models/teacher
    depends_on: [data-dir]

  checkpoint-share:
    type: mount
    machine: intel
    source: "lambda:/data/albor/checkpoints"
    path: /data/albor/checkpoints
    fstype: nfs
    options: "rw,sync,no_subtree_check"
    depends_on: [data-dir]

  logit-share:
    type: mount
    machine: lambda
    source: "intel:/data/albor/teacher-logits"
    path: /data/albor/teacher-logits
    fstype: nfs
    options: "ro,sync"
    depends_on: [data-dir]

  # ═══════════════════════════════════════════════════════════
  # DATA PIPELINE (task resources — call apr subcommands)
  # ═══════════════════════════════════════════════════════════

  ingest-local:
    type: task
    machine: lambda
    command: >
      alimentar import local ../depyler/examples/ ../depyler/tdd-book/tests/
        --lang python --output ./data/local/depyler.parquet &&
      alimentar import local ../hf-ground-truth-corpus/
        --lang python --output ./data/local/hf-gtc.parquet &&
      alimentar import local ../jax-ground-truth-corpus/
        --lang python --output ./data/local/jax-gtc.parquet &&
      alimentar import local ../vllm-ground-truth-corpus/
        --lang python --output ./data/local/vllm-gtc.parquet
    output_artifacts: ["./data/local/*.parquet"]
    depends_on: [data-dir]

  ingest-external:
    type: task
    machine: lambda
    command: >
      alimentar import hf bigcode/starcoderdata --lang python
        --output ./data/starcoder-python/ &&
      alimentar import hf HuggingFaceFW/fineweb-edu
        --output ./data/fineweb-edu/
    output_artifacts: ["./data/starcoder-python/", "./data/fineweb-edu/"]
    depends_on: [data-dir]

  data-mix:
    type: task
    machine: lambda
    command: >
      alimentar quality check ./data/ --profile ml-training &&
      alimentar mix
        --input ./data/local/depyler.parquet --weight 0.025 --upsample 10
        --input ./data/local/hf-gtc.parquet --weight 0.025 --upsample 10
        --input ./data/local/jax-gtc.parquet --weight 0.025 --upsample 10
        --input ./data/local/vllm-gtc.parquet --weight 0.025 --upsample 10
        --input ./data/starcoder-python/ --weight 0.40
        --input ./data/fineweb-edu/ --weight 0.20
        --input ./data/processed/python-docs.parquet --weight 0.10
        --output ./data/mixed/ --seed 42 --shuffle
    output_artifacts: ["./data/mixed/"]
    depends_on: [ingest-local, ingest-external]

  tokenize:
    type: task
    machine: lambda
    command: >
      apr tokenize plan --input ./data/mixed/*.parquet --vocab-size 32768
        --output ./models/albor-tokenizer/ &&
      apr tokenize apply --input ./data/mixed/*.parquet --vocab-size 32768
        --output ./models/albor-tokenizer/ --seed 42 &&
      apr tokenize apply --tokenizer ./models/albor-tokenizer/
        --input ./data/mixed/*.parquet --output ./data/tokenized/
        --max-seq-len 2048
    output_artifacts: ["./models/albor-tokenizer/", "./data/tokenized/"]
    depends_on: [data-mix]

  # ═══════════════════════════════════════════════════════════
  # TRAINING (task resources — long-running, checkpoint-aware)
  # ═══════════════════════════════════════════════════════════

  train-50m:
    type: task
    machine: lambda
    command: >
      apr train plan configs/train/pretrain-50m.yaml &&
      apr train apply configs/train/pretrain-50m.yaml --seed 42
    output_artifacts: ["./checkpoints/albor-base-50m/"]
    completion_check: "test -f ./checkpoints/albor-base-50m/checkpoint-best.safetensors"
    depends_on: [tokenize, cuda-driver]

  train-350m:
    type: task
    machine: lambda
    command: >
      apr train plan configs/train/pretrain-350m.yaml &&
      apr train apply configs/train/pretrain-350m.yaml --seed 42
    output_artifacts: ["./checkpoints/albor-base-350m/"]
    completion_check: "test -f ./checkpoints/albor-base-350m/checkpoint-best.safetensors"
    depends_on: [train-50m]

  # ═══════════════════════════════════════════════════════════
  # DISTILLATION (cross-machine: intel produces logits, lambda trains)
  # ═══════════════════════════════════════════════════════════

  distill-logits:
    type: task
    machine: intel
    command: >
      apr distill plan configs/train/distill.yaml &&
      apr distill apply configs/train/distill.yaml --stage precompute
    output_artifacts: ["./data/teacher-logits/"]
    completion_check: "test -d ./data/teacher-logits/ && ls ./data/teacher-logits/*.parquet"
    depends_on: [train-350m, teacher-model, logit-share]

  distill:
    type: task
    machine: lambda
    command: >
      apr distill apply configs/train/distill.yaml --stage train --seed 42
    output_artifacts: ["./checkpoints/albor-distill/"]
    completion_check: "test -f ./checkpoints/albor-distill/checkpoint-best.safetensors"
    depends_on: [distill-logits]

  # ═══════════════════════════════════════════════════════════
  # POST-TRAINING LADDER (sequential, each depends on previous)
  # ═══════════════════════════════════════════════════════════

  finetune:
    type: task
    machine: lambda
    command: >
      apr finetune plan configs/train/finetune-lora.yaml &&
      apr finetune apply configs/train/finetune-lora.yaml
    output_artifacts: ["./checkpoints/albor-instruct/"]
    depends_on: [distill]

  merge:
    type: task
    machine: lambda
    command: >
      apr merge plan --models albor-distill-350m,albor-instruct-350m
        --method slerp --weight 0.6 --output ./checkpoints/albor-merged/ &&
      apr merge apply --models albor-distill-350m,albor-instruct-350m
        --method slerp --weight 0.6 --output ./checkpoints/albor-merged/
    output_artifacts: ["./checkpoints/albor-merged/"]
    depends_on: [finetune]

  prune:
    type: task
    machine: lambda
    command: >
      apr prune plan --model ./checkpoints/albor-merged-350m/
        --method wanda --sparsity 0.5 --output ./checkpoints/albor-pruned/ &&
      apr prune apply --model ./checkpoints/albor-merged-350m/
        --method wanda --sparsity 0.5 --output ./checkpoints/albor-pruned/
    output_artifacts: ["./checkpoints/albor-pruned/"]
    depends_on: [merge]

  quantize:
    type: task
    machine: lambda
    command: >
      apr quantize plan --model ./checkpoints/albor-merged-350m/
        --method q4_k --output ./checkpoints/albor-q4/ &&
      apr quantize apply --model ./checkpoints/albor-merged-350m/
        --method q4_k --output ./checkpoints/albor-q4/
    output_artifacts: ["./checkpoints/albor-q4/"]
    depends_on: [merge]

  # ═══════════════════════════════════════════════════════════
  # EVALUATION (can run on intel concurrently with training)
  # ═══════════════════════════════════════════════════════════

  eval-code:
    type: task
    machine: lambda
    command: >
      apr eval plan --model ./checkpoints/albor-merged-350m/
        --tasks humaneval,humaneval_fim,mbpp,ds1000 &&
      apr eval apply --model ./checkpoints/albor-merged-350m/
        --tasks humaneval,humaneval_fim,mbpp,ds1000
        --output ./eval/python-code-results.json --seed 42
    output_artifacts: ["./eval/python-code-results.json"]
    depends_on: [merge]

  eval-general:
    type: task
    machine: intel
    command: >
      apr eval apply --model ./checkpoints/albor-merged-350m/
        --tasks arc_easy,hellaswag,piqa,lambada
        --output ./eval/general-results.json --seed 42
    output_artifacts: ["./eval/general-results.json"]
    depends_on: [merge, checkpoint-share]

  # ═══════════════════════════════════════════════════════════
  # RELEASE
  # ═══════════════════════════════════════════════════════════

  export:
    type: task
    machine: lambda
    command: >
      apr export plan --model ./checkpoints/albor-q4/ --format gguf &&
      apr export apply --model ./checkpoints/albor-q4/ --format gguf
        --output ./release/albor-350m-q4_k.gguf &&
      apr export apply --model ./checkpoints/albor-merged-350m/
        --format safetensors
        --output ./release/albor-350m.safetensors
    output_artifacts: ["./release/"]
    depends_on: [quantize, eval-code]

  publish:
    type: task
    machine: lambda
    command: >
      apr publish plan --model ./release/ --hub paiml/albor-350m &&
      apr publish apply --model ./release/ --hub paiml/albor-350m
    depends_on: [export, eval-general]

policy:
  failure: stop_on_first
  parallel_machines: true
  retry: 2
  bashrs_lint: true            # Validate all task command: fields via bashrs
```

### 10.3 Pipeline Workflow

```bash
# Show full DAG with time/resource estimates (no side effects)
apr pipeline plan configs/pipeline/albor.yaml

# Execute everything (resumable — skips converged steps)
apr pipeline apply configs/pipeline/albor.yaml

# Check what's done, what's pending, what failed
apr pipeline status

# Detect unauthorized changes to converged resources
apr pipeline drift

# Re-run only failed steps (everything else is NoOp)
apr pipeline apply configs/pipeline/albor.yaml

# Force re-run a specific resource and its dependents
apr pipeline apply configs/pipeline/albor.yaml --target train-350m --force
```

### 10.4 The `task` Resource Type (ALB-027)

The `task` resource is what makes forjar a pipeline orchestrator, not just an
infrastructure tool. It runs an arbitrary command, tracks completion, and
hashes output artifacts for idempotency.

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Shell command to execute (bashrs-validated at plan time) |
| `output_artifacts` | list[string] | Paths to hash for idempotency (glob-supported) |
| `completion_check` | string | Optional shell expression to verify completion (e.g., checkpoint exists) |
| `timeout` | duration | Max wall time before Andon stop (default: none) |
| `resume_command` | string | Optional command for resuming interrupted long-running tasks |

**Idempotency for ML tasks**: A `task` resource is considered converged when:
1. The `command` exited 0 on a previous run, AND
2. The BLAKE3 hash of `output_artifacts` matches the lock file, AND
3. The `completion_check` (if set) passes

If any of these fail, the task is re-run. For training jobs that crashed
mid-run, the `command` itself includes `--resume` logic (e.g., `apr train
apply` auto-detects and resumes from the latest checkpoint).

### 10.5 Why Not Makefile / Shell Scripts

| Approach | DAG | State | Resume | Multi-Machine | Lint |
|----------|-----|-------|--------|---------------|------|
| **`apr pipeline` (forjar)** | Kahn's toposort | BLAKE3 lock files | Automatic (skip converged) | Native SSH dispatch | bashrs at plan time |
| Makefile | File timestamps only | None | Manual | None (SSH in recipes) | None |
| Shell scripts | Sequential only | None | Manual | Manual SSH | ShellCheck (external) |

The Makefile and shell scripts are eliminated. One manifest. One DAG. One tool.

