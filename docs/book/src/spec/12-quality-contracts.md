# 12. Provable Quality & Design by Contract

Every computational kernel used in Albor must have a provable-contracts YAML
specification with Popperian falsification tests, property-based probar tests,
and Kani bounded model checking harnesses. This is not optional — it is a
first-class deliverable alongside the model.

### 12.1 Verification Ladder

Four levels of assurance, from cheapest to most rigorous:

```
Level 4: Kani bounded model check    ─── PROOF (exhaustive for inputs ≤ N)
Level 3: probar property tests       ─── HIGH CONFIDENCE (10,000+ random inputs)
Level 2: Falsification tests         ─── TARGETED (specific edge cases)
Level 1: Type system                 ─── BY CONSTRUCTION (Rust compiler)
Level 0: Code review                 ─── HUMAN (necessary but insufficient)
```

**Requirement**: Every kernel reaches at least Level 3. Critical kernels
(softmax, attention, cross-entropy, KD loss) reach Level 4.

### 12.2 Contract Registry for Albor

Albor requires contracts for every kernel in the training + post-training pipeline.
Many already exist in provable-contracts; new ones must be written.

#### Existing Contracts (bind to aprender implementations)

| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| `softmax-kernel-v1.yaml` | softmax | 6 (normalization, positivity, monotonicity, SIMD parity, translation invariance, bound) | Exists, 289 bindings |
| `rmsnorm-kernel-v1.yaml` | RMSNorm | 5 (finiteness, scale invariance, SIMD parity, idempotency) | Exists |
| `attention-kernel-v1.yaml` | scaled dot-product attention | Multiple (causal mask, score bounds, gradient flow) | Exists |
| `rope-kernel-v1.yaml` | Rotary Position Embedding | Multiple (rotation invariant, frequency spectrum) | Exists |
| `gelu-kernel-v1.yaml` | GELU activation | Bound, monotonicity, SIMD parity | Exists |
| `matmul-kernel-v1.yaml` | matrix multiplication | Associativity, SIMD parity, bound | Exists |
| `cross-entropy-kernel-v1.yaml` | cross-entropy loss | Non-negativity, gradient correctness | Exists |
| `adamw-kernel-v1.yaml` | AdamW optimizer | Bias correction, weight decay decoupling | Exists |
| `gqa-kernel-v1.yaml` | Grouped Query Attention | Equivalence to MHA when groups=heads | Exists |
| `swiglu-kernel-v1.yaml` | SwiGLU FFN | Gating invariants | Exists |

#### New Contracts Required for Albor (ALB-013 through ALB-017)

| Contract (NEW) | Key Equations | Key Obligations | Priority |
|----------------|---------------|-----------------|----------|
| `knowledge-distillation-kernel-v1.yaml` | KD_loss = α·KL(σ(z_t/T) ∥ σ(z_s/T))·T² + (1-α)·CE(y, z_s) | KL non-negativity, temperature scaling invariant, gradient correctness, α interpolation bound | Critical |
| `bpe-tokenizer-kernel-v1.yaml` | BPE merge rules, byte-pair encoding | Roundtrip invariant: decode(encode(x)) = x, vocab coverage, merge ordering | High |
| `model-merging-kernel-v1.yaml` | SLERP: interp(θ, w₁, w₂) on unit sphere; TIES: trim + elect + disjoint merge | SLERP interpolation bound (‖result‖ ≈ 1), TIES sparsity guarantee | Medium |
| `pruning-kernel-v1.yaml` | WANDA: score = |w| · ‖x‖₂; magnitude: score = |w| | Sparsity invariant (exactly k% weights zeroed), score ordering preserved | Medium |
| `gradient-accumulation-kernel-v1.yaml` | G_accum = (1/N)·Σ g_i ≈ g_full | Numerical equivalence within tolerance, loss scaling correctness | High |

### 12.3 Contract Workflow for Each Kernel

```bash
# 1. Write or validate YAML contract
pv validate contracts/knowledge-distillation-kernel-v1.yaml

# 2. Generate trait stubs + failing tests
pv scaffold contracts/knowledge-distillation-kernel-v1.yaml

# 3. Generate property-based tests (wired to actual aprender code)
pv probar contracts/knowledge-distillation-kernel-v1.yaml \
  --binding contracts/aprender/binding.yaml

# 4. Generate Kani bounded model checking harnesses
pv kani contracts/knowledge-distillation-kernel-v1.yaml

# 5. Run falsification sweep
pv audit contracts/knowledge-distillation-kernel-v1.yaml \
  --binding contracts/aprender/binding.yaml

# 6. Verify full contract status
pv status contracts/knowledge-distillation-kernel-v1.yaml
```

### 12.4 Falsification Tests: Albor-Specific

Every claim in this specification must be falsifiable. Below are the concrete
falsification tests for Albor's key properties.

#### Training Correctness

```yaml
# FALSIFY-ALBOR-001: Loss decreases monotonically (smoothed)
- id: FALSIFY-ALBOR-001
  rule: "Training convergence"
  prediction: "EMA(loss, window=100) is monotonically decreasing after warmup"
  test: "Load training log, compute EMA, assert no sustained increase >5% over 500 steps"
  if_fails: "Learning rate too high, data corruption, or gradient computation bug"

# FALSIFY-ALBOR-002: Gradient norms are bounded
- id: FALSIFY-ALBOR-002
  rule: "Training stability"
  prediction: "Global gradient norm < 10.0 after clipping for all steps"
  test: "Parse training log, assert max gradient norm across all steps"
  if_fails: "Gradient clipping not applied, loss spike, or NaN propagation"

# FALSIFY-ALBOR-003: Checkpoint determinism
- id: FALSIFY-ALBOR-003
  rule: "Reproducibility"
  prediction: "Two runs with seed=42 produce identical checkpoints at step 1000"
  test: "Train twice, BLAKE3 hash both checkpoints, assert equality"
  if_fails: "Non-deterministic operation (async GPU, HashMap ordering, etc.)"
```

#### Distillation Correctness

```yaml
# FALSIFY-ALBOR-004: KL divergence is non-negative
- id: FALSIFY-ALBOR-004
  rule: "KD loss validity"
  prediction: "KL(teacher || student) >= 0 for all batches"
  test: "proptest with 10000 random logit pairs, assert KL >= -1e-7"
  if_fails: "Log-domain computation error or softmax numerical instability"

# FALSIFY-ALBOR-005: Distillation improves over base
- id: FALSIFY-ALBOR-005
  rule: "Distillation value"
  prediction: "albor-distill avg benchmark > albor-base avg benchmark"
  test: "Run full eval suite on both, paired t-test with p < 0.05"
  if_fails: "Teacher logits corrupted, temperature too high/low, or alpha miscalibrated"

# FALSIFY-ALBOR-006: Teacher logit integrity
- id: FALSIFY-ALBOR-006
  rule: "Data pipeline integrity"
  prediction: "Pre-computed teacher logits match live teacher inference within 1e-4"
  test: "Sample 100 batches, run live teacher inference, compare against stored logits"
  if_fails: "Serialization precision loss, wrong batch ordering, or teacher model mismatch"
```

#### Post-Training Invariants

```yaml
# FALSIFY-ALBOR-007: Merge interpolation bound
- id: FALSIFY-ALBOR-007
  rule: "SLERP correctness"
  prediction: "‖SLERP(w1, w2, t)‖ ≈ ‖w1‖ for t ∈ [0,1] (unit sphere)"
  test: "proptest with 10000 random weight pairs and t values"
  if_fails: "SLERP implementation uses LERP instead, or normalization missing"

# FALSIFY-ALBOR-008: Pruning sparsity guarantee
- id: FALSIFY-ALBOR-008
  rule: "WANDA correctness"
  prediction: "Exactly 50% of weights are zero after prune --sparsity 0.5"
  test: "Count zero weights, assert within ±0.1% of target sparsity"
  if_fails: "Pruning threshold computation error or layer exclusion bug"

# FALSIFY-ALBOR-009: Quantization round-trip
- id: FALSIFY-ALBOR-009
  rule: "Q4 fidelity"
  prediction: "Perplexity(Q4 model) < 1.05 × Perplexity(fp16 model)"
  test: "Evaluate both on held-out set, assert ratio < 1.05"
  if_fails: "Quantization calibration data insufficient or block size wrong"
```

### 12.5 Verification DAG (Albor End-to-End)

Like the Qwen 3.5 verification DAG in provable-contracts, Albor composes
sub-contracts into a full model verification:

```
softmax ← attention ← gqa
                        ↑
rmsnorm ──────────────── albor-forward ← training-loop
                        ↑                      ↑
gelu ← swiglu ──────────┘                     │
                                               │
rope ──────────────────── albor-forward        │
                                               │
matmul ← gqa                                   │
                                               │
cross-entropy ─────────── training-loss ────────┘
                              ↑
adamw ─────────── optimizer-step ──────── training-loop
                                               │
gradient-accumulation ─────────────────────────┘
                                               │
knowledge-distillation ── distill-loss ── distill-loop
                              ↑
bpe-tokenizer ─── data-pipeline ─── training-loop

model-merging ─── post-training ─── albor-merged
pruning ────────── post-training ─── albor-pruned
```

Each node in this DAG is a contract. `pv graph contracts/ --format mermaid`
renders the full dependency graph. A change to any sub-contract triggers
re-verification of all dependents.
