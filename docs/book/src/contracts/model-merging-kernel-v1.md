# model-merging-kernel-v1

**Version:** 1.0.0

Model merging kernel — SLERP, TIES, and DARE weight interpolation

## References

- Shoemake (1985) Animating Rotation with Quaternion Curves (SLERP)
- Yadav et al. (2023) TIES-Merging: Resolving Interference When Merging Models
- Yu et al. (2023) Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)

## Equations

### dare

$$
tau_tilde_i = m_i * tau_i / (1-p) where m_i ~ Bernoulli(1-p)
$$

**Domain:** $tau_i (task vector), p in [0, 1) (drop probability)$

**Codomain:** $tau_tilde_i: rescaled sparse task vector$

**Invariants:**

- $E[tau_tilde] = tau (unbiased estimator)$
- $Sparsity approximately p$

### slerp

$$
SLERP(w1, w2, t) = sin((1-t)*Omega)/sin(Omega) * w1 + sin(t*Omega)/sin(Omega) * w2
$$

**Domain:** $w1, w2 in R^n (weight vectors), t in [0, 1], cos(Omega) = w1.w2 / (||w1|| * ||w2||)$

**Codomain:** $result in R^n with ||result|| approximately ||w1||$

**Invariants:**

- $SLERP(w1, w2, 0) = w1 (left boundary)$
- $SLERP(w1, w2, 1) = w2 (right boundary)$
- $||SLERP(w1, w2, t)|| approximately ||w1|| for normalized inputs$

### ties

$$
w_merged = w_base + lambda * elect(trim(tau_1, ..., tau_n))
$$

**Domain:** $tau_i = w_i - w_base (task vectors), trim ratio k in [0,1]$

**Codomain:** $w_merged in R^n$

**Invariants:**

- $After trim(k\%), exactly k\% of delta weights are zeroed per layer$
- $Sign election resolves conflicts by majority vote$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | SLERP interpolation bound | $\|\|SLERP(w1, w2, t)\|\| within 1\% of \|\|w1\|\| for normalized inputs$ |
| 2 | invariant | SLERP boundary conditions | $SLERP(w1, w2, 0) = w1 and SLERP(w1, w2, 1) = w2$ |
| 3 | invariant | TIES trim sparsity | $After trim(k\%), exactly k\% of deltas are zero$ |
| 4 | invariant | DARE unbiased estimator | $E[tau_tilde] = tau over many samples$ |
| 5 | invariant | Architecture compatibility check | $Merge rejects incompatible architectures with clear error$ |

## Kernel Phases

1. **validate_architectures**: Verify all input models have same architecture — *hidden_size, num_layers, vocab_size match*
2. **compute_task_vectors**: Compute delta from base: tau_i = w_i - w_base — *tau has same shape as w*
3. **merge_weights**: Apply SLERP/TIES/DARE to combine weights — *output weights are finite*

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-MERGE-001 | SLERP interpolation bound | \|\|SLERP(w1, w2, t)\|\| within 1% of \|\|w1\|\| for normalized inputs | SLERP uses LERP instead, or normalization missing |
| FALSIFY-MERGE-002 | SLERP boundary | SLERP(w1, w2, 0) = w1 exactly (within fp tolerance) | Off-by-one in interpolation parameter |
| FALSIFY-MERGE-003 | DARE unbiased | Average of 10000 DARE samples within 1e-2 of original | Rescaling factor (1-p) not applied correctly |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-MERGE-001 | MERGE-BND-001 | 4 | stub_float |

## QA Gate

**Model Merging Contract** (F-MERGE-001)

Weight merging correctness for Albor post-training

**Checks:** slerp_bound, slerp_boundary, dare_unbiased

**Pass criteria:** All 3 falsification tests pass + Kani SLERP harness verifies

