# pruning-kernel-v1

**Version:** 1.0.0

Pruning kernel — WANDA and magnitude-based weight pruning

## References

- Sun et al. (2023) A Simple and Effective Pruning Approach for Large Language Models (WANDA)
- Han et al. (2015) Learning both Weights and Connections for Efficient Neural Networks

## Equations

### magnitude_score

$$
score(w_ij) = |w_ij|
$$

**Domain:** $w_ij: weight value$

**Codomain:** $score in [0, +inf)$

**Invariants:**

- $score >= 0$
- $score = 0 iff w_ij = 0$

### sparsity

$$
s = |{w : w = 0}| / |w|
$$

**Domain:** $w: weight tensor$

**Codomain:** $s in [0, 1]$

**Invariants:**

- $s = 0 means no pruning$
- $s = 1 means all weights zeroed$
- $After pruning with target s, achieved sparsity within 0.1\% of s$

### wanda_score

$$
score(w_ij) = |w_ij| * ||X_j||_2
$$

**Domain:** $w_ij: weight, X_j: activation column vector$

**Codomain:** $score in [0, +inf)$

**Invariants:**

- $score >= 0 (product of norms)$
- $score = 0 iff w_ij = 0 or X_j = 0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Sparsity target met | $Achieved sparsity within +/-0.1\% of target$ |
| 2 | ordering | Score ordering preserved | $All pruned weights have score <= all surviving weights$ |
| 3 | invariant | WANDA activation dependency | $Same weight magnitude + different activation norms => different WANDA scores$ |
| 4 | invariant | Zero sparsity is identity | $prune(model, sparsity=0) returns original model unchanged$ |
| 5 | invariant | Full sparsity zeroes all | $prune(model, sparsity=1.0) zeroes all prunable weights$ |
| 6 | invariant | Embedding layer excluded | $Embedding and output projection weights untouched by pruning$ |

## Kernel Phases

1. **compute_scores**: Compute importance score for each weight — *scores are non-negative*
2. **determine_threshold**: Find threshold score for target sparsity — *threshold partitions weights into keep/prune sets*
3. **apply_mask**: Zero out weights below threshold — *sparsity matches target within tolerance*

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-PRUNE-001 | Sparsity guarantee | Exactly 50% of weights zero after prune --sparsity 0.5 | Threshold computation error or layer exclusion bug |
| FALSIFY-PRUNE-002 | Score ordering | All pruned weights have score <= all surviving weights | Sorting or partitioning algorithm bug |
| FALSIFY-PRUNE-003 | Identity at zero sparsity | Pruning with sparsity=0 returns original weights | Off-by-one in threshold or mask computation |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-PRUNE-001 | PRUNE-INV-001 | 16 | stub_float |

## QA Gate

**Pruning Contract** (F-PRUNE-001)

Weight pruning correctness for Albor model compression

**Checks:** sparsity_guarantee, score_ordering, identity_at_zero

**Pass criteria:** All 3 falsification tests pass + Kani sparsity harness verifies

