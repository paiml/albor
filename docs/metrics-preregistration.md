# Pre-registered Metrics

**Registered**: 2026-02-28 (before Phase 3 training)

## Primary Metrics

| Metric | Target | Measurement | Registered |
|--------|--------|-------------|------------|
| Training loss | < 3.0 at convergence | Cross-entropy loss on train set | 2026-02-28 |
| Validation perplexity | < 50 | exp(cross-entropy) on held-out val set | 2026-02-28 |
| HumanEval pass@1 | > 5% | Functional correctness on 20 problems | 2026-02-28 |
| Checkpoint load time | < 5s | realizar SafeTensors loading | 2026-02-28 |

## Training Stability Metrics

| Metric | Bound | Contract |
|--------|-------|----------|
| Gradient norm | ≤ max_grad_norm (1.0) | C-EMBED-GRAD-001 |
| Loss monotonicity | Decreasing over 50-step windows | FALSIFY-ALBOR-001 |
| No NaN in weights | 0 NaN values in all parameters | C-HYPERPARAMS-001 |

## Effect Size Thresholds

| Comparison | Minimum effect size | Statistical test |
|-----------|-------------------|------------------|
| 50M vs 350M perplexity | Cohen's d > 0.8 (large) | Paired t-test, α=0.05 |
| Pre vs post distillation | Cohen's d > 0.5 (medium) | Paired t-test, α=0.05 |
