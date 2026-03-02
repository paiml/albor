# Equation Model Card: Training

## Governing Equations

### Cross-Entropy Loss
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

### AdamW Optimizer
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t)$$
$$\hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_t = \theta_{t-1} - \eta (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_{t-1})$$

### RMSNorm
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

### Rotary Position Embedding (RoPE)
$$f(x, m) = R_{\Theta,m}^d x$$
where $R_{\Theta,m}^d$ is the rotation matrix at position $m$.

## Numerical Bounds

| Equation | Parameter | Bound | Verified |
|----------|-----------|-------|----------|
| AdamW bias correction | $1/(1-\beta_2^t)$ | $< 1/\epsilon$ for $t > 0$ | C-HYPERPARAMS-001 |
| Gradient norm | $\|g\|$ | $\leq$ max_grad_norm | C-EMBED-GRAD-001 |
| RMSNorm | denominator | $> \epsilon$ | ALB-038 FIXED |

## Implementation References

| Equation | Contract | Implementation |
|----------|----------|----------------|
| Cross-entropy loss | training-gpu-kernel-v1 | entrenar `src/cuda/loss.rs` |
| AdamW | gradient-accumulation-kernel-v1 | entrenar `src/optim/adamw.rs` |
| RMSNorm | training-gpu-kernel-v1 | entrenar `src/cuda/norms.rs` |
