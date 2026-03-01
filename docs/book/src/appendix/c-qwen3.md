# Appendix C: Qwen3-Coder-Next Architecture Details

| Layer Pattern | Count | Description |
|---------------|-------|-------------|
| Gated DeltaNet → MoE | 36 (3 per block × 12 blocks) | Linear attention with gating, routed to 10/512 experts |
| Gated Attention → MoE | 12 (1 per block × 12 blocks) | Standard GQA with gating, routed to 10/512 experts |
| **Total layers** | **48** | |

This hybrid architecture means realizar needs to support:
- DeltaNet (linear attention variant) — likely a new gap
- MoE routing (top-k expert selection) — may partially exist
- Gated variants of both attention types
