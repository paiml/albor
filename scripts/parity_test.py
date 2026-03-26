#!/usr/bin/env python3
"""Training parity test — ONE step comparison between PyTorch and entrenar.

Runs one forward+backward pass on identical input tokens and dumps:
- Per-layer gradient norms
- Total gradient norm
- Loss value
- Weight update magnitude after one optimizer step

Usage:
    PYTHONUNBUFFERED=1 uv run --extra-index-url https://download.pytorch.org/whl/cu128 \
        --with 'torch,transformers' scripts/parity_test.py

Then compare output with entrenar's step-0 diagnostics in the training log.
"""
import json, math, sys
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from torch.optim import AdamW

SEED = 123
SEQ_LEN = 1024
# Fixed deterministic input
INPUT_IDS = list(range(100, 100 + SEQ_LEN))

def main():
    torch.manual_seed(SEED)
    config = LlamaConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        num_key_value_heads=4, intermediate_size=4096, vocab_size=32768,
        max_position_embeddings=1024, rms_norm_eps=1e-5, rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params on {device}", flush=True)

    # Capture init weight norms
    print(f"\n=== Init Weight Norms ===", flush=True)
    for name, p in model.named_parameters():
        if any(k in name for k in [".0.", "embed", "norm.weight", "lm_head"]):
            print(f"  {name}: norm={p.data.norm().item():.4f} std={p.data.std().item():.6f}", flush=True)

    # Forward
    ids = torch.tensor([INPUT_IDS], device=device)
    output = model(input_ids=ids, labels=ids)
    loss = output.loss
    print(f"\n=== Forward Pass ===", flush=True)
    print(f"Loss: {loss.item():.6f} (ppl={math.exp(min(loss.item(),20)):.1f})", flush=True)
    print(f"Logits: mean={output.logits.mean().item():.6f} std={output.logits.std().item():.6f} norm={output.logits.norm().item():.2f}", flush=True)

    # Backward (NO grad_accum division — raw gradients)
    loss.backward()

    print(f"\n=== Gradient Norms (per layer) ===", flush=True)
    total_norm_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            total_norm_sq += gn ** 2
            if any(k in name for k in [".0.", "embed", "norm.weight", "lm_head"]):
                print(f"  {name}: grad_norm={gn:.6f}", flush=True)

    total_norm = total_norm_sq ** 0.5
    print(f"\nTotal grad norm: {total_norm:.4f}", flush=True)

    # Compare with entrenar
    # Entrenar at step 0: gnorm=0.654 (per micro-batch, with scale=1/(seq_len*ga))
    # PyTorch here: raw gradient (no scale), so we also compute scaled version
    seq_len_effective = SEQ_LEN - 1  # label shift
    ga = 8
    scaled_norm = total_norm / seq_len_effective  # mean reduction
    scaled_ga_norm = scaled_norm / ga  # with grad_accum
    print(f"\nScaled by mean reduction (/seq_len): {scaled_norm:.4f}", flush=True)
    print(f"Scaled by mean+GA (/{seq_len_effective}/{ga}): {scaled_ga_norm:.4f}", flush=True)
    print(f"\nEntrenar reports gnorm=0.654 at step 0", flush=True)
    print(f"PyTorch scaled+GA: {scaled_ga_norm:.4f}", flush=True)
    if scaled_ga_norm > 0:
        print(f"Ratio PyTorch/Entrenar: {scaled_ga_norm/0.654:.2f}x", flush=True)

    # One optimizer step
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    optimizer.step()

    print(f"\n=== After One Optimizer Step ===", flush=True)
    for name, p in model.named_parameters():
        if any(k in name for k in [".0.self_attn.q", "embed_tokens"]):
            print(f"  {name}: norm={p.data.norm().item():.4f}", flush=True)

    print(f"\nDone. Compare these values with entrenar step-0 diagnostics.", flush=True)


if __name__ == "__main__":
    main()
