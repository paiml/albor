"""
C-BACKPARITY-001: Per-parameter gradient norm comparison.
Golden reference generator for backward pass parity testing.

Usage: uv run scripts/gradient_parity.py
Output: JSON with per-parameter gradient norms after one forward+backward step.
"""
import torch
import math
import json
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(123)

config = LlamaConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=4,
    intermediate_size=4096,
    vocab_size=32768,
    max_position_embeddings=1024,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)
model = LlamaForCausalLM(config).cuda()

# Re-init with N(0, 0.02) to match entrenar C-INIT-001
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)

# Fixed input: tokens [100..1123] (1024 tokens)
input_ids = torch.arange(100, 1124, dtype=torch.long, device='cuda').unsqueeze(0)

model.train()
model.zero_grad()

out = model(input_ids=input_ids, labels=input_ids)
loss = out.loss
print(f"Loss: {loss.item():.6f}")
loss.backward()

# Collect per-parameter gradient norms
results = {"loss": loss.item(), "parameters": {}}

for name, param in model.named_parameters():
    if param.grad is not None:
        gnorm = param.grad.norm().item()
        results["parameters"][name] = {
            "gnorm": gnorm,
            "shape": list(param.shape),
            "numel": param.numel(),
        }
        # Print summary for key params
        if "layers.0." in name or "embed" in name or "lm_head" in name or "model.norm" in name:
            print(f"  {name}: gnorm={gnorm:.6f} shape={list(param.shape)}")

# Aggregate by layer
layer_gnorms = {}
for name, info in results["parameters"].items():
    if "layers." in name:
        parts = name.split(".")
        layer_idx = int(parts[2])
        if layer_idx not in layer_gnorms:
            layer_gnorms[layer_idx] = 0.0
        layer_gnorms[layer_idx] += info["gnorm"] ** 2
    
for layer_idx in sorted(layer_gnorms.keys()):
    layer_gnorm = math.sqrt(layer_gnorms[layer_idx])
    print(f"  layer {layer_idx:2d}: total_gnorm={layer_gnorm:.6f}")

total_gnorm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f"\nTotal gnorm: {total_gnorm:.6f}")
results["total_gnorm"] = total_gnorm

# Save JSON
with open("scripts/golden_gradients.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to scripts/golden_gradients.json")
