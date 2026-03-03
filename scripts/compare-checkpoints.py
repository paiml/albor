#!/usr/bin/env python3
"""R-031: Compare two model checkpoints on the same evaluation data.

Loads two entrenar SafeTensors checkpoints, runs perplexity evaluation on
the same held-out data, and prints a side-by-side metrics comparison table.

Usage:
    # Compare two checkpoints on validation data
    python scripts/compare-checkpoints.py \
        --model-a checkpoints/albor-base-350m/ \
        --model-b checkpoints/albor-base-350m-v2/ \
        --data data/pretokenized-2048/val/val.parquet \
        --max-sequences 50 --seq-len 512

    # Compare with labels
    python scripts/compare-checkpoints.py \
        --model-a checkpoints/step-1000/ --label-a "Step 1000" \
        --model-b checkpoints/step-5000/ --label-b "Step 5000" \
        --data data/pretokenized-2048/val/val.parquet

    # Output as JSON
    python scripts/compare-checkpoints.py \
        --model-a ckpt-a/ --model-b ckpt-b/ \
        --data val.parquet --json

Requirements: pip install numpy safetensors pyarrow
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def load_config(model_dir: Path) -> dict:
    """Load config.json from checkpoint directory."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


def load_safetensors(path: Path) -> dict:
    """Load SafeTensors file into numpy arrays."""
    from safetensors.numpy import load_file
    return load_file(str(path))


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def rope_embed(x, seq_len, head_dim, theta=500000.0):
    """Apply RoPE positional embeddings."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    positions = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    x1 = x[..., :half]
    x2 = x[..., half:]
    out1 = x1 * cos_a - x2 * sin_a
    out2 = x1 * sin_a + x2 * cos_a
    return np.concatenate([out1, out2], axis=-1)


def forward_transformer(tokens, weights, config):
    """Run transformer forward pass, return logits."""
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv = config.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    eps = config.get("rms_norm_eps", 1e-5)
    theta = config.get("rope_theta", 500000.0)

    seq_len = len(tokens)

    # Embedding
    embed_w = weights["model.embed_tokens.weight"]
    hidden = embed_w[tokens]

    # Transformer blocks
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"

        # Input norm
        normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], eps)

        # Self-attention
        wq = weights[f"{prefix}.self_attn.q_proj.weight"]
        wk = weights[f"{prefix}.self_attn.k_proj.weight"]
        wv = weights[f"{prefix}.self_attn.v_proj.weight"]
        wo = weights[f"{prefix}.self_attn.o_proj.weight"]

        q = normed @ wq
        k = normed @ wk
        v = normed @ wv

        q = q.reshape(seq_len, num_heads, head_dim)
        k = k.reshape(seq_len, num_kv, head_dim)
        v = v.reshape(seq_len, num_kv, head_dim)

        q = rope_embed(q, seq_len, head_dim, theta)
        k = rope_embed(k, seq_len, head_dim, theta)

        # GQA: repeat KV heads
        if num_kv < num_heads:
            reps = num_heads // num_kv
            k = np.repeat(k, reps, axis=1)
            v = np.repeat(v, reps, axis=1)

        # Scaled dot-product attention with causal mask
        scores = np.einsum("shd,thd->sht", q, k) / math.sqrt(head_dim)
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
        scores = scores + mask[:, :, None]
        attn = softmax(scores, axis=1)
        attn_out = np.einsum("sht,thd->shd", attn, v)
        attn_out = attn_out.reshape(seq_len, hidden_size)
        hidden = hidden + attn_out @ wo

        # Post-attention norm + FFN
        normed2 = rms_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], eps)
        w_gate = weights[f"{prefix}.mlp.gate_proj.weight"]
        w_up = weights[f"{prefix}.mlp.up_proj.weight"]
        w_down = weights[f"{prefix}.mlp.down_proj.weight"]

        gate = normed2 @ w_gate
        up = normed2 @ w_up
        # SiLU activation
        silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
        ffn_out = (silu_gate * up) @ w_down
        hidden = hidden + ffn_out

    # Final norm + LM head
    norm_w = weights["model.norm.weight"]
    hidden = rms_norm(hidden, norm_w, eps)
    lm_head = weights.get("lm_head.weight", embed_w)
    logits = hidden @ lm_head
    return logits


def find_weights_path(model_dir):
    """Find the weights file in a checkpoint directory."""
    weights_path = model_dir / "model.safetensors"
    if weights_path.exists():
        return weights_path
    candidates = sorted(model_dir.glob("model-step-*.safetensors"))
    if candidates:
        return candidates[-1]
    print(f"ERROR: No weights found in {model_dir}", file=sys.stderr)
    sys.exit(1)


def find_token_column(table):
    """Find the token column name in a Parquet table."""
    for name in ["input_ids", "tokens", "token_ids"]:
        if name in table.column_names:
            return name
    return None


def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss for a sequence."""
    log_probs = logits - np.log(
        np.sum(np.exp(logits - np.max(logits, axis=-1, keepdims=True)), axis=-1, keepdims=True)
    )
    loss = sum(-log_probs[t, targets[t]] for t in range(len(targets)))
    return float(loss), len(targets)


def load_sequences(data_path, max_sequences):
    """Load pre-tokenized sequences from Parquet."""
    import pyarrow.parquet as pq
    table = pq.read_table(str(data_path))
    token_col = find_token_column(table)
    if not token_col:
        print(f"ERROR: No token column found in {data_path}", file=sys.stderr)
        sys.exit(1)
    return table[token_col].to_pylist()[:max_sequences]


def evaluate_perplexity(model_dir, data_path, max_sequences, seq_len):
    """Evaluate perplexity of a checkpoint on data."""
    config = load_config(model_dir)
    weights_path = find_weights_path(model_dir)
    weights = load_safetensors(weights_path)
    sequences = load_sequences(data_path, max_sequences)

    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    for i, seq in enumerate(sequences):
        tokens = np.array(seq[:seq_len], dtype=np.int64)
        if len(tokens) < 2:
            continue
        logits = forward_transformer(tokens[:-1], weights, config)
        loss, n_tok = cross_entropy_loss(logits, tokens[1:])
        total_loss += loss
        total_tokens += n_tok

    elapsed = time.time() - start
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "perplexity": float(ppl),
        "avg_loss": float(avg_loss),
        "total_tokens": total_tokens,
        "sequences": len(sequences),
        "elapsed_s": round(elapsed, 1),
        "weights": str(weights_path),
        "num_params": sum(w.size for w in weights.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two model checkpoints")
    parser.add_argument("--model-a", required=True, type=Path, help="First model directory")
    parser.add_argument("--model-b", required=True, type=Path, help="Second model directory")
    parser.add_argument("--label-a", default=None, help="Label for model A")
    parser.add_argument("--label-b", default=None, help="Label for model B")
    parser.add_argument("--data", required=True, type=Path, help="Evaluation data (Parquet)")
    parser.add_argument("--max-sequences", type=int, default=50, help="Max sequences to evaluate")
    parser.add_argument("--seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    label_a = args.label_a or args.model_a.name
    label_b = args.label_b or args.model_b.name

    print(f"=== Model Comparison ===", file=sys.stderr)
    print(f"  A: {label_a} ({args.model_a})", file=sys.stderr)
    print(f"  B: {label_b} ({args.model_b})", file=sys.stderr)
    print(f"  Data: {args.data} (max {args.max_sequences} seqs, len {args.seq_len})", file=sys.stderr)
    print(file=sys.stderr)

    print(f"Evaluating model A ({label_a})...", file=sys.stderr)
    result_a = evaluate_perplexity(args.model_a, args.data, args.max_sequences, args.seq_len)

    print(f"\nEvaluating model B ({label_b})...", file=sys.stderr)
    result_b = evaluate_perplexity(args.model_b, args.data, args.max_sequences, args.seq_len)

    # Comparison
    ppl_diff = result_b["perplexity"] - result_a["perplexity"]
    ppl_pct = (ppl_diff / result_a["perplexity"] * 100) if result_a["perplexity"] > 0 else 0
    winner = label_a if result_a["perplexity"] < result_b["perplexity"] else label_b

    if args.json:
        output = {
            "model_a": {"label": label_a, **result_a},
            "model_b": {"label": label_b, **result_b},
            "comparison": {
                "ppl_diff": round(ppl_diff, 2),
                "ppl_pct_change": round(ppl_pct, 2),
                "winner": winner,
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print()
        print(f"{'Metric':<25} {'A: ' + label_a:<25} {'B: ' + label_b:<25} {'Delta':<15}")
        print("-" * 90)
        print(f"{'Perplexity':<25} {result_a['perplexity']:<25.2f} {result_b['perplexity']:<25.2f} {ppl_diff:+.2f} ({ppl_pct:+.1f}%)")
        print(f"{'Avg Loss':<25} {result_a['avg_loss']:<25.4f} {result_b['avg_loss']:<25.4f} {result_b['avg_loss'] - result_a['avg_loss']:+.4f}")
        print(f"{'Params':<25} {result_a['num_params']:<25,} {result_b['num_params']:<25,}")
        print(f"{'Tokens Evaluated':<25} {result_a['total_tokens']:<25,} {result_b['total_tokens']:<25,}")
        print(f"{'Eval Time (s)':<25} {result_a['elapsed_s']:<25.1f} {result_b['elapsed_s']:<25.1f}")
        print("-" * 90)
        print(f"Winner: {winner} (lower PPL is better)")


if __name__ == "__main__":
    main()
