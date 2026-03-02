#!/usr/bin/env python3
"""Evaluate model perplexity on held-out data.

Pure-Python Qwen2/LLaMA transformer inference for perplexity evaluation.
Loads entrenar SafeTensors checkpoints directly.

NOTE: `apr eval` has a weight-loading bug (ALB-037, GitHub #35) where
realizar ignores loaded weights during inference. This script provides
a workaround using pure NumPy inference.

IMPORTANT: Data must be pre-tokenized with the SAME tokenizer used during
training. Using mismatched tokenizer IDs will produce garbage results.

Usage:
    # 50M model (v2 tokenizer, pre-tokenized data)
    python scripts/eval-perplexity.py checkpoints/albor-base-50m/ \
        --data data/pretokenized-128/train/train.parquet \
        --max-sequences 50 --seq-len 128

    # 350M model (v2 tokenizer, pre-tokenized data)
    python scripts/eval-perplexity.py checkpoints/albor-base-350m/ \
        --data data/pretokenized-2048/val/val.parquet \
        --max-sequences 100 --seq-len 2048

Requirements: pip install numpy safetensors pyarrow
"""

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm as used in LLaMA/Qwen2."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def silu(x):
    """SiLU/Swish activation."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def rope_freqs(dim, max_len, base=10000.0):
    """Precompute RoPE frequency table."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # [max_len, dim//2]
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin, start_pos=0):
    """Apply RoPE to query or key tensor.

    x: [seq_len, n_heads, head_dim]
    """
    seq_len, n_heads, head_dim = x.shape
    half = head_dim // 2

    x1 = x[:, :, :half]
    x2 = x[:, :, half:]

    c = cos[start_pos:start_pos + seq_len, :half].reshape(seq_len, 1, half)
    s = sin[start_pos:start_pos + seq_len, :half].reshape(seq_len, 1, half)

    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return np.concatenate([out1, out2], axis=-1)


class TransformerModel:
    """Minimal Qwen2/LLaMA-style transformer for perplexity evaluation."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self._load_config()
        self._load_weights()

        # Precompute RoPE
        head_dim = self.hidden_size // self.num_heads
        self.rope_cos, self.rope_sin = rope_freqs(
            head_dim, self.max_position_embeddings
        )

    def _load_config(self):
        """Load architecture config."""
        config_path = self.checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self.hidden_size = cfg["hidden_size"]
            self.num_layers = cfg["num_hidden_layers"]
            self.num_heads = cfg["num_attention_heads"]
            self.num_kv_heads = cfg["num_key_value_heads"]
            self.intermediate_size = cfg["intermediate_size"]
            self.vocab_size = cfg["vocab_size"]
            self.max_position_embeddings = cfg.get("max_position_embeddings", 2048)
            self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
        else:
            raise FileNotFoundError(f"config.json not found in {self.checkpoint_dir}")

    def _load_weights(self):
        """Load SafeTensors weights with proper reshaping."""
        st_path = self.checkpoint_dir / "model.safetensors"
        bak_path = self.checkpoint_dir / "model.safetensors.bak"

        # Prefer backup (original) if it exists and we need to reshape
        load_path = bak_path if bak_path.exists() else st_path

        with open(load_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode())
            data = f.read()

        self.weights = {}
        head_dim = self.hidden_size // self.num_heads

        for name, info in header.items():
            if name == "__metadata__":
                continue

            raw = data[info["data_offsets"][0]:info["data_offsets"][1]]
            dtype = np.float32 if info["dtype"] == "F32" else np.float16
            arr = np.frombuffer(raw, dtype=dtype).copy()

            # Infer and apply proper shape
            shape = self._infer_shape(name, arr.size, head_dim)
            if shape:
                arr = arr.reshape(shape)

            self.weights[name] = arr.astype(np.float32)

        print(f"Loaded {len(self.weights)} tensors from {load_path.name}")

    def _infer_shape(self, name, flat_size, head_dim):
        """Infer proper 2D shape from tensor name pattern."""
        h, ffn, v, kv = self.hidden_size, self.intermediate_size, self.vocab_size, self.num_kv_heads
        kv_dim = kv * head_dim
        shape_table = {
            "embed_tokens": (v, h), "lm_head": (v, h),
            "q_proj": (h, h), "k_proj": (kv_dim, h), "v_proj": (kv_dim, h), "o_proj": (h, h),
            "gate_proj": (ffn, h), "up_proj": (ffn, h), "down_proj": (h, ffn),
        }
        for pattern, shape in shape_table.items():
            if pattern in name:
                return shape
        if "norm" in name or "bias" in name:
            return (flat_size,)
        return None

    def _get(self, name):
        """Get weight tensor."""
        return self.weights[name]

    def forward(self, input_ids):
        """Forward pass returning logits.

        input_ids: [seq_len] numpy array of token IDs
        Returns: [seq_len, vocab_size] logits
        """
        seq_len = len(input_ids)

        # Embedding lookup
        x = self._get("model.embed_tokens.weight")[input_ids]  # [seq_len, hidden]

        head_dim = self.hidden_size // self.num_heads
        kv_head_dim = head_dim
        gqa_groups = self.num_heads // self.num_kv_heads

        for i in range(self.num_layers):
            prefix = f"model.layers.{i}"

            # Pre-attention norm
            normed = rms_norm(
                x,
                self._get(f"{prefix}.input_layernorm.weight"),
                self.rms_norm_eps,
            )

            # QKV projections
            q = normed @ self._get(f"{prefix}.self_attn.q_proj.weight").T
            k = normed @ self._get(f"{prefix}.self_attn.k_proj.weight").T
            v = normed @ self._get(f"{prefix}.self_attn.v_proj.weight").T

            # Reshape for multi-head attention
            q = q.reshape(seq_len, self.num_heads, head_dim)
            k = k.reshape(seq_len, self.num_kv_heads, head_dim)
            v = v.reshape(seq_len, self.num_kv_heads, head_dim)

            # Apply RoPE
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

            # Expand KV heads for GQA
            if gqa_groups > 1:
                k = np.repeat(k, gqa_groups, axis=1)
                v = np.repeat(v, gqa_groups, axis=1)

            # Attention: [seq, heads, head_dim] -> scores
            scale = 1.0 / math.sqrt(head_dim)
            # q: [seq, heads, dim], k: [seq, heads, dim]
            scores = np.einsum("shd,thd->sht", q, k) * scale

            # Causal mask
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores += mask[:, :, np.newaxis].transpose(0, 2, 1)
            # Fix: scores is [s, h, t], mask should be [s, 1, t]
            # Actually let me redo this properly
            scores = np.einsum("shd,thd->hst", q, k) * scale  # [heads, seq, seq]
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores = scores + mask[np.newaxis, :, :]

            attn = softmax(scores, axis=-1)  # [heads, seq, seq]

            # Apply attention to values
            # v: [seq, heads, dim] -> [heads, seq, dim]
            v_t = v.transpose(1, 0, 2)  # [heads, seq, dim]
            out = np.einsum("hst,htd->hsd", attn, v_t)  # [heads, seq, dim]
            out = out.transpose(1, 0, 2).reshape(seq_len, -1)  # [seq, hidden]

            # Output projection
            attn_out = out @ self._get(f"{prefix}.self_attn.o_proj.weight").T

            # Residual
            x = x + attn_out

            # Post-attention norm + MLP
            normed = rms_norm(
                x,
                self._get(f"{prefix}.post_attention_layernorm.weight"),
                self.rms_norm_eps,
            )

            # SwiGLU MLP
            gate = normed @ self._get(f"{prefix}.mlp.gate_proj.weight").T
            up = normed @ self._get(f"{prefix}.mlp.up_proj.weight").T
            mlp_out = silu(gate) * up
            mlp_out = mlp_out @ self._get(f"{prefix}.mlp.down_proj.weight").T

            x = x + mlp_out

        # Final norm
        x = rms_norm(x, self._get("model.norm.weight"), self.rms_norm_eps)

        # LM head (tied embeddings)
        lm_weight = self.weights.get(
            "lm_head.weight",
            self._get("model.embed_tokens.weight"),
        )
        logits = x @ lm_weight.T  # [seq_len, vocab_size]

        return logits


def compute_perplexity(model, sequences, seq_len):
    """Compute perplexity over a set of token sequences."""
    total_ce = 0.0
    total_tokens = 0

    for i, seq in enumerate(sequences):
        tokens = np.array(seq[:seq_len], dtype=np.int64)
        if len(tokens) < 2:
            continue

        t0 = time.time()
        logits = model.forward(tokens)
        dt = time.time() - t0

        # Cross-entropy: compare logits[:-1] with targets tokens[1:]
        log_probs = logits[:-1] - np.log(
            np.sum(np.exp(logits[:-1] - np.max(logits[:-1], axis=-1, keepdims=True)),
                   axis=-1, keepdims=True)
        ) - np.max(logits[:-1], axis=-1, keepdims=True)
        # Actually compute log_softmax properly
        max_logits = np.max(logits[:-1], axis=-1, keepdims=True)
        shifted = logits[:-1] - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp

        targets = tokens[1:]
        token_log_probs = log_probs[np.arange(len(targets)), targets]
        ce = -np.mean(token_log_probs)

        total_ce += ce * len(targets)
        total_tokens += len(targets)

        if (i + 1) % 10 == 0 or i == 0:
            ppl_so_far = math.exp(total_ce / total_tokens)
            print(f"  [{i+1}/{len(sequences)}] ce={ce:.4f} "
                  f"running_ppl={ppl_so_far:.2f} ({dt:.2f}s/seq)")

    avg_ce = total_ce / total_tokens
    ppl = math.exp(avg_ce)
    return ppl, avg_ce, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--data", type=Path, required=True,
                        help="Parquet file with pre-tokenized data (input_ids column)")
    parser.add_argument("--text-data", type=Path,
                        help="Parquet file with text data (text column)")
    parser.add_argument("--max-sequences", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Perplexity threshold for pass/fail")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint_dir}...")
    model = TransformerModel(args.checkpoint_dir)
    print(f"  Architecture: hidden={model.hidden_size}, layers={model.num_layers}, "
          f"heads={model.num_heads}, kv_heads={model.num_kv_heads}")

    print(f"\nLoading data from {args.data}...")
    import pyarrow.parquet as pq
    table = pq.read_table(args.data)

    if "input_ids" in table.column_names:
        sequences = [row.as_py() for row in table.column("input_ids")]
    elif "text" in table.column_names:
        raise NotImplementedError("Text data requires tokenizer — use pre-tokenized data")
    else:
        raise ValueError(f"Expected 'input_ids' or 'text' column, got {table.column_names}")

    sequences = sequences[:args.max_sequences]
    print(f"  Sequences: {len(sequences)}, seq_len cap: {args.seq_len}")

    print(f"\nComputing perplexity...")
    t0 = time.time()
    ppl, ce, n_tokens = compute_perplexity(model, sequences, args.seq_len)
    elapsed = time.time() - t0

    passed = ppl <= args.threshold
    status = "PASS" if passed else "FAIL"

    print(f"\n{'='*60}")
    print(f"  Perplexity:     {ppl:.2f} ({status}: threshold {args.threshold})")
    print(f"  Cross-entropy:  {ce:.4f}")
    print(f"  Tokens:         {n_tokens:,}")
    print(f"  Time:           {elapsed:.1f}s ({n_tokens/elapsed:.0f} tok/s)")
    print(f"  Sequences:      {len(sequences)}")
    print(f"{'='*60}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
