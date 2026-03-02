#!/usr/bin/env python3
"""Convert entrenar checkpoint to realizar-compatible format.

entrenar saves SafeTensors with 1D (flattened) tensor shapes.
realizar expects proper 2D shapes for matrix operations.

This script:
1. Reads the entrenar checkpoint
2. Reshapes all weight tensors to proper 2D shapes
3. Creates lm_head.weight if missing (tied embeddings → explicit copy)
4. Writes a new SafeTensors file with correct shapes

Usage:
    python scripts/convert-checkpoint.py \\
        checkpoints/albor-base-50m/ \\
        --config configs/train/pretrain-50m.yaml

    python scripts/convert-checkpoint.py \\
        checkpoints/albor-base-350m/ \\
        --hidden-size 1024 --layers 24 --heads 16 --kv-heads 4 --ffn 4096
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


def load_safetensors_raw(path: Path) -> tuple[dict, bytes, int]:
    """Load safetensors header and raw data."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))
        data = f.read()
    return header, data, header_size


def save_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]):
    """Save tensors as safetensors file.

    tensors: dict of name -> (numpy_array, dtype_str)
    dtype_str: "F32" or "F16"
    """
    # Build header
    header = {}
    offset = 0
    tensor_data_parts = []

    for name in sorted(tensors.keys()):
        arr, dtype_str = tensors[name]
        raw = arr.tobytes()
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        tensor_data_parts.append(raw)
        offset += len(raw)

    header["__metadata__"] = {
        "version": "0.1.0",
        "architecture": "Qwen2ForCausalLM",
        "converted_by": "albor/scripts/convert-checkpoint.py",
    }

    header_bytes = json.dumps(header).encode("utf-8")
    # Pad to 8-byte alignment
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for part in tensor_data_parts:
            f.write(part)


def _build_shape_table(hidden, ffn, vocab, heads, kv_heads):
    """Build lookup table mapping tensor name patterns to shapes."""
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim
    return {
        "embed_tokens": (vocab, hidden),
        "lm_head": (vocab, hidden),
        "q_proj": (hidden, hidden),
        "k_proj": (kv_dim, hidden),
        "v_proj": (kv_dim, hidden),
        "o_proj": (hidden, hidden),
        "gate_proj": (ffn, hidden),
        "up_proj": (ffn, hidden),
        "down_proj": (hidden, ffn),
    }


def infer_shape(name, flat_size, hidden, ffn, vocab, heads, kv_heads):
    """Infer proper 2D shape from tensor name and flat size."""
    table = _build_shape_table(hidden, ffn, vocab, heads, kv_heads)

    for pattern, shape in table.items():
        if pattern in name:
            return shape

    if "norm" in name or "bias" in name or "layernorm" in name:
        return (flat_size,)

    raise ValueError(f"Cannot infer shape for {name} with size {flat_size}")


DTYPE_MAP = {"F32": np.float32, "F16": np.float16}


def extract_tensors(header, data):
    """Extract all tensors from safetensors header + data."""
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        np_dtype = DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype {dtype_str} for {name}")
        raw = data[info["data_offsets"][0]:info["data_offsets"][1]]
        tensors[name] = (np.frombuffer(raw, dtype=np_dtype).copy(), dtype_str)
    return tensors


def reshape_tensor(name, arr, hidden, ffn, vocab, heads, kv_heads):
    """Reshape a single tensor, returning new array."""
    target = infer_shape(name, arr.size, hidden, ffn, vocab, heads, kv_heads)
    if math.prod(target) != arr.size:
        return arr  # cannot reshape, keep as-is
    return arr.reshape(target)


def convert_checkpoint(checkpoint_dir, hidden, layers, heads, kv_heads, ffn, vocab):
    """Convert entrenar checkpoint to realizar format."""
    src = checkpoint_dir / "model.safetensors"
    backup = checkpoint_dir / "model.safetensors.bak"

    if not src.exists():
        sys.exit(f"ERROR: {src} not found")

    print(f"Loading {src}...")
    header, data, _ = load_safetensors_raw(src)

    if header.get("__metadata__", {}).get("converted_by"):
        print("Already converted. Skipping.")
        return

    tensors_in = extract_tensors(header, data)

    print(f"Backing up to {backup}...")
    src.rename(backup)

    # Reshape all tensors
    tensors_out = {}
    for name, (arr, dtype) in sorted(tensors_in.items()):
        reshaped = reshape_tensor(name, arr, hidden, ffn, vocab, heads, kv_heads)
        tensors_out[name] = (reshaped, dtype)
        print(f"  {name}: [{arr.size}] -> {list(reshaped.shape)}")

    # Add lm_head if missing (tied embeddings)
    if "lm_head.weight" not in tensors_out and "model.embed_tokens.weight" in tensors_out:
        embed_arr, embed_dtype = tensors_out["model.embed_tokens.weight"]
        tensors_out["lm_head.weight"] = (embed_arr.copy(), embed_dtype)
        print(f"  lm_head.weight: CREATED (tied copy) {list(embed_arr.shape)}")

    print(f"\nSaving converted checkpoint...")
    save_safetensors(src, tensors_out)
    print(f"Conversion complete: {len(tensors_out)} tensors, backup: {backup}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert entrenar checkpoint to realizar-compatible format"
    )
    parser.add_argument("checkpoint_dir", type=Path,
                        help="Checkpoint directory containing model.safetensors")
    parser.add_argument("--config", type=Path,
                        help="Training YAML config (reads architecture from it)")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--vocab", type=int, default=32768)
    args = parser.parse_args()

    # Read architecture from config if provided
    if args.config:
        try:
            import yaml
        except ImportError:
            sys.exit("ERROR: PyYAML required when using --config. Install: pip install pyyaml")
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        arch = cfg["model"]["architecture"]
        hidden = args.hidden_size or arch["hidden_size"]
        layers = args.layers or arch["num_hidden_layers"]
        heads = args.heads or arch["num_attention_heads"]
        kv_heads = args.kv_heads or arch["num_key_value_heads"]
        ffn = args.ffn or arch["intermediate_size"]
        vocab = args.vocab or arch.get("vocab_size", 32768)
    else:
        hidden = args.hidden_size
        layers = args.layers
        heads = args.heads
        kv_heads = args.kv_heads
        ffn = args.ffn
        vocab = args.vocab

        if not all([hidden, layers, heads, kv_heads, ffn]):
            sys.exit("ERROR: Provide --config or all of: "
                     "--hidden-size --layers --heads --kv-heads --ffn")

    convert_checkpoint(args.checkpoint_dir, hidden, layers, heads, kv_heads, ffn, vocab)


if __name__ == "__main__":
    main()
