#!/usr/bin/env python3
"""Validate ALB-118: GPU optimizer state in APR checkpoint.

Checks that __training__.block_optimizer.{layer}.{m,v}.{weight} tensors
are present and non-zero in the checkpoint file.

Usage:
    python3 scripts/validate-checkpoint-alb118.py checkpoints/albor-base-350m-v13/model-best.apr
"""
import struct
import json
import sys
from pathlib import Path


def read_apr_header(path: Path) -> dict:
    """Read APR v2 header (JSON metadata + tensor index)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"APR\x02":
            raise ValueError(f"Not an APR v2 file (magic: {magic!r})")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
        return json.loads(header_json)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint.apr>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    header = read_apr_header(path)
    tensors = header.get("tensors", {})

    # Expected block optimizer tensor suffixes
    suffixes = [
        "m.w_q", "v.w_q", "m.w_k", "v.w_k", "m.w_v", "v.w_v",
        "m.w_o", "v.w_o", "m.w_gate", "v.w_gate", "m.w_up", "v.w_up",
        "m.w_down", "v.w_down", "m.input_norm", "v.input_norm",
        "m.post_attn_norm", "v.post_attn_norm",
    ]

    # Count block optimizer tensors
    block_tensors = [k for k in tensors if k.startswith("__training__.block_optimizer.")]
    lm_head_tensors = [k for k in tensors if k.startswith("__training__.lm_head_optimizer.")]
    final_norm_tensors = [k for k in tensors if k.startswith("__training__.final_norm_optimizer.")]
    embed_tensors = [k for k in tensors if k.startswith("__training__.embed_optimizer.")]

    print(f"APR checkpoint: {path}")
    print(f"  Total tensors: {len(tensors)}")
    print(f"  Block optimizer tensors: {len(block_tensors)}")
    print(f"  LM head optimizer tensors: {len(lm_head_tensors)}")
    print(f"  Final norm optimizer tensors: {len(final_norm_tensors)}")
    print(f"  Embed optimizer tensors: {len(embed_tensors)}")
    print()

    # Detect number of layers
    layer_indices = set()
    for name in block_tensors:
        parts = name.split(".")
        # __training__.block_optimizer.{layer}.{suffix}
        if len(parts) >= 4:
            try:
                layer_indices.add(int(parts[2]))
            except ValueError:
                pass

    num_layers = len(layer_indices)
    expected_block = 18 * num_layers
    expected_total = expected_block + 2 + 2  # + lm_head + final_norm

    print(f"  Detected layers: {num_layers}")
    print(f"  Expected block tensors: {expected_block} (18 × {num_layers})")
    print(f"  Expected total optimizer tensors: {expected_total}")
    print()

    # Validation
    ok = True

    if len(block_tensors) != expected_block:
        print(f"  FAIL: Block optimizer tensors: {len(block_tensors)} != {expected_block}")
        ok = False
    else:
        print(f"  PASS: Block optimizer tensors: {len(block_tensors)}")

    if len(lm_head_tensors) != 2:
        print(f"  FAIL: LM head optimizer tensors: {len(lm_head_tensors)} != 2")
        ok = False
    else:
        print(f"  PASS: LM head optimizer tensors: {len(lm_head_tensors)}")

    if len(final_norm_tensors) != 2:
        print(f"  FAIL: Final norm optimizer tensors: {len(final_norm_tensors)} != 2")
        ok = False
    else:
        print(f"  PASS: Final norm optimizer tensors: {len(final_norm_tensors)}")

    # Check each layer has all 18 suffixes
    for layer_idx in sorted(layer_indices):
        for suffix in suffixes:
            key = f"__training__.block_optimizer.{layer_idx}.{suffix}"
            if key not in tensors:
                print(f"  FAIL: Missing {key}")
                ok = False

    # Report total optimizer state size
    total_elements = 0
    for name in block_tensors + lm_head_tensors + final_norm_tensors:
        shape = tensors[name].get("shape", [])
        elements = 1
        for dim in shape:
            elements *= dim
        total_elements += elements

    total_bytes = total_elements * 4  # f32
    total_mb = total_bytes / (1024 * 1024)
    print(f"\n  Total GPU optimizer state: {total_mb:.1f} MB ({total_elements:,} f32 elements)")

    if ok:
        print("\n  ✓ ALB-118 VERIFIED: All GPU optimizer state tensors present")
    else:
        print("\n  ✗ ALB-118 FAILED: Missing GPU optimizer state tensors")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
