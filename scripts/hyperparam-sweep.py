#!/usr/bin/env python3
"""R-027: Hyperparameter sweep infrastructure.

Generates multiple training YAML configs from a sweep specification.
Supports grid search and random search over learning rate, batch size,
warmup steps, weight decay, and other training hyperparameters.

Usage:
    # Grid search over LR and weight decay
    python scripts/hyperparam-sweep.py \
        --base-config configs/train/pretrain-350m.yaml \
        --sweep-lr 1e-4 3e-4 1e-3 \
        --sweep-wd 0.01 0.1 \
        --output-dir configs/sweep/

    # Random search (10 random configs)
    python scripts/hyperparam-sweep.py \
        --base-config configs/train/pretrain-350m.yaml \
        --random 10 \
        --lr-range 1e-5 1e-3 \
        --wd-range 0.01 0.2 \
        --output-dir configs/sweep/

    # Then run each config:
    for cfg in configs/sweep/*.yaml; do
        apr train apply --task pretrain --config "$cfg"
    done

Requirements: pip install pyyaml
"""

import argparse
import itertools
import math
import random
import sys
from pathlib import Path

import yaml


def load_base_config(path: Path) -> dict:
    """Load base YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def update_nested(config: dict, key_path: str, value) -> dict:
    """Update a nested key in a config dict (e.g., 'optimizer.lr')."""
    keys = key_path.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    return config


def grid_search(base_config, sweep_params, output_dir):
    """Generate all combinations from sweep parameters."""
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    configs = []

    for combo in itertools.product(*values):
        config = yaml.safe_load(yaml.dump(base_config))  # deep copy
        name_parts = []
        for key, val in zip(keys, combo):
            update_nested(config, key, val)
            short_key = key.split(".")[-1]
            name_parts.append(f"{short_key}={val}")

        name = "_".join(name_parts)
        # Update output dir to be unique per config
        if "training" in config:
            orig_out = config["training"].get("output_dir", "checkpoints/sweep")
            config["training"]["output_dir"] = f"{orig_out}/{name}"

        out_path = output_dir / f"sweep_{name}.yaml"
        configs.append((out_path, config))

    return configs


def random_search(base_config, param_ranges, n_samples, seed, output_dir):
    """Generate N random configs sampling from parameter ranges."""
    rng = random.Random(seed)
    configs = []

    for i in range(n_samples):
        config = yaml.safe_load(yaml.dump(base_config))  # deep copy
        name_parts = []
        for key, (lo, hi, log_scale) in param_ranges.items():
            if log_scale:
                val = math.exp(rng.uniform(math.log(lo), math.log(hi)))
                val = float(f"{val:.2e}")
            else:
                val = round(rng.uniform(lo, hi), 6)
            update_nested(config, key, val)
            short_key = key.split(".")[-1]
            name_parts.append(f"{short_key}={val}")

        name = f"r{i:03d}_" + "_".join(name_parts)
        if "training" in config:
            orig_out = config["training"].get("output_dir", "checkpoints/sweep")
            config["training"]["output_dir"] = f"{orig_out}/{name}"

        out_path = output_dir / f"sweep_{name}.yaml"
        configs.append((out_path, config))

    return configs


def build_random_ranges(args):
    """Build parameter ranges dict from CLI args."""
    ranges = {}
    if args.lr_range:
        ranges["optimizer.lr"] = (args.lr_range[0], args.lr_range[1], True)
    else:
        ranges["optimizer.lr"] = (1e-5, 1e-3, True)
    if args.wd_range:
        ranges["optimizer.weight_decay"] = (args.wd_range[0], args.wd_range[1], False)
    if args.warmup_range:
        ranges["training.warmup_steps"] = (args.warmup_range[0], args.warmup_range[1], False)
    return ranges


def build_grid_params(args):
    """Build grid parameter dict from CLI args."""
    params = {}
    if args.sweep_lr:
        params["optimizer.lr"] = args.sweep_lr
    if args.sweep_wd:
        params["optimizer.weight_decay"] = args.sweep_wd
    if args.sweep_warmup:
        params["training.warmup_steps"] = args.sweep_warmup
    if args.sweep_batch:
        params["data.batch_size"] = args.sweep_batch
    return params


def write_configs(configs, output_dir):
    """Write generated configs to YAML files."""
    for out_path, config in configs:
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  Generated: {out_path}")
    print(f"\n{len(configs)} configs generated in {output_dir}/")
    print(f"\nTo run sweep:")
    print(f"  for cfg in {output_dir}/*.yaml; do")
    print(f'    apr train apply --task pretrain --config "$cfg"')
    print(f"  done")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate hyperparameter sweep configs")
    parser.add_argument("--base-config", required=True, type=Path, help="Base training YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("configs/sweep"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sweep-lr", nargs="+", type=float, help="Grid: learning rates")
    parser.add_argument("--sweep-wd", nargs="+", type=float, help="Grid: weight decay values")
    parser.add_argument("--sweep-warmup", nargs="+", type=int, help="Grid: warmup steps")
    parser.add_argument("--sweep-batch", nargs="+", type=int, help="Grid: batch sizes")
    parser.add_argument("--random", type=int, default=0, help="Number of random configs")
    parser.add_argument("--lr-range", nargs=2, type=float, help="Random: LR range [lo hi]")
    parser.add_argument("--wd-range", nargs=2, type=float, help="Random: weight decay range")
    parser.add_argument("--warmup-range", nargs=2, type=int, help="Random: warmup steps range")
    return parser.parse_args()


def main():
    args = parse_args()
    base_config = load_base_config(args.base_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.random > 0:
        ranges = build_random_ranges(args)
        configs = random_search(base_config, ranges, args.random, args.seed, args.output_dir)
    else:
        params = build_grid_params(args)
        if not params:
            print("ERROR: No sweep parameters specified. Use --sweep-lr, --sweep-wd, etc.", file=sys.stderr)
            sys.exit(1)
        configs = grid_search(base_config, params, args.output_dir)

    write_configs(configs, args.output_dir)


if __name__ == "__main__":
    main()
