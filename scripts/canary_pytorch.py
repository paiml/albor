#!/usr/bin/env python3
"""PyTorch canary — ground truth for albor training convergence.

Answers the question: "Can a 350M LLaMA model reach low val_ppl on
codeparrot-clean with these hyperparameters?" If PyTorch can't do it,
neither can entrenar — the problem is config/data, not the stack.

Usage:
    # Install deps (one-time)
    pip install torch transformers datasets

    # Run canary (GPU, ~30 min for 5K steps)
    python3 scripts/canary_pytorch.py --steps 5000

    # Quick smoke test (CPU, 50 steps)
    python3 scripts/canary_pytorch.py --steps 50 --device cpu

    # Compare with entrenar
    python3 scripts/canary_pytorch.py --steps 5000 --compare logs/v15-training.log

Config mirrors pretrain-350m-v15.yaml exactly:
  - 24 layers, 1024 hidden, 16 heads, 4 KV heads, 4096 FFN
  - AdamW lr=3e-4, beta1=0.9, beta2=0.95, wd=0.1
  - Cosine LR with 2000-step warmup
  - Batch=4, seq=1024, grad_accum=8
  - codeparrot-clean data
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

# Match albor config exactly
CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "intermediate_size": 4096,
    "vocab_size": 32768,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "lr": 3e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "batch_size": 2,  # Reduced from 4 — OOM with 4 even with grad checkpointing
    "seq_len": 1024,
    "grad_accum": 16,  # 2×16 = 32 = same effective batch as 4×8
    "seed": 123,  # Match v15
}

OUTPUT_DIR = "checkpoints/canary-pytorch"


def main():
    parser = argparse.ArgumentParser(description="PyTorch canary for albor")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--compare", type=str, help="Path to entrenar log for comparison")
    parser.add_argument("--data-dir", default="data/pretokenized-1024-v3/train/")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default: use CONFIG)")
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW

    seed = args.seed if args.seed is not None else CONFIG["seed"]
    torch.manual_seed(seed)
    print(f"Seed: {seed}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(args.device)

    # --- Build LLaMA model from scratch ---
    print(f"Building 350M LLaMA model...")
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(
            hidden_size=CONFIG["hidden_size"],
            num_hidden_layers=CONFIG["num_hidden_layers"],
            num_attention_heads=CONFIG["num_attention_heads"],
            num_key_value_heads=CONFIG["num_key_value_heads"],
            intermediate_size=CONFIG["intermediate_size"],
            vocab_size=CONFIG["vocab_size"],
            max_position_embeddings=CONFIG["max_position_embeddings"],
            rms_norm_eps=CONFIG["rms_norm_eps"],
            rope_theta=CONFIG["rope_theta"],
            tie_word_embeddings=True,  # Match entrenar (tied lm_head)
        )
        model = LlamaForCausalLM(config)
        model.gradient_checkpointing_enable()
        model = model.to(device)
    except ImportError:
        print("ERROR: pip install transformers torch")
        return

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} ({params/1e6:.0f}M)")

    # --- Load data ---
    print(f"Loading data from {args.data_dir}...")
    try:
        import pyarrow.parquet as pq
        data_path = Path(args.data_dir)
        shards = sorted(data_path.glob("*.parquet"))
        if not shards:
            print(f"ERROR: No parquet files in {args.data_dir}")
            return
        # Load first shard only for canary
        table = pq.read_table(shards[0])
        col = "input_ids" if "input_ids" in table.column_names else "token_ids"
        sequences = [row.as_py() for row in table[col]]
        print(f"  Loaded {len(sequences)} sequences from {shards[0].name}")

        # Load held-out val set
        val_dir = Path(args.data_dir).parent / "val"
        val_shards = sorted(val_dir.glob("*.parquet"))
        if val_shards:
            val_table = pq.read_table(val_shards[0])
            val_col = "input_ids" if "input_ids" in val_table.column_names else "token_ids"
            val_sequences = [row.as_py() for row in val_table[val_col]]
            print(f"  Loaded {len(val_sequences)} val sequences from {val_shards[0].name}")
        else:
            val_sequences = sequences[:1000]  # fallback: first 1K train seqs
            print(f"  No val dir found, using first 1K train sequences")
    except ImportError:
        print("ERROR: pip install pyarrow")
        return

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        betas=(CONFIG["beta1"], CONFIG["beta2"]),
        weight_decay=CONFIG["weight_decay"],
    )

    # --- Cosine LR schedule ---
    def get_lr(step):
        if step < CONFIG["warmup_steps"]:
            return CONFIG["lr"] * step / CONFIG["warmup_steps"]
        progress = (step - CONFIG["warmup_steps"]) / max(1, args.steps - CONFIG["warmup_steps"])
        return CONFIG["lr"] * 0.1 + 0.5 * CONFIG["lr"] * 0.9 * (1 + math.cos(math.pi * progress))

    # --- Training loop ---
    print(f"Training for {args.steps} steps (eval every {args.eval_interval})...")
    log = []
    batch_idx = 0
    model.train()
    start = time.time()

    for step in range(args.steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(CONFIG["grad_accum"]):
            # Get batch
            batch_seqs = []
            for _ in range(CONFIG["batch_size"]):
                seq = sequences[batch_idx % len(sequences)][:CONFIG["seq_len"]]
                if len(seq) < CONFIG["seq_len"]:
                    seq = seq + [0] * (CONFIG["seq_len"] - len(seq))
                batch_seqs.append(seq)
                batch_idx += 1

            input_ids = torch.tensor(batch_seqs, device=device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / CONFIG["grad_accum"]
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # LR schedule
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()

        # Log
        if step % 100 == 0 or step == args.steps - 1:
            elapsed = time.time() - start
            toks = (step + 1) * CONFIG["batch_size"] * CONFIG["grad_accum"] * CONFIG["seq_len"]
            tok_s = toks / max(elapsed, 1)
            entry = {"step": step, "loss": total_loss, "lr": lr, "tok_s": tok_s}
            log.append(entry)
            print(f"  step={step:>5d} loss={total_loss:.4f} lr={lr:.2e} tok/s={tok_s:.0f}", flush=True)

        # Eval
        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                eval_seqs = val_sequences  # Use held-out val set
                for vi in range(min(250, len(eval_seqs) // CONFIG["batch_size"])):
                    vseqs = []
                    for _ in range(CONFIG["batch_size"]):
                        seq = eval_seqs[(vi * CONFIG["batch_size"] + _) % len(eval_seqs)][:CONFIG["seq_len"]]
                        if len(seq) < CONFIG["seq_len"]:
                            seq = seq + [0] * (CONFIG["seq_len"] - len(seq))
                        vseqs.append(seq)
                    ids = torch.tensor(vseqs, device=device)
                    out = model(input_ids=ids, labels=ids)
                    val_loss += out.loss.item()
                    val_batches += 1
            avg_val = val_loss / max(val_batches, 1)
            val_ppl = math.exp(min(avg_val, 20))
            print(f"  [EVAL] step={step} val_loss={avg_val:.4f} val_ppl={val_ppl:.1f}", flush=True)
            log.append({"step": step, "val_loss": avg_val, "val_ppl": val_ppl, "type": "eval"})
            model.train()

    # Save log
    log_path = Path(OUTPUT_DIR) / "canary_log.jsonl"
    with open(log_path, "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")
    print(f"\nLog saved to {log_path}")

    # Compare with entrenar if requested
    if args.compare:
        print(f"\n=== Comparison with {args.compare} ===")
        canary_evals = [e for e in log if e.get("type") == "eval"]
        # Parse entrenar log
        entrenar_evals = []
        with open(args.compare) as f:
            for line in f:
                if "[eval]" in line and "val_ppl=" in line:
                    step = int(line.split("step=")[1].split()[0])
                    ppl = float(line.split("val_ppl=")[1].split()[0])
                    entrenar_evals.append({"step": step, "val_ppl": ppl})

        print(f"{'Step':>6} {'PyTorch ppl':>12} {'Entrenar ppl':>13} {'Delta':>8}")
        for ce in canary_evals:
            s = ce["step"]
            ee = next((e for e in entrenar_evals if e["step"] == s), None)
            if ee:
                delta = (ee["val_ppl"] - ce["val_ppl"]) / ce["val_ppl"] * 100
                print(f"{s:>6d} {ce['val_ppl']:>12.1f} {ee['val_ppl']:>13.1f} {delta:>+7.1f}%")


if __name__ == "__main__":
    main()
