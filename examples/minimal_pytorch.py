#!/usr/bin/env python3
"""
Minimal PyTorch Example using MZSAEAttention
MZahran Sparse Attention Engine (MZSAE)
"""

import time
import torch
import mzsae
from mzsae import MZSAEAttention, load_config

def main():
    print(f"=== MZSAE Attention PyTorch Demo (v{mzsae.__version__}) ===")

    # 1. Load Hardware Configuration
    cfg = load_config("auto")
    print(f"Hardware Profile: {cfg.hardware.chip_name} ({cfg.hardware.memory_bandwidth_gbps} GB/s bandwidth)")

    # 2. Instantiate PyTorch Module
    embed_dim = 1536
    num_heads = 12
    num_kv_heads = 2
    head_dim = 128

    attn = MZSAEAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hardware_profile="auto",
    )
    print(f"MZSAEAttention initialized: {num_heads} query heads, {num_kv_heads} KV heads (GQA)")

    # 3. Simulate Prompt Prefill (64 tokens)
    seq_len = 64
    print(f"\n--- Step 1: Prefill Sequence (Length = {seq_len}) ---")
    x_prefill = torch.randn(1, seq_len, embed_dim)
    t0 = time.perf_counter()
    out_prefill = attn(x_prefill, causal=True, use_cache=True)
    t1 = time.perf_counter()
    print(f"Prefill Output Shape: {out_prefill.shape} | Time: {(t1 - t0)*1000:.2f} ms")

    # 4. Simulate Autoregressive Token Generation (Decode Step)
    print(f"\n--- Step 2: Autoregressive Single-Token Decode Steps ---")
    for step in range(5):
        x_tok = torch.randn(1, 1, embed_dim)
        t_start = time.perf_counter()
        out_tok = attn(x_tok, causal=True, use_cache=True)
        t_step = time.perf_counter() - t_start
        print(f"  Step {step + 1}: Output Shape: {out_tok.shape} | Decode Latency: {t_step*1000:.3f} ms")

    print("\n[SUCCESS] PyTorch MZSAEAttention demo completed cleanly.")

if __name__ == "__main__":
    main()
