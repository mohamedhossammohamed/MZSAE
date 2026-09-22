#!/usr/bin/env python3
"""
Minimal MLX Example using MZSAEEngine
MZahran Sparse Attention Engine (MZSAE)
"""

import time
import numpy as np
import mlx.core as mx
import mzsae
from mzsae import MZSAEEngine, load_config

def main():
    print(f"=== MZSAE Attention MLX Demo (v{mzsae.__version__}) ===")

    # 1. Load Hardware Profile
    cfg = load_config("auto")
    print(f"Hardware Profile: {cfg.hardware.chip_name} ({cfg.hardware.memory_bandwidth_gbps} GB/s bandwidth)")

    # 2. Instantiate Engine
    num_q_heads = 12
    num_kv_heads = 2
    head_dim = 128

    engine = MZSAEEngine(
        config=cfg,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    print(f"MZSAEEngine initialized with {engine.backend.device_name}")

    # 3. Create Key/Value prefill in MLX
    seq_len = 128
    print(f"\n--- Step 1: Ingest Prefill Sequence via MLX (Length = {seq_len}) ---")
    k_mlx = mx.random.normal((seq_len, num_kv_heads, head_dim)).astype(mx.float16)
    v_mlx = mx.random.normal((seq_len, num_kv_heads, head_dim)).astype(mx.float16)
    mx.eval(k_mlx, v_mlx)

    # Convert to NumPy / zero-copy memory view for MZSAE compressed cache ingestion
    t0 = time.perf_counter()
    engine.cache.ingest_prefill(np.array(k_mlx), np.array(v_mlx))
    t1 = time.perf_counter()
    print(f"Prefill compressed into Dual-Plane Cache in {(t1 - t0)*1000:.2f} ms")
    print(f"Active Plane-1 Compressed Bytes: {engine.plane1_bytes:,} B")
    print(f"Active Plane-2 Sentinel Bytes:   {engine.plane2_bytes:,} B")

    # 4. Single-token decode steps with MLX Query
    print(f"\n--- Step 2: Autoregressive Selective Decode Steps ---")
    for step in range(5):
        q_mlx = mx.random.normal((num_q_heads, head_dim)).astype(mx.float32)
        mx.eval(q_mlx)

        q_np = np.array(q_mlx)
        t_start = time.perf_counter()
        out_np, telemetry = engine.selective_decode(q_np, tau=16.0, return_telemetry=True)
        t_step = time.perf_counter() - t_start

        out_mlx = mx.array(out_np)
        mx.eval(out_mlx)
        print(f"  Step {step + 1}: Output Shape: {out_mlx.shape} | Decode Latency: {t_step*1000:.3f} ms | Pruned: {telemetry['pruning_ratio']*100:.1f}%")

    print("\n[SUCCESS] MLX MZSAE demo completed cleanly.")

if __name__ == "__main__":
    main()
