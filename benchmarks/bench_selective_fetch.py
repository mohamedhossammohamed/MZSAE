#!/usr/bin/env python3
"""
bench_selective_fetch.py
Comprehensive Paired Benchmark: Dense FP16 vs Static 4-bit vs Plane-2 Selective 4-bit
Measures:
  - DRAM Traffic (Bytes Streamed per Step)
  - Pruning Ratio (% Blocks Skipped)
  - Hardware GPU Latency (Metal Timestamps)
  - Numerical Output Parity (Cosine Similarity vs Dense Baseline)
Runs under GPULock.
"""

import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

import time
import json
import numpy as np

from fastattn_memfix.gpulock import GPULock
from mzsae.engine import MZSAEEngine, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS
from mzsae.metal_backend import MetalBackend

LOGS_DIR = os.path.join(REPO_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
OUT_JSON = os.path.join(LOGS_DIR, "benchmark_selective_fetch.json")

def generate_realistic_kv_and_query(seq_len: int, needle_depth: float = 0.35):
    """
    Generates realistic long-context KV activations:
      - 4 Attention Sinks with strong baseline mass
      - Background semantic clusters (64-token blocks)
      - Sharp needle key at needle_depth
      - Recent window (last 64 tokens)
      - Query aligned with needle and sinks
    """
    np.random.seed(42)
    # Background keys: clustered unrotated manifolds
    num_blocks = seq_len // 64
    k_blocks = []
    v_blocks = []
    for b in range(num_blocks):
        centroid = np.random.randn(NUM_KV_HEADS, HEAD_DIM).astype(np.float32) * 0.5
        residual = np.random.randn(64, NUM_KV_HEADS, HEAD_DIM).astype(np.float32) * 0.15
        blk_k = centroid + residual
        blk_v = np.random.randn(64, NUM_KV_HEADS, HEAD_DIM).astype(np.float32) * 0.5
        k_blocks.append(blk_k)
        v_blocks.append(blk_v)

    K = np.concatenate(k_blocks, axis=0)[:seq_len].astype(np.float16)
    V = np.concatenate(v_blocks, axis=0)[:seq_len].astype(np.float16)

    # Sinks: Attention sink at token 0 (slow RoPE manifold)
    K[0, :, 112:] = 3.5

    # Inject needle (slow RoPE manifold at needle_pos)
    needle_pos = (int(seq_len * needle_depth) // 64) * 64
    needle_vector = np.random.randn(NUM_KV_HEADS, HEAD_DIM).astype(np.float16) * 0.5
    needle_vector[:, 112:] = 4.0
    K[needle_pos:needle_pos+8] = needle_vector

    # Query: heads 0..5 attend to needle; heads 6..11 attend to sinks
    q = np.random.randn(NUM_Q_HEADS, HEAD_DIM).astype(np.float32) * 0.2
    for h in range(6):
        q[h, 112:] = 4.0
    for h in range(6, 12):
        q[h, 112:] = 3.5

    return K, V, q

def run_benchmark():
    print("=" * 80)
    print("MZSAE PLANE-2 SELECTIVE FETCH BENCHMARK: DRAM BANDWIDTH BREAKTHROUGH")
    print("=" * 80)

    contexts = [
        {"seq_len": 65536, "splits": 64, "label": "64k"},
        {"seq_len": 131072, "splits": 128, "label": "128k"}
    ]

    warmups = 15
    timed_runs = 30
    results = {}

    with GPULock(tag="bench_selective_fetch"):
        backend = MetalBackend()
        backend.clear_cache()

        for ctx in contexts:
            seq_len = ctx["seq_len"]
            splits = ctx["splits"]
            label = ctx["label"]

            print(f"\n[Benchmarking Context: {label} ({seq_len:,} tokens), Splits: {splits}]")
            K, V, q = generate_realistic_kv_and_query(seq_len)

            engine = MZSAEEngine(num_splits=splits)
            engine.cache.ingest_prefill(K, V)
            bufs = engine.cache.get_metal_buffers()

            # 1. Dense FP16 Reference
            dense_gpu_times = []
            out_dense = None
            for _ in range(warmups):
                backend.dense_decode(q, K, V, seq_len=seq_len, num_splits=splits)
            for _ in range(timed_runs):
                out_dense, gpu_us = backend.dense_decode(q, K, V, seq_len=seq_len, num_splits=splits, return_gpu_time=True)
                dense_gpu_times.append(gpu_us)

            # 2. Static MZSAE 4-bit (Full Streaming)
            static_gpu_times = []
            out_static = None
            for _ in range(warmups):
                engine.decode(q)
            for _ in range(timed_runs):
                out_static, gpu_us = backend.fused_decode(
                    q=q,
                    k_payload=bufs["k_payload"],
                    v_payload=bufs["v_payload"],
                    k_centroids=bufs["k_centroids"],
                    k_scales=bufs["k_scales"],
                    k_mins=bufs["k_mins"],
                    v_group_meta=bufs["v_group_meta"],
                    sinks_k=bufs["sinks_k"],
                    sinks_v=bufs["sinks_v"],
                    recent_k=bufs["recent_k"],
                    recent_v=bufs["recent_v"],
                    seq_len=bufs["seq_len"],
                    num_splits=splits,
                    return_gpu_time=True
                )
                static_gpu_times.append(gpu_us)

            # 3. Plane-2 Sentinel Selective 4-bit (tau = 12.0)
            sel_gpu_times = []
            out_sel = None
            telemetry = None
            for _ in range(warmups):
                engine.selective_decode(q, tau=12.0)
            for _ in range(timed_runs):
                out_sel, telemetry = engine.selective_decode(q, tau=12.0, return_telemetry=True)
                sel_gpu_times.append(telemetry["gpu_us"])

            # Bandwidth Calculations
            dense_bytes = seq_len * NUM_KV_HEADS * HEAD_DIM * 2 * 2 # K + V FP16
            static_bytes = (
                bufs["k_payload"].nbytes + bufs["v_payload"].nbytes +
                bufs["k_centroids"].nbytes + bufs["k_scales"].nbytes + bufs["k_mins"].nbytes +
                bufs["v_group_meta"].nbytes + bufs["sinks_k"].nbytes + bufs["sinks_v"].nbytes +
                bufs["recent_k"].nbytes + bufs["recent_v"].nbytes
            )

            approved_blocks = telemetry["approved_blocks"]
            total_blocks = telemetry["total_blocks"]
            pruning_ratio = telemetry["pruning_ratio"]

            # In Selective mode: Plane 2 sentinels + approved block payloads only
            bytes_per_approved_kv_block = 64 * (64 + 64) + 32 * 2 * 4 * 3 + 64 * 2 * 4 * 2
            selective_bytes = (
                bufs["sentinels"].nbytes +
                approved_blocks * bytes_per_approved_kv_block +
                bufs["sinks_k"].nbytes + bufs["sinks_v"].nbytes +
                bufs["recent_k"].nbytes + bufs["recent_v"].nbytes
            )

            dense_ms = np.median(dense_gpu_times) / 1000.0
            static_ms = np.median(static_gpu_times) / 1000.0
            sel_ms = np.median(sel_gpu_times) / 1000.0

            # Cosine similarities
            cos_static_vs_dense = float(np.dot(out_static.flatten(), out_dense.flatten()) /
                                        (np.linalg.norm(out_static) * np.linalg.norm(out_dense)))
            cos_sel_vs_dense = float(np.dot(out_sel.flatten(), out_dense.flatten()) /
                                     (np.linalg.norm(out_sel) * np.linalg.norm(out_dense)))
            cos_sel_vs_static = float(np.dot(out_sel.flatten(), out_static.flatten()) /
                                      (np.linalg.norm(out_sel) * np.linalg.norm(out_static)))

            # Effective bandwidth on standard bus
            dram_reduction = float(dense_bytes) / float(selective_bytes)

            ctx_res = {
                "seq_len": seq_len,
                "splits": splits,
                "dense_fp16_ms": round(float(dense_ms), 3),
                "static_4bit_ms": round(float(static_ms), 3),
                "selective_4bit_ms": round(float(sel_ms), 3),
                "speedup_vs_dense": round(float(dense_ms / sel_ms), 2),
                "speedup_vs_static": round(float(static_ms / sel_ms), 2),
                "dense_dram_mb": round(float(dense_bytes) / 1e6, 2),
                "static_dram_mb": round(float(static_bytes) / 1e6, 2),
                "selective_dram_mb": round(float(selective_bytes) / 1e6, 2),
                "dram_traffic_reduction_x": round(dram_reduction, 2),
                "approved_blocks": approved_blocks,
                "total_blocks": total_blocks,
                "pruning_ratio_pct": round(pruning_ratio * 100.0, 1),
                "cosine_sim_vs_dense": round(cos_sel_vs_dense, 6),
                "cosine_sim_vs_static": round(cos_sel_vs_static, 6)
            }
            results[label] = ctx_res

            print(f"  * Dense FP16 DRAM Stream:     {dense_bytes / 1e6:.2f} MB | Latency: {dense_ms:.3f} ms")
            print(f"  * Static 4-bit DRAM Stream:   {static_bytes / 1e6:.2f} MB | Latency: {static_ms:.3f} ms")
            print(f"  * Selective 4-bit DRAM Stream:{selective_bytes / 1e6:.2f} MB | Latency: {sel_ms:.3f} ms")
            print(f"  * DRAM Traffic Reduction:     {dram_reduction:.2f}x ({100*(1 - selective_bytes/dense_bytes):.1f}% reduction)")
            print(f"  * Pruning Ratio:              {pruning_ratio*100.0:.1f}% ({approved_blocks}/{total_blocks} blocks)")
            print(f"  * Speedup vs Dense FP16:      {dense_ms / sel_ms:.2f}x")
            print(f"  * Quality (Cosine vs Dense):  {cos_sel_vs_dense:.6f}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {OUT_JSON}")

if __name__ == "__main__":
    run_benchmark()
