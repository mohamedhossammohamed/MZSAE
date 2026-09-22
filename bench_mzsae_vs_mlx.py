#!/usr/bin/env python3
"""
bench_mzsae_vs_mlx.py
FIX SET 2: Paired and interleaved timing benchmark of MZSAE Fused Kernel vs MLX SDPA at 64k and 128k.
Runs under GPU lock.
"""
import os
import sys
import time
import json
import ctypes
import numpy as np
import mlx.core as mx

from fastattn_memfix.gpulock import GPULock

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DYLIB_PATH = os.path.join(REPO_DIR, "libmzsae_metal.dylib")
OUT_JSON = os.path.join(REPO_DIR, "logs", "benchmark_fused_vs_mlx.json")

def pack_kv(K, V, num_sinks=4, recent_win=64, block_size=64):
    T, nkv, D = K.shape
    body_tokens = T - num_sinks - recent_win
    num_blocks = (body_tokens + block_size - 1) // block_size
    
    sinks_k = K[:num_sinks].copy()
    sinks_v = V[:num_sinks].copy()
    recent_k = K[T-recent_win:].copy()
    recent_v = V[T-recent_win:].copy()
    
    k_payload = np.zeros((body_tokens, nkv, 64), dtype=np.uint8)
    v_payload = np.zeros((body_tokens, nkv, 64), dtype=np.uint8)
    
    k_centroids = np.zeros((num_blocks, nkv, 32, 4), dtype=np.float16)
    k_scales    = np.zeros((num_blocks, nkv, 32, 4), dtype=np.float16)
    k_mins      = np.zeros((num_blocks, nkv, 32, 4), dtype=np.float16)
    v_group_meta= np.zeros((body_tokens, nkv, 2, 4), dtype=np.float16)
    
    k_body = K[num_sinks:T-recent_win].astype(np.float32)
    v_body = V[num_sinks:T-recent_win].astype(np.float32)
    
    # Pack K
    for b in range(num_blocks):
        b_start = b * block_size
        b_end = min(b_start + block_size, body_tokens)
        blk = k_body[b_start:b_end]
        mu = blk.mean(axis=0)
        res = blk - mu
        res_min = res.min(axis=0)
        res_max = res.max(axis=0)
        scale = np.maximum((res_max - res_min) / 15.0, 1e-8)
        
        mu_h = mu.astype(np.float16)
        scale_h = scale.astype(np.float16)
        res_min_h = res_min.astype(np.float16)
        
        k_centroids[b] = mu_h.reshape(nkv, 32, 4)
        k_scales[b]    = scale_h.reshape(nkv, 32, 4)
        k_mins[b]      = res_min_h.reshape(nkv, 32, 4)
        
        q = np.clip(np.round((res - res_min) / scale), 0, 15).astype(np.uint8)
        for t in range(b_end - b_start):
            q_tok = q[t]
            k_payload[b_start + t] = q_tok[:, 0::2] | (q_tok[:, 1::2] << 4)
            
    # Pack V
    for t in range(body_tokens):
        vt = v_body[t]
        for g in range(2):
            g_vals = vt[:, g*64:(g+1)*64]
            mu_g = g_vals.mean(axis=-1, keepdims=True)
            res_g = g_vals - mu_g
            min_g = res_g.min(axis=-1, keepdims=True)
            max_g = res_g.max(axis=-1, keepdims=True)
            sc_g = np.maximum((max_g - min_g) / 15.0, 1e-8)
            
            v_group_meta[t, :, g, 0] = mu_g.squeeze(-1).astype(np.float16)
            v_group_meta[t, :, g, 1] = min_g.squeeze(-1).astype(np.float16)
            v_group_meta[t, :, g, 2] = sc_g.squeeze(-1).astype(np.float16)
            
            q_g = np.clip(np.round((res_g - min_g) / sc_g), 0, 15).astype(np.uint8)
            v_payload[t, :, g*32:(g+1)*32] = q_g[:, 0::2] | (q_g[:, 1::2] << 4)
            
    return (sinks_k, sinks_v, recent_k, recent_v,
            k_payload, v_payload, k_centroids, k_scales, k_mins, v_group_meta)

def main():
    print("=" * 75)
    print("FIX SET 2: TIMING BENCHMARK - MZSAE FUSED KERNEL VS MLX SDPA")
    print("=" * 75)
    
    with GPULock(tag="bench_v2_timing"):
        lib = ctypes.CDLL(DYLIB_PATH)
        assert lib.mzsae_metal_init(None) == 0
        
        lib.mzsae_metal_clear_cache.argtypes = []
        lib.mzsae_metal_clear_cache.restype = None

        lib.mzsae_metal_fused_decode.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32
        ]
        lib.mzsae_metal_fused_decode.restype = ctypes.c_float
        
        nq, nkv, d = 12, 2, 128
        contexts = [65536, 131072]
        benchmark_results = {}
        
        for seq_len in contexts:
            label = f"{seq_len // 1024}k"
            print(f"\n--- Benchmarking Context: {label} ({seq_len} tokens) ---")
            
            # Clear runtime buffer map cache to ensure no stale address aliasing
            lib.mzsae_metal_clear_cache()

            # Prepare non-trivial data with temporal variance (not compile-away zeros/ones)
            np.random.seed(42 + seq_len)
            K_np = ((np.random.randn(seq_len, nkv, d) * 0.15) + (np.sin(np.arange(seq_len)[:, None, None] * 0.01) * 0.2)).astype(np.float16)
            V_np = ((np.random.randn(seq_len, nkv, d) * 0.15) + (np.cos(np.arange(seq_len)[:, None, None] * 0.01) * 0.2)).astype(np.float16)
            Q_np = ((np.random.randn(nq, d) * 0.15) + 0.05).astype(np.float32)
            
            # MLX setup
            scale = 1.0 / np.sqrt(d)
            q_mx = mx.array(Q_np.reshape(1, nq, 1, d), dtype=mx.float16)
            k_mx = mx.array(K_np.transpose(1, 0, 2).reshape(1, nkv, seq_len, d))
            v_mx = mx.array(V_np.transpose(1, 0, 2).reshape(1, nkv, seq_len, d))
            k_rep = mx.repeat(k_mx, 6, axis=1)
            v_rep = mx.repeat(v_mx, 6, axis=1)
            mx.eval(q_mx, k_rep, v_rep)
            mx.synchronize()
            
            def op_mlx():
                t0 = time.perf_counter()
                out = mx.fast.scaled_dot_product_attention(q_mx, k_rep, v_rep, scale=scale)
                mx.eval(out)
                mx.synchronize()
                return (time.perf_counter() - t0) * 1e6
                
            # MZSAE packed setup
            (sinks_k, sinks_v, recent_k, recent_v,
             k_payload, v_payload, k_centroids, k_scales, k_mins, v_group_meta) = pack_kv(K_np, V_np)
             
            # Verify and print actual buffer byte lengths passed to Metal runtime
            print(f"  [Buffer Scaling Verification] seq_len={seq_len}:")
            print(f"    * k_payload:    {k_payload.nbytes:>10,} bytes ({k_payload.shape})")
            print(f"    * v_payload:    {v_payload.nbytes:>10,} bytes ({v_payload.shape})")
            print(f"    * k_centroids:  {k_centroids.nbytes:>10,} bytes ({k_centroids.shape})")
            print(f"    * k_scales:     {k_scales.nbytes:>10,} bytes ({k_scales.shape})")
            print(f"    * k_mins:       {k_mins.nbytes:>10,} bytes ({k_mins.shape})")
            print(f"    * v_group_meta: {v_group_meta.nbytes:>10,} bytes ({v_group_meta.shape})")

            out_fused = np.zeros((nq, d), dtype=np.float32)
            q_ptr = Q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_ptr = out_fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            def run_mzsae(splits):
                gpu_us = lib.mzsae_metal_fused_decode(
                    q_ptr,
                    ctypes.c_void_p(k_payload.ctypes.data),
                    ctypes.c_void_p(v_payload.ctypes.data),
                    ctypes.c_void_p(k_centroids.ctypes.data),
                    ctypes.c_void_p(k_scales.ctypes.data),
                    ctypes.c_void_p(k_mins.ctypes.data),
                    ctypes.c_void_p(v_group_meta.ctypes.data),
                    ctypes.c_void_p(sinks_k.ctypes.data),
                    ctypes.c_void_p(sinks_v.ctypes.data),
                    ctypes.c_void_p(recent_k.ctypes.data),
                    ctypes.c_void_p(recent_v.ctypes.data),
                    seq_len, splits, out_ptr,
                    ctypes.c_uint32(recent_k.shape[0])
                )
                return float(gpu_us)
                
            # 1. Tune split count among [16, 32, 64, 128]
            print("  Tuning sequence split counts (16, 32, 64, 128)...")
            split_times = {}
            for sp in [16, 32, 64, 128]:
                for _ in range(5): run_mzsae(sp)
                times = [run_mzsae(sp) for _ in range(15)]
                split_times[sp] = np.median(times)
                print(f"    * splits={sp:3d}: median GPU time = {split_times[sp]:.2f} us")
                
            best_splits = min(split_times, key=split_times.get)
            print(f"  Selected best splits count: {best_splits} ({split_times[best_splits]:.2f} us)")
            
            # 2. Interleaved comparison vs MLX (20 warmups, 50 runs)
            print("  Running paired interleaved benchmark vs MLX (20 warmups, 50 timed runs)...")
            for _ in range(20):
                op_mlx()
                run_mzsae(best_splits)
                
            mlx_times = []
            mzsae_times = []
            
            for _ in range(50):
                # Interleaved: MLX then MZSAE
                mlx_us = op_mlx()
                mzsae_us = run_mzsae(best_splits)
                mlx_times.append(mlx_us)
                mzsae_times.append(mzsae_us)
                
            med_mlx = float(np.median(mlx_times))
            med_mzsae = float(np.median(mzsae_times))
            speedup = med_mlx / med_mzsae
            
            print(f"  RESULTS for {label}:")
            print(f"    - MLX SDPA Median:     {med_mlx:8.2f} us ({med_mlx/1000:.3f} ms)")
            print(f"    - MZSAE GPU Median:    {med_mzsae:8.2f} us ({med_mzsae/1000:.3f} ms) [splits={best_splits}]")
            print(f"    - Speedup vs MLX:      {speedup:8.2f}x ({'BEATS MLX' if speedup > 1.0 else 'BEHIND MLX'})")
            
            # Calculate real compression ratio from buffer sizes
            dense_bytes = seq_len * nkv * d * 2 * 2 # K and V in FP16
            body_toks = seq_len - 4 - 64
            n_blocks = (body_toks + 64 - 1) // 64
            compressed_bytes = (
                4 * nkv * d * 2 * 2 + # sinks
                64 * nkv * d * 2 * 2 + # recent
                body_toks * nkv * 64 + # k_payload
                body_toks * nkv * 64 + # v_payload
                n_blocks * nkv * 32 * 4 * 2 * 3 + # k centroids, scales, mins
                body_toks * nkv * 16 # v_group_meta
            )
            comp_ratio = dense_bytes / compressed_bytes
            print(f"    - Buffer Memory:       {dense_bytes / 1024**2:.2f} MB (dense) -> {compressed_bytes / 1024**2:.2f} MB (compressed)")
            print(f"    - Compression Ratio:   {comp_ratio:.2f}x")
            
            benchmark_results[label] = {
                "seq_len": seq_len,
                "best_splits": best_splits,
                "mlx_med_us": round(med_mlx, 2),
                "mzsae_med_us": round(med_mzsae, 2),
                "speedup_vs_mlx": round(speedup, 2),
                "compression_ratio": round(comp_ratio, 2),
                "dense_mb": round(dense_bytes / 1024**2, 2),
                "compressed_mb": round(compressed_bytes / 1024**2, 2)
            }
            
            # Free memory
            del q_mx, k_mx, v_mx, k_rep, v_rep, K_np, V_np, Q_np
            
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\n[benchmark] Saved results to {OUT_JSON}")

if __name__ == "__main__":
    main()
