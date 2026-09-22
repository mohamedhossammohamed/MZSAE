#!/usr/bin/env python3
"""
fastattn_memfix/replay_harness.py
Memory-Guarded Replay Harness for Qwen2.5-1.5B on Apple Silicon M4.
Guarantees:
  - 1 model in memory maximum
  - Explicit float16 loading with parameter size check (< 3.5 GB)
  - Hard context cap <= 32768
  - Active background memory watchdog (10 GB abort limit)
  - Per-layer cleanup with gc.collect() and torch.mps.empty_cache()
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import gc
import json
import math
import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastattn_memfix.mem_watchdog import (
    start_memory_watchdog,
    stop_memory_watchdog,
    set_current_operation,
    assert_model_memory_safe,
    assert_context_length_safe,
    cleanup_layer_memory,
    get_current_rss_bytes,
    get_peak_rss_bytes,
    get_current_mps_bytes,
    get_peak_mps_bytes,
)
from fastattn_memfix.gpulock import GPULock

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_DIR, "fastattn_verify4", "model_gguf")
GGUF_FILE = "qwen2.5-1.5b.gguf"
DOC_A = os.path.join(REPO_DIR, "work_docs", "docA.txt")

def quantize_mzsae_cand_f(k: torch.Tensor, v: torch.Tensor, num_sinks: int = 4, recent_win: int = 64):
    """Causal block 64 per-channel K, group 64 V, block centroid, sinks+recent window preserved."""
    T = k.shape[-2]
    if T <= num_sinks + recent_win:
        return k, v
    
    body_start = num_sinks
    body_end = T - recent_win
    k_body = k[..., body_start:body_end, :].clone()
    v_body = v[..., body_start:body_end, :].clone()
    
    # 1. Keys: causal 64-token temporal blocks, per-channel affine residual around block centroid
    k_rec = torch.empty_like(k_body)
    T_b = k_body.shape[-2]
    for s0 in range(0, T_b, 64):
        s1 = min(s0 + 64, T_b)
        blk = k_body[..., s0:s1, :]
        mu = blk.mean(dim=-2, keepdim=True)
        res = blk - mu
        res_min = res.min(dim=-2, keepdim=True).values
        res_max = res.max(dim=-2, keepdim=True).values
        scale = torch.clamp((res_max - res_min) / 15.0, min=1e-8)
        q = torch.clamp(torch.round((res - res_min) / scale), 0, 15)
        k_rec[..., s0:s1, :] = mu + (res_min + q * scale)

    # 2. Values: per-token group 64 affine residual
    orig_v_shape = v_body.shape
    v_reshaped = v_body.view(*orig_v_shape[:-1], 2, 64)
    v_mu = v_reshaped.mean(dim=-1, keepdim=True)
    v_res = v_reshaped - v_mu
    v_min = v_res.min(dim=-1, keepdim=True).values
    v_max = v_res.max(dim=-1, keepdim=True).values
    v_scale = torch.clamp((v_max - v_min) / 15.0, min=1e-8)
    v_q = torch.clamp(torch.round((v_res - v_min) / v_scale), 0, 15)
    v_rec = (v_mu + (v_min + v_q * v_scale)).view(orig_v_shape)

    k_out = k.clone()
    v_out = v.clone()
    k_out[..., body_start:body_end, :] = k_rec
    v_out[..., body_start:body_end, :] = v_rec
    return k_out, v_out

def nll_of(logit_row: torch.Tensor, target_id: int) -> float:
    lp = torch.log_softmax(logit_row.float(), dim=-1)
    return float(-lp[int(target_id)].item())

@torch.inference_mode()
def run_eval_sequence(model, tokens: torch.Tensor, mode: str = "f16", eval_steps: int = 256, chunk_size: int = 1024) -> dict:
    """Evaluates an autoregressive sequence with strict memory guards and chunked prefill."""
    seq_len = len(tokens)
    assert_context_length_safe(seq_len)
    set_current_operation(f"eval_sequence_mode_{mode}_len_{seq_len}")

    prefill_len = seq_len - eval_steps
    assert prefill_len > 64, f"Sequence length {seq_len} too short for {eval_steps} eval steps"

    ids = tokens.unsqueeze(0).to("mps")
    set_current_operation(f"chunked_prefill_mode_{mode}")

    cache = None
    last_logit = None
    for start_idx in range(0, prefill_len, chunk_size):
        end_idx = min(start_idx + chunk_size, prefill_len)
        chunk_ids = ids[:, start_idx:end_idx]
        out = model(input_ids=chunk_ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if end_idx == prefill_len:
            last_logit = out.logits[0, -1, :].clone()
        del out, chunk_ids
        cleanup_layer_memory()

        if start_idx == 0: # just processed first 1024 tokens
            rss_gb = get_current_rss_bytes() / (1024**3)
            mps_gb = get_current_mps_bytes() / (1024**3)
            print(f"[smoke_test] Milestone [AFTER ~1000 TOKENS ({end_idx})]: RSS={rss_gb:.3f} GB, MPS={mps_gb:.3f} GB", flush=True)
        elif start_idx <= 4096 and end_idx >= 4096:
            rss_gb = get_current_rss_bytes() / (1024**3)
            mps_gb = get_current_mps_bytes() / (1024**3)
            print(f"[smoke_test] Milestone [AFTER ~4000 TOKENS ({end_idx})]: RSS={rss_gb:.3f} GB, MPS={mps_gb:.3f} GB", flush=True)

    assert last_logit is not None
    decode_nlls = [nll_of(last_logit, tokens[prefill_len])]
    del last_logit
    cleanup_layer_memory()

    # Compress cache if mode != 'f16'
    if mode != "f16":
        set_current_operation(f"compress_cache_mode_{mode}")
        num_layers = len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)
        for l in range(num_layers):
            if hasattr(cache, "layers"):
                k = cache.layers[l].keys
                v = cache.layers[l].values
            else:
                k = cache.key_cache[l]
                v = cache.value_cache[l]

            orig_dtype = k.dtype
            orig_dev = k.device
            kf = k.detach().cpu().float()
            vf = v.detach().cpu().float()

            kr, vr = quantize_mzsae_cand_f(kf, vf)

            if hasattr(cache, "layers"):
                cache.layers[l].keys = kr.to(device=orig_dev, dtype=orig_dtype)
                cache.layers[l].values = vr.to(device=orig_dev, dtype=orig_dtype)
            else:
                cache.key_cache[l] = kr.to(device=orig_dev, dtype=orig_dtype)
                cache.value_cache[l] = vr.to(device=orig_dev, dtype=orig_dtype)
            
            del kf, vf, kr, vr
            cleanup_layer_memory()

    # Eval decode steps
    set_current_operation(f"decode_steps_mode_{mode}")
    for s in range(1, eval_steps):
        step_inp = ids[:, prefill_len + s - 1:prefill_len + s]
        out = model(input_ids=step_inp, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        dnll = nll_of(out.logits[0, -1], tokens[prefill_len + s])
        decode_nlls.append(dnll)
        del out, step_inp
        if s % 32 == 0:
            cleanup_layer_memory()

    rss_8192_gb = get_current_rss_bytes() / (1024**3)
    mps_8192_gb = get_current_mps_bytes() / (1024**3)
    print(f"[smoke_test] Milestone [AFTER {seq_len} TOKENS (FULL SEQ)]: RSS={rss_8192_gb:.3f} GB, MPS={mps_8192_gb:.3f} GB", flush=True)

    mean_nll = sum(decode_nlls) / len(decode_nlls)
    ppl = math.exp(mean_nll)

    del cache, ids
    cleanup_layer_memory()

    return {
        "mode": mode,
        "seq_len": seq_len,
        "eval_steps": eval_steps,
        "ppl": ppl,
        "mean_nll": mean_nll,
        "peak_rss_gb": get_peak_rss_bytes() / (1024**3)
    }
