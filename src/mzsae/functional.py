"""
Functional API: Drop-in Replacement for flash_attn_with_kvcache
MZahran Sparse Attention Engine (MZSAE)
Author: Mohammed Hossam Zahran
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .engine import MZSAEKVCache
from .metal_backend import MetalBackend

def mzsae_with_kvcache(
    q: Any,                     # [batch=1, 1, num_heads, head_dim] or [num_heads, head_dim]
    k_unrot_new: Any,           # [batch=1, 1, num_kv_heads, head_dim] or [num_kv_heads, head_dim]
    v_new: Any,                 # [batch=1, 1, num_kv_heads, head_dim] or [num_kv_heads, head_dim]
    cache: MZSAEKVCache,
    tau: float = 16.0,
    rope_base: float = 1000000.0,
    metal_backend: Optional[MetalBackend] = None,
    num_splits: int = 64,
    use_selective: bool = True,
    return_telemetry: bool = False
) -> Any:
    """
    Drop-in functional operator equivalent to flash_attn_with_kvcache.
    Executes Fused Multi-Split GQA Attention decode or Plane-2 Sentinel Selective Fetch on Apple Silicon Metal GPU.
    """
    is_torch = False
    device = "cpu"
    dtype = None

    if HAS_TORCH and isinstance(q, torch.Tensor):
        is_torch = True
        device = q.device
        dtype = q.dtype
        q_np = q.detach().cpu().to(torch.float32).numpy()
        k_np = k_unrot_new.detach().cpu().to(torch.float16).numpy()
        v_np = v_new.detach().cpu().to(torch.float16).numpy()
    else:
        q_np = np.asarray(q, dtype=np.float32)
        k_np = np.asarray(k_unrot_new, dtype=np.float16)
        v_np = np.asarray(v_new, dtype=np.float16)

    orig_shape = q_np.shape
    if q_np.ndim == 4:
        q_np = q_np[0, 0]
        k_np = k_np[0, 0]
        v_np = v_np[0, 0]
    elif q_np.ndim == 3:
        q_np = q_np[0]
        k_np = k_np[0]
        v_np = v_np[0]

    # Append new incoming KV token to cache
    cache.append_kv(k_np, v_np)

    if metal_backend is None:
        metal_backend = MetalBackend()

    bufs = cache.get_metal_buffers()
    has_body_blocks = bufs["sentinels"].shape[0] > 0

    if use_selective and has_body_blocks:
        out_arr, telemetry = metal_backend.selective_decode(
            q=q_np,
            k_payload=bufs["k_payload"],
            v_payload=bufs["v_payload"],
            sentinels=bufs["sentinels"],
            k_centroids=bufs["k_centroids"],
            k_scales=bufs["k_scales"],
            k_mins=bufs["k_mins"],
            v_group_meta=bufs["v_group_meta"],
            sinks_k=bufs["sinks_k"],
            sinks_v=bufs["sinks_v"],
            recent_k=bufs["recent_k"],
            recent_v=bufs["recent_v"],
            seq_len=bufs["seq_len"],
            num_splits=num_splits,
            tau=tau
        )
    else:
        out_arr = metal_backend.fused_decode(
            q=q_np,
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
            num_splits=num_splits
        )
        telemetry = {"gpu_us": 0.0, "approved_blocks": 0, "total_blocks": 0, "pruning_ratio": 0.0}

    if len(orig_shape) == 4:
        out_arr = out_arr[np.newaxis, np.newaxis, ...]
    elif len(orig_shape) == 3:
        out_arr = out_arr[np.newaxis, ...]

    if is_torch:
        out_val = torch.from_numpy(out_arr).to(device=device, dtype=dtype)
    else:
        out_val = out_arr

    if return_telemetry:
        return out_val, telemetry
    return out_val
