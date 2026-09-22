"""
Centroid-Residual Quantization (CRQ) Pipeline & Sentinel Extraction
MZahran Sparse Attention Engine (MZSAE)
"""

import struct
from typing import Optional, Any, Dict, Tuple
import numpy as np
from .rope import decouple_rope_spectrum

BLOCK_SIZE = 32
HEAD_DIM = 128
SLOW_DIMS = 16


def pack_2bit_residuals(codes: np.ndarray) -> np.ndarray:
    """
    Packs 2-bit integer codes into uint8 array (4 codes per byte).
    """
    flat = codes.reshape(-1)
    assert len(flat) % 4 == 0, f"Array length {len(flat)} must be divisible by 4"
    c0 = flat[0::4]
    c1 = flat[1::4]
    c2 = flat[2::4]
    c3 = flat[3::4]
    packed = (c0 & 3) | ((c1 & 3) << 2) | ((c2 & 3) << 4) | ((c3 & 3) << 6)
    return packed.astype(np.uint8)


def unpack_2bit_residuals(packed: np.ndarray, num_elements: int) -> np.ndarray:
    """
    Unpacks uint8 byte stream into 2-bit codes {0, 1, 2, 3}.
    """
    c0 = packed & 3
    c1 = (packed >> 2) & 3
    c2 = (packed >> 4) & 3
    c3 = (packed >> 6) & 3
    unpacked = np.empty((packed.size, 4), dtype=np.uint8)
    unpacked[:, 0] = c0
    unpacked[:, 1] = c1
    unpacked[:, 2] = c2
    unpacked[:, 3] = c3
    return unpacked.reshape(-1)[:num_elements]


def crq_quantize_block(
    keys_unrotated: np.ndarray,
    head_dim: int = HEAD_DIM,
    slow_dims: int = SLOW_DIMS,
    block_size: int = BLOCK_SIZE,
    weight_config: Optional[Any] = None,
) -> dict:
    """
    Quantizes a block of unrotated keys into:
    - FP16 Centroid
    - 2-bit packed residuals
    - Symmetric scale (float16)
    - Slow-RoPE sentinel s_slow, residual radius R_delta, C_fast, sigma2_k
    Adjusts dynamic range and sentinel scaling based on model weight quantization format.
    """
    assert keys_unrotated.shape == (block_size, head_dim), (
        f"Expected shape ({block_size}, {head_dim}), got {keys_unrotated.shape}"
    )

    if weight_config is not None and getattr(weight_config, "is_ternary", lambda: False)():
        centroid = (
            np.mean(keys_unrotated, axis=0)
            * getattr(weight_config, "dynamic_range_scale", 1.0)
        ).astype(np.float32)
    else:
        centroid = np.mean(keys_unrotated, axis=0).astype(np.float32)

    residuals = keys_unrotated - centroid

    if weight_config is not None and getattr(weight_config, "is_low_bit", lambda: False)():
        max_abs = float(np.max(np.abs(residuals))) * getattr(weight_config, "dynamic_range_scale", 1.0)
    else:
        max_abs = float(np.max(np.abs(residuals)))

    scale = max(max_abs / 1.5, 1e-7)

    norm_res = residuals / scale
    codes = np.zeros_like(norm_res, dtype=np.uint8)
    codes[norm_res < -1.0] = 0
    codes[(norm_res >= -1.0) & (norm_res < 0.0)] = 1
    codes[(norm_res >= 0.0) & (norm_res < 1.0)] = 2
    codes[norm_res >= 1.0] = 3

    packed_residuals = pack_2bit_residuals(codes)

    spec = decouple_rope_spectrum(head_dim, slow_dims)
    slow_idx = spec["slow_indices"]
    fast_idx = spec["fast_indices"]

    s_slow = centroid[slow_idx].astype(np.float32)
    res_slow = residuals[:, slow_idx]
    R_delta = float(np.max(np.linalg.norm(res_slow, axis=-1)))

    k_fast = keys_unrotated[:, fast_idx]
    C_fast = float(np.max(np.linalg.norm(k_fast, axis=-1)))

    if weight_config is not None:
        sentinel_scale = getattr(weight_config, "sentinel_scale", 1.0)
        s_slow = (s_slow * sentinel_scale).astype(np.float32)
        R_delta = R_delta * sentinel_scale
        C_fast = C_fast * sentinel_scale

    sigma2_k = float(np.var(residuals))

    return {
        "centroid": centroid,
        "packed_residuals": packed_residuals,
        "scale": np.float16(scale),
        "s_slow": s_slow,
        "R_delta": R_delta,
        "C_fast": C_fast,
        "sigma2_k": sigma2_k,
    }


def crq_dequantize_block(
    centroid: np.ndarray,
    packed_residuals: np.ndarray,
    scale: float,
    head_dim: int = HEAD_DIM,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """
    Reconstructs unrotated keys from centroid and packed residuals.
    """
    codes = unpack_2bit_residuals(packed_residuals, block_size * head_dim)
    codes = codes.reshape(block_size, head_dim)
    norm_res = codes.astype(np.float32) - 1.5
    residuals = norm_res * scale
    return centroid + residuals


def extract_sentinel_descriptor(quantized_block: dict, timestep: int) -> bytes:
    """
    Serializes block metadata into exactly 64 bytes matching MZSAE_SentinelDesc.
    """
    s_slow = quantized_block["s_slow"].astype(np.float16)
    R_delta = float(quantized_block["R_delta"])
    C_fast = float(quantized_block["C_fast"])
    a_cum = 0.0
    t_last = int(timestep)
    sigma2_k = float(quantized_block["sigma2_k"])
    V_pred = 0.0
    flags = 1  # MZSAE_BLOCK_FLAG_ACTIVE

    header = s_slow.tobytes()  # 16 * 2 = 32 bytes
    meta = struct.pack(
        "<fffIffQ", R_delta, C_fast, a_cum, t_last, sigma2_k, V_pred, flags
    )  # 32 bytes
    desc = header + meta

    assert len(desc) == 64, f"Descriptor size must be 64 bytes, got {len(desc)}"
    return desc
