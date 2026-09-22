"""
Unit Tests for Centroid-Residual Quantization (CRQ)
Section 4.2 & Brick 2 Verification
"""

import pytest
import numpy as np
from src.mzsae.crq import (
    crq_quantize_block,
    crq_dequantize_block,
    pack_2bit_residuals,
    unpack_2bit_residuals,
    extract_sentinel_descriptor,
    BLOCK_SIZE,
    HEAD_DIM
)

def test_pack_unpack_bit_exactness():
    np.random.seed(42)
    original_codes = np.random.randint(0, 4, size=(BLOCK_SIZE, HEAD_DIM), dtype=np.uint8)
    packed = pack_2bit_residuals(original_codes)
    assert packed.shape == (1024,)
    assert packed.dtype == np.uint8

    unpacked = unpack_2bit_residuals(packed, BLOCK_SIZE * HEAD_DIM)
    unpacked = unpacked.reshape(BLOCK_SIZE, HEAD_DIM)
    np.testing.assert_array_equal(original_codes, unpacked)

def test_crq_reconstruction_bounded_error():
    np.random.seed(42)
    # Simulate clustered key vectors within a 32-token semantic block
    base_centroid = np.random.randn(HEAD_DIM).astype(np.float32)
    residuals = np.random.randn(BLOCK_SIZE, HEAD_DIM).astype(np.float32) * 0.2
    keys_unrotated = base_centroid + residuals

    q_block = crq_quantize_block(keys_unrotated)
    keys_rec = crq_dequantize_block(
        q_block["centroid"],
        q_block["packed_residuals"],
        float(q_block["scale"]),
        HEAD_DIM
    )

    # 2-bit quantization has maximum normalized residual error of 0.5 * scale
    max_err = np.max(np.abs(keys_unrotated - keys_rec))
    expected_bound = 0.5 * float(q_block["scale"]) + 1e-4
    assert max_err <= expected_bound, f"Reconstruction error {max_err} exceeded bound {expected_bound}"

    # Centroid must match the mean exactly
    np.testing.assert_allclose(q_block["centroid"], np.mean(keys_unrotated, axis=0), rtol=1e-5)

def test_descriptor_exact_64_bytes():
    np.random.seed(42)
    keys = np.random.randn(BLOCK_SIZE, HEAD_DIM).astype(np.float32)
    q_block = crq_quantize_block(keys)
    desc = extract_sentinel_descriptor(q_block, timestep=42)
    assert len(desc) == 64, f"Plane 2 descriptor must be exactly 64 bytes (1 cache line), got {len(desc)}"
