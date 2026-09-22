"""
Unit Tests for RoPE Frequency Decoupling and Givens Rotations
Section 4.1 Verification
"""

import pytest
import numpy as np
from src.mzsae.rope import (
    compute_rope_frequencies,
    decouple_rope_spectrum,
    apply_rope_givens,
)

def test_rope_spectrum_properties():
    head_dim = 128
    base = 10000.0
    slow_dims = 16

    spec = decouple_rope_spectrum(head_dim, slow_dims, base)
    
    assert spec["fast_pairs"] == 56
    assert spec["slow_pairs"] == 8
    assert len(spec["slow_indices"]) == 16
    assert len(spec["fast_indices"]) == 112

    # High frequency pair 0 must have frequency 1.0 (base^0 = 1)
    np.testing.assert_allclose(spec["theta_fast"][0], 1.0, rtol=1e-5)

    # Slow frequency pairs must be much smaller than 1.0
    slowest_freq = spec["theta_slow"][-1]
    assert slowest_freq < 0.001, f"Slowest frequency {slowest_freq} must be << 1"

    # Theorem Section 4.1: Across block of size B=32, Delta phi = 31 * theta_j approx 0
    max_slow_delta_phi = 31.0 * float(spec["theta_slow"][0])
    print(f"Max slow manifold phase change across 32 tokens: {max_slow_delta_phi:.4f} rad")
    # For slow manifold, phase change is bounded
    assert max_slow_delta_phi < 1.0

def test_givens_rotations_norm_preservation():
    """RoPE is an orthogonal rotation matrix; vector norm must be strictly preserved."""
    np.random.seed(42)
    x = np.random.randn(10, 128).astype(np.float32)
    positions = np.arange(10, 20)

    original_norms = np.linalg.norm(x, axis=-1)
    x_rot = apply_rope_givens(x, positions)
    rotated_norms = np.linalg.norm(x_rot, axis=-1)

    np.testing.assert_allclose(original_norms, rotated_norms, rtol=1e-5, atol=1e-5)

def test_givens_rotations_at_position_zero():
    """At position 0, angle is 0, cos=1, sin=0, so x_rot must equal x identically."""
    np.random.seed(42)
    x = np.random.randn(5, 128).astype(np.float32)
    positions = np.zeros(5, dtype=np.int32)

    x_rot = apply_rope_givens(x, positions)
    np.testing.assert_allclose(x, x_rot, rtol=1e-6, atol=1e-6)
