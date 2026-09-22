"""
RoPE Frequency Spectrum Decoupling & In-Flight Givens Rotations
MZahran Sparse Attention Engine (MZSAE)
"""

import numpy as np


def compute_rope_frequencies(head_dim: int, base: float = 10000.0) -> np.ndarray:
    """
    Computes theta_j = base^{-2j / head_dim} for j in {0, ..., head_dim/2 - 1}.
    """
    assert head_dim % 2 == 0, f"Head dimension {head_dim} must be even"
    j = np.arange(head_dim // 2, dtype=np.float64)
    theta = np.power(base, -2.0 * j / float(head_dim))
    return theta.astype(np.float32)


def decouple_rope_spectrum(
    head_dim: int = 128, slow_dims: int = 16, base: float = 10000.0
) -> dict:
    """
    Partitions the head_dim vector space into fast and slow frequency subspaces.
    Returns:
      theta: full frequencies
      theta_fast: frequencies for fast pairs (first (head_dim - slow_dims)//2 pairs)
      theta_slow: frequencies for slow pairs (last slow_dims//2 pairs)
      slow_indices: coordinate indices for slow subspace
      fast_indices: coordinate indices for fast subspace
      slow_pairs: count of slow pairs
      fast_pairs: count of fast pairs
    """
    theta = compute_rope_frequencies(head_dim, base)
    num_pairs = head_dim // 2
    slow_pairs = slow_dims // 2
    fast_pairs = num_pairs - slow_pairs

    theta_fast = theta[:fast_pairs]
    theta_slow = theta[fast_pairs:]

    fast_indices = np.arange(0, fast_pairs * 2)
    slow_indices = np.arange(fast_pairs * 2, head_dim)

    return {
        "theta": theta,
        "theta_fast": theta_fast,
        "theta_slow": theta_slow,
        "fast_indices": fast_indices,
        "slow_indices": slow_indices,
        "slow_pairs": slow_pairs,
        "fast_pairs": fast_pairs,
    }


def apply_rope_givens(
    x: np.ndarray, positions: np.ndarray, base: float = 10000.0
) -> np.ndarray:
    """
    Applies Givens rotations to vectors x at given positions.
    Args:
      x: array of shape (..., head_dim)
      positions: array of shape (..., ) corresponding to token indices
      base: RoPE base
    Returns:
      Rotated vector with identical shape.
    """
    head_dim = x.shape[-1]
    theta = compute_rope_frequencies(head_dim, base)

    pos = np.asarray(positions, dtype=np.float32)[..., np.newaxis]
    angles = pos * theta
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)

    x_rot = np.empty_like(x, dtype=np.float32)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]

    x_rot[..., 0::2] = x0 * cos_a - x1 * sin_a
    x_rot[..., 1::2] = x0 * sin_a + x1 * cos_a

    return x_rot
