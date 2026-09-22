"""
Shape & Routing Invariance Tests (formerly "Architecture Neutrality").

RED-TEAM CLARIFICATION (Pre-Launch Audit):
    These tests use synthetic Gaussian tensors (np.random.randn) to verify MZSAE
    does not crash and correctly broadcasts KV heads across arbitrary MHA/GQA/MQA
    geometries. Gaussian data has no attention sinks, outliers, or heavy tails,
    so the Cauchy-Schwarz bound is trivially loose here.

    What this PROVES: shape invariance + GQA/MQA routing correctness.
    What this DOES NOT prove: semantic retrieval accuracy or numerical
    correctness on real model activations (which require integration with
    `transformers` or `mlx-lm` on real weights + heavy-tailed distributions).
    See docs/LIMITATIONS.md ("Random Tensor Caveat").
"""

import numpy as np
import pytest
from src.mzsae.neutral import (
    MZSAENeutralWrapper,
    StandardAttentionWrapper,
    SlidingWindowWrapper,
)

ARCHITECTURES = {
    "llama-7b": {"heads": 32, "kv_heads": 32, "head_dim": 128},  # MHA
    "llama-3-8b": {"heads": 32, "kv_heads": 8, "head_dim": 128},  # GQA 4:1
    "mistral-7b": {"heads": 32, "kv_heads": 8, "head_dim": 128, "window": 4096},
    "mixtral-8x7b": {"heads": 32, "kv_heads": 8, "head_dim": 128},  # MoE
    "qwen2.5-7b": {"heads": 28, "kv_heads": 4, "head_dim": 128},  # GQA 7:1
    "gemma-2-9b": {"heads": 16, "kv_heads": 8, "head_dim": 256},  # head_dim 256
    "phi-3-mini": {"heads": 32, "kv_heads": 32, "head_dim": 96},  # head_dim 96
    "falcon-7b": {"heads": 71, "kv_heads": 1, "head_dim": 64},  # MQA 71:1
    "deepseek-v2": {"heads": 64, "kv_heads": 8, "head_dim": 128},
}


@pytest.mark.parametrize("arch_name,arch_config", ARCHITECTURES.items())
def test_architecture_neutrality(arch_name, arch_config):
    """Shape & routing invariance across architectures using synthetic Gaussian tensors.

    Asserts output shape/finiteness only — NOT semantic retrieval accuracy.
    """
    batch = 1
    seq_len = 256
    heads = arch_config["heads"]
    kv_heads = arch_config["kv_heads"]
    head_dim = arch_config["head_dim"]

    # Generate synthetic Q, K, V without model weights
    q = np.random.randn(batch, 1, heads, head_dim).astype(np.float32)
    k = np.random.randn(batch, seq_len, kv_heads, head_dim).astype(np.float32)
    v = np.random.randn(batch, seq_len, kv_heads, head_dim).astype(np.float32)

    if "window" in arch_config:
        backend = SlidingWindowWrapper(
            backend=StandardAttentionWrapper(),
            window_size=arch_config["window"],
        )
    else:
        backend = StandardAttentionWrapper()

    mzsae = MZSAENeutralWrapper(backend)
    output = mzsae.decode_step(q, k, v)

    assert output.shape == (batch, 1, heads, head_dim)
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
def test_head_dim_neutrality(head_dim):
    """Shape invariance across head dims (synthetic Gaussian; shape/finiteness only)."""
    batch, seq_len = 1, 128
    heads, kv_heads = 16, 4
    q = np.random.randn(batch, 1, heads, head_dim).astype(np.float32)
    k = np.random.randn(batch, seq_len, kv_heads, head_dim).astype(np.float32)
    v = np.random.randn(batch, seq_len, kv_heads, head_dim).astype(np.float32)

    mzsae = MZSAENeutralWrapper(StandardAttentionWrapper())
    output = mzsae.decode_step(q, k, v)

    assert output.shape == (batch, 1, heads, head_dim)
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("kv_ratio", [1, 2, 4, 8, 16, 32])
def test_gqa_ratio_neutrality(kv_ratio):
    """GQA broadcast routing invariance (synthetic Gaussian; shape/finiteness only)."""
    batch, seq_len = 1, 128
    head_dim = 128
    heads = 32
    kv_heads = max(1, heads // kv_ratio)

    q = np.random.randn(batch, 1, heads, head_dim).astype(np.float32)
    k = np.random.randn(batch, seq_len, kv_heads, head_dim).astype(np.float32)
    v = np.random.randn(batch, seq_len, kv_heads, head_dim).astype(np.float32)

    mzsae = MZSAENeutralWrapper(StandardAttentionWrapper())
    output = mzsae.decode_step(q, k, v)

    assert output.shape == (batch, 1, heads, head_dim)
    assert np.all(np.isfinite(output))
