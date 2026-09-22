"""
MoE Attention Neutrality Tests
Verifies MZSAE works across Mixture-of-Experts (MoE) architectures with shared KV cache.
"""

import numpy as np
from src.mzsae.neutral import (
    MZSAENeutralWrapper,
    StandardAttentionWrapper,
    MoEAttentionWrapper,
)


def test_moe_attention_neutrality():
    """
    MoE models use standard attention with expert-shared KV caches.
    MZSAE operates on attention scores and functions without expert routing modifications.
    """
    num_experts = 8
    batch, seq_len = 1, 256
    heads, kv_heads, head_dim = 32, 8, 128

    # In MoE (Mixtral, DeepSeek), KV cache is shared across experts
    k_shared = np.random.randn(batch, seq_len, kv_heads, head_dim).astype(np.float32)
    v_shared = np.random.randn(batch, seq_len, kv_heads, head_dim).astype(np.float32)

    backend = MoEAttentionWrapper(
        backend=StandardAttentionWrapper(),
        num_experts=num_experts,
        top_k=2,
    )
    mzsae = MZSAENeutralWrapper(backend)

    q = np.random.randn(batch, 1, heads, head_dim).astype(np.float32)
    output = mzsae.decode_step(q, k_shared, v_shared)

    assert output.shape == (batch, 1, heads, head_dim)
    assert np.all(np.isfinite(output))
