"""
Tests for Functional API and 1-Line HuggingFace Model Patcher
MZahran Sparse Attention Engine (MZSAE)
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.mzsae.crq import HEAD_DIM, BLOCK_SIZE
from src.mzsae.functional import MZSAEKVCache, mzsae_with_kvcache
from src.mzsae.hf_patch import patch_model, unpatch_model, patch_attention_module


def test_mzsae_kv_cache_prefill_and_reset():
    num_kv_heads = 2
    cache = MZSAEKVCache(capacity_blocks=64, head_dim=HEAD_DIM, num_kv_heads=num_kv_heads)
    assert cache.total_tokens == 0
    assert cache.total_active_blocks == 0

    seq_len = 96  # Exactly 3 blocks of 32
    k_unrot = np.random.randn(1, seq_len, num_kv_heads, HEAD_DIM).astype(np.float32)
    v = np.random.randn(1, seq_len, num_kv_heads, HEAD_DIM).astype(np.float32)

    cache.ingest_prefill(k_unrot, v)
    assert cache.total_tokens >= 96
    assert cache.total_active_blocks == 3

    cache.reset()
    assert cache.total_tokens == 0
    assert cache.total_active_blocks == 0


def test_mzsae_with_kvcache_numpy():
    num_heads = 4
    num_kv_heads = 2
    cache = MZSAEKVCache(capacity_blocks=64, head_dim=HEAD_DIM, num_kv_heads=num_kv_heads)

    # Prefill 64 tokens
    k_prefill = np.random.randn(1, 64, num_kv_heads, HEAD_DIM).astype(np.float32)
    v_prefill = np.random.randn(1, 64, num_kv_heads, HEAD_DIM).astype(np.float32)
    cache.ingest_prefill(k_prefill, v_prefill)

    # Decode 1 token
    q = np.random.randn(1, 1, num_heads, HEAD_DIM).astype(np.float32)
    k_new = np.random.randn(1, 1, num_kv_heads, HEAD_DIM).astype(np.float32)
    v_new = np.random.randn(1, 1, num_kv_heads, HEAD_DIM).astype(np.float32)

    out = mzsae_with_kvcache(q, k_new, v_new, cache=cache, tau=16.0)
    assert out.shape == (1, 1, num_heads, HEAD_DIM)
    assert np.all(np.isfinite(out))
    assert np.linalg.norm(out) > 0


def test_mzsae_with_kvcache_torch():
    num_heads = 8
    num_kv_heads = 2
    cache = MZSAEKVCache(capacity_blocks=64, head_dim=HEAD_DIM, num_kv_heads=num_kv_heads)

    # Prefill 128 tokens
    k_prefill = torch.randn(1, 128, num_kv_heads, HEAD_DIM, dtype=torch.float32)
    v_prefill = torch.randn(1, 128, num_kv_heads, HEAD_DIM, dtype=torch.float32)
    cache.ingest_prefill(k_prefill, v_prefill)

    # Multiple sequential decode steps
    for step in range(5):
        q = torch.randn(1, 1, num_heads, HEAD_DIM, dtype=torch.float32)
        k_new = torch.randn(1, 1, num_kv_heads, HEAD_DIM, dtype=torch.float32)
        v_new = torch.randn(1, 1, num_kv_heads, HEAD_DIM, dtype=torch.float32)

        out = mzsae_with_kvcache(q, k_new, v_new, cache=cache, tau=16.0)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 1, num_heads, HEAD_DIM)
        assert torch.all(torch.isfinite(out))


class MockAttention(nn.Module):
    """Mock HuggingFace Attention Module (Llama/Qwen style)."""
    def __init__(self, hidden_size=256, num_heads=2, num_kv_heads=1, head_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attn_out = v.repeat(1, 1, self.num_heads // self.num_kv_heads)
        out = self.o_proj(attn_out)
        return (out, None, None)


class MockModel(nn.Module):
    """Mock Transformer Model with attention layers."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockAttention() for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x, _, _ = layer(x)
        return x


def test_patch_and_unpatch_model():
    model = MockModel()
    model.eval()

    # Verify original forward
    prompt = torch.randn(1, 64, 256)
    orig_out = model(prompt)
    assert orig_out.shape == (1, 64, 256)

    # Patch model with MZSAE
    patch_model(model, tau=16.0)

    # Prefill pass (q_len = 64)
    prefill_out = model(prompt)
    assert prefill_out.shape == (1, 64, 256)

    # Decode pass (q_len = 1)
    decode_token = torch.randn(1, 1, 256)
    for _ in range(3):
        decode_out = model(decode_token)
        assert decode_out.shape == (1, 1, 256)
        assert torch.all(torch.isfinite(decode_out))

    # Unpatch and verify restoration
    unpatch_model(model)
    for layer in model.layers:
        assert not hasattr(layer, "_original_forward")

    restored_out = model(prompt)
    assert restored_out.shape == (1, 64, 256)


def test_mzsae_with_kvcache_selective():
    num_heads = 12
    num_kv_heads = 2
    cache = MZSAEKVCache(capacity_blocks=64, head_dim=HEAD_DIM, num_kv_heads=num_kv_heads)

    # Prefill 256 tokens (4 sinks + 128 body [2 blocks] + 124 recent)
    k_prefill = torch.randn(1, 256, num_kv_heads, HEAD_DIM, dtype=torch.float32)
    v_prefill = torch.randn(1, 256, num_kv_heads, HEAD_DIM, dtype=torch.float32)
    cache.ingest_prefill(k_prefill, v_prefill)

    # Decode with selective fetch and telemetry
    q = torch.randn(1, 1, num_heads, HEAD_DIM, dtype=torch.float32)
    k_new = torch.randn(1, 1, num_kv_heads, HEAD_DIM, dtype=torch.float32)
    v_new = torch.randn(1, 1, num_kv_heads, HEAD_DIM, dtype=torch.float32)

    out, tel = mzsae_with_kvcache(q, k_new, v_new, cache=cache, tau=16.0, use_selective=True, return_telemetry=True)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1, num_heads, HEAD_DIM)
    assert torch.all(torch.isfinite(out))
    assert isinstance(tel, dict)
    assert "approved_blocks" in tel
    assert "total_blocks" in tel
    assert "pruning_ratio" in tel

