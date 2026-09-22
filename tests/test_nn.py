"""
Tests for PyTorch MZSAEAttention Module
"""

import pytest
import torch
from src.mzsae.nn import MZSAEAttention
from src.mzsae.config import load_config


def test_mzsae_attention_layer_forward():
    attn = MZSAEAttention(
        embed_dim=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        hardware_profile="auto",
    )
    x = torch.randn(1, 16, 256)
    out = attn(x, causal=True)
    assert out.shape == (1, 16, 256)
    assert not torch.isnan(out).any()


def test_mzsae_attention_functional_operator():
    attn = MZSAEAttention(
        embed_dim=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        hardware_profile="auto",
    )
    q = torch.randn(1, 16, 4, 64)
    k = torch.randn(1, 16, 2, 64)
    v = torch.randn(1, 16, 2, 64)

    out = attn(q, k, v, causal=True)
    assert out.shape == (1, 16, 4, 64)
    assert not torch.isnan(out).any()


def test_mzsae_attention_single_token_decode():
    attn = MZSAEAttention(
        embed_dim=1536,
        num_heads=12,
        num_kv_heads=2,
        head_dim=128,
        hardware_profile="auto",
    )
    # Prefill 64 tokens into cache
    k_prefill = torch.randn(1, 64, 2, 128)
    v_prefill = torch.randn(1, 64, 2, 128)
    q_prefill = torch.randn(1, 64, 12, 128)
    _ = attn(q_prefill, k_prefill, v_prefill, use_cache=True)

    # Decode 1 token
    q_dec = torch.randn(1, 1, 12, 128)
    k_dec = torch.randn(1, 1, 2, 128)
    v_dec = torch.randn(1, 1, 2, 128)
    out_dec = attn(q_dec, k_dec, v_dec, use_cache=True)
    assert out_dec.shape == (1, 1, 12, 128)
    assert not torch.isnan(out_dec).any()
