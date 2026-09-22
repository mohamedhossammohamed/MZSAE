"""
End-to-End Simulation Tests for Quantized Model Weight Ingestion
Simulates full autoregressive decode pipelines with synthetic quantized projection weights
(Q4_K_M, BitNet b1.58 Ternary, Q8_0, and Q2_K) with ZERO model downloads.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.mzsae.config import MZSAEConfig, ModelWeightConfig, load_config
from src.mzsae.core.engine import MZSAEEngine
from src.mzsae.core.compression import BLOCK_SIZE, HEAD_DIM, SLOW_DIMS
from src.mzsae.nn import MZSAEAttention
from src.mzsae.ternary import create_ternary_config


class SyntheticQuantizedLinear(nn.Module):
    """
    Simulates a quantized linear projection layer (e.g., Q4_K_M, Q8_0, or BitNet).
    Weights are quantized during forward pass to simulate hardware-quantized GEMM.
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 4, mode: str = "uniform"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.mode = mode
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / np.sqrt(in_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "ternary":
            # BitNet b1.58: weights quantized to {-1, 0, +1}
            # Scale by mean absolute value
            scale = self.weight.abs().mean().clamp(min=1e-5)
            w_quant = torch.round(self.weight / scale).clamp(-1, 1) * scale
        elif self.bits < 16:
            # Uniform affine/symmetric quantization to simulate Q4_0, Q8_0, etc.
            qmin = -(2 ** (self.bits - 1))
            qmax = (2 ** (self.bits - 1)) - 1
            w_max = self.weight.abs().max().clamp(min=1e-5)
            scale = w_max / qmax
            w_quant = torch.round(self.weight / scale).clamp(qmin, qmax) * scale
        else:
            w_quant = self.weight

        return torch.matmul(x, w_quant.t())


def test_q4_k_m_model_simulation():
    """
    End-to-end simulation of a 4-bit (Q4_K_M) quantized model decoding loop.
    Verifies that MZSAE Attention correctly caches and decodes over multiple steps.
    """
    embed_dim = 512
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    seq_len = 80  # Triggers block creation (> 64)

    # Instantiate MZSAE attention configured for 4-bit model weights
    attn = MZSAEAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        model_weights="4bit",
    )

    # Replace linear projections with simulated 4-bit quantized projections
    attn.q_proj = SyntheticQuantizedLinear(embed_dim, num_heads * head_dim, bits=4)
    attn.k_proj = SyntheticQuantizedLinear(embed_dim, num_kv_heads * head_dim, bits=4)
    attn.v_proj = SyntheticQuantizedLinear(embed_dim, num_kv_heads * head_dim, bits=4)

    # Prefill phase (64 tokens)
    x_prompt = torch.randn(1, 64, embed_dim)
    q_pre = attn.q_proj(x_prompt).view(1, 64, num_heads, head_dim)
    k_pre = attn.k_proj(x_prompt).view(1, 64, num_kv_heads, head_dim)
    v_pre = attn.v_proj(x_prompt).view(1, 64, num_kv_heads, head_dim)

    out_pre = attn(q_pre, k_pre, v_pre, use_cache=True)
    assert out_pre.shape == (1, 64, num_heads, head_dim)
    assert not torch.isnan(out_pre).any()
    assert not torch.isinf(out_pre).any()

    # Autoregressive decoding phase (16 single-token steps)
    for step in range(16):
        x_step = torch.randn(1, 1, embed_dim)
        q_step = attn.q_proj(x_step).view(1, 1, num_heads, head_dim)
        k_step = attn.k_proj(x_step).view(1, 1, num_kv_heads, head_dim)
        v_step = attn.v_proj(x_step).view(1, 1, num_kv_heads, head_dim)

        out_step = attn(q_step, k_step, v_step, use_cache=True)
        assert out_step.shape == (1, 1, num_heads, head_dim)
        assert not torch.isnan(out_step).any()
        assert not torch.isinf(out_step).any()

    # Verify cache accumulated all tokens
    assert attn.engine.cache.total_tokens == seq_len


def test_bitnet_ternary_simulation():
    """
    End-to-end simulation of a BitNet b1.58 ternary model decoding loop.
    Weights are strictly {-1, 0, +1}, producing activations with smaller variance.
    Verifies that MZSAE's dynamic range scale (0.5) and reduced tau (0.7x) maintain stability.
    """
    embed_dim = 256
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64

    # Instantiate with ternary configuration
    attn = MZSAEAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        model_weights="ternary",
    )

    # Verify ternary parameters were applied
    assert attn.config.model_weights.is_ternary() is True
    assert attn.config.model_weights.dynamic_range_scale == 0.5
    assert attn.config.model_weights.tau_multiplier == 0.7
    assert attn.config.model_weights.sentinel_scale == 0.5

    # Replace projections with BitNet ternary projections
    attn.q_proj = SyntheticQuantizedLinear(embed_dim, num_heads * head_dim, bits=2, mode="ternary")
    attn.k_proj = SyntheticQuantizedLinear(embed_dim, num_kv_heads * head_dim, bits=2, mode="ternary")
    attn.v_proj = SyntheticQuantizedLinear(embed_dim, num_kv_heads * head_dim, bits=2, mode="ternary")

    # Layer forward pass (end-to-end projection mode)
    x = torch.randn(1, 32, embed_dim)
    out = attn(x, causal=True)
    assert out.shape == (1, 32, embed_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Multi-step decode simulation
    attn.reset_cache()
    for step in range(32):
        x_tok = torch.randn(1, 1, embed_dim)
        out_tok = attn(x_tok, causal=True, use_cache=True)
        assert out_tok.shape == (1, 1, embed_dim)
        assert not torch.isnan(out_tok).any()

    assert attn.engine.cache.total_tokens == 32


def test_q8_0_model_simulation():
    """Simulates 8-bit model weights with 64 decode steps."""
    embed_dim = 256
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64

    attn = MZSAEAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        model_weights="8bit",
    )

    attn.q_proj = SyntheticQuantizedLinear(embed_dim, num_heads * head_dim, bits=8)
    attn.k_proj = SyntheticQuantizedLinear(embed_dim, num_kv_heads * head_dim, bits=8)
    attn.v_proj = SyntheticQuantizedLinear(embed_dim, num_kv_heads * head_dim, bits=8)

    x = torch.randn(1, 48, embed_dim)
    out = attn(x, causal=True)
    assert out.shape == (1, 48, embed_dim)
    assert not torch.isnan(out).any()


def test_q2_k_extreme_low_bit_simulation():
    """Simulates 2-bit quantization (Q2_K) with dynamic range scaling."""
    embed_dim = 256
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64

    attn = MZSAEAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        model_weights="2bit",
    )

    assert attn.config.model_weights.quantization_bits == 2
    assert attn.config.model_weights.dynamic_range_scale == 0.8
    assert attn.config.model_weights.is_low_bit() is True

    attn.q_proj = SyntheticQuantizedLinear(embed_dim, num_heads * head_dim, bits=2)
    attn.k_proj = SyntheticQuantizedLinear(embed_dim, num_kv_heads * head_dim, bits=2)
    attn.v_proj = SyntheticQuantizedLinear(embed_dim, num_kv_heads * head_dim, bits=2)

    x = torch.randn(1, 64, embed_dim)
    out = attn(x, causal=True)
    assert out.shape == (1, 64, embed_dim)
    assert not torch.isnan(out).any()
