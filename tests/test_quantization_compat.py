"""
Unit & Compatibility Tests for Model Weight Quantization Support
Verifies MZSAE compatibility across FP16, BF16, 8-bit, 4-bit, 3-bit, 2-bit, and Ternary BitNet b1.58.
All tests run with zero model downloads using synthetic scaled tensors.
"""

import pytest
import numpy as np
import torch

from src.mzsae.config import (
    ModelWeightConfig,
    MZSAEConfig,
    load_config,
)
from src.mzsae.core.compression import (
    crq_quantize_block,
    crq_dequantize_block,
    extract_sentinel_descriptor,
    BLOCK_SIZE,
    HEAD_DIM,
    SLOW_DIMS,
)
from src.mzsae.core.engine import MZSAEEngine
from src.mzsae.core.cache import MZSAEKVCache
from src.mzsae.nn import MZSAEAttention
from src.mzsae.ternary import (
    create_ternary_config,
    adjust_tau_for_ternary,
    adjust_sentinel_scale_for_ternary,
)
from src.mzsae.rope import decouple_rope_spectrum, apply_rope_givens


QUANT_FORMATS = [
    ("fp16", 16, 1.0, 1.0, 1.0),
    ("bf16", 16, 1.0, 1.0, 1.0),
    ("8bit", 8, 1.0, 1.0, 1.0),
    ("4bit", 4, 1.0, 1.0, 1.0),
    ("3bit", 3, 0.9, 0.9, 0.9),
    ("2bit", 2, 0.8, 0.8, 0.8),
    ("ternary", 2, 0.5, 0.7, 0.5),
]


@pytest.mark.parametrize("q_type,bits,dr_scale,tau_mult,sent_scale", QUANT_FORMATS)
def test_model_weight_config_properties(q_type, bits, dr_scale, tau_mult, sent_scale):
    """Verifies factory construction and property flags across all quantization formats."""
    cfg = ModelWeightConfig.from_type(q_type)
    assert cfg.quantization_type == q_type
    assert cfg.quantization_bits == bits
    assert np.isclose(cfg.dynamic_range_scale, dr_scale)
    assert np.isclose(cfg.tau_multiplier, tau_mult)
    assert np.isclose(cfg.sentinel_scale, sent_scale)

    if q_type == "ternary":
        assert cfg.is_ternary() is True
        assert cfg.is_low_bit() is True
    elif bits <= 4:
        assert cfg.is_low_bit() is True
        assert cfg.is_ternary() is False
    else:
        assert cfg.is_low_bit() is False
        assert cfg.is_ternary() is False


@pytest.mark.parametrize("q_type,bits,dr_scale,tau_mult,sent_scale", QUANT_FORMATS)
def test_crq_quantization_with_weight_config(q_type, bits, dr_scale, tau_mult, sent_scale):
    """Verifies CRQ block quantization and reconstruction under each quantization setting."""
    weight_cfg = ModelWeightConfig.from_type(q_type)

    # Simulate post-projection keys with scale reflecting quantization dynamic range
    scale = dr_scale
    keys_unrotated = (np.random.randn(BLOCK_SIZE, HEAD_DIM).astype(np.float32) * scale)

    q_block = crq_quantize_block(
        keys_unrotated,
        head_dim=HEAD_DIM,
        slow_dims=SLOW_DIMS,
        block_size=BLOCK_SIZE,
        weight_config=weight_cfg,
    )

    assert "centroid" in q_block
    assert "packed_residuals" in q_block
    assert "scale" in q_block
    assert "s_slow" in q_block
    assert "R_delta" in q_block
    assert "C_fast" in q_block
    assert np.all(np.isfinite(q_block["centroid"]))
    assert np.isfinite(q_block["scale"])

    desc = extract_sentinel_descriptor(q_block, timestep=0)
    assert len(desc) == 64
    assert isinstance(desc, bytes)

    # Dequantize and check finite reconstruction
    recon = crq_dequantize_block(
        q_block["centroid"],
        q_block["packed_residuals"],
        float(q_block["scale"]),
        HEAD_DIM,
    )
    assert recon.shape == (BLOCK_SIZE, HEAD_DIM)
    assert np.all(np.isfinite(recon))


@pytest.mark.parametrize("q_type,bits,dr_scale,tau_mult,sent_scale", QUANT_FORMATS)
def test_engine_decode_with_weight_config(q_type, bits, dr_scale, tau_mult, sent_scale):
    """Verifies MZSAEEngine execution and cache operations with each weight format."""
    cfg = load_config("default")
    cfg.model_weights = ModelWeightConfig.from_type(q_type)

    engine = MZSAEEngine(
        config=cfg,
        use_metal=False,  # Test CPU path for deterministic parity
    )

    # Ingest blocks with scaled synthetic projections
    scale = dr_scale
    for b in range(4):
        k = np.random.randn(BLOCK_SIZE, HEAD_DIM).astype(np.float32) * scale
        v = np.random.randn(BLOCK_SIZE, HEAD_DIM).astype(np.float32) * scale
        engine.ingest_block(k, v, logical_id=b)

    # Single-token decode step
    q = np.random.randn(HEAD_DIM).astype(np.float32)
    out, telemetry = engine.decode_step(q)

    assert out.shape == (HEAD_DIM,)
    assert np.all(np.isfinite(out))
    assert isinstance(telemetry, dict)


@pytest.mark.parametrize("q_type,bits,dr_scale,tau_mult,sent_scale", QUANT_FORMATS)
def test_nn_module_with_quantization_format(q_type, bits, dr_scale, tau_mult, sent_scale):
    """Verifies PyTorch MZSAEAttention module instantiated with model_weights."""
    attn = MZSAEAttention(
        embed_dim=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        model_weights=q_type,
    )

    assert attn.config.model_weights.quantization_type == q_type
    assert attn.config.model_weights.quantization_bits == bits

    x = torch.randn(1, 16, 256)
    out = attn(x, causal=True)
    assert out.shape == (1, 16, 256)
    assert not torch.isnan(out).any()


def test_ternary_specific():
    """Specific tests for BitNet b1.58 ternary model adaptation."""
    base_cfg = load_config("default")
    assert base_cfg.eviction.default_tau == 16.0

    ternary_cfg = create_ternary_config(base_cfg)
    assert ternary_cfg.model_weights.is_ternary() is True
    assert ternary_cfg.model_weights.quantization_bits == 2
    assert ternary_cfg.model_weights.dynamic_range_scale == 0.5
    assert ternary_cfg.model_weights.tau_multiplier == 0.7
    assert ternary_cfg.model_weights.sentinel_scale == 0.5

    # Check tau adjustment helper
    adj_tau = adjust_tau_for_ternary(16.0)
    assert np.isclose(adj_tau, 11.2)

    # Check sentinel scale helper
    assert adjust_sentinel_scale_for_ternary() == 0.5

    # Check engine instantiates with reduced tau
    engine = MZSAEEngine(config=ternary_cfg, use_metal=False)
    assert np.isclose(engine.tau, 16.0 * 0.7)
