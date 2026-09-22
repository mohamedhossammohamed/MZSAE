"""
Unit Tests for Metal Hardware Runtime & Kernel Execution
Brick 3 & Metal Shader Verification
"""

import pytest
import numpy as np
from src.mzsae.engine import MZSAEEngine
from src.mzsae.crq import BLOCK_SIZE, HEAD_DIM

def test_metal_device_discovery():
    engine = MZSAEEngine(use_metal=True)
    if not engine.use_metal:
        pytest.skip("Metal GPU backend not available on this platform")
    
    assert engine.metal_backend is not None
    print(f"Detected Metal Hardware Device: {engine.metal_backend.device_name}")
    assert "Apple" in engine.metal_backend.device_name

def test_metal_vs_cpu_reference_numerical_parity():
    """
    Compares the Metal GPU fused decode kernel with the CPU reference simulator.
    Both should match within FP16 reconstruction tolerance.
    """
    np.random.seed(99)
    num_blocks = 8
    seq_len = num_blocks * BLOCK_SIZE # 256 tokens

    # Generate synthetic unrotated keys and values
    keys = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)
    vals = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)

    # 1. Initialize Metal engine
    engine_metal = MZSAEEngine(capacity_blocks=16, use_metal=True, tau=24.0)
    if not engine_metal.use_metal:
        pytest.skip("Metal backend unavailable")
    engine_metal.ingest_kv_chunk(keys, vals)

    # 2. Initialize CPU reference engine
    engine_cpu = MZSAEEngine(capacity_blocks=16, use_metal=False, tau=24.0)
    engine_cpu.ingest_kv_chunk(keys, vals)

    # 3. Execute decode on query
    query = np.random.randn(HEAD_DIM).astype(np.float32)

    out_metal, tel_metal = engine_metal.decode_step(query)
    out_cpu, tel_cpu = engine_cpu.decode_step(query)

    print(f"Metal Approved Blocks: {tel_metal['approved_blocks']} / {num_blocks}")
    print(f"CPU Approved Blocks:   {tel_cpu['approved_blocks']} / {num_blocks}")

    # The approved count should match
    assert tel_metal["approved_blocks"] == tel_cpu["approved_blocks"]

    # Numerical difference between Metal GPU output and CPU reference
    max_diff = np.max(np.abs(out_metal - out_cpu))
    print(f"Metal vs CPU Max Output Difference: {max_diff:.5f}")
    assert max_diff < 0.05, f"Metal and CPU outputs diverged: max diff {max_diff}"
