"""
Tests for Adaptive Hybrid Dispatch (Limitation §4 Resolution)
Verifies automatic short-context dense bypass and long-context sparse selective decode.
"""

import pytest
import numpy as np
from mzsae.core.engine import MZSAEEngine, HEAD_DIM, BLOCK_SIZE
from mzsae.config import MZSAEConfig, KernelConfig


def test_adaptive_dispatch_short_context_routes_to_dense():
    """At context < sparse_threshold_tokens (e.g. 512 < 2048), engine should use dense bypass."""
    seq_len = 512
    keys = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)
    vals = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)

    engine = MZSAEEngine(
        use_metal=False,
        sparse_threshold_tokens=2048,
        adaptive=True,
    )
    engine.ingest_kv_chunk(keys, vals)

    query = np.random.randn(HEAD_DIM).astype(np.float32)
    out, tel = engine.decode_step(query)

    assert tel["dispatch_mode"] == "dense_bypass"
    assert tel["pruning_ratio"] == 0.0
    assert tel["approved_blocks"] == tel["total_blocks"]
    assert np.all(np.isfinite(out))


def test_adaptive_dispatch_long_context_routes_to_sparse():
    """At context >= sparse_threshold_tokens (e.g. 2048 >= 2048), engine should use sparse selective decode."""
    seq_len = 2048
    keys = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)
    vals = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)

    engine = MZSAEEngine(
        use_metal=False,
        sparse_threshold_tokens=2048,
        adaptive=True,
        tau=16.0,
    )
    engine.ingest_kv_chunk(keys, vals)

    query = np.random.randn(HEAD_DIM).astype(np.float32)
    out, tel = engine.decode_step(query)

    assert tel["dispatch_mode"] == "sparse_selective"
    assert "approved_blocks" in tel
    assert "total_blocks" in tel
    assert np.all(np.isfinite(out))


def test_adaptive_dispatch_force_sparse_override():
    """When force_sparse=True, engine should run sparse selective decode even at short contexts."""
    seq_len = 256
    keys = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)
    vals = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)

    engine = MZSAEEngine(
        use_metal=False,
        sparse_threshold_tokens=2048,
        adaptive=True,
    )
    engine.ingest_kv_chunk(keys, vals)

    query = np.random.randn(HEAD_DIM).astype(np.float32)
    out, tel = engine.decode_step(query, force_sparse=True)

    assert tel["dispatch_mode"] == "sparse_selective"


def test_adaptive_dispatch_disabled():
    """When adaptive=False, engine always uses sparse selective decode."""
    seq_len = 256
    keys = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)
    vals = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)

    engine = MZSAEEngine(
        use_metal=False,
        sparse_threshold_tokens=2048,
        adaptive=False,
    )
    engine.ingest_kv_chunk(keys, vals)

    query = np.random.randn(HEAD_DIM).astype(np.float32)
    out, tel = engine.decode_step(query)

    assert tel["dispatch_mode"] == "sparse_selective"


def test_adaptive_dispatch_numerical_parity_metal_cpu():
    """Verifies that dense bypass yields numerical parity across Metal and CPU reference."""
    seq_len = 512
    keys = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)
    vals = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)

    engine_metal = MZSAEEngine(
        use_metal=True,
        sparse_threshold_tokens=2048,
        adaptive=True,
    )
    if not engine_metal.use_metal:
        pytest.skip("Metal backend unavailable")

    engine_cpu = MZSAEEngine(
        use_metal=False,
        sparse_threshold_tokens=2048,
        adaptive=True,
    )

    engine_metal.ingest_kv_chunk(keys, vals)
    engine_cpu.ingest_kv_chunk(keys, vals)

    query = np.random.randn(HEAD_DIM).astype(np.float32)
    out_metal, tel_metal = engine_metal.decode_step(query)
    out_cpu, tel_cpu = engine_cpu.decode_step(query)

    assert tel_metal["dispatch_mode"] == "dense_bypass"
    assert tel_cpu["dispatch_mode"] == "dense_bypass"

    max_diff = float(np.max(np.abs(out_metal - out_cpu)))
    assert max_diff < 1e-3, f"Parity mismatch in dense bypass mode: max diff {max_diff}"
