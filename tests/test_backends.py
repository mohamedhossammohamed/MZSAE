"""
Tests for Backend Dispatcher and CPU Reference Execution
"""

import numpy as np
import pytest
from src.mzsae.backends.dispatcher import get_backend, is_metal_available
from src.mzsae.backends.cpu_reference import CPUReferenceBackend
from src.mzsae.core.cache import MZSAEKVCache


def test_cpu_reference_decode():
    backend = CPUReferenceBackend(head_dim=128)
    assert backend.is_available
    assert "CPU" in backend.device_name

    cache = MZSAEKVCache(num_q_heads=12, num_kv_heads=2, head_dim=128)
    K = np.random.randn(128, 2, 128).astype(np.float16)
    V = np.random.randn(128, 2, 128).astype(np.float16)
    cache.ingest_prefill(K, V)

    bufs = cache.get_metal_buffers()
    q = np.random.randn(12, 128).astype(np.float32)

    out = backend.fused_decode(
        q=q,
        k_payload=bufs["k_payload"],
        v_payload=bufs["v_payload"],
        k_centroids=bufs["k_centroids"],
        k_scales=bufs["k_scales"],
        k_mins=bufs["k_mins"],
        v_group_meta=bufs["v_group_meta"],
        sinks_k=bufs["sinks_k"],
        sinks_v=bufs["sinks_v"],
        recent_k=bufs["recent_k"],
        recent_v=bufs["recent_v"],
        seq_len=bufs["seq_len"],
    )
    assert out.shape == (12, 128)
    assert np.all(np.isfinite(out))


def test_cpu_reference_selective_decode():
    backend = CPUReferenceBackend(head_dim=128)
    cache = MZSAEKVCache(num_q_heads=12, num_kv_heads=2, head_dim=128)
    K = np.random.randn(128, 2, 128).astype(np.float16)
    V = np.random.randn(128, 2, 128).astype(np.float16)
    cache.ingest_prefill(K, V)

    bufs = cache.get_metal_buffers()
    q = np.random.randn(12, 128).astype(np.float32)

    out, telemetry = backend.selective_decode(
        q=q,
        k_payload=bufs["k_payload"],
        v_payload=bufs["v_payload"],
        sentinels=bufs["sentinels"],
        k_centroids=bufs["k_centroids"],
        k_scales=bufs["k_scales"],
        k_mins=bufs["k_mins"],
        v_group_meta=bufs["v_group_meta"],
        sinks_k=bufs["sinks_k"],
        sinks_v=bufs["sinks_v"],
        recent_k=bufs["recent_k"],
        recent_v=bufs["recent_v"],
        seq_len=bufs["seq_len"],
        tau=16.0,
    )
    assert out.shape == (12, 128)
    assert "pruning_ratio" in telemetry
    assert "approved_blocks" in telemetry


def test_dispatcher_selection():
    b_cpu = get_backend("cpu")
    assert isinstance(b_cpu, CPUReferenceBackend)

    b_auto = get_backend("auto")
    assert b_auto.is_available

    b_cuda = get_backend("cuda")
    assert b_cuda.is_available

    with pytest.raises(ValueError):
        get_backend("invalid_unknown_backend")
