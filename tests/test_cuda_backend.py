"""
Unit Tests for NVIDIA CUDA Backend, L2 Cache Persistence, and DGX Cluster Topology
MZSAE Enterprise NVIDIA Extension
"""

import pytest
import numpy as np
import torch

from src.mzsae.config import load_config
from src.mzsae.backends.dispatcher import get_backend, is_cuda_available
from src.mzsae.backends.cuda import (
    CUDABackend,
    L2CachePersistenceManager,
    DGXTopologyManager,
    TPShardingConfig,
    CUDAGraphManager,
)
from src.mzsae.crq import BLOCK_SIZE, HEAD_DIM


def test_cuda_hardware_profiles_load():
    """Validates that all new enterprise NVIDIA YAML configuration profiles load correctly."""
    # 1. NVIDIA H100 SXM5
    h100_cfg = load_config("configs/nvidia_h100.yaml")
    assert h100_cfg.hardware.chip_name == "NVIDIA H100 SXM5"
    assert h100_cfg.hardware.memory_bandwidth_gbps == 3350.0
    assert h100_cfg.hardware.l2_cache_mb == 50.0
    assert h100_cfg.kernel.num_splits == 128

    # 2. NVIDIA GH200 Grace Hopper
    gh200_cfg = load_config("configs/nvidia_gh200.yaml")
    assert "GH200" in gh200_cfg.hardware.chip_name
    assert gh200_cfg.hardware.nvlink_c2c_coherent is True
    assert gh200_cfg.hardware.nvlink_c2c_bandwidth_gbps == 900.0

    # 3. NVIDIA Blackwell B200
    b200_cfg = load_config("configs/nvidia_b200.yaml")
    assert "B200" in b200_cfg.hardware.chip_name
    assert b200_cfg.hardware.memory_bandwidth_gbps == 8000.0
    assert b200_cfg.hardware.l2_cache_mb == 60.0
    assert b200_cfg.kernel.num_splits == 256


def test_cuda_backend_instantiation():
    """Tests CUDABackend initialization and device name telemetry."""
    backend = CUDABackend(device_id=0)
    assert backend.is_available is True
    assert backend.device_name is not None
    assert "CUDA" in backend.device_name


def test_l2_cache_persistence_manager():
    """Validates L2 Cache eviction window calculation and telemetry."""
    l2_mgr = L2CachePersistenceManager(device_id=0, persistent_fraction=0.80)
    l2_size_mb = l2_mgr.get_device_l2_size_mb()
    assert l2_size_mb >= 40.0 # Modern datacenter GPUs have >=40MB L2

    # Test pinning tensor in L2
    dummy_sentinels = torch.zeros((16, 2, 64), dtype=torch.uint8)
    if torch.cuda.is_available():
        dummy_sentinels = dummy_sentinels.cuda()

    pinned = l2_mgr.pin_sentinels_in_l2(dummy_sentinels)
    telemetry = l2_mgr.get_telemetry()
    assert "device_total_l2_mb" in telemetry
    assert telemetry["device_total_l2_mb"] >= 40.0

    l2_mgr.reset_l2_window()
    assert l2_mgr.is_active is False


def test_dgx_topology_manager_sharding():
    """Tests Tensor Parallelism (TP) KV head partitioning on an 8-GPU DGX node."""
    # DGX H100 with 8 GPUs: 32 query heads, 8 KV heads
    tp_config = TPShardingConfig(
        world_size=8,
        rank=2, # GPU 2
        total_q_heads=32,
        total_kv_heads=8,
        head_dim=128
    )
    dgx = DGXTopologyManager(tp_config=tp_config, is_grace_hopper=False)

    summary = dgx.get_topology_summary()
    assert summary["world_size"] == 8
    assert summary["rank"] == 2
    assert summary["local_q_heads"] == 4   # 32 / 8 = 4 query heads per GPU
    assert summary["local_kv_heads"] == 1  # 8 / 8 = 1 KV head per GPU
    assert summary["kv_head_range"] == (2, 3)

    # Shard global query tensor [32, 128] -> [4, 128]
    q_global = torch.randn(32, 128)
    q_shard = dgx.shard_query_tensor(q_global)
    assert q_shard.shape == (4, 128)
    assert torch.equal(q_shard, q_global[8:12])

    # Shard global KV cache [1, 64, 8, 128] -> [1, 64, 1, 128]
    k_global = torch.randn(1, 64, 8, 128)
    k_shard = dgx.shard_kv_tensor(k_global)
    assert k_shard.shape == (1, 64, 1, 128)


def test_dgx_topology_grace_hopper_zero_copy():
    """Tests coherent memory zero-copy allocation on Grace Hopper GH200."""
    dgx = DGXTopologyManager(is_grace_hopper=True)
    buf, meta = dgx.allocate_zero_copy_ring_buffer(
        buffer_bytes=1024 * 256,
        device=torch.device("cpu")
    )
    assert buf.numel() == 1024 * 256
    assert meta["bus_bandwidth_gbps"] == 900.0


def test_cuda_graph_manager():
    """Tests CUDA Graph Manager capture structure."""
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mgr = CUDAGraphManager(device=dev)
    assert mgr.is_captured is False

    # Calling replay before capture is safe and no-op
    mgr.replay()
    assert mgr.is_captured is False


def test_dispatcher_cuda_routing():
    """Tests that dispatcher properly recognizes 'cuda' and hardware profile configurations."""
    backend_cuda = get_backend("cuda")
    assert isinstance(backend_cuda, CUDABackend)

    h100_cfg = load_config("configs/nvidia_h100.yaml")
    backend_from_cfg = get_backend("cuda", config=h100_cfg)
    assert isinstance(backend_from_cfg, CUDABackend)


def test_cuda_backend_execution_decode():
    """Tests end-to-end execution of selective_decode on the CUDABackend."""
    from src.mzsae.core.cache import MZSAEKVCache

    backend = CUDABackend()
    cache = MZSAEKVCache(num_q_heads=12, num_kv_heads=2, head_dim=128)
    K = np.random.randn(256, 2, 128).astype(np.float16)
    V = np.random.randn(256, 2, 128).astype(np.float16)
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
        num_splits=16,
        tau=16.0,
    )

    assert out.shape == (12, 128)
    assert not np.isnan(out).any()
    assert "dram_traffic_reduction" in telemetry
    assert "backend" in telemetry
    assert "l2_pinned" in telemetry
