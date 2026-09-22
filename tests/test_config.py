"""
Tests for MZSAE Configuration Subsystem
"""

import os
import pytest
from src.mzsae.config import (
    load_config,
    auto_detect_hardware,
    MZSAEConfig,
    HardwareProfile,
    KernelConfig,
    EvictionConfig,
    CacheConfig,
    ModelWeightConfig,
)


def test_load_named_configs():
    for name in ["default", "apple_m1", "apple_m4", "apple_m4_max", "nvidia_future", "bitnet_b158"]:
        cfg = load_config(name)
        assert isinstance(cfg, MZSAEConfig)
        assert isinstance(cfg.hardware, HardwareProfile)
        assert isinstance(cfg.kernel, KernelConfig)
        assert isinstance(cfg.eviction, EvictionConfig)
        assert isinstance(cfg.cache, CacheConfig)
        assert isinstance(cfg.model_weights, ModelWeightConfig)
        assert cfg.hardware.memory_bandwidth_gbps > 0


def test_load_yaml_filepath():
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "apple_m4.yaml")
    cfg = load_config(path)
    assert cfg.hardware.chip_name == "Apple M4"
    assert cfg.hardware.memory_bandwidth_gbps == 120.0
    assert cfg.kernel.num_splits == 64


def test_auto_detect_hardware():
    hw = auto_detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.device_type in ("apple_silicon", "cpu")
    assert hw.memory_bandwidth_gbps > 0
    assert hw.recommended_threadgroup_size in (64, 256)


def test_load_config_auto():
    cfg = load_config("auto")
    assert isinstance(cfg, MZSAEConfig)
    assert cfg.hardware.memory_bandwidth_gbps > 0


def test_config_dict_roundtrip():
    cfg = load_config("apple_m4")
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["hardware"]["chip_name"] == "Apple M4"
    assert "model_weights" in d

    cfg_restored = MZSAEConfig.from_dict(d)
    assert cfg_restored.hardware.chip_name == cfg.hardware.chip_name
    assert cfg_restored.kernel.head_dim == cfg.kernel.head_dim
    assert cfg_restored.model_weights.quantization_type == cfg.model_weights.quantization_type


def test_load_nonexistent_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_profile_xyz")
