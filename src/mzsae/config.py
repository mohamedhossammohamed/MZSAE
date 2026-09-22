"""
Hardware-Agnostic Configuration System
MZahran Sparse Attention Engine (MZSAE)
"""

import os
import sys
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class HardwareProfile:
    """Hardware capabilities and physical memory hierarchy profile."""
    device_type: str = "apple_silicon"  # 'apple_silicon', 'cpu', 'cuda'
    chip_name: str = "Apple M4"
    memory_bandwidth_gbps: float = 120.0
    l2_cache_mb: float = 16.0
    sram_size_kb: float = 32.0
    recommended_threadgroup_size: int = 256


@dataclass
class KernelConfig:
    """Low-level kernel execution parameters."""
    block_size_tokens: int = 64
    head_dim: int = 128
    num_kv_heads: int = 2
    num_query_heads: int = 12
    slow_rope_dims: int = 16
    sentinel_bytes: int = 64
    threadgroup_size: int = 256
    num_splits: int = 64


@dataclass
class EvictionConfig:
    """Biologically-inspired eviction and gating hyperparameters."""
    default_tau: float = 16.0
    acc_veto_threshold: float = 0.40
    ring_buffer_kb: int = 128
    sleep_batch_size: int = 32
    sleep_steps: int = 10


@dataclass
class CacheConfig:
    """Dual-plane KV cache sizing and retention constraints."""
    num_sinks: int = 4
    recent_win: int = 64
    max_active_blocks: Optional[int] = None
    capacity_blocks: Optional[int] = None


@dataclass
class ModelWeightConfig:
    """Configuration for the model's weight quantization format."""
    quantization_type: str = "fp16"  # "fp16", "bf16", "8bit", "4bit", "3bit", "2bit", "ternary"
    quantization_bits: int = 16
    dynamic_range_scale: float = 1.0  # Scale factor for centroid/residual computation
    tau_multiplier: float = 1.0  # Multiplier for pruning threshold
    sentinel_scale: float = 1.0  # Scale for Cauchy-Schwarz bounds

    def is_ternary(self) -> bool:
        return self.quantization_type.lower() in ("ternary", "bitnet", "bitnet_b158", "b1.58")

    def is_low_bit(self) -> bool:
        return self.quantization_bits <= 4 or self.is_ternary()

    @classmethod
    def from_type(cls, quant_type: str) -> "ModelWeightConfig":
        """Factory method to construct optimal ModelWeightConfig from quantization format string."""
        qt = quant_type.lower()
        if qt in ("fp16", "float16"):
            return cls(quantization_type="fp16", quantization_bits=16, dynamic_range_scale=1.0, tau_multiplier=1.0, sentinel_scale=1.0)
        elif qt in ("bf16", "bfloat16"):
            return cls(quantization_type="bf16", quantization_bits=16, dynamic_range_scale=1.0, tau_multiplier=1.0, sentinel_scale=1.0)
        elif qt in ("8bit", "q8_0", "int8"):
            return cls(quantization_type="8bit", quantization_bits=8, dynamic_range_scale=1.0, tau_multiplier=1.0, sentinel_scale=1.0)
        elif qt in ("4bit", "q4_k_m", "q4_0", "int4", "awq", "gptq"):
            return cls(quantization_type="4bit", quantization_bits=4, dynamic_range_scale=1.0, tau_multiplier=1.0, sentinel_scale=1.0)
        elif qt in ("3bit", "q3_k_m", "int3"):
            return cls(quantization_type="3bit", quantization_bits=3, dynamic_range_scale=0.9, tau_multiplier=0.9, sentinel_scale=0.9)
        elif qt in ("2bit", "q2_k", "int2"):
            return cls(quantization_type="2bit", quantization_bits=2, dynamic_range_scale=0.8, tau_multiplier=0.8, sentinel_scale=0.8)
        elif qt in ("ternary", "bitnet", "bitnet_b158", "b1.58"):
            return cls(quantization_type="ternary", quantization_bits=2, dynamic_range_scale=0.5, tau_multiplier=0.7, sentinel_scale=0.5)
        else:
            return cls(quantization_type=quant_type)


@dataclass
class MZSAEConfig:
    """Unified configuration container for MZSAE runtime."""
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    model_weights: ModelWeightConfig = field(default_factory=ModelWeightConfig)
    use_metal: bool = True
    use_cpu_fallback: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MZSAEConfig":
        hw_data = data.get("hardware", {})
        k_data = data.get("kernel", {})
        e_data = data.get("eviction", {})
        c_data = data.get("cache", {})
        mw_data = data.get("model_weights", {})

        return cls(
            hardware=HardwareProfile(**hw_data) if hw_data else HardwareProfile(),
            kernel=KernelConfig(**k_data) if k_data else KernelConfig(),
            eviction=EvictionConfig(**e_data) if e_data else EvictionConfig(),
            cache=CacheConfig(**c_data) if c_data else CacheConfig(),
            model_weights=ModelWeightConfig(**mw_data) if mw_data else ModelWeightConfig(),
            use_metal=data.get("use_metal", True),
            use_cpu_fallback=data.get("use_cpu_fallback", True),
        )


def auto_detect_hardware() -> HardwareProfile:
    """
    Detects host platform and returns the matching HardwareProfile.
    """
    if platform.system() != "Darwin":
        return HardwareProfile(
            device_type="cpu",
            chip_name="Generic Host CPU",
            memory_bandwidth_gbps=50.0,
            l2_cache_mb=8.0,
            sram_size_kb=0.0,
            recommended_threadgroup_size=64,
        )

    # Darwin / Apple Silicon detection
    chip_desc = "Apple Silicon"
    try:
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
        chip_desc = out.decode("utf-8").strip()
    except Exception:
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.model"], stderr=subprocess.DEVNULL)
            chip_desc = out.decode("utf-8").strip()
        except Exception:
            pass

    chip_lower = chip_desc.lower()
    if "m4 max" in chip_lower:
        return HardwareProfile(
            device_type="apple_silicon",
            chip_name=chip_desc,
            memory_bandwidth_gbps=410.0,
            l2_cache_mb=48.0,
            sram_size_kb=32.0,
            recommended_threadgroup_size=256,
        )
    elif "m4 pro" in chip_lower:
        return HardwareProfile(
            device_type="apple_silicon",
            chip_name=chip_desc,
            memory_bandwidth_gbps=273.0,
            l2_cache_mb=32.0,
            sram_size_kb=32.0,
            recommended_threadgroup_size=256,
        )
    elif "m4" in chip_lower:
        return HardwareProfile(
            device_type="apple_silicon",
            chip_name=chip_desc,
            memory_bandwidth_gbps=120.0,
            l2_cache_mb=16.0,
            sram_size_kb=32.0,
            recommended_threadgroup_size=256,
        )
    elif "m3 max" in chip_lower or "m2 max" in chip_lower or "m1 max" in chip_lower:
        return HardwareProfile(
            device_type="apple_silicon",
            chip_name=chip_desc,
            memory_bandwidth_gbps=300.0,
            l2_cache_mb=32.0,
            sram_size_kb=32.0,
            recommended_threadgroup_size=256,
        )
    elif "m1" in chip_lower or "m2" in chip_lower or "m3" in chip_lower:
        return HardwareProfile(
            device_type="apple_silicon",
            chip_name=chip_desc,
            memory_bandwidth_gbps=68.0,
            l2_cache_mb=12.0,
            sram_size_kb=32.0,
            recommended_threadgroup_size=256,
        )
    else:
        return HardwareProfile(
            device_type="apple_silicon",
            chip_name=chip_desc,
            memory_bandwidth_gbps=100.0,
            l2_cache_mb=16.0,
            sram_size_kb=32.0,
            recommended_threadgroup_size=256,
        )


def _find_config_file(path_or_name: str) -> str:
    """Resolves config name or filepath to an absolute path."""
    if os.path.exists(path_or_name):
        return os.path.abspath(path_or_name)

    name = path_or_name
    if not name.endswith(".yaml") and not name.endswith(".yml"):
        name_yaml = f"{name}.yaml"
    else:
        name_yaml = name

    search_dirs = [
        os.path.join(os.getcwd(), "configs"),
        os.path.join(os.path.dirname(__file__), "..", "..", "configs"),
        os.path.join(os.path.dirname(__file__), "configs"),
    ]

    for d in search_dirs:
        candidate = os.path.join(d, name_yaml)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Configuration '{path_or_name}' not found. Searched paths: {search_dirs}"
    )


def load_config(path_or_name: Union[str, Dict[str, Any]] = "auto") -> MZSAEConfig:
    """
    Loads and parses an MZSAE configuration.
    Args:
      path_or_name: 'auto', profile name (e.g. 'apple_m4'), YAML path, or dict
    Returns:
      MZSAEConfig instance
    """
    if isinstance(path_or_name, dict):
        return MZSAEConfig.from_dict(path_or_name)

    if path_or_name == "auto":
        hw = auto_detect_hardware()
        cfg = MZSAEConfig(hardware=hw)
        if hw.device_type != "apple_silicon":
            cfg.use_metal = False
        return cfg

    config_path = _find_config_file(path_or_name)
    if not HAS_YAML:
        raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return MZSAEConfig.from_dict(data or {})
