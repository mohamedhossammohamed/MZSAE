"""
MZahran Sparse Attention Engine (MZSAE)
Hardware-Aware Sparse Continual Attention Runtime
Author: Mohammed Hossam Zahran
"""

from .version import __version__
from .config import (
    MZSAEConfig,
    HardwareProfile,
    KernelConfig,
    EvictionConfig,
    CacheConfig,
    ModelWeightConfig,
    load_config,
    auto_detect_hardware,
)
from .core.engine import MZSAEEngine
from .core.cache import MZSAEKVCache
from .nn import MZSAEAttention
from .api import create_engine
from .backends.base import MZSAEBackend
from .backends.dispatcher import get_backend, is_metal_available
from .backends.metal.runtime import MetalBackend
from .backends.cpu_reference import CPUReferenceBackend
from .functional import mzsae_with_kvcache
from .hf_patch import patch_model, unpatch_model
from .errors import (
    MZSAEError,
    BackendNotAvailableError,
    ConfigError,
    DeviceMismatchError,
)
from .ternary import (
    create_ternary_config,
    adjust_tau_for_ternary,
    adjust_sentinel_scale_for_ternary,
)
from .neutral import (
    AttentionBackend,
    StandardAttentionWrapper,
    FlashAttentionWrapper,
    SlidingWindowWrapper,
    MoEAttentionWrapper,
    MZSAENeutralWrapper,
)

__all__ = [
    "__version__",
    "MZSAEEngine",
    "MZSAEAttention",
    "MZSAEConfig",
    "HardwareProfile",
    "KernelConfig",
    "EvictionConfig",
    "CacheConfig",
    "ModelWeightConfig",
    "load_config",
    "auto_detect_hardware",
    "create_engine",
    "MZSAEKVCache",
    "MZSAEBackend",
    "get_backend",
    "is_metal_available",
    "MetalBackend",
    "CPUReferenceBackend",
    "mzsae_with_kvcache",
    "patch_model",
    "unpatch_model",
    "MZSAEError",
    "BackendNotAvailableError",
    "ConfigError",
    "DeviceMismatchError",
    "AttentionBackend",
    "StandardAttentionWrapper",
    "FlashAttentionWrapper",
    "SlidingWindowWrapper",
    "MoEAttentionWrapper",
    "MZSAENeutralWrapper",
    "create_ternary_config",
    "adjust_tau_for_ternary",
    "adjust_sentinel_scale_for_ternary",
]
