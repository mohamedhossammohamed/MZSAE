"""
Special handling for ternary weight models (BitNet b1.58).
MZahran Sparse Attention Engine (MZSAE)
"""

from copy import deepcopy
from .config import MZSAEConfig


def adjust_tau_for_ternary(base_tau: float) -> float:
    """Ternary models need less aggressive pruning due to tighter score distributions."""
    return base_tau * 0.7


def adjust_sentinel_scale_for_ternary() -> float:
    """Ternary K/V values have smaller dynamic range."""
    return 0.5


def create_ternary_config(base_config: MZSAEConfig) -> MZSAEConfig:
    """Create a config optimized for ternary models (BitNet b1.58)."""
    config = deepcopy(base_config)
    config.model_weights.quantization_type = "ternary"
    config.model_weights.quantization_bits = 2
    config.model_weights.dynamic_range_scale = 0.5
    config.model_weights.tau_multiplier = 0.7
    config.model_weights.sentinel_scale = 0.5
    return config
