"""Core attention, cache, quantization, and lifecycle primitives."""

from .cache import (
    MZSAEKVCache,
    HEAD_DIM,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    NUM_SINKS,
    RECENT_WIN,
    BLOCK_SIZE,
)
from .engine import MZSAEEngine
from .veto import DirectionalVeto
from .td_policy import TDAttnPolicy, TelemetryRingBuffer
from .rope import (
    compute_rope_frequencies,
    decouple_rope_spectrum,
    apply_rope_givens,
)
from .compression import (
    pack_2bit_residuals,
    unpack_2bit_residuals,
    crq_quantize_block,
    crq_dequantize_block,
    extract_sentinel_descriptor,
    SLOW_DIMS,
)
from .eviction import EvictionManager

__all__ = [
    "MZSAEKVCache",
    "MZSAEEngine",
    "DirectionalVeto",
    "TDAttnPolicy",
    "TelemetryRingBuffer",
    "compute_rope_frequencies",
    "decouple_rope_spectrum",
    "apply_rope_givens",
    "pack_2bit_residuals",
    "unpack_2bit_residuals",
    "crq_quantize_block",
    "crq_dequantize_block",
    "extract_sentinel_descriptor",
    "EvictionManager",
    "HEAD_DIM",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "NUM_SINKS",
    "RECENT_WIN",
    "BLOCK_SIZE",
    "SLOW_DIMS",
]
