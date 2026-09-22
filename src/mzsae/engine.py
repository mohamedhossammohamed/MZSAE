"""
Backward-compatibility re-export shim for MZSAE engine.
Canonical implementation located in mzsae.core.engine and mzsae.core.cache.
"""

from .core.cache import (
    MZSAEKVCache,
    HEAD_DIM,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    NUM_SINKS,
    RECENT_WIN,
    BLOCK_SIZE,
)
from .core.engine import MZSAEEngine
from .core.veto import DirectionalVeto
from .core.td_policy import TDAttnPolicy, TelemetryRingBuffer
from .backends.metal.runtime import MetalBackend

__all__ = [
    "MZSAEKVCache",
    "MZSAEEngine",
    "HEAD_DIM",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "NUM_SINKS",
    "RECENT_WIN",
    "BLOCK_SIZE",
    "DirectionalVeto",
    "TDAttnPolicy",
    "TelemetryRingBuffer",
    "MetalBackend",
]
