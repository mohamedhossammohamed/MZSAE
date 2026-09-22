"""
Backward-compatibility re-export shim for CRQ pipeline.
Canonical implementation located in mzsae.core.compression.
"""

from .core.compression import (
    BLOCK_SIZE,
    HEAD_DIM,
    SLOW_DIMS,
    pack_2bit_residuals,
    unpack_2bit_residuals,
    crq_quantize_block,
    crq_dequantize_block,
    extract_sentinel_descriptor,
)
from .core.rope import decouple_rope_spectrum

__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "SLOW_DIMS",
    "pack_2bit_residuals",
    "unpack_2bit_residuals",
    "crq_quantize_block",
    "crq_dequantize_block",
    "extract_sentinel_descriptor",
    "decouple_rope_spectrum",
]
