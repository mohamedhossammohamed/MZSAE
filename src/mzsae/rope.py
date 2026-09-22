"""
Backward-compatibility re-export shim for RoPE frequency decoupling.
Canonical implementation located in mzsae.core.rope.
"""

from .core.rope import (
    compute_rope_frequencies,
    decouple_rope_spectrum,
    apply_rope_givens,
)

__all__ = [
    "compute_rope_frequencies",
    "decouple_rope_spectrum",
    "apply_rope_givens",
]
