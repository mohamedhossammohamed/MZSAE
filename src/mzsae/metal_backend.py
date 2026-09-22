"""
Backward-compatibility re-export shim for Metal backend.
Canonical implementation located in mzsae.backends.metal.runtime.
"""

from .backends.metal.runtime import MetalBackend

__all__ = [
    "MetalBackend",
]
