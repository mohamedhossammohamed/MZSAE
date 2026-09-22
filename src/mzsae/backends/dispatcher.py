"""
Backend Dispatcher and Hardware Selector
MZahran Sparse Attention Engine (MZSAE)
"""

import sys
import platform
from typing import Optional, Union, Dict, Type, Any

from .base import MZSAEBackend
from .cpu_reference import CPUReferenceBackend


def is_metal_available() -> bool:
    """Checks if macOS Apple Silicon Metal runtime can be loaded."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        from .metal.runtime import MetalBackend

        backend = MetalBackend()
        return backend.is_available
    except Exception:
        return False


def get_backend(
    name: str = "auto",
    config: Optional[Any] = None,
    head_dim: int = 128,
) -> MZSAEBackend:
    """
    Instantiates and returns the appropriate hardware backend.
    Args:
      name: 'auto', 'metal', 'cpu', 'cuda'
      config: Optional MZSAEConfig instance
      head_dim: Token head dimension (default 128)
    """
    target = name.lower()

    if target == "auto":
        if is_metal_available():
            target = "metal"
        else:
            target = "cpu"

    if target == "metal":
        from .metal.runtime import MetalBackend

        # Metal C-ABI kernels are compiled for standard head_dim=128
        if head_dim != 128:
            return CPUReferenceBackend(head_dim=head_dim)

        try:
            return MetalBackend()
        except Exception as e:
            # Fallback to CPU if metal failed and config allows fallback
            if config is not None and getattr(config, "use_cpu_fallback", True):
                return CPUReferenceBackend(head_dim=head_dim)
            raise RuntimeError(
                f"Failed to initialize Metal backend: {e}. Set use_cpu_fallback=True or specify backend='cpu'."
            ) from e

    elif target in ("cpu", "cpu_reference"):
        return CPUReferenceBackend(head_dim=head_dim)

    elif target == "cuda":
        from .cuda import CUDABackend

        return CUDABackend()

    else:
        raise ValueError(
            f"Unknown backend '{name}'. Available options: 'auto', 'metal', 'cpu', 'cuda'."
        )
