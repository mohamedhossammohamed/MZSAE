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


def is_cuda_available() -> bool:
    """Checks if NVIDIA CUDA GPU device and runtime are available."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def get_backend(
    name: str = "auto",
    config: Optional[Any] = None,
    head_dim: int = 128,
) -> MZSAEBackend:
    """
    Instantiates and returns the appropriate hardware backend.
    Priority order for 'auto':
      1. NVIDIA CUDA (if torch.cuda.is_available())
      2. Apple Silicon Metal (if macOS arm64 with Metal framework)
      3. CPU Reference (Fallback)
    Args:
      name: 'auto', 'cuda', 'metal', 'cpu'
      config: Optional MZSAEConfig instance
      head_dim: Token head dimension (default 128)
    """
    target = name.lower()

    if target == "auto":
        if is_cuda_available():
            target = "cuda"
        elif is_metal_available():
            target = "metal"
        else:
            target = "cpu"

    if target == "cuda":
        from .cuda import CUDABackend

        is_gh = False
        if config is not None:
            hw = getattr(config, "hardware", None)
            if hw is not None:
                is_gh = getattr(hw, "nvlink_c2c_coherent", False)

        return CUDABackend(is_grace_hopper=is_gh)

    elif target == "metal":
        from .metal.runtime import MetalBackend

        # Metal C-ABI kernels are compiled for standard head_dim=128
        if head_dim != 128:
            return CPUReferenceBackend(head_dim=head_dim)

        try:
            return MetalBackend()
        except Exception as e:
            # Fallback to CPU if metal failed and config allows fallback
            use_fallback = (
                getattr(config, "use_cpu_fallback", True) if config is not None else True
            )
            if use_fallback:
                return CPUReferenceBackend(head_dim=head_dim)
            raise RuntimeError(
                f"Failed to initialize Metal backend: {e}. Set use_cpu_fallback=True or specify backend='cpu'."
            ) from e

    elif target in ("cpu", "cpu_reference"):
        return CPUReferenceBackend(head_dim=head_dim)

    else:
        raise ValueError(
            f"Unknown backend '{name}'. Available options: 'auto', 'cuda', 'metal', 'cpu'."
        )
