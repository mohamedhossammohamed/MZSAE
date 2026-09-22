"""CUDA backend placeholder for MZSAE."""

from ..base import MZSAEBackend


class CUDABackend(MZSAEBackend):
    """
    Placeholder for NVIDIA CUDA backend.
    """

    def __init__(self):
        raise NotImplementedError(
            "CUDA backend is planned for v1.3.0. Currently supported backends: 'metal', 'cpu'."
        )

    def fused_decode(self, *args, **kwargs):
        raise NotImplementedError

    def selective_decode(self, *args, **kwargs):
        raise NotImplementedError

    def dense_decode(self, *args, **kwargs):
        raise NotImplementedError

    def clear_cache(self):
        pass

    @property
    def device_name(self) -> str:
        return "NVIDIA CUDA (Unimplemented)"

    @property
    def is_available(self) -> bool:
        return False


__all__ = ["CUDABackend"]
