"""
Abstract Backend Interface for MZSAE
MZahran Sparse Attention Engine (MZSAE)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class MZSAEBackend(ABC):
    """
    Abstract Hardware Execution Backend for MZSAE.
    Backends implement hardware-specific attention decode, selective fetch,
    and memory management.
    """

    @abstractmethod
    def fused_decode(
        self,
        q: np.ndarray,
        k_payload: np.ndarray,
        v_payload: np.ndarray,
        k_centroids: np.ndarray,
        k_scales: np.ndarray,
        k_mins: np.ndarray,
        v_group_meta: np.ndarray,
        sinks_k: np.ndarray,
        sinks_v: np.ndarray,
        recent_k: np.ndarray,
        recent_v: np.ndarray,
        seq_len: int,
        num_splits: int = 64,
        recent_win: int = 64,
    ) -> np.ndarray:
        """Executes multi-split compressed fused attention decode."""
        pass

    @abstractmethod
    def selective_decode(
        self,
        q: np.ndarray,
        k_payload: np.ndarray,
        v_payload: np.ndarray,
        sentinels: np.ndarray,
        k_centroids: np.ndarray,
        k_scales: np.ndarray,
        k_mins: np.ndarray,
        v_group_meta: np.ndarray,
        sinks_k: np.ndarray,
        sinks_v: np.ndarray,
        recent_k: np.ndarray,
        recent_v: np.ndarray,
        seq_len: int,
        num_splits: int = 64,
        tau: float = 16.0,
        return_gpu_time: bool = False,
        recent_win: int = 64,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Executes Plane-2 sentinel selective fetch decode."""
        pass

    @abstractmethod
    def dense_decode(
        self,
        q: np.ndarray,
        dense_k_fp16: np.ndarray,
        dense_v_fp16: np.ndarray,
        seq_len: int,
        num_splits: int = 64,
    ) -> np.ndarray:
        """Executes dense uncompressed attention decode."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clears hardware pipeline caches."""
        pass

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Returns human-readable hardware device name."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the backend runtime is successfully loaded and functional."""
        pass
