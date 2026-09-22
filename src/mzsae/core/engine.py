"""
MZSAE Engine: Hardware-Aware Sparse Attention Engine
MZahran Sparse Attention Engine (MZSAE)
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

from .cache import (
    MZSAEKVCache,
    HEAD_DIM,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    NUM_SINKS,
    RECENT_WIN,
    BLOCK_SIZE,
)
from .veto import DirectionalVeto
from .td_policy import TDAttnPolicy, TelemetryRingBuffer
from ..backends.dispatcher import get_backend
from ..backends.base import MZSAEBackend


class MZSAEEngine:
    """
    High-level MZSAE Engine managing dual-plane KV cache, biological eviction veto,
    and hardware-accelerated attention execution.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        num_q_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        num_splits: Optional[int] = None,
        capacity_blocks: Optional[int] = None,
        use_metal: Optional[bool] = None,
        tau: Optional[float] = None,
        backend: Optional[MZSAEBackend] = None,
    ):
        k_cfg = getattr(config, "kernel", None) if config is not None else None
        e_cfg = getattr(config, "eviction", None) if config is not None else None
        c_cfg = getattr(config, "cache", None) if config is not None else None
        mw_cfg = getattr(config, "model_weights", None) if config is not None else None

        if num_q_heads is None:
            num_q_heads = getattr(k_cfg, "num_query_heads", NUM_Q_HEADS) if k_cfg else NUM_Q_HEADS
        if num_kv_heads is None:
            num_kv_heads = getattr(k_cfg, "num_kv_heads", NUM_KV_HEADS) if k_cfg else NUM_KV_HEADS
        if head_dim is None:
            head_dim = getattr(k_cfg, "head_dim", HEAD_DIM) if k_cfg else HEAD_DIM
        if num_splits is None:
            num_splits = getattr(k_cfg, "num_splits", 64) if k_cfg else 64
        if capacity_blocks is None and c_cfg is not None:
            capacity_blocks = getattr(c_cfg, "max_active_blocks", None)
        if use_metal is None:
            use_metal = getattr(config, "use_metal", True) if config is not None else True
        if tau is None:
            tau = getattr(e_cfg, "default_tau", 16.0) if e_cfg else 16.0

        if mw_cfg is not None:
            tau = tau * getattr(mw_cfg, "tau_multiplier", 1.0)

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_splits = num_splits
        self.capacity_blocks = capacity_blocks
        self.use_metal = use_metal
        self.tau = tau
        self.weight_config = getattr(config, "model_weights", None) if config is not None else None

        self.cache = MZSAEKVCache(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            capacity_blocks=capacity_blocks,
            weight_config=self.weight_config,
        )

        if backend is not None:
            self.backend = backend
        else:
            backend_type = "metal" if use_metal else "cpu"
            self.backend = get_backend(backend_type, config=config, head_dim=head_dim)

        try:
            from ..backends.metal.runtime import MetalBackend

            self.use_metal = isinstance(self.backend, MetalBackend)
        except Exception:
            self.use_metal = False

        self.veto = DirectionalVeto(cos_threshold=0.40)
        self.policy = TDAttnPolicy()
        self.telemetry = TelemetryRingBuffer(max_records=1024)

    @property
    def metal_backend(self) -> Optional[MZSAEBackend]:
        return self.backend

    def ingest_kv_chunk(self, keys: Any, vals: Any, start_pos: int = 0):
        self.cache.ingest_prefill(keys, vals)

    def ingest_block(
        self,
        k_block: Any,
        v_block: Any,
        logical_id: Optional[Any] = None,
        start_pos: Optional[int] = None,
    ):
        """Ingests a discrete block of KV activations with logical ID tracking."""
        self.cache.ingest_block(
            k_block, v_block, logical_id=logical_id, start_pos=start_pos
        )

    def evict_logical_block(self, logical_id: Any) -> bool:
        """Evicts a specific logical block from the cache."""
        return self.cache.evict_logical_block(logical_id)

    def sleep_cycle(self, batch_size: int = 32, steps: int = 10) -> float:
        """
        Executes an offline sleep replay cycle (hippocampal-neocortical consolidation).
        Replays buffered transitions from TelemetryRingBuffer to train TDAttnPolicy via TD(0).
        """
        if self.telemetry.size == 0:
            return 0.0
        final_loss = 0.0
        for _ in range(steps):
            states, actions, rewards, next_states = (
                self.telemetry.sample_batch(batch_size=batch_size)
            )
            if states.shape[0] > 0:
                final_loss = self.policy.td_update(
                    states, rewards, next_states
                )
        return final_loss

    def decode_step(
        self, q: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        is_1d = q.ndim == 1
        if is_1d:
            q_in = np.tile(q[np.newaxis, :], (self.num_q_heads, 1)).astype(
                np.float32
            )
        else:
            q_in = q.astype(np.float32)

        out, telemetry = self.selective_decode(
            q_in, tau=self.tau, return_telemetry=True
        )
        if is_1d:
            return out[0], telemetry
        return out, telemetry

    def decode(self, q: np.ndarray) -> np.ndarray:
        """
        Executes single-token decode for query of shape [num_q_heads, head_dim].
        """
        bufs = self.cache.get_metal_buffers()
        return self.backend.fused_decode(
            q=q,
            k_payload=bufs["k_payload"],
            v_payload=bufs["v_payload"],
            k_centroids=bufs["k_centroids"],
            k_scales=bufs["k_scales"],
            k_mins=bufs["k_mins"],
            v_group_meta=bufs["v_group_meta"],
            sinks_k=bufs["sinks_k"],
            sinks_v=bufs["sinks_v"],
            recent_k=bufs["recent_k"],
            recent_v=bufs["recent_v"],
            seq_len=bufs["seq_len"],
            num_splits=self.num_splits,
        )

    def selective_decode(
        self,
        q: np.ndarray,
        tau: float = 16.0,
        return_telemetry: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Executes single-token selective decode using Plane-2 Sentinel gating.
        """
        bufs = self.cache.get_metal_buffers()
        out, telemetry = self.backend.selective_decode(
            q=q,
            k_payload=bufs["k_payload"],
            v_payload=bufs["v_payload"],
            sentinels=bufs["sentinels"],
            k_centroids=bufs["k_centroids"],
            k_scales=bufs["k_scales"],
            k_mins=bufs["k_mins"],
            v_group_meta=bufs["v_group_meta"],
            sinks_k=bufs["sinks_k"],
            sinks_v=bufs["sinks_v"],
            recent_k=bufs["recent_k"],
            recent_v=bufs["recent_v"],
            seq_len=bufs["seq_len"],
            num_splits=self.num_splits,
            tau=tau,
            return_gpu_time=True,
        )
        if return_telemetry:
            return out, telemetry
        return out

    @property
    def plane1_bytes(self) -> int:
        bufs = self.cache.get_metal_buffers()
        return int(
            bufs["k_payload"].nbytes
            + bufs["v_payload"].nbytes
            + bufs["k_centroids"].nbytes
            + bufs["k_scales"].nbytes
            + bufs["k_mins"].nbytes
            + bufs["v_group_meta"].nbytes
        )

    @property
    def plane2_bytes(self) -> int:
        bufs = self.cache.get_metal_buffers()
        return int(bufs["sentinels"].nbytes)
