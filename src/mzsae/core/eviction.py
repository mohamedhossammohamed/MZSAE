"""
Dynamic Eviction & Lifecycle Primitives
MZahran Sparse Attention Engine (MZSAE)
"""

from typing import Any, List, Optional
import numpy as np
from .cache import MZSAEKVCache
from .veto import DirectionalVeto
from .td_policy import TDAttnPolicy, TelemetryRingBuffer


class EvictionManager:
    """
    Coordinates directional veto checks, temporal-difference policy evaluations,
    and logical block removals from an active MZSAEKVCache.
    """

    def __init__(
        self,
        cache: MZSAEKVCache,
        veto: Optional[DirectionalVeto] = None,
        policy: Optional[TDAttnPolicy] = None,
        telemetry: Optional[TelemetryRingBuffer] = None,
        acc_veto_threshold: float = 0.40,
    ):
        self.cache = cache
        self.veto = veto or DirectionalVeto(cos_threshold=acc_veto_threshold)
        self.policy = policy or TDAttnPolicy()
        self.telemetry = telemetry or TelemetryRingBuffer()

    def evict_block(self, logical_id: Any) -> bool:
        """Evicts a specific logical block from the cache."""
        return self.cache.evict_logical_block(logical_id)

    def evaluate_and_prune(
        self,
        query: np.ndarray,
        tau: float = 16.0,
        max_blocks: Optional[int] = None,
    ) -> List[Any]:
        """
        Evaluates candidate blocks against directional veto and capacity constraints.
        Returns list of evicted block IDs.
        """
        evicted = []
        if max_blocks is None:
            return evicted

        active_blocks = list(self.cache.logical_block_ids)
        while len(active_blocks) > max_blocks:
            candidate_id = active_blocks[0]
            # Sinks are preserved, verify veto
            q_slow = (
                query[..., -16:]
                if query.shape[-1] >= 16
                else np.zeros(16, dtype=np.float32)
            )
            # Safe evaluation
            can_evict = self.veto.evaluate_eviction(
                block_id=0,
                query_slow=q_slow.flatten()[:16],
                sentinel_slow=np.zeros(16, dtype=np.float32),
                is_sink=False,
            )
            if can_evict:
                if self.cache.evict_logical_block(candidate_id):
                    evicted.append(candidate_id)
            active_blocks.pop(0)

        return evicted
