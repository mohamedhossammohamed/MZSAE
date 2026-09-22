"""
Unified Dual-Plane Compressed KV Cache
MZahran Sparse Attention Engine (MZSAE)
"""

import struct
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

HEAD_DIM = 128
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
NUM_SINKS = 4
RECENT_WIN = 64
BLOCK_SIZE = 64


class MZSAEKVCache:
    """
    Unified Dual-Plane Compressed KV Cache.
    Implements FIX SET 1 format:
      - 4 FP16 sinks preserved
      - 64 FP16 recent tokens in local window
      - 64-token temporal blocks with causal centroid and per-channel affine residual for Keys
      - 64-element channel groups for Values
    """

    def __init__(
        self,
        num_q_heads: int = NUM_Q_HEADS,
        num_kv_heads: int = NUM_KV_HEADS,
        head_dim: int = HEAD_DIM,
        num_sinks: int = NUM_SINKS,
        recent_win: int = RECENT_WIN,
        block_size: int = BLOCK_SIZE,
        capacity_blocks: Optional[int] = None,
        weight_config: Optional[Any] = None,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_sinks = num_sinks
        self.recent_win = recent_win
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self.weight_config = weight_config

        self.seq_len = 0
        self.sinks_k = np.zeros((num_sinks, num_kv_heads, head_dim), dtype=np.float16)
        self.sinks_v = np.zeros((num_sinks, num_kv_heads, head_dim), dtype=np.float16)

        # Ring buffer for recent tokens
        self.recent_k_buf: List[np.ndarray] = []
        self.recent_v_buf: List[np.ndarray] = []

        # Raw history for CPU reference
        self.raw_k_history: List[np.ndarray] = []
        self.raw_v_history: List[np.ndarray] = []

        # Compressed body blocks
        self.k_payload_blocks: List[np.ndarray] = (
            []
        )  # each (block_size, num_kv_heads, 64) uint8
        self.v_payload_blocks: List[np.ndarray] = (
            []
        )  # each (block_size, num_kv_heads, 64) uint8
        self.k_centroids: List[np.ndarray] = (
            []
        )  # each (num_kv_heads, 32, 4) float16
        self.k_scales: List[np.ndarray] = []  # each (num_kv_heads, 32, 4) float16
        self.k_mins: List[np.ndarray] = []  # each (num_kv_heads, 32, 4) float16
        self.v_group_meta_blocks: List[np.ndarray] = (
            []
        )  # each (block_size, num_kv_heads, 2, 4) float16
        self.sentinels: List[np.ndarray] = []  # each (num_kv_heads, 64) uint8
        self.body_block_ids: List[Any] = []
        self.recent_block_id: Optional[Any] = None
        self._pending_recent_blk: Optional[Tuple[np.ndarray, np.ndarray, Any]] = (
            None
        )

    @property
    def total_tokens(self) -> int:
        return self.seq_len

    @property
    def logical_block_ids(self) -> List[Any]:
        ids = list(self.body_block_ids)
        if self.recent_block_id is not None:
            ids.append(self.recent_block_id)
        return ids

    @property
    def total_active_blocks(self) -> int:
        n_act = len(self.body_block_ids) + (
            1 if self.recent_block_id is not None else 0
        )
        if n_act > 0:
            return n_act
        if self.capacity_blocks is not None and self.seq_len > 0:
            return (self.seq_len + 31) // 32
        return len(self.sentinels)

    @property
    def stride(self) -> int:
        """Byte footprint of a single Plane-1 compressed block."""
        return 2688

    @property
    def active_blocks(self) -> List[Any]:
        """List of active logical block IDs."""
        return self.logical_block_ids

    def reset(self):
        self.seq_len = 0
        self.recent_k_buf.clear()
        self.recent_v_buf.clear()
        self.raw_k_history.clear()
        self.raw_v_history.clear()
        self.k_payload_blocks.clear()
        self.v_payload_blocks.clear()
        self.k_centroids.clear()
        self.k_scales.clear()
        self.k_mins.clear()
        self.v_group_meta_blocks.clear()
        self.sentinels.clear()
        self.body_block_ids.clear()
        self.recent_block_id = None
        self._pending_recent_blk = None
        self.sinks_k.fill(0)
        self.sinks_v.fill(0)

    def _compress_and_store_block(
        self,
        k_blk: np.ndarray,
        v_blk: np.ndarray,
        logical_id: Optional[Any] = None,
    ):
        """Compresses a contiguous 64-token block into FIX SET 1 format."""
        bsz = k_blk.shape[0]
        nkv = self.num_kv_heads

        # 1. Keys: causal block centroid and per-channel scale/min
        if self.weight_config is not None and getattr(self.weight_config, "is_ternary", lambda: False)():
            mu = (k_blk.astype(np.float32).mean(axis=0) * getattr(self.weight_config, "dynamic_range_scale", 1.0)).astype(np.float32)
        else:
            mu = k_blk.astype(np.float32).mean(axis=0)  # [nkv, D]
        res = k_blk.astype(np.float32) - mu
        res_min = res.min(axis=0)
        res_max = res.max(axis=0)
        scale = np.maximum((res_max - res_min) / 15.0, 1e-8)

        mu_h = mu.astype(np.float16)
        scale_h = scale.astype(np.float16)
        min_h = res_min.astype(np.float16)

        self.k_centroids.append(mu_h.reshape(nkv, 32, 4))
        self.k_scales.append(scale_h.reshape(nkv, 32, 4))
        self.k_mins.append(min_h.reshape(nkv, 32, 4))

        q_k = np.clip(np.round((res - res_min) / scale), 0, 15).astype(np.uint8)
        k_pay = q_k[:, :, 0::2] | (q_k[:, :, 1::2] << 4)  # [bsz, nkv, 64]
        self.k_payload_blocks.append(k_pay)

        # 2. Values: 64-element channel groups
        v_pay = np.zeros((bsz, nkv, 64), dtype=np.uint8)
        v_meta = np.zeros((bsz, nkv, 2, 4), dtype=np.float16)

        for t in range(bsz):
            vt = v_blk[t].astype(np.float32)  # [nkv, 128]
            for g in range(2):
                g_vals = vt[:, g * 64 : (g + 1) * 64]
                mu_g = g_vals.mean(axis=-1, keepdims=True)
                res_g = g_vals - mu_g
                min_g = res_g.min(axis=-1, keepdims=True)
                max_g = res_g.max(axis=-1, keepdims=True)
                sc_g = np.maximum((max_g - min_g) / 15.0, 1e-8)

                v_meta[t, :, g, 0] = mu_g.squeeze(-1).astype(np.float16)
                v_meta[t, :, g, 1] = min_g.squeeze(-1).astype(np.float16)
                v_meta[t, :, g, 2] = sc_g.squeeze(-1).astype(np.float16)

                q_g = np.clip(np.round((res_g - min_g) / sc_g), 0, 15).astype(
                    np.uint8
                )
                v_pay[t, :, g * 32 : (g + 1) * 32] = (
                    q_g[:, 0::2] | (q_g[:, 1::2] << 4)
                )

        self.v_payload_blocks.append(v_pay)
        self.v_group_meta_blocks.append(v_meta)

        # 3. Plane-2 Sentinels (64 bytes per block per KV head)
        sentinel_scale = getattr(self.weight_config, "sentinel_scale", 1.0) if self.weight_config is not None else 1.0
        blk_sent = np.zeros((nkv, 64), dtype=np.uint8)
        for h in range(nkv):
            s_slow = (mu[h, 112:] * sentinel_scale).astype(np.float16)
            res_slow = res[:, h, 112:]
            r_delta = (
                float(np.max(np.linalg.norm(res_slow, axis=-1))) * sentinel_scale
                if bsz > 0
                else 0.0
            )
            k_fast = k_blk[:, h, :112]
            c_fast = (
                float(np.max(np.linalg.norm(k_fast, axis=-1))) * sentinel_scale
                if bsz > 0
                else 0.0
            )
            sig2 = float(np.var(res[:, h]))
            header = s_slow.tobytes()
            meta = struct.pack(
                "<fffIffQ", r_delta, c_fast, 0.0, int(self.seq_len), sig2, 0.0, 1
            )
            desc = header + meta
            blk_sent[h] = np.frombuffer(desc, dtype=np.uint8)
        self.sentinels.append(blk_sent)

    def ingest_block(
        self,
        k_blk: Any,
        v_blk: Any,
        logical_id: Optional[Any] = None,
        start_pos: Optional[int] = None,
    ):
        """
        Ingests a discrete block of KV activations with logical block ID tracking.
        Maintains sinks (first block), body blocks (compressed), and recent window (most recent block).
        """
        if hasattr(k_blk, "detach"):
            k_blk = k_blk.detach().cpu().numpy()
        if hasattr(v_blk, "detach"):
            v_blk = v_blk.detach().cpu().numpy()

        k_blk = np.asarray(k_blk, dtype=np.float32)
        v_blk = np.asarray(v_blk, dtype=np.float32)

        if k_blk.ndim == 2:
            k_blk = np.repeat(k_blk[:, np.newaxis, :], self.num_kv_heads, axis=1)
            v_blk = np.repeat(v_blk[:, np.newaxis, :], self.num_kv_heads, axis=1)
        elif (
            k_blk.ndim == 3
            and k_blk.shape[1] != self.num_kv_heads
            and k_blk.shape[1] == 1
        ):
            k_blk = np.repeat(k_blk, self.num_kv_heads, axis=1)
            v_blk = np.repeat(v_blk, self.num_kv_heads, axis=1)

        bsz = k_blk.shape[0]
        if logical_id is None:
            logical_id = len(self.body_block_ids) + (
                1 if self.recent_block_id is not None else 0
            )

        if self.recent_block_id is None:
            num_s = min(self.num_sinks, bsz)
            self.sinks_k[:num_s] = k_blk[:num_s].astype(np.float16)
            self.sinks_v[:num_s] = v_blk[:num_s].astype(np.float16)
            self.recent_block_id = logical_id
            self._pending_recent_blk = (k_blk, v_blk, logical_id)
            self.recent_k_buf = [k_blk[t].astype(np.float16) for t in range(bsz)]
            self.recent_v_buf = [v_blk[t].astype(np.float16) for t in range(bsz)]
        else:
            prev_k, prev_v, prev_id = self._pending_recent_blk
            self._compress_and_store_block(prev_k, prev_v)
            self.body_block_ids.append(prev_id)

            self.recent_block_id = logical_id
            self._pending_recent_blk = (k_blk, v_blk, logical_id)
            self.recent_k_buf = [k_blk[t].astype(np.float16) for t in range(bsz)]
            self.recent_v_buf = [v_blk[t].astype(np.float16) for t in range(bsz)]

        self.seq_len = (
            self.num_sinks
            + len(self.k_payload_blocks) * self.block_size
            + len(self.recent_k_buf)
        )

    def evict_logical_block(self, logical_id: Any) -> bool:
        """
        Evicts a specific logical block by ID, updating payload, centroids, and sentinels.
        Returns True if evicted, False if block not found.
        """
        if self.recent_block_id == logical_id:
            self.recent_block_id = None
            self._pending_recent_blk = None
            self.recent_k_buf.clear()
            self.recent_v_buf.clear()
            self.seq_len = (
                self.num_sinks + len(self.k_payload_blocks) * self.block_size
            )
            return True

        if logical_id in self.body_block_ids:
            idx = self.body_block_ids.index(logical_id)
            self.body_block_ids.pop(idx)
            self.k_payload_blocks.pop(idx)
            self.v_payload_blocks.pop(idx)
            self.k_centroids.pop(idx)
            self.k_scales.pop(idx)
            self.k_mins.pop(idx)
            self.v_group_meta_blocks.pop(idx)
            self.sentinels.pop(idx)
            self.seq_len = (
                self.num_sinks
                + len(self.k_payload_blocks) * self.block_size
                + len(self.recent_k_buf)
            )
            return True

        return False

    def append_kv(
        self,
        k_tok: np.ndarray,
        v_tok: np.ndarray,
        timestep: Optional[int] = None,
    ):
        """
        Appends 1 token of KV (shape [num_kv_heads, head_dim] or [head_dim]).
        """
        k_h = np.ascontiguousarray(k_tok, dtype=np.float16)
        v_h = np.ascontiguousarray(v_tok, dtype=np.float16)

        if k_h.ndim == 1:
            k_h = np.repeat(k_h[np.newaxis, :], self.num_kv_heads, axis=0)
            v_h = np.repeat(v_h[np.newaxis, :], self.num_kv_heads, axis=0)
        elif k_h.ndim == 2 and k_h.shape[0] != self.num_kv_heads:
            if k_h.shape[0] == 1:
                k_h = np.repeat(k_h, self.num_kv_heads, axis=0)
                v_h = np.repeat(v_h, self.num_kv_heads, axis=0)

        if self.seq_len < self.num_sinks:
            self.sinks_k[self.seq_len] = k_h
            self.sinks_v[self.seq_len] = v_h
        else:
            self.recent_k_buf.append(k_h)
            self.recent_v_buf.append(v_h)
            # If recent window exceeds recent_win + block_size, compress oldest block
            if len(self.recent_k_buf) >= self.recent_win + self.block_size:
                k_blk = np.stack(self.recent_k_buf[: self.block_size], axis=0)
                v_blk = np.stack(self.recent_v_buf[: self.block_size], axis=0)
                self._compress_and_store_block(k_blk, v_blk)
                self.recent_k_buf = self.recent_k_buf[self.block_size :]
                self.recent_v_buf = self.recent_v_buf[self.block_size :]

        self.seq_len += 1

    def ingest_prefill(self, K: Any, V: Any):
        """
        Ingests a prefill sequence.
        Accepts numpy arrays or torch tensors with shape:
          - [seq_len, head_dim]
          - [seq_len, num_kv_heads, head_dim]
          - [batch, seq_len, num_kv_heads, head_dim]
        """
        if hasattr(K, "detach"):
            K = K.detach().cpu().numpy()
        if hasattr(V, "detach"):
            V = V.detach().cpu().numpy()

        if K.ndim == 4:
            K = K[0]
            V = V[0]
        if K.ndim == 2:
            K = np.repeat(K[:, np.newaxis, :], self.num_kv_heads, axis=1)
            V = np.repeat(V[:, np.newaxis, :], self.num_kv_heads, axis=1)
        elif K.ndim == 3 and K.shape[1] != self.num_kv_heads:
            if K.shape[1] == 1:
                K = np.repeat(K, self.num_kv_heads, axis=1)
                V = np.repeat(V, self.num_kv_heads, axis=1)

        T = K.shape[0]
        self.reset()
        self.seq_len = T
        self.raw_k_history = [K[t].astype(np.float16) for t in range(T)]
        self.raw_v_history = [V[t].astype(np.float16) for t in range(T)]

        num_sinks = min(self.num_sinks, T)
        self.sinks_k[:num_sinks] = K[:num_sinks].astype(np.float16)
        self.sinks_v[:num_sinks] = V[:num_sinks].astype(np.float16)

        rem = T - num_sinks
        if rem > 0:
            if rem <= self.recent_win:
                for t in range(num_sinks, T):
                    self.recent_k_buf.append(K[t].astype(np.float16))
                    self.recent_v_buf.append(V[t].astype(np.float16))
            else:
                num_body_blocks = (rem - self.recent_win) // self.block_size
                num_body_tokens = num_body_blocks * self.block_size
                body_end = num_sinks + num_body_tokens
                if num_body_tokens > 0:
                    k_body = K[num_sinks:body_end]
                    v_body = V[num_sinks:body_end]
                    for b_start in range(0, num_body_tokens, self.block_size):
                        b_end = b_start + self.block_size
                        self._compress_and_store_block(
                            k_body[b_start:b_end], v_body[b_start:b_end]
                        )
                for t in range(body_end, T):
                    self.recent_k_buf.append(K[t].astype(np.float16))
                    self.recent_v_buf.append(V[t].astype(np.float16))

    def get_metal_buffers(self) -> Dict[str, np.ndarray]:
        """Assembles contiguous arrays ready for Metal dispatch."""
        body_tokens = sum(b.shape[0] for b in self.k_payload_blocks)

        if body_tokens > 0:
            k_payload = np.ascontiguousarray(
                np.concatenate(self.k_payload_blocks, axis=0)
            )
            v_payload = np.ascontiguousarray(
                np.concatenate(self.v_payload_blocks, axis=0)
            )
            sentinels = np.ascontiguousarray(np.stack(self.sentinels, axis=0))
            k_centroids = np.ascontiguousarray(np.stack(self.k_centroids, axis=0))
            k_scales = np.ascontiguousarray(np.stack(self.k_scales, axis=0))
            k_mins = np.ascontiguousarray(np.stack(self.k_mins, axis=0))
            v_group_meta = np.ascontiguousarray(
                np.concatenate(self.v_group_meta_blocks, axis=0)
            )
        else:
            k_payload = np.zeros((0, self.num_kv_heads, 64), dtype=np.uint8)
            v_payload = np.zeros((0, self.num_kv_heads, 64), dtype=np.uint8)
            sentinels = np.zeros((0, self.num_kv_heads, 64), dtype=np.uint8)
            k_centroids = np.zeros((0, self.num_kv_heads, 32, 4), dtype=np.float16)
            k_scales = np.zeros((0, self.num_kv_heads, 32, 4), dtype=np.float16)
            k_mins = np.zeros((0, self.num_kv_heads, 32, 4), dtype=np.float16)
            v_group_meta = np.zeros(
                (0, self.num_kv_heads, 2, 4), dtype=np.float16
            )

        if len(self.recent_k_buf) > 0:
            recent_k = np.ascontiguousarray(np.stack(self.recent_k_buf, axis=0))
            recent_v = np.ascontiguousarray(np.stack(self.recent_v_buf, axis=0))
        else:
            recent_k = np.zeros(
                (0, self.num_kv_heads, self.head_dim), dtype=np.float16
            )
            recent_v = np.zeros(
                (0, self.num_kv_heads, self.head_dim), dtype=np.float16
            )

        return {
            "sinks_k": self.sinks_k,
            "sinks_v": self.sinks_v,
            "recent_k": recent_k,
            "recent_v": recent_v,
            "k_payload": k_payload,
            "v_payload": v_payload,
            "sentinels": sentinels,
            "k_centroids": k_centroids,
            "k_scales": k_scales,
            "k_mins": k_mins,
            "v_group_meta": v_group_meta,
            "seq_len": self.seq_len,
        }
