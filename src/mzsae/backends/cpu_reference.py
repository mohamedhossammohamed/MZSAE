"""
Pure NumPy CPU Reference Backend
MZahran Sparse Attention Engine (MZSAE)
"""

import struct
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import MZSAEBackend
from ..core.rope import apply_rope_givens


class CPUReferenceBackend(MZSAEBackend):
    """
    Bit-exact CPU reference implementation of MZSAE attention decode.
    Simulates Plane-2 sentinel selective fetch and dequantization without GPU dependencies.
    """

    def __init__(self, head_dim: int = 128):
        self._head_dim = head_dim
        self._device_name = "CPU Reference (NumPy SIMD)"

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def is_available(self) -> bool:
        return True

    def clear_cache(self) -> None:
        pass

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
        return_gpu_time: bool = False,
    ) -> Any:
        # Fused decode on CPU processes all blocks without sentinel pruning
        num_blocks = k_centroids.shape[0] if k_centroids.ndim > 0 else 0
        num_kv_heads = sinks_k.shape[1] if sinks_k.ndim > 1 else 2
        sentinels = np.zeros((num_blocks, num_kv_heads, 64), dtype=np.uint8)
        # Force all blocks approved
        out, _ = self._decode_internal(
            q=q,
            k_payload=k_payload,
            v_payload=v_payload,
            sentinels=sentinels,
            k_centroids=k_centroids,
            k_scales=k_scales,
            k_mins=k_mins,
            v_group_meta=v_group_meta,
            sinks_k=sinks_k,
            sinks_v=sinks_v,
            recent_k=recent_k,
            recent_v=recent_v,
            seq_len=seq_len,
            tau=1e9,  # Approve all
            recent_win=recent_win,
        )
        if return_gpu_time:
            return out, 0.0
        return out

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
        return self._decode_internal(
            q=q,
            k_payload=k_payload,
            v_payload=v_payload,
            sentinels=sentinels,
            k_centroids=k_centroids,
            k_scales=k_scales,
            k_mins=k_mins,
            v_group_meta=v_group_meta,
            sinks_k=sinks_k,
            sinks_v=sinks_v,
            recent_k=recent_k,
            recent_v=recent_v,
            seq_len=seq_len,
            tau=tau,
            recent_win=recent_win,
        )

    def dense_decode(
        self,
        q: np.ndarray,
        dense_k_fp16: np.ndarray,
        dense_v_fp16: np.ndarray,
        seq_len: int,
        num_splits: int = 64,
    ) -> np.ndarray:
        T = seq_len
        nq, d = q.shape
        inv_sqrt_d = 1.0 / np.sqrt(d)
        nkv = dense_k_fp16.shape[1] if dense_k_fp16.ndim == 3 else 1
        heads_per_kv = nq // nkv

        out = np.zeros((nq, d), dtype=np.float32)
        pos = np.arange(T)

        for h in range(nq):
            kv_h = h // heads_per_kv
            k_head = dense_k_fp16[:T, kv_h, :] if dense_k_fp16.ndim == 3 else dense_k_fp16[:T]
            v_head = dense_v_fp16[:T, kv_h, :] if dense_v_fp16.ndim == 3 else dense_v_fp16[:T]

            k_rot = apply_rope_givens(k_head.astype(np.float32), pos, base=1000000.0)
            scores = (k_rot @ q[h].astype(np.float32)) * inv_sqrt_d
            m_val = np.max(scores)
            weights = np.exp(scores - m_val)
            weights /= np.sum(weights)
            out[h] = weights @ v_head.astype(np.float32)

        return out

    def _decode_internal(
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
        tau: float = 16.0,
        recent_win: int = 64,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        T = seq_len
        nq, d = q.shape
        inv_sqrt_d = 1.0 / np.sqrt(d)
        num_sinks = sinks_k.shape[0] if sinks_k is not None and sinks_k.ndim > 0 else 0
        num_kv_heads = sinks_k.shape[1] if sinks_k is not None and sinks_k.ndim > 1 else 2
        heads_per_kv = nq // num_kv_heads
        num_blocks = sentinels.shape[0] if sentinels is not None and sentinels.ndim > 0 else 0

        # Pass 0: local max over sinks and recent tokens
        local_max = np.full((nq,), -np.inf, dtype=np.float32)
        for h in range(nq):
            kv_h = h // heads_per_kv
            if num_sinks > 0 and T > 0:
                s_count = min(num_sinks, T)
                s_rot = apply_rope_givens(
                    sinks_k[:s_count, kv_h, :].astype(np.float32),
                    np.arange(s_count),
                    base=1000000.0,
                )
                local_max[h] = max(local_max[h], float(np.max(s_rot @ q[h])) * inv_sqrt_d)

            if recent_k is not None and recent_k.shape[0] > 0:
                rec_count = recent_k.shape[0]
                rec_pos = np.arange(T - rec_count, T)
                r_rot = apply_rope_givens(
                    recent_k[:, kv_h, :].astype(np.float32),
                    rec_pos,
                    base=1000000.0,
                )
                local_max[h] = max(local_max[h], float(np.max(r_rot @ q[h])) * inv_sqrt_d)

        # Sentinel pruning
        approved_mask = np.zeros((num_blocks, num_kv_heads), dtype=bool)
        for kv_h in range(num_kv_heads):
            q_base = kv_h * heads_per_kv
            for b in range(num_blocks):
                raw = sentinels[b, kv_h].tobytes()
                s_slow = np.frombuffer(raw[:32], dtype=np.float16).astype(np.float32)
                r_delta, c_fast = struct.unpack("<ff", raw[32:40])
                flags = struct.unpack("<Q", raw[56:64])[0] if len(raw) >= 64 else 0

                if flags & 2:  # MZSAE_BLOCK_FLAG_FORCED
                    approved_mask[b, kv_h] = True
                    continue

                for h_rel in range(heads_per_kv):
                    qh = q[q_base + h_rel]
                    q_slow = qh[d - 16 :]
                    dot_slow = float(np.dot(q_slow, s_slow))
                    q_slow_norm = float(np.linalg.norm(q_slow))
                    u_b = (dot_slow + q_slow_norm * r_delta + c_fast) * inv_sqrt_d
                    ref_m = local_max[q_base + h_rel]
                    if ref_m > -1e10:
                        if u_b >= (ref_m - tau):
                            approved_mask[b, kv_h] = True
                            break
                    else:
                        approved_mask[b, kv_h] = True
                        break

        total_blocks = num_blocks * num_kv_heads
        approved_blocks = int(np.sum(approved_mask))

        # Reconstruct active full K, V arrays
        k_list = []
        v_list = []
        if num_sinks > 0 and T > 0:
            s_c = min(num_sinks, T)
            k_list.append(sinks_k[:s_c].astype(np.float32))
            v_list.append(sinks_v[:s_c].astype(np.float32))

        for b in range(num_blocks):
            kp = k_payload[b * 64 : (b + 1) * 64] if k_payload.shape[0] >= (b + 1) * 64 else k_payload
            kc = k_centroids[b].astype(np.float32).reshape(num_kv_heads, 128)
            ks = k_scales[b].astype(np.float32).reshape(num_kv_heads, 128)
            km = k_mins[b].astype(np.float32).reshape(num_kv_heads, 128)
            bsz_b = min(64, kp.shape[0])

            kb_deq = np.zeros((bsz_b, num_kv_heads, d), dtype=np.float32)
            for t_idx in range(bsz_b):
                for h_idx in range(num_kv_heads):
                    raw_bytes = kp[t_idx, h_idx]
                    q0 = raw_bytes & 0x0F
                    q1 = (raw_bytes >> 4) & 0x0F
                    q_unp = np.empty(128, dtype=np.float32)
                    q_unp[0::2] = q0
                    q_unp[1::2] = q1
                    kb_deq[t_idx, h_idx] = kc[h_idx] + km[h_idx] + q_unp * ks[h_idx]
            k_list.append(kb_deq)

            vp = v_payload[b * 64 : (b + 1) * 64] if v_payload.shape[0] >= (b + 1) * 64 else v_payload
            vm = v_group_meta[b * 64 : (b + 1) * 64].astype(np.float32) if v_group_meta.shape[0] >= (b + 1) * 64 else v_group_meta
            vb_deq = np.zeros((bsz_b, num_kv_heads, d), dtype=np.float32)
            for t_idx in range(bsz_b):
                for h_idx in range(num_kv_heads):
                    raw_bytes = vp[t_idx, h_idx]
                    q0 = raw_bytes & 0x0F
                    q1 = (raw_bytes >> 4) & 0x0F
                    q_unp = np.empty(128, dtype=np.float32)
                    q_unp[0::2] = q0
                    q_unp[1::2] = q1
                    for g in range(2):
                        mu_g = vm[t_idx, h_idx, g, 0]
                        min_g = vm[t_idx, h_idx, g, 1]
                        sc_g = vm[t_idx, h_idx, g, 2]
                        vb_deq[t_idx, h_idx, g * 64 : (g + 1) * 64] = (
                            mu_g + min_g + q_unp[g * 64 : (g + 1) * 64] * sc_g
                        )
            v_list.append(vb_deq)

        if recent_k is not None and recent_k.shape[0] > 0:
            k_list.append(recent_k.astype(np.float32))
            v_list.append(recent_v.astype(np.float32))

        if len(k_list) > 0:
            K_full = np.concatenate(k_list, axis=0)[:T]
            V_full = np.concatenate(v_list, axis=0)[:T]
        else:
            K_full = np.zeros((T, num_kv_heads, d), dtype=np.float32)
            V_full = np.zeros((T, num_kv_heads, d), dtype=np.float32)

        out = np.zeros((nq, d), dtype=np.float32)
        for h in range(nq):
            kv_h = h // heads_per_kv
            pos = np.arange(T)
            k_rot = apply_rope_givens(K_full[:, kv_h, :], pos, base=1000000.0)
            scores = (k_rot @ q[h]) * inv_sqrt_d

            # Mask out pruned body blocks
            body_start = num_sinks
            body_end = max(T - (recent_k.shape[0] if recent_k is not None else 0), num_sinks)
            for b in range(num_blocks):
                if not approved_mask[b, kv_h]:
                    b_start = body_start + b * 64
                    b_end = min(b_start + 64, body_end)
                    if b_start < body_end:
                        scores[b_start:b_end] = -np.inf

            m_val = np.max(scores)
            weights = np.exp(scores - m_val)
            denom = np.sum(weights)
            if denom > 0:
                weights /= denom
            out[h] = weights @ V_full[:, kv_h, :]

        telemetry = {
            "gpu_us": 0.0,
            "approved_blocks": approved_blocks,
            "total_blocks": total_blocks,
            "pruning_ratio": 1.0 - (float(approved_blocks) / max(total_blocks, 1)),
        }
        return out, telemetry
