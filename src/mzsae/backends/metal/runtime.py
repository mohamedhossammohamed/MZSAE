"""
Metal C-ABI Driver Bridge
MZahran Sparse Attention Engine (MZSAE)
"""

import os
import ctypes
import numpy as np
from typing import Optional, Tuple, Dict, Any

from ..base import MZSAEBackend


class MetalBackend(MZSAEBackend):
    """Python ctypes bridge to libmzsae_metal.dylib on Apple Silicon."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            current_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(current_dir, "libmzsae_metal.dylib"),
                os.path.join(current_dir, "..", "..", "libmzsae_metal.dylib"),
                os.path.join(
                    current_dir, "..", "..", "..", "libmzsae_metal.dylib"
                ),
                os.path.join(os.getcwd(), "libmzsae_metal.dylib"),
                os.path.join(
                    os.getcwd(), "src", "mzsae", "libmzsae_metal.dylib"
                ),
            ]
            for c in candidates:
                if os.path.exists(c):
                    lib_path = os.path.abspath(c)
                    break

        if not lib_path or not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"libmzsae_metal.dylib not found at {lib_path}. Run make first."
            )

        self.lib = ctypes.CDLL(lib_path)
        self._setup_prototypes()

        current_dir = os.path.dirname(__file__)
        shader_candidates = [
            os.path.join(current_dir, "mzsae_kernels.metal"),
            os.path.join(current_dir, "kernels", "mzsae_kernels.metal"),
            os.path.join(current_dir, "..", "..", "mzsae_kernels.metal"),
            os.path.join(
                current_dir,
                "..",
                "..",
                "backends",
                "metal",
                "kernels",
                "mzsae_kernels.metal",
            ),
            os.path.join(
                current_dir, "..", "..", "..", "src", "metal", "mzsae_kernels.metal"
            ),
            os.path.join(os.getcwd(), "src", "metal", "mzsae_kernels.metal"),
        ]
        shader_bytes = None
        for sc in shader_candidates:
            if os.path.exists(sc):
                with open(sc, "r", encoding="utf-8") as f:
                    shader_bytes = f.read().encode("utf-8")
                break

        res = self.lib.mzsae_metal_init(shader_bytes)
        if res != 0:
            raise RuntimeError(
                f"Failed to initialize Metal runtime, code {res}"
            )

        self._device_name = self.lib.mzsae_metal_get_device_name().decode(
            "utf-8"
        )
        self.max_working_set = (
            self.lib.mzsae_metal_get_recommended_max_working_set()
        )

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def is_available(self) -> bool:
        return self.lib is not None

    def _setup_prototypes(self):
        self.lib.mzsae_metal_init.argtypes = [ctypes.c_char_p]
        self.lib.mzsae_metal_init.restype = ctypes.c_int

        self.lib.mzsae_metal_shutdown.argtypes = []
        self.lib.mzsae_metal_shutdown.restype = None

        self.lib.mzsae_metal_alloc_raw.argtypes = [ctypes.c_size_t]
        self.lib.mzsae_metal_alloc_raw.restype = ctypes.c_void_p

        self.lib.mzsae_metal_free.argtypes = [ctypes.c_void_p]
        self.lib.mzsae_metal_free.restype = None

        self.lib.mzsae_metal_get_device_name.argtypes = []
        self.lib.mzsae_metal_get_device_name.restype = ctypes.c_char_p

        self.lib.mzsae_metal_get_recommended_max_working_set.argtypes = []
        self.lib.mzsae_metal_get_recommended_max_working_set.restype = (
            ctypes.c_uint64
        )

        self.lib.mzsae_metal_clear_cache.argtypes = []
        self.lib.mzsae_metal_clear_cache.restype = None

        self.lib.mzsae_metal_fused_decode.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # q [num_q_heads, 128]
            ctypes.c_void_p,  # k_payload
            ctypes.c_void_p,  # v_payload
            ctypes.c_void_p,  # k_centroids
            ctypes.c_void_p,  # k_scales
            ctypes.c_void_p,  # k_mins
            ctypes.c_void_p,  # v_group_meta
            ctypes.c_void_p,  # sinks_k
            ctypes.c_void_p,  # sinks_v
            ctypes.c_void_p,  # recent_k
            ctypes.c_void_p,  # recent_v
            ctypes.c_uint32,  # seq_len
            ctypes.c_uint32,  # num_splits
            ctypes.POINTER(ctypes.c_float),  # final_out
            ctypes.c_uint32,  # recent_win
        ]
        self.lib.mzsae_metal_fused_decode.restype = ctypes.c_float

        self.lib.mzsae_metal_dense_decode.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # q [num_q_heads, 128]
            ctypes.c_void_p,  # dense_k_fp16
            ctypes.c_void_p,  # dense_v_fp16
            ctypes.c_uint32,  # seq_len
            ctypes.c_uint32,  # num_splits
            ctypes.POINTER(ctypes.c_float),  # final_out
        ]
        self.lib.mzsae_metal_dense_decode.restype = ctypes.c_float

        self.lib.mzsae_metal_selective_decode.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # q [num_q_heads, 128]
            ctypes.c_void_p,  # k_payload
            ctypes.c_void_p,  # v_payload
            ctypes.c_void_p,  # sentinels
            ctypes.c_void_p,  # k_centroids
            ctypes.c_void_p,  # k_scales
            ctypes.c_void_p,  # k_mins
            ctypes.c_void_p,  # v_group_meta
            ctypes.c_void_p,  # sinks_k
            ctypes.c_void_p,  # sinks_v
            ctypes.c_void_p,  # recent_k
            ctypes.c_void_p,  # recent_v
            ctypes.c_uint32,  # seq_len
            ctypes.c_uint32,  # num_splits
            ctypes.c_float,  # tau
            ctypes.POINTER(ctypes.c_float),  # final_out
            ctypes.POINTER(ctypes.c_uint32),  # out_approved_blocks
            ctypes.POINTER(ctypes.c_uint32),  # out_total_blocks
            ctypes.c_uint32,  # recent_win
        ]
        self.lib.mzsae_metal_selective_decode.restype = ctypes.c_float

    def clear_cache(self):
        """Clears the Metal runtime internal buffer allocation cache."""
        self.lib.mzsae_metal_clear_cache()

    def _prepare_q_out(
        self, q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        orig_nq, d = q.shape
        if orig_nq == 12:
            q_c = np.ascontiguousarray(q, dtype=np.float32)
            out_c = np.zeros((12, d), dtype=np.float32)
        elif orig_nq < 12:
            q_pad = np.zeros((12, d), dtype=np.float32)
            q_pad[:orig_nq] = q
            q_c = np.ascontiguousarray(q_pad, dtype=np.float32)
            out_c = np.zeros((12, d), dtype=np.float32)
        else:
            q_c = np.ascontiguousarray(q, dtype=np.float32)
            out_c = np.zeros((orig_nq, d), dtype=np.float32)
        return q_c, out_c, orig_nq

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
        return_gpu_time: bool = False,
        recent_win: Optional[int] = None,
    ):
        nq, d = q.shape
        if recent_win is None:
            recent_win = (
                recent_k.shape[0]
                if (recent_k is not None and recent_k.ndim > 0)
                else 64
            )

        if nq > 12:
            final_out = np.zeros((nq, d), dtype=np.float32)
            total_gpu_us = 0.0
            for start in range(0, nq, 12):
                end = min(start + 12, nq)
                sub_out, sub_gpu = self.fused_decode(
                    q=q[start:end],
                    k_payload=k_payload,
                    v_payload=v_payload,
                    k_centroids=k_centroids,
                    k_scales=k_scales,
                    k_mins=k_mins,
                    v_group_meta=v_group_meta,
                    sinks_k=sinks_k,
                    sinks_v=sinks_v,
                    recent_k=recent_k,
                    recent_v=recent_v,
                    seq_len=seq_len,
                    num_splits=num_splits,
                    return_gpu_time=True,
                    recent_win=recent_win,
                )
                final_out[start:end] = sub_out
                total_gpu_us += sub_gpu
            if return_gpu_time:
                return final_out, total_gpu_us
            return final_out

        q_c, out_c, orig_nq = self._prepare_q_out(q)

        gpu_us = self.lib.mzsae_metal_fused_decode(
            q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(k_payload.ctypes.data),
            ctypes.c_void_p(v_payload.ctypes.data),
            ctypes.c_void_p(k_centroids.ctypes.data),
            ctypes.c_void_p(k_scales.ctypes.data),
            ctypes.c_void_p(k_mins.ctypes.data),
            ctypes.c_void_p(v_group_meta.ctypes.data),
            ctypes.c_void_p(sinks_k.ctypes.data),
            ctypes.c_void_p(sinks_v.ctypes.data),
            ctypes.c_void_p(recent_k.ctypes.data),
            ctypes.c_void_p(recent_v.ctypes.data),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(num_splits),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(recent_win),
        )
        res = out_c[:orig_nq] if orig_nq < 12 else out_c
        if return_gpu_time:
            return res, float(gpu_us)
        return res

    def dense_decode(
        self,
        q: np.ndarray,
        dense_k_fp16: np.ndarray,
        dense_v_fp16: np.ndarray,
        seq_len: int,
        num_splits: int = 64,
        return_gpu_time: bool = False,
    ):
        nq, d = q.shape
        if nq > 12:
            final_out = np.zeros((nq, d), dtype=np.float32)
            total_gpu_us = 0.0
            for start in range(0, nq, 12):
                end = min(start + 12, nq)
                sub_out, sub_gpu = self.dense_decode(
                    q=q[start:end],
                    dense_k_fp16=dense_k_fp16,
                    dense_v_fp16=dense_v_fp16,
                    seq_len=seq_len,
                    num_splits=num_splits,
                    return_gpu_time=True,
                )
                final_out[start:end] = sub_out
                total_gpu_us += sub_gpu
            if return_gpu_time:
                return final_out, total_gpu_us
            return final_out

        q_c, out_c, orig_nq = self._prepare_q_out(q)
        k_c = np.ascontiguousarray(dense_k_fp16, dtype=np.float16)
        v_c = np.ascontiguousarray(dense_v_fp16, dtype=np.float16)

        gpu_us = self.lib.mzsae_metal_dense_decode(
            q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(k_c.ctypes.data),
            ctypes.c_void_p(v_c.ctypes.data),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(num_splits),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        res = out_c[:orig_nq] if orig_nq < 12 else out_c
        if return_gpu_time:
            return res, float(gpu_us)
        return res

    def flash_attn_dense(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        if k.ndim == 2:
            k = np.repeat(k[:, np.newaxis, :], 2, axis=1)
            v = np.repeat(v[:, np.newaxis, :], 2, axis=1)
        if q.ndim == 1:
            q = np.repeat(q[np.newaxis, :], 12, axis=0)
        seq_len = k.shape[0]
        out = self.dense_decode(
            q=q, dense_k_fp16=k, dense_v_fp16=v, seq_len=seq_len
        )
        return out[0]

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
        recent_win: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        nq, d = q.shape
        if recent_win is None:
            recent_win = (
                recent_k.shape[0]
                if (recent_k is not None and recent_k.ndim > 0)
                else 64
            )

        if nq > 12:
            final_out = np.zeros((nq, d), dtype=np.float32)
            total_telemetry = {
                "gpu_us": 0.0,
                "latency_us": 0.0,
                "approved_blocks": 0,
                "total_blocks": 0,
                "pruning_ratio": 0.0,
            }
            for start in range(0, nq, 12):
                end = min(start + 12, nq)
                sub_out, sub_tel = self.selective_decode(
                    q=q[start:end],
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
                    num_splits=num_splits,
                    tau=tau,
                    recent_win=recent_win,
                )
                final_out[start:end] = sub_out
                total_telemetry["gpu_us"] += sub_tel["gpu_us"]
                total_telemetry["latency_us"] += sub_tel.get("latency_us", sub_tel["gpu_us"])
                total_telemetry["approved_blocks"] = max(
                    total_telemetry["approved_blocks"],
                    sub_tel["approved_blocks"],
                )
                total_telemetry["total_blocks"] = max(
                    total_telemetry["total_blocks"], sub_tel["total_blocks"]
                )
            total_telemetry["pruning_ratio"] = (
                1.0
                - (
                    float(total_telemetry["approved_blocks"])
                    / max(total_telemetry["total_blocks"], 1)
                )
            )
            return final_out, total_telemetry

        q_c, out_c, orig_nq = self._prepare_q_out(q)
        sent_c = np.ascontiguousarray(sentinels, dtype=np.uint8)
        approved = ctypes.c_uint32(0)
        total = ctypes.c_uint32(0)

        gpu_us = self.lib.mzsae_metal_selective_decode(
            q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(k_payload.ctypes.data),
            ctypes.c_void_p(v_payload.ctypes.data),
            ctypes.c_void_p(sent_c.ctypes.data),
            ctypes.c_void_p(k_centroids.ctypes.data),
            ctypes.c_void_p(k_scales.ctypes.data),
            ctypes.c_void_p(k_mins.ctypes.data),
            ctypes.c_void_p(v_group_meta.ctypes.data),
            ctypes.c_void_p(sinks_k.ctypes.data),
            ctypes.c_void_p(sinks_v.ctypes.data),
            ctypes.c_void_p(recent_k.ctypes.data),
            ctypes.c_void_p(recent_v.ctypes.data),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(num_splits),
            ctypes.c_float(tau),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(approved),
            ctypes.byref(total),
            ctypes.c_uint32(recent_win),
        )
        res = out_c[:orig_nq] if orig_nq < 12 else out_c
        app_val = int(approved.value)
        tot_val = int(total.value)
        telemetry = {
            "gpu_us": float(gpu_us),
            "latency_us": float(gpu_us),
            "approved_blocks": app_val,
            "total_blocks": tot_val,
            "pruning_ratio": (
                1.0 - (float(app_val) / max(tot_val, 1)) if tot_val > 0 else 0.0
            ),
        }
        return res, telemetry
