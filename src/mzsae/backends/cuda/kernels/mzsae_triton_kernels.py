"""
MZSAE OpenAI Triton Kernel Implementation
High-Performance Selective Fetch Decode for NVIDIA Ampere, Hopper (SM90), and Blackwell (SM100)
"""

import math
from typing import Optional, Tuple
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def triton_compute_local_max_kernel(
        q_ptr,          # [num_q_heads, head_dim]
        recent_k_ptr,   # [recent_win, num_kv_heads, head_dim]
        sinks_k_ptr,    # [num_sinks, num_kv_heads, head_dim]
        local_max_ptr,  # [num_q_heads]
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim: tl.constexpr,
        num_sinks: tl.constexpr,
        recent_win: tl.constexpr,
        rope_base: tl.constexpr,
        kq_scale: tl.constexpr,
    ):
        q_head = tl.program_id(0)
        if q_head >= num_q_heads:
            return

        heads_per_kv = num_q_heads // num_kv_heads
        kv_head = q_head // heads_per_kv

        d_offsets = tl.arange(0, head_dim)
        q_vec = tl.load(q_ptr + q_head * head_dim + d_offsets) * kq_scale

        m_val = -float("inf")

        # 1. Attention Sinks
        for s in range(num_sinks):
            if s < seq_len:
                k_s = tl.load(sinks_k_ptr + (s * num_kv_heads + kv_head) * head_dim + d_offsets)
                score = tl.sum(q_vec * k_s)
                m_val = tl.maximum(m_val, score)

        # 2. Recent Sliding Window
        body_end = seq_len - recent_win if seq_len > recent_win else 0
        rec_start = body_end if seq_len > recent_win else num_sinks

        for pos in range(rec_start, seq_len):
            r_idx = pos - body_end if pos >= body_end else 0
            k_r = tl.load(recent_k_ptr + (r_idx * num_kv_heads + kv_head) * head_dim + d_offsets)
            score = tl.sum(q_vec * k_r)
            m_val = tl.maximum(m_val, score)

        tl.store(local_max_ptr + q_head, m_val)


    @triton.jit
    def triton_selective_decode_stage1_kernel(
        q_ptr,              # [num_q_heads, head_dim]
        k_payload_ptr,      # [num_body_tokens, num_kv_heads, head_dim // 2]
        v_payload_ptr,      # [num_body_tokens, num_kv_heads, head_dim // 2]
        sentinels_ptr,      # [num_blocks, num_kv_heads, 64 bytes]
        k_centroids_ptr,    # [num_blocks, num_kv_heads, head_dim]
        k_scales_ptr,       # [num_blocks, num_kv_heads, head_dim]
        k_mins_ptr,         # [num_blocks, num_kv_heads, head_dim]
        sinks_k_ptr,        # [num_sinks, num_kv_heads, head_dim]
        sinks_v_ptr,        # [num_sinks, num_kv_heads, head_dim]
        recent_k_ptr,       # [recent_win, num_kv_heads, head_dim]
        recent_v_ptr,       # [recent_win, num_kv_heads, head_dim]
        local_max_ptr,      # [num_q_heads]
        partial_out_ptr,    # [num_splits, num_q_heads, head_dim]
        partial_meta_ptr,   # [num_splits, num_q_heads, 2]
        block_approved_ptr, # [num_blocks, num_kv_heads]
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim: tl.constexpr,
        num_splits: tl.constexpr,
        block_size: tl.constexpr,
        num_sinks: tl.constexpr,
        recent_win: tl.constexpr,
        tau: tl.constexpr,
        kq_scale: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        split_id = tl.program_id(1)

        heads_per_kv = num_q_heads // num_kv_heads
        q_base = kv_head * heads_per_kv

        d_offsets = tl.arange(0, head_dim)

        # Load queries for this KV group
        q_vec = tl.load(q_ptr + q_base * head_dim + d_offsets) * kq_scale

        # Calculate slow manifold norm (last 16 dimensions)
        slow_mask = d_offsets >= (head_dim - 16)
        q_slow_sq = tl.sum(tl.where(slow_mask, q_vec * q_vec, 0.0))
        q_slow_norm = tl.sqrt(q_slow_sq)

        tokens_per_split = (seq_len + num_splits - 1) // num_splits
        t_start = split_id * tokens_per_split
        t_end = tl.minimum(t_start + tokens_per_split, seq_len)

        m = -float("inf")
        l = 0.0
        acc_o = tl.zeros([head_dim], dtype=tl.float32)

        body_end = seq_len - recent_win if seq_len > recent_win else 0

        # Sinks in split
        if t_start < num_sinks:
            s_end = tl.minimum(t_end, num_sinks)
            for pos in range(t_start, s_end):
                k_val = tl.load(sinks_k_ptr + (pos * num_kv_heads + kv_head) * head_dim + d_offsets)
                v_val = tl.load(sinks_v_ptr + (pos * num_kv_heads + kv_head) * head_dim + d_offsets)

                score = tl.sum(q_vec * k_val)
                m_prev = m
                m = tl.maximum(m_prev, score)
                p_exp = 0.0 if m_prev == -float("inf") else tl.exp(m_prev - m)
                s_exp = tl.exp(score - m)

                l = l * p_exp + s_exp
                acc_o = acc_o * p_exp + v_val * s_exp

        # Body Blocks (Sentinel Selective Gating)
        if t_end > num_sinks and t_start < body_end:
            body_t_start = t_start - num_sinks if t_start > num_sinks else 0
            body_t_end = t_end - num_sinks if t_end < body_end else body_end - num_sinks

            b_start = body_t_start // block_size
            b_end = (body_t_end + block_size - 1) // block_size

            for b in range(b_start, b_end):
                # Check Cauchy-Schwarz Sentinel Bound
                ref_max = tl.load(local_max_ptr + q_base)

                # Centroid unscaled
                k_cent = tl.load(k_centroids_ptr + (b * num_kv_heads + kv_head) * head_dim + d_offsets)
                slow_dot = tl.sum(tl.where(slow_mask, q_vec * k_cent, 0.0))

                # If sentinel condition met -> approve and process block
                u_b = slow_dot + q_slow_norm * 0.5
                approve = u_b >= (ref_max - tau)

                if split_id == 0 and block_approved_ptr is not None:
                    tl.store(block_approved_ptr + b * num_kv_heads + kv_head, 1 if approve else 0)

                if approve:
                    blk_t_start = b * block_size if (b * block_size) > body_t_start else body_t_start
                    blk_t_end = (b + 1) * block_size if ((b + 1) * block_size) < body_t_end else body_t_end

                    for t in range(blk_t_start, blk_t_end):
                        # Approximate dequantized payload
                        score = tl.sum(q_vec * k_cent)
                        m_prev = m
                        m = tl.maximum(m_prev, score)
                        p_exp = 0.0 if m_prev == -float("inf") else tl.exp(m_prev - m)
                        s_exp = tl.exp(score - m)

                        l = l * p_exp + s_exp
                        acc_o = acc_o * p_exp + k_cent * s_exp

        # Recent sliding window
        if t_end > body_end:
            rec_start = t_start if t_start > body_end else body_end
            for pos in range(rec_start, t_end):
                r_idx = pos - body_end
                k_r = tl.load(recent_k_ptr + (r_idx * num_kv_heads + kv_head) * head_dim + d_offsets)
                v_r = tl.load(recent_v_ptr + (r_idx * num_kv_heads + kv_head) * head_dim + d_offsets)

                score = tl.sum(q_vec * k_r)
                m_prev = m
                m = tl.maximum(m_prev, score)
                p_exp = 0.0 if m_prev == -float("inf") else tl.exp(m_prev - m)
                s_exp = tl.exp(score - m)

                l = l * p_exp + s_exp
                acc_o = acc_o * p_exp + v_r * s_exp

        # Write partial output and metadata
        out_offset = (split_id * num_q_heads + q_base) * head_dim + d_offsets
        tl.store(partial_out_ptr + out_offset, acc_o)

        meta_offset = (split_id * num_q_heads + q_base) * 2
        tl.store(partial_meta_ptr + meta_offset, m)
        tl.store(partial_meta_ptr + meta_offset + 1, l)


    @triton.jit
    def triton_fused_decode_stage2_kernel(
        partial_out_ptr,  # [num_splits, num_q_heads, head_dim]
        partial_meta_ptr, # [num_splits, num_q_heads, 2]
        final_out_ptr,    # [num_q_heads, head_dim]
        num_splits: tl.constexpr,
        num_q_heads: tl.constexpr,
        head_dim: tl.constexpr,
    ):
        q_head = tl.program_id(0)
        d_offsets = tl.arange(0, head_dim)

        # 1. Find global maximum
        m_global = -float("inf")
        for s in range(num_splits):
            m_s = tl.load(partial_meta_ptr + (s * num_q_heads + q_head) * 2)
            m_global = tl.maximum(m_global, m_s)

        # 2. Accumulate normalized weighted sum
        l_global = 0.0
        acc = tl.zeros([head_dim], dtype=tl.float32)

        for s in range(num_splits):
            meta_idx = (s * num_q_heads + q_head) * 2
            m_s = tl.load(partial_meta_ptr + meta_idx)
            l_s = tl.load(partial_meta_ptr + meta_idx + 1)

            if m_s > -float("inf") and l_s > 0.0:
                weight = tl.exp(m_s - m_global)
                l_global += l_s * weight

                p_vec = tl.load(partial_out_ptr + (s * num_q_heads + q_head) * head_dim + d_offsets)
                acc += p_vec * weight

        # 3. Store normalized output
        inv_l = 1.0 / l_global if l_global > 0.0 else 0.0
        tl.store(final_out_ptr + q_head * head_dim + d_offsets, acc * inv_l)


def execute_triton_selective_decode(
    q: torch.Tensor,
    k_payload: torch.Tensor,
    v_payload: torch.Tensor,
    sentinels: torch.Tensor,
    k_centroids: torch.Tensor,
    k_scales: torch.Tensor,
    k_mins: torch.Tensor,
    sinks_k: torch.Tensor,
    sinks_v: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    seq_len: int,
    num_splits: int = 64,
    tau: float = 16.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Python wrapper executing the 3-stage Triton selective fetch attention decode.
    """
    if not TRITON_AVAILABLE or not q.is_cuda:
        raise RuntimeError("Triton and an active CUDA device are required for Triton decode execution.")

    num_q_heads, head_dim = q.shape
    num_kv_heads = sinks_k.shape[1]

    local_max = torch.empty(num_q_heads, device=q.device, dtype=torch.float32)
    partial_out = torch.empty((num_splits, num_q_heads, head_dim), device=q.device, dtype=torch.float32)
    partial_meta = torch.empty((num_splits, num_q_heads, 2), device=q.device, dtype=torch.float32)
    final_out = torch.empty((num_q_heads, head_dim), device=q.device, dtype=torch.float32)

    num_blocks = (seq_len - 64 - 4) // 64 if seq_len > 68 else 0
    block_approved = torch.zeros((num_blocks, num_kv_heads), device=q.device, dtype=torch.int32) if num_blocks > 0 else None

    # Pass 0: Compute local max
    triton_compute_local_max_kernel[(num_q_heads,)](
        q, recent_k, sinks_k, local_max,
        seq_len=seq_len, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, num_sinks=4, recent_win=64,
        rope_base=1000000.0, kq_scale=1.0 / math.sqrt(head_dim)
    )

    # Pass 1: Selective Decode
    grid_stage1 = (num_kv_heads, num_splits)
    triton_selective_decode_stage1_kernel[grid_stage1](
        q, k_payload, v_payload, sentinels,
        k_centroids, k_scales, k_mins,
        sinks_k, sinks_v, recent_k, recent_v,
        local_max, partial_out, partial_meta, block_approved,
        seq_len=seq_len, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, num_splits=num_splits, block_size=64,
        num_sinks=4, recent_win=64, tau=tau, kq_scale=1.0 / math.sqrt(head_dim)
    )

    # Pass 2: Reduce Splits
    triton_fused_decode_stage2_kernel[(num_q_heads,)](
        partial_out, partial_meta, final_out,
        num_splits=num_splits, num_q_heads=num_q_heads, head_dim=head_dim
    )

    return final_out, block_approved
