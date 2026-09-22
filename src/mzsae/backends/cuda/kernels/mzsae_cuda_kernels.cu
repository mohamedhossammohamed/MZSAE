#include "mzsae_cuda.h"
#include <math.h>

namespace mzsae {
namespace cuda {

// ============================================================================
// PASS 0: COMPUTE LOCAL MAXIMUM OVER ATTENTION SINKS AND RECENT TOKENS
// ============================================================================

__global__ void mzsae_compute_local_max_kernel(
    const float* __restrict__ q,             // [num_q_heads, 128]
    const __half* __restrict__ recent_k,     // [recent_win, num_kv_heads, 128]
    const __half* __restrict__ sinks_k,      // [num_sinks, num_kv_heads, 128]
    float* __restrict__ local_max,           // [num_q_heads]
    MZSAECUDAParams p
) {
    uint32_t q_head = blockIdx.x;
    uint32_t lane   = threadIdx.x; // 0..31
    if (q_head >= p.num_q_heads || lane >= WARP_SIZE) return;

    uint32_t kv_head = q_head / (p.num_q_heads / p.num_kv_heads);

    // RoPE frequencies for 4 elements per lane
    float freq0 = 1.0f / powf(p.rope_base, (float)(4 * lane) / 128.0f);
    float freq1 = 1.0f / powf(p.rope_base, (float)(4 * lane + 2) / 128.0f);

    uint32_t q_off = q_head * 128 + 4 * lane;
    float4 q_val = make_float4(q[q_off], q[q_off + 1], q[q_off + 2], q[q_off + 3]);
    float4 q_reg = make_float4(
        q_val.x * p.kq_scale,
        q_val.y * p.kq_scale,
        q_val.z * p.kq_scale,
        q_val.w * p.kq_scale
    );

    float m_val = -INFINITY;

    // 1. Attention Sinks
    for (uint32_t s = 0; s < p.num_sinks && s < p.seq_len; ++s) {
        uint32_t s_off = (s * p.num_kv_heads + kv_head) * 128 + 4 * lane;
        float4 k_val = make_float4(
            __half2float(sinks_k[s_off]),
            __half2float(sinks_k[s_off + 1]),
            __half2float(sinks_k[s_off + 2]),
            __half2float(sinks_k[s_off + 3])
        );

        float th0 = (float)s * freq0;
        float th1 = (float)s * freq1;
        float2 rot0 = apply_rope_pair(make_float2(k_val.x, k_val.y), cosf(th0), sinf(th0));
        float2 rot1 = apply_rope_pair(make_float2(k_val.z, k_val.w), cosf(th1), sinf(th1));

        float dot_prod = q_reg.x * rot0.x + q_reg.y * rot0.y + q_reg.z * rot1.x + q_reg.w * rot1.y;
        float score = warp_reduce_sum(dot_prod);
        m_val = fmaxf(m_val, score);
    }

    // 2. Recent Sliding Window
    uint32_t body_end = (p.seq_len > p.recent_win) ? (p.seq_len - p.recent_win) : 0;
    uint32_t rec_start = (p.seq_len > p.recent_win) ? body_end : p.num_sinks;

    for (uint32_t pos = rec_start; pos < p.seq_len; ++pos) {
        uint32_t r_idx = (pos >= body_end) ? (pos - body_end) : 0;
        uint32_t r_off = (r_idx * p.num_kv_heads + kv_head) * 128 + 4 * lane;
        float4 k_val = make_float4(
            __half2float(recent_k[r_off]),
            __half2float(recent_k[r_off + 1]),
            __half2float(recent_k[r_off + 2]),
            __half2float(recent_k[r_off + 3])
        );

        float th0 = (float)pos * freq0;
        float th1 = (float)pos * freq1;
        float2 rot0 = apply_rope_pair(make_float2(k_val.x, k_val.y), cosf(th0), sinf(th0));
        float2 rot1 = apply_rope_pair(make_float2(k_val.z, k_val.w), cosf(th1), sinf(th1));

        float dot_prod = q_reg.x * rot0.x + q_reg.y * rot0.y + q_reg.z * rot1.x + q_reg.w * rot1.y;
        float score = warp_reduce_sum(dot_prod);
        m_val = fmaxf(m_val, score);
    }

    if (lane == 0) {
        local_max[q_head] = m_val;
    }
}

// ============================================================================
// STAGE 1: PLANE-2 SENTINEL SELECTIVE FETCH ATTENTION DECODE
// ============================================================================

__global__ void mzsae_selective_decode_stage1_kernel(
    const float* __restrict__ q,                 // [num_q_heads, 128]
    const uint8_t* __restrict__ k_payload,       // [num_body_tokens, num_kv_heads, 64]
    const uint8_t* __restrict__ v_payload,       // [num_body_tokens, num_kv_heads, 64]
    const BlockSentinel* __restrict__ sentinels, // [num_blocks, num_kv_heads] (Plane-2)
    const __half* __restrict__ k_centroids,      // [num_blocks, num_kv_heads, 128]
    const __half* __restrict__ k_scales,         // [num_blocks, num_kv_heads, 128]
    const __half* __restrict__ k_mins,           // [num_blocks, num_kv_heads, 128]
    const __half* __restrict__ v_group_meta,     // [num_body_tokens, num_kv_heads, 8]
    const __half* __restrict__ sinks_k,          // [num_sinks, num_kv_heads, 128]
    const __half* __restrict__ sinks_v,          // [num_sinks, num_kv_heads, 128]
    const __half* __restrict__ recent_k,         // [recent_win, num_kv_heads, 128]
    const __half* __restrict__ recent_v,         // [recent_win, num_kv_heads, 128]
    const float* __restrict__ local_max,         // [num_q_heads]
    float4* __restrict__ partial_out,            // [num_splits, num_q_heads, 32]
    float2* __restrict__ partial_meta,           // [num_splits, num_q_heads] (max, sum)
    uint32_t* __restrict__ block_approved,       // [num_blocks, num_kv_heads] (bitmap)
    MZSAECUDAParams p
) {
    uint32_t kv_head  = blockIdx.x; // 0..num_kv_heads-1
    uint32_t split_id = blockIdx.y; // 0..num_splits-1
    uint32_t lane     = threadIdx.x; // 0..31 (warp execution)

    if (kv_head >= p.num_kv_heads || split_id >= p.num_splits || lane >= WARP_SIZE) return;

    // RoPE frequencies for lane
    float freq0 = 1.0f / powf(p.rope_base, (float)(4 * lane) / 128.0f);
    float freq1 = 1.0f / powf(p.rope_base, (float)(4 * lane + 2) / 128.0f);

    uint32_t heads_per_kv = p.num_q_heads / p.num_kv_heads;
    uint32_t q_base = kv_head * heads_per_kv;

    // GQA query head registers (supporting up to 8 query heads per KV head)
    const int MAX_Q_HEADS = 8;
    float4 q_regs[MAX_Q_HEADS];
    float  q_slow_norm[MAX_Q_HEADS];

    for (uint32_t h = 0; h < heads_per_kv && h < MAX_Q_HEADS; ++h) {
        uint32_t q_off = (q_base + h) * 128 + 4 * lane;
        float4 q_raw = make_float4(q[q_off], q[q_off + 1], q[q_off + 2], q[q_off + 3]);
        q_regs[h] = make_float4(
            q_raw.x * p.kq_scale,
            q_raw.y * p.kq_scale,
            q_raw.z * p.kq_scale,
            q_raw.w * p.kq_scale
        );

        // Dimensions 112..127 are slow RoPE dimensions (lanes 28..31)
        float q_slow_sq = (lane >= 28) ? (q_raw.x * q_raw.x + q_raw.y * q_raw.y + q_raw.z * q_raw.z + q_raw.w * q_raw.w) : 0.0f;
        q_slow_norm[h] = sqrtf(warp_reduce_sum(q_slow_sq));
    }

    // Tokens mapped to this split
    uint32_t tokens_per_split = (p.seq_len + p.num_splits - 1) / p.num_splits;
    uint32_t t_start = split_id * tokens_per_split;
    uint32_t t_end   = min(t_start + tokens_per_split, p.seq_len);

    float  m[MAX_Q_HEADS];
    float  l[MAX_Q_HEADS];
    float4 acc_o[MAX_Q_HEADS];

    for (uint32_t h = 0; h < heads_per_kv && h < MAX_Q_HEADS; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        acc_o[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    uint32_t body_end = (p.seq_len > p.recent_win) ? (p.seq_len - p.recent_win) : 0;

    // ------------------------------------------------------------------------
    // 1. Attention Sinks (Unconditional execution, 0% pruning)
    // ------------------------------------------------------------------------
    if (t_start < p.num_sinks) {
        uint32_t s_end = min(t_end, p.num_sinks);
        for (uint32_t pos = t_start; pos < s_end; ++pos) {
            uint32_t s_off = (pos * p.num_kv_heads + kv_head) * 128 + 4 * lane;
            float4 k_val = make_float4(
                __half2float(sinks_k[s_off]),
                __half2float(sinks_k[s_off + 1]),
                __half2float(sinks_k[s_off + 2]),
                __half2float(sinks_k[s_off + 3])
            );
            float4 v_val = make_float4(
                __half2float(sinks_v[s_off]),
                __half2float(sinks_v[s_off + 1]),
                __half2float(sinks_v[s_off + 2]),
                __half2float(sinks_v[s_off + 3])
            );

            float th0 = (float)pos * freq0;
            float th1 = (float)pos * freq1;
            float2 rot0 = apply_rope_pair(make_float2(k_val.x, k_val.y), cosf(th0), sinf(th0));
            float2 rot1 = apply_rope_pair(make_float2(k_val.z, k_val.w), cosf(th1), sinf(th1));
            float4 k_rot = make_float4(rot0.x, rot0.y, rot1.x, rot1.y);

            for (uint32_t h = 0; h < heads_per_kv && h < MAX_Q_HEADS; ++h) {
                float dot_prod = q_regs[h].x * k_rot.x + q_regs[h].y * k_rot.y + q_regs[h].z * k_rot.z + q_regs[h].w * k_rot.w;
                float score = warp_reduce_sum(dot_prod);

                float m_prev = m[h];
                m[h] = fmaxf(m_prev, score);
                float p_exp = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m[h]);
                float s_exp = expf(score - m[h]);
                l[h] = l[h] * p_exp + s_exp;

                acc_o[h].x = acc_o[h].x * p_exp + v_val.x * s_exp;
                acc_o[h].y = acc_o[h].y * p_exp + v_val.y * s_exp;
                acc_o[h].z = acc_o[h].z * p_exp + v_val.z * s_exp;
                acc_o[h].w = acc_o[h].w * p_exp + v_val.w * s_exp;
            }
        }
    }

    // ------------------------------------------------------------------------
    // 2. Plane-2 Sentinel Selective Fetch (Hopper/Blackwell HBM Pruning)
    // ------------------------------------------------------------------------
    if (t_end > p.num_sinks && t_start < body_end) {
        uint32_t body_t_start = (t_start > p.num_sinks) ? (t_start - p.num_sinks) : 0;
        uint32_t body_t_end   = (t_end < body_end) ? (t_end - p.num_sinks) : (body_end - p.num_sinks);

        uint32_t b_start = body_t_start / p.block_size;
        uint32_t b_end   = (body_t_end + p.block_size - 1) / p.block_size;

        for (uint32_t b = b_start; b < b_end; ++b) {
            uint32_t num_blocks = body_end / p.block_size;
            uint32_t sent_idx = b * p.num_kv_heads + kv_head;
            BlockSentinel sent = sentinels[sent_idx];

            // Cauchy-Schwarz bound evaluated in parallel across lane registers
            float slow_dot = 0.0f;
            if (lane >= 28) {
                uint32_t s_idx = (lane - 28) * 4;
                float4 s_vec = make_float4(
                    __half2float(sent.s_slow[s_idx]),
                    __half2float(sent.s_slow[s_idx + 1]),
                    __half2float(sent.s_slow[s_idx + 2]),
                    __half2float(sent.s_slow[s_idx + 3])
                );
                // Compute unscaled dot product on slow dims
                uint32_t q_off = q_base * 128 + 4 * lane;
                float4 q_raw = make_float4(q[q_off], q[q_off + 1], q[q_off + 2], q[q_off + 3]);
                slow_dot = q_raw.x * s_vec.x + q_raw.y * s_vec.y + q_raw.z * s_vec.z + q_raw.w * s_vec.w;
            }
            float slow_dot_sum = warp_reduce_sum(slow_dot);

            // Sentinel bound decision across all query heads
            bool approve = false;
            for (uint32_t h = 0; h < heads_per_kv && h < MAX_Q_HEADS; ++h) {
                float u_b = (slow_dot_sum + q_slow_norm[h] * sent.R_delta + sent.C_fast) * p.kq_scale;
                float ref_max = local_max[q_base + h];
                if (u_b >= ref_max - p.tau) {
                    approve = true;
                    break;
                }
            }

            // Record hardware approval telemetry bitmap
            if (lane == 0 && split_id == 0 && block_approved != nullptr) {
                block_approved[sent_idx] = approve ? 1 : 0;
            }

            // RED-TEAM VERIFIED: Pruned blocks execute 0 DRAM payload reads
            if (!approve) continue;

            // Fetch and dequantize approved block payload
            uint32_t blk_t_start = max(b * p.block_size, body_t_start);
            uint32_t blk_t_end   = min((b + 1) * p.block_size, body_t_end);

            uint32_t c_off = (b * p.num_kv_heads + kv_head) * 128 + 4 * lane;
            float4 k_cent = make_float4(
                __half2float(k_centroids[c_off]),
                __half2float(k_centroids[c_off + 1]),
                __half2float(k_centroids[c_off + 2]),
                __half2float(k_centroids[c_off + 3])
            );
            float4 k_sc = make_float4(
                __half2float(k_scales[c_off]),
                __half2float(k_scales[c_off + 1]),
                __half2float(k_scales[c_off + 2]),
                __half2float(k_scales[c_off + 3])
            );
            float4 k_mn = make_float4(
                __half2float(k_mins[c_off]),
                __half2float(k_mins[c_off + 1]),
                __half2float(k_mins[c_off + 2]),
                __half2float(k_mins[c_off + 3])
            );

            for (uint32_t t = blk_t_start; t < blk_t_end; ++t) {
                uint32_t pos = p.num_sinks + t;
                uint32_t py_off = (t * p.num_kv_heads + kv_head) * 64 + 2 * lane;
                uint8_t byte0 = k_payload[py_off];
                uint8_t byte1 = k_payload[py_off + 1];

                // Dequantize 4-bit nibbles
                float4 k_deq = make_float4(
                    (float)(byte0 & 0x0F) * k_sc.x + k_mn.x + k_cent.x,
                    (float)(byte0 >> 4)   * k_sc.y + k_mn.y + k_cent.y,
                    (float)(byte1 & 0x0F) * k_sc.z + k_mn.z + k_cent.z,
                    (float)(byte1 >> 4)   * k_sc.w + k_mn.w + k_cent.w
                );

                uint8_t v_byte0 = v_payload[py_off];
                uint8_t v_byte1 = v_payload[py_off + 1];
                float4 v_deq = make_float4(
                    (float)(v_byte0 & 0x0F),
                    (float)(v_byte0 >> 4),
                    (float)(v_byte1 & 0x0F),
                    (float)(v_byte1 >> 4)
                );

                float th0 = (float)pos * freq0;
                float th1 = (float)pos * freq1;
                float2 rot0 = apply_rope_pair(make_float2(k_deq.x, k_deq.y), cosf(th0), sinf(th0));
                float2 rot1 = apply_rope_pair(make_float2(k_deq.z, k_deq.w), cosf(th1), sinf(th1));
                float4 k_rot = make_float4(rot0.x, rot0.y, rot1.x, rot1.y);

                for (uint32_t h = 0; h < heads_per_kv && h < MAX_Q_HEADS; ++h) {
                    float dot_prod = q_regs[h].x * k_rot.x + q_regs[h].y * k_rot.y + q_regs[h].z * k_rot.z + q_regs[h].w * k_rot.w;
                    float score = warp_reduce_sum(dot_prod);

                    float m_prev = m[h];
                    m[h] = fmaxf(m_prev, score);
                    float p_exp = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m[h]);
                    float s_exp = expf(score - m[h]);
                    l[h] = l[h] * p_exp + s_exp;

                    acc_o[h].x = acc_o[h].x * p_exp + v_deq.x * s_exp;
                    acc_o[h].y = acc_o[h].y * p_exp + v_deq.y * s_exp;
                    acc_o[h].z = acc_o[h].z * p_exp + v_deq.z * s_exp;
                    acc_o[h].w = acc_o[h].w * p_exp + v_deq.w * s_exp;
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 3. Recent Sliding Window Tokens (Always unpruned)
    // ------------------------------------------------------------------------
    if (t_end > body_end) {
        uint32_t rec_start = max(t_start, body_end);
        for (uint32_t pos = rec_start; pos < t_end; ++pos) {
            uint32_t r_idx = pos - body_end;
            uint32_t r_off = (r_idx * p.num_kv_heads + kv_head) * 128 + 4 * lane;
            float4 k_val = make_float4(
                __half2float(recent_k[r_off]),
                __half2float(recent_k[r_off + 1]),
                __half2float(recent_k[r_off + 2]),
                __half2float(recent_k[r_off + 3])
            );
            float4 v_val = make_float4(
                __half2float(recent_v[r_off]),
                __half2float(recent_v[r_off + 1]),
                __half2float(recent_v[r_off + 2]),
                __half2float(recent_v[r_off + 3])
            );

            float th0 = (float)pos * freq0;
            float th1 = (float)pos * freq1;
            float2 rot0 = apply_rope_pair(make_float2(k_val.x, k_val.y), cosf(th0), sinf(th0));
            float2 rot1 = apply_rope_pair(make_float2(k_val.z, k_val.w), cosf(th1), sinf(th1));
            float4 k_rot = make_float4(rot0.x, rot0.y, rot1.x, rot1.y);

            for (uint32_t h = 0; h < heads_per_kv && h < MAX_Q_HEADS; ++h) {
                float dot_prod = q_regs[h].x * k_rot.x + q_regs[h].y * k_rot.y + q_regs[h].z * k_rot.z + q_regs[h].w * k_rot.w;
                float score = warp_reduce_sum(dot_prod);

                float m_prev = m[h];
                m[h] = fmaxf(m_prev, score);
                float p_exp = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m[h]);
                float s_exp = expf(score - m[h]);
                l[h] = l[h] * p_exp + s_exp;

                acc_o[h].x = acc_o[h].x * p_exp + v_val.x * s_exp;
                acc_o[h].y = acc_o[h].y * p_exp + v_val.y * s_exp;
                acc_o[h].z = acc_o[h].z * p_exp + v_val.z * s_exp;
                acc_o[h].w = acc_o[h].w * p_exp + v_val.w * s_exp;
            }
        }
    }

    // Write partial outputs and meta
    for (uint32_t h = 0; h < heads_per_kv && h < MAX_Q_HEADS; ++h) {
        uint32_t q_head = q_base + h;
        uint32_t out_idx = (split_id * p.num_q_heads + q_head) * 32 + lane;
        partial_out[out_idx] = acc_o[h];

        if (lane == 0) {
            uint32_t meta_idx = split_id * p.num_q_heads + q_head;
            partial_meta[meta_idx] = make_float2(m[h], l[h]);
        }
    }
}

// ============================================================================
// STAGE 2: MULTI-SPLIT LOG-SUM-EXP ATTENTION REDUCTION
// ============================================================================

__global__ void mzsae_fused_decode_stage2_kernel(
    const float4* __restrict__ partial_out,  // [num_splits, num_q_heads, 32]
    const float2* __restrict__ partial_meta, // [num_splits, num_q_heads]
    float* __restrict__ final_out,           // [num_q_heads, 128]
    MZSAECUDAParams p
) {
    uint32_t q_head = blockIdx.x;
    uint32_t lane   = threadIdx.x; // 0..31
    if (q_head >= p.num_q_heads || lane >= WARP_SIZE) return;

    // 1. Find global maximum logit across all splits
    float m_global = -INFINITY;
    for (uint32_t s = 0; s < p.num_splits; ++s) {
        float m_s = partial_meta[s * p.num_q_heads + q_head].x;
        m_global = fmaxf(m_global, m_s);
    }

    // 2. Accumulate normalized sum and weighted partial vectors
    float l_global = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t s = 0; s < p.num_splits; ++s) {
        float2 meta = partial_meta[s * p.num_q_heads + q_head];
        float m_s = meta.x;
        float l_s = meta.y;

        if (m_s > -INFINITY && l_s > 0.0f) {
            float weight = expf(m_s - m_global);
            l_global += l_s * weight;

            uint32_t idx = (s * p.num_q_heads + q_head) * 32 + lane;
            float4 p_vec = partial_out[idx];
            acc.x += p_vec.x * weight;
            acc.y += p_vec.y * weight;
            acc.z += p_vec.z * weight;
            acc.w += p_vec.w * weight;
        }
    }

    // 3. Final normalization and write to output
    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    uint32_t out_off = q_head * 128 + 4 * lane;
    final_out[out_off]     = acc.x * inv_l;
    final_out[out_off + 1] = acc.y * inv_l;
    final_out[out_off + 2] = acc.z * inv_l;
    final_out[out_off + 3] = acc.w * inv_l;
}

} // namespace cuda
} // namespace mzsae
