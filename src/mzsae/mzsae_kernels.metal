#include <metal_stdlib>
using namespace metal;

struct MZSAEParams {
    uint seq_len;
    uint num_q_heads;    // 12
    uint num_kv_heads;   // 2
    uint head_dim;       // 128
    uint num_sinks;      // 4
    uint recent_win;     // 64
    uint num_splits;     // 16, 32, 64, or 128
    float rope_base;     // 1000000.0f
    float kq_scale;      // 1.0f / sqrt(128.0f)
    uint block_size;     // 64
};

// In-register RoPE rotation for adjacent pair format (llama.cpp / Qwen)
inline float4 apply_rope_fast(float4 k_unrot, float2 cs0, float2 cs1) {
    float4 k_rot;
    k_rot.x = k_unrot.x * cs0.x - k_unrot.y * cs0.y;
    k_rot.y = k_unrot.x * cs0.y + k_unrot.y * cs0.x;
    k_rot.z = k_unrot.z * cs1.x - k_unrot.w * cs1.y;
    k_rot.w = k_unrot.z * cs1.y + k_unrot.w * cs1.x;
    return k_rot;
}

// ============================================================================
// STAGE 1: FUSED COMPRESSED DECODE (GQA 6:1, Single K/V Dequant, Multi-Split)
// ============================================================================
kernel void mzsae_fused_decode_stage1(
    const device float   * q             [[buffer(0)]],   // [num_q_heads, 128]
    const device uchar   * k_payload     [[buffer(1)]],   // [num_body_tokens, num_kv_heads, 64]
    const device uchar   * v_payload     [[buffer(2)]],   // [num_body_tokens, num_kv_heads, 64]
    const device half4   * k_centroids   [[buffer(3)]],   // [num_blocks, num_kv_heads, 32]
    const device half4   * k_scales      [[buffer(4)]],   // [num_blocks, num_kv_heads, 32]
    const device half4   * k_mins        [[buffer(5)]],   // [num_blocks, num_kv_heads, 32]
    const device half4   * v_group_meta  [[buffer(6)]],   // [num_body_tokens, num_kv_heads, 2]
    const device half    * sinks_k       [[buffer(7)]],   // [num_sinks, num_kv_heads, 128]
    const device half    * sinks_v       [[buffer(8)]],   // [num_sinks, num_kv_heads, 128]
    const device half    * recent_k      [[buffer(9)]],   // [recent_win, num_kv_heads, 128]
    const device half    * recent_v      [[buffer(10)]],  // [recent_win, num_kv_heads, 128]
    device float4        * partial_out   [[buffer(11)]],  // [num_splits, num_q_heads, 32]
    device float2        * partial_meta  [[buffer(12)]],  // [num_splits, num_q_heads] (max, sum)
    constant MZSAEParams & p             [[buffer(13)]],
    uint3 tg_pos                         [[threadgroup_position_in_grid]],
    uint  simd_lane_id                   [[thread_index_in_simdgroup]]
) {
    uint kv_head  = tg_pos.x; // 0..1
    uint split_id = tg_pos.y; // 0..num_splits-1
    uint lane     = simd_lane_id; // 0..31

    // Precompute thread frequencies for RoPE once
    float freq0 = 1.0f / pow(p.rope_base, (float)(4 * lane) / 128.0f);
    float freq1 = 1.0f / pow(p.rope_base, (float)(4 * lane + 2) / 128.0f);

    // GQA: 6 query heads per KV head loaded into registers
    uint q_base = kv_head * 6;
    float4 q_regs[6];
    for (uint h = 0; h < 6; ++h) {
        uint q_off = (q_base + h) * 128 + 4 * lane;
        q_regs[h] = float4(q[q_off], q[q_off + 1], q[q_off + 2], q[q_off + 3]) * p.kq_scale;
    }

    // Token range for this split
    uint tokens_per_split = (p.seq_len + p.num_splits - 1) / p.num_splits;
    uint t_start = split_id * tokens_per_split;
    uint t_end   = min(t_start + tokens_per_split, p.seq_len);

    float m[6];
    float l[6];
    float4 acc_o[6];
    for (uint h = 0; h < 6; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        acc_o[h] = float4(0.0f);
    }

    uint body_end = (p.seq_len > p.recent_win) ? (p.seq_len - p.recent_win) : 0;

    uint curr_blk = 0xFFFFFFFF;
    half4 k_scale = half4(1.0h);
    half4 k_base  = half4(0.0h);

    for (uint pos = t_start; pos < t_end; ++pos) {
        float4 k_val;
        float4 v_val;

        if (pos < p.num_sinks) {
            uint s_off = (pos * p.num_kv_heads + kv_head) * 128 + 4 * lane;
            k_val = float4(sinks_k[s_off], sinks_k[s_off + 1], sinks_k[s_off + 2], sinks_k[s_off + 3]);
            v_val = float4(sinks_v[s_off], sinks_v[s_off + 1], sinks_v[s_off + 2], sinks_v[s_off + 3]);
        } else if (pos >= body_end) {
            uint r_idx = pos - body_end;
            uint r_off = (r_idx * p.num_kv_heads + kv_head) * 128 + 4 * lane;
            k_val = float4(recent_k[r_off], recent_k[r_off + 1], recent_k[r_off + 2], recent_k[r_off + 3]);
            v_val = float4(recent_v[r_off], recent_v[r_off + 1], recent_v[r_off + 2], recent_v[r_off + 3]);
        } else {
            uint t_body = pos - p.num_sinks;
            uint blk_idx = t_body / p.block_size;

            if (blk_idx != curr_blk) {
                curr_blk = blk_idx;
                uint blk_off = (blk_idx * p.num_kv_heads + kv_head) * 32 + lane;
                half4 k_cent = k_centroids[blk_off];
                k_scale = k_scales[blk_off];
                half4 k_min = k_mins[blk_off];
                k_base = k_cent + k_min;
            }

            uint tok_kv_idx = t_body * p.num_kv_heads + kv_head;
            uint p_off = tok_kv_idx * 64 + 2 * lane;
            const device uint16_t * kp16 = (const device uint16_t*)(k_payload + p_off);
            uint16_t raw_k = *kp16;
            half4 q_k = half4(raw_k & 0x0F, (raw_k >> 4) & 0x0F, (raw_k >> 8) & 0x0F, (raw_k >> 12) & 0x0F);
            k_val = float4(fma(q_k, k_scale, k_base));

            // Value dequantization (64-element groups)
            uint v_meta_off = tok_kv_idx * 2 + (lane / 16);
            half4 v_meta = v_group_meta[v_meta_off];
            half4 v_base = half4(v_meta.x + v_meta.y);
            half v_scale = v_meta.z;

            const device uint16_t * vp16 = (const device uint16_t*)(v_payload + p_off);
            uint16_t raw_v = *vp16;
            half4 q_v = half4(raw_v & 0x0F, (raw_v >> 4) & 0x0F, (raw_v >> 8) & 0x0F, (raw_v >> 12) & 0x0F);
            v_val = float4(fma(q_v, v_scale, v_base));
        }

        // Fast RoPE rotation in registers
        float th0 = (float)pos * freq0;
        float th1 = (float)pos * freq1;
        float4 k_rot = apply_rope_fast(k_val, float2(cos(th0), sin(th0)), float2(cos(th1), sin(th1)));

        // Reuse K and V across all 6 query heads
        for (uint h = 0; h < 6; ++h) {
            float score = simd_sum(dot(q_regs[h], k_rot));
            float m_prev = m[h];
            m[h] = max(m_prev, score);
            float p_exp = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m[h]);
            float s_exp = exp(score - m[h]);
            l[h] = l[h] * p_exp + s_exp;
            acc_o[h] = acc_o[h] * p_exp + v_val * s_exp;
        }
    }

    // Write partial outputs for all 6 query heads
    for (uint h = 0; h < 6; ++h) {
        uint q_head = q_base + h;
        uint out_idx = (split_id * p.num_q_heads + q_head) * 32 + lane;
        partial_out[out_idx] = acc_o[h];

        if (lane == 0) {
            uint meta_idx = split_id * p.num_q_heads + q_head;
            partial_meta[meta_idx] = float2(m[h], l[h]);
        }
    }
}

// ============================================================================
// STAGE 2: MERGE SPLITS ACROSS SEQUENCE PARTITIONS
// ============================================================================
kernel void mzsae_fused_decode_stage2(
    const device float4  * partial_out  [[buffer(0)]],   // [num_splits, num_q_heads, 32]
    const device float2  * partial_meta [[buffer(1)]],   // [num_splits, num_q_heads] (max, sum)
    device float         * final_out    [[buffer(2)]],   // [num_q_heads, 128]
    constant MZSAEParams & p            [[buffer(3)]],
    uint  q_head                        [[threadgroup_position_in_grid]],
    uint  simd_lane_id                  [[thread_index_in_simdgroup]]
) {
    uint lane = simd_lane_id;

    // 1. Global max across splits
    float global_max = -INFINITY;
    for (uint s = 0; s < p.num_splits; ++s) {
        float m_s = partial_meta[s * p.num_q_heads + q_head].x;
        global_max = max(global_max, m_s);
    }

    // 2. Rescale and accumulate
    float global_sum = 0.0f;
    float4 global_acc = float4(0.0f);

    for (uint s = 0; s < p.num_splits; ++s) {
        float2 meta = partial_meta[s * p.num_q_heads + q_head];
        float m_s = meta.x;
        float l_s = meta.y;

        if (m_s > -INFINITY && l_s > 0.0f) {
            float factor = exp(m_s - global_max);
            global_sum += l_s * factor;

            uint part_idx = (s * p.num_q_heads + q_head) * 32 + lane;
            global_acc += partial_out[part_idx] * factor;
        }
    }

    float4 o_val = (global_sum > 0.0f) ? (global_acc / global_sum) : float4(0.0f);

    uint out_offset = q_head * 128 + 4 * lane;
    final_out[out_offset + 0] = o_val.x;
    final_out[out_offset + 1] = o_val.y;
    final_out[out_offset + 2] = o_val.z;
    final_out[out_offset + 3] = o_val.w;
}

// ============================================================================
// TUNED DENSE REFERENCE KERNELS (For Verification & Baseline Benchmarking)
// ============================================================================
kernel void tuned_dense_decode_stage1(
    const device float   * q             [[buffer(0)]],   // [num_q_heads, 128]
    const device half4   * K             [[buffer(1)]],   // [seq_len, num_kv_heads, 32]
    const device half4   * V             [[buffer(2)]],   // [seq_len, num_kv_heads, 32]
    device float4        * partial_out   [[buffer(3)]],   // [num_splits, num_q_heads, 32]
    device float2        * partial_meta  [[buffer(4)]],   // [num_splits, num_q_heads]
    constant MZSAEParams & p             [[buffer(5)]],
    uint3 tg_pos                         [[threadgroup_position_in_grid]],
    uint  simd_lane_id                   [[thread_index_in_simdgroup]]
) {
    uint kv_head  = tg_pos.x;
    uint split_id = tg_pos.y;
    uint lane     = simd_lane_id;

    float freq0 = 1.0f / pow(p.rope_base, (float)(4 * lane) / 128.0f);
    float freq1 = 1.0f / pow(p.rope_base, (float)(4 * lane + 2) / 128.0f);

    uint q_base = kv_head * 6;
    float4 q_regs[6];
    for (uint h = 0; h < 6; ++h) {
        uint q_off = (q_base + h) * 128 + 4 * lane;
        q_regs[h] = float4(q[q_off], q[q_off + 1], q[q_off + 2], q[q_off + 3]) * p.kq_scale;
    }

    uint tokens_per_split = (p.seq_len + p.num_splits - 1) / p.num_splits;
    uint t_start = split_id * tokens_per_split;
    uint t_end   = min(t_start + tokens_per_split, p.seq_len);

    float m[6];
    float l[6];
    float4 acc_o[6];
    for (uint h = 0; h < 6; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        acc_o[h] = float4(0.0f);
    }

    for (uint pos = t_start; pos < t_end; ++pos) {
        uint kv_offset = (pos * p.num_kv_heads + kv_head) * 32 + lane;
        float4 k_val = float4(K[kv_offset]);
        float4 v_val = float4(V[kv_offset]);

        float th0 = (float)pos * freq0;
        float th1 = (float)pos * freq1;
        float4 k_rot = apply_rope_fast(k_val, float2(cos(th0), sin(th0)), float2(cos(th1), sin(th1)));

        for (uint h = 0; h < 6; ++h) {
            float score = simd_sum(dot(q_regs[h], k_rot));
            float m_prev = m[h];
            m[h] = max(m_prev, score);
            float p_exp = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m[h]);
            float s_exp = exp(score - m[h]);
            l[h] = l[h] * p_exp + s_exp;
            acc_o[h] = acc_o[h] * p_exp + v_val * s_exp;
        }
    }

    for (uint h = 0; h < 6; ++h) {
        uint q_head = q_base + h;
        uint out_idx = (split_id * p.num_q_heads + q_head) * 32 + lane;
        partial_out[out_idx] = acc_o[h];

        if (lane == 0) {
            uint meta_idx = split_id * p.num_q_heads + q_head;
            partial_meta[meta_idx] = float2(m[h], l[h]);
        }
    }
}

// ============================================================================
// STAGE 1 (SELECTIVE): PLANE-2 SENTINEL SELECTIVE FETCH DECODE
// Evaluates L2-resident Sentinels, skips 70-85% of blocks with 0 DRAM traffic
// ============================================================================

struct BlockSentinel {
    half     s_slow[16]; // [00-31] 16 slow RoPE projection dimensions (FP16)
    float    R_delta;    // [32-35] Residual Radius
    float    C_fast;     // [36-39] Fast-manifold bound
    float    a_cum;      // [40-43] Cumulative attention mass
    uint     t_last;     // [44-47] Last accessed timestep
    float    sigma2_k;   // [48-51] Spatial key variance
    float    V_pred;     // [52-55] Predicted value score
    uint64_t flags;      // [56-63] Flags & locks
};

struct MZSAESelectiveParams {
    uint seq_len;
    uint num_q_heads;    // 12
    uint num_kv_heads;   // 2
    uint head_dim;       // 128
    uint num_sinks;      // 4
    uint recent_win;     // 64
    uint num_splits;     // 16, 32, 64, or 128
    float rope_base;     // 1000000.0f
    float kq_scale;      // 1.0f / sqrt(128.0f)
    uint block_size;     // 64
    float tau;           // pruning threshold (e.g. 16.0f)
};

kernel void mzsae_compute_local_max(
    const device float                * q           [[buffer(0)]],  // [num_q_heads, 128]
    const device half                 * recent_k    [[buffer(1)]],  // [recent_win, num_kv_heads, 128]
    const device half                 * sinks_k     [[buffer(2)]],  // [num_sinks, num_kv_heads, 128]
    device float                      * local_max   [[buffer(3)]],  // [num_q_heads]
    constant MZSAESelectiveParams     & p           [[buffer(4)]],
    uint  q_head                                    [[threadgroup_position_in_grid]],
    uint  simd_lane_id                              [[thread_index_in_simdgroup]]
) {
    uint lane = simd_lane_id;
    uint kv_head = q_head / 6;

    float freq0 = 1.0f / pow(p.rope_base, (float)(4 * lane) / 128.0f);
    float freq1 = 1.0f / pow(p.rope_base, (float)(4 * lane + 2) / 128.0f);

    uint q_off = q_head * 128 + 4 * lane;
    float4 q_reg = float4(q[q_off], q[q_off + 1], q[q_off + 2], q[q_off + 3]) * p.kq_scale;

    float m_val = -INFINITY;

    // Sinks
    uint num_sinks = min(p.num_sinks, p.seq_len);
    for (uint s = 0; s < num_sinks; ++s) {
        uint s_off = (s * p.num_kv_heads + kv_head) * 128 + 4 * lane;
        float4 k_val = float4(sinks_k[s_off], sinks_k[s_off + 1], sinks_k[s_off + 2], sinks_k[s_off + 3]);
        float th0 = (float)s * freq0;
        float th1 = (float)s * freq1;
        float4 k_rot = apply_rope_fast(k_val, float2(cos(th0), sin(th0)), float2(cos(th1), sin(th1)));
        float score = simd_sum(dot(q_reg, k_rot));
        m_val = max(m_val, score);
    }

    // Recent window
    uint body_end = (p.seq_len > p.recent_win) ? (p.seq_len - p.recent_win) : 0;
    uint num_recent = p.seq_len - body_end;
    for (uint r = 0; r < num_recent; ++r) {
        uint pos = body_end + r;
        uint r_off = (r * p.num_kv_heads + kv_head) * 128 + 4 * lane;
        float4 k_val = float4(recent_k[r_off], recent_k[r_off + 1], recent_k[r_off + 2], recent_k[r_off + 3]);
        float th0 = (float)pos * freq0;
        float th1 = (float)pos * freq1;
        float4 k_rot = apply_rope_fast(k_val, float2(cos(th0), sin(th0)), float2(cos(th1), sin(th1)));
        float score = simd_sum(dot(q_reg, k_rot));
        m_val = max(m_val, score);
    }

    if (lane == 0) {
        local_max[q_head] = m_val;
    }
}

kernel void mzsae_selective_decode_stage1(
    const device float                * q             [[buffer(0)]],   // [num_q_heads, 128]
    const device uchar                * k_payload     [[buffer(1)]],   // [num_body_tokens, num_kv_heads, 64]
    const device uchar                * v_payload     [[buffer(2)]],   // [num_body_tokens, num_kv_heads, 64]
    const device BlockSentinel        * sentinels     [[buffer(3)]],   // [num_blocks, num_kv_heads]
    const device half4                * k_centroids   [[buffer(4)]],   // [num_blocks, num_kv_heads, 32]
    const device half4                * k_scales      [[buffer(5)]],   // [num_blocks, num_kv_heads, 32]
    const device half4                * k_mins        [[buffer(6)]],   // [num_blocks, num_kv_heads, 32]
    const device half4                * v_group_meta  [[buffer(7)]],   // [num_body_tokens, num_kv_heads, 2]
    const device half                 * sinks_k       [[buffer(8)]],   // [num_sinks, num_kv_heads, 128]
    const device half                 * sinks_v       [[buffer(9)]],   // [num_sinks, num_kv_heads, 128]
    const device half                 * recent_k      [[buffer(10)]],  // [recent_win, num_kv_heads, 128]
    const device half                 * recent_v      [[buffer(11)]],  // [recent_win, num_kv_heads, 128]
    const device float                * local_max     [[buffer(12)]],  // [num_q_heads]
    device float4                     * partial_out   [[buffer(13)]],  // [num_splits, num_q_heads, 32]
    device float2                     * partial_meta  [[buffer(14)]],  // [num_splits, num_q_heads]
    device uint                       * block_approved[[buffer(15)]],  // [num_blocks, num_kv_heads] (boolean bitmap)
    constant MZSAESelectiveParams     & p             [[buffer(16)]],
    uint3 tg_pos                                      [[threadgroup_position_in_grid]],
    uint  simd_lane_id                                [[thread_index_in_simdgroup]]
) {
    uint kv_head  = tg_pos.x; // 0..1
    uint split_id = tg_pos.y; // 0..num_splits-1
    uint lane     = simd_lane_id; // 0..31

    float freq0 = 1.0f / pow(p.rope_base, (float)(4 * lane) / 128.0f);
    float freq1 = 1.0f / pow(p.rope_base, (float)(4 * lane + 2) / 128.0f);

    uint q_base = kv_head * 6;
    float4 q_regs[6];
    float q_slow_norm[6];
    float q_fast_norm[6];

    for (uint h = 0; h < 6; ++h) {
        uint q_off = (q_base + h) * 128 + 4 * lane;
        float4 q_raw = float4(q[q_off], q[q_off + 1], q[q_off + 2], q[q_off + 3]);
        q_regs[h] = q_raw * p.kq_scale;

        float q_slow_sq = (lane >= 28) ? dot(q_raw, q_raw) : 0.0f;
        float q_fast_sq = (lane < 28) ? dot(q_raw, q_raw) : 0.0f;
        q_slow_norm[h] = sqrt(simd_sum(q_slow_sq));
        q_fast_norm[h] = sqrt(simd_sum(q_fast_sq));
    }

    uint tokens_per_split = (p.seq_len + p.num_splits - 1) / p.num_splits;
    uint t_start = split_id * tokens_per_split;
    uint t_end   = min(t_start + tokens_per_split, p.seq_len);

    float m[6];
    float l[6];
    float4 acc_o[6];
    for (uint h = 0; h < 6; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        acc_o[h] = float4(0.0f);
    }

    uint body_end = (p.seq_len > p.recent_win) ? (p.seq_len - p.recent_win) : 0;

    // 1. Sinks in this split (always unpruned)
    if (t_start < p.num_sinks) {
        uint s_end = min(t_end, p.num_sinks);
        for (uint pos = t_start; pos < s_end; ++pos) {
            uint s_off = (pos * p.num_kv_heads + kv_head) * 128 + 4 * lane;
            float4 k_val = float4(sinks_k[s_off], sinks_k[s_off + 1], sinks_k[s_off + 2], sinks_k[s_off + 3]);
            float4 v_val = float4(sinks_v[s_off], sinks_v[s_off + 1], sinks_v[s_off + 2], sinks_v[s_off + 3]);

            float th0 = (float)pos * freq0;
            float th1 = (float)pos * freq1;
            float4 k_rot = apply_rope_fast(k_val, float2(cos(th0), sin(th0)), float2(cos(th1), sin(th1)));

            for (uint h = 0; h < 6; ++h) {
                float score = simd_sum(dot(q_regs[h], k_rot));
                float m_prev = m[h];
                m[h] = max(m_prev, score);
                float p_exp = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m[h]);
                float s_exp = exp(score - m[h]);
                l[h] = l[h] * p_exp + s_exp;
                acc_o[h] = acc_o[h] * p_exp + v_val * s_exp;
            }
        }
    }

    // 2. Body Blocks in this split (Selective Sentinel Pruning)
    if (t_end > p.num_sinks && t_start < body_end) {
        uint body_t_start = (t_start > p.num_sinks) ? (t_start - p.num_sinks) : 0;
        uint body_t_end   = (t_end < body_end) ? (t_end - p.num_sinks) : (body_end - p.num_sinks);

        uint b_start = body_t_start / p.block_size;
        uint b_end   = (body_t_end + p.block_size - 1) / p.block_size;

        for (uint b = b_start; b < b_end; ++b) {
            uint desc_idx = b * p.num_kv_heads + kv_head;
            BlockSentinel desc = sentinels[desc_idx];

            bool approve = false;
            if (desc.flags & 2ULL) {
                approve = true;
            } else {
                float s_val = (lane < 16) ? float(desc.s_slow[lane]) : 0.0f;
                for (uint h = 0; h < 6; ++h) {
                    float q_elem = 0.0f;
                    if (lane < 16) {
                        uint q_off = (q_base + h) * 128 + 112 + lane;
                        q_elem = q[q_off];
                    }
                    float dot_slow = simd_sum(q_elem * s_val);
                    float u_bound = (dot_slow + q_slow_norm[h] * desc.R_delta + desc.C_fast) * p.kq_scale;

                    float ref_m = (local_max != nullptr) ? local_max[q_base + h] : m[h];
                    if (ref_m > -1e10f) {
                        if (u_bound >= (ref_m - p.tau)) {
                            approve = true;
                            break;
                        }
                    } else {
                        approve = true;
                        break;
                    }
                }
            }

            // RED-TEAM VERIFIED: Pruned blocks execute 0 DRAM payload reads.
            // `continue` branches past ALL Plane-1 device_loads below
            // (k_centroids/k_scales/k_mins fetch + k_payload/v_payload
            // dequant loop). Only the 64B Plane-2 sentinel (L2-resident)
            // + q-register reads above this point are touched. Sinks and
            // recent-window paths are intentionally always unpruned. There
            // is no hidden fallback that reloads a pruned block.
            // Telemetry note: block_approved bitmap is written by lane 0
            // ONLY for approved blocks and summed on GPU (splitStatsBuffer),
            // so approved/total counts are kernel-observed, but DRAM-byte
            // savings derived from them remain estimates (see LIMITATIONS.md).
            if (!approve) {
                // ZERO DRAM ACCESS: Skip all tokens in this block
                continue;
            }

            if (lane == 0 && block_approved != nullptr) {
                block_approved[b * p.num_kv_heads + kv_head] = 1;
            }

            uint blk_off = (b * p.num_kv_heads + kv_head) * 32 + lane;
            half4 k_cent = k_centroids[blk_off];
            half4 k_scale = k_scales[blk_off];
            half4 k_min = k_mins[blk_off];
            half4 k_base = k_cent + k_min;

            uint b_tok_start = max(b * p.block_size, body_t_start);
            uint b_tok_end   = min((b + 1) * p.block_size, body_t_end);

            for (uint t_body = b_tok_start; t_body < b_tok_end; ++t_body) {
                uint pos = t_body + p.num_sinks;

                uint tok_kv_idx = t_body * p.num_kv_heads + kv_head;
                uint p_off = tok_kv_idx * 64 + 2 * lane;
                const device uint16_t * kp16 = (const device uint16_t*)(k_payload + p_off);
                uint16_t raw_k = *kp16;
                half4 q_k = half4(raw_k & 0x0F, (raw_k >> 4) & 0x0F, (raw_k >> 8) & 0x0F, (raw_k >> 12) & 0x0F);
                float4 k_val = float4(fma(q_k, k_scale, k_base));

                uint v_meta_off = tok_kv_idx * 2 + (lane / 16);
                half4 v_meta = v_group_meta[v_meta_off];
                half4 v_base = half4(v_meta.x + v_meta.y);
                half v_scale = v_meta.z;

                const device uint16_t * vp16 = (const device uint16_t*)(v_payload + p_off);
                uint16_t raw_v = *vp16;
                half4 q_v = half4(raw_v & 0x0F, (raw_v >> 4) & 0x0F, (raw_v >> 8) & 0x0F, (raw_v >> 12) & 0x0F);
                float4 v_val = float4(fma(q_v, v_scale, v_base));

                float th0 = (float)pos * freq0;
                float th1 = (float)pos * freq1;
                float4 k_rot = apply_rope_fast(k_val, float2(cos(th0), sin(th0)), float2(cos(th1), sin(th1)));

                for (uint h = 0; h < 6; ++h) {
                    float score = simd_sum(dot(q_regs[h], k_rot));
                    float m_prev = m[h];
                    m[h] = max(m_prev, score);
                    float p_exp = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m[h]);
                    float s_exp = exp(score - m[h]);
                    l[h] = l[h] * p_exp + s_exp;
                    acc_o[h] = acc_o[h] * p_exp + v_val * s_exp;
                }
            }
        }
    }

    // 3. Recent window in this split (always unpruned)
    if (t_end > body_end) {
        uint r_start = max(t_start, body_end);
        for (uint pos = r_start; pos < t_end; ++pos) {
            uint r_idx = pos - body_end;
            uint r_off = (r_idx * p.num_kv_heads + kv_head) * 128 + 4 * lane;
            float4 k_val = float4(recent_k[r_off], recent_k[r_off + 1], recent_k[r_off + 2], recent_k[r_off + 3]);
            float4 v_val = float4(recent_v[r_off], recent_v[r_off + 1], recent_v[r_off + 2], recent_v[r_off + 3]);

            float th0 = (float)pos * freq0;
            float th1 = (float)pos * freq1;
            float4 k_rot = apply_rope_fast(k_val, float2(cos(th0), sin(th0)), float2(cos(th1), sin(th1)));

            for (uint h = 0; h < 6; ++h) {
                float score = simd_sum(dot(q_regs[h], k_rot));
                float m_prev = m[h];
                m[h] = max(m_prev, score);
                float p_exp = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m[h]);
                float s_exp = exp(score - m[h]);
                l[h] = l[h] * p_exp + s_exp;
                acc_o[h] = acc_o[h] * p_exp + v_val * s_exp;
            }
        }
    }

    // Write partial outputs
    for (uint h = 0; h < 6; ++h) {
        uint q_head = q_base + h;
        uint out_idx = (split_id * p.num_q_heads + q_head) * 32 + lane;
        partial_out[out_idx] = acc_o[h];

        if (lane == 0) {
            uint meta_idx = split_id * p.num_q_heads + q_head;
            partial_meta[meta_idx] = float2(m[h], l[h]);
        }
    }
}

