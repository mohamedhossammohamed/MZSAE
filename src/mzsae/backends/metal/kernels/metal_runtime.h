#ifndef MZSAE_METAL_RUNTIME_H
#define MZSAE_METAL_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Metal runtime and compile shaders. Returns 0 on success.
int mzsae_metal_init(const char* shader_source);
void mzsae_metal_shutdown(void);

// Buffer allocation in unified memory
void* mzsae_metal_alloc_raw(size_t bytes);
void  mzsae_metal_free(void* ptr);
void  mzsae_metal_clear_cache(void);

// Fused multi-split 4-bit MZSAE decode (returns GPU execution time in microseconds)
float mzsae_metal_fused_decode(
    const float* q,               // [num_q_heads, 128]
    const void*  k_payload,       // [num_body_tokens, num_kv_heads, 64]
    const void*  v_payload,       // [num_body_tokens, num_kv_heads, 64]
    const void*  k_centroids,     // [num_blocks, num_kv_heads, 32 * sizeof(half4)]
    const void*  k_scales,        // [num_blocks, num_kv_heads, 32 * sizeof(half4)]
    const void*  k_mins,          // [num_blocks, num_kv_heads, 32 * sizeof(half4)]
    const void*  v_group_meta,    // [num_body_tokens, num_kv_heads, 2 * sizeof(half4)]
    const void*  sinks_k,         // [num_sinks, num_kv_heads, 128 * sizeof(half)]
    const void*  sinks_v,         // [num_sinks, num_kv_heads, 128 * sizeof(half)]
    const void*  recent_k,        // [recent_win, num_kv_heads, 128 * sizeof(half)]
    const void*  recent_v,        // [recent_win, num_kv_heads, 128 * sizeof(half)]
    uint32_t     seq_len,
    uint32_t     num_splits,
    float*       final_out,       // [num_q_heads, 128]
    uint32_t     recent_win
);

// Tuned dense reference decode (returns GPU execution time in microseconds)
float mzsae_metal_dense_decode(
    const float* q,               // [num_q_heads, 128]
    const void*  dense_k_fp16,    // [seq_len, num_kv_heads, 128 * sizeof(half)]
    const void*  dense_v_fp16,    // [seq_len, num_kv_heads, 128 * sizeof(half)]
    uint32_t     seq_len,
    uint32_t     num_splits,
    float*       final_out        // [num_q_heads, 128]
);

// Plane-2 Sentinel Selective Fetch decode (returns GPU execution time in microseconds)
float mzsae_metal_selective_decode(
    const float* q,               // [num_q_heads, 128]
    const void*  k_payload,       // [num_body_tokens, num_kv_heads, 64]
    const void*  v_payload,       // [num_body_tokens, num_kv_heads, 64]
    const void*  sentinels,       // [num_blocks, num_kv_heads, 64 bytes]
    const void*  k_centroids,     // [num_blocks, num_kv_heads, 32 * sizeof(half4)]
    const void*  k_scales,        // [num_blocks, num_kv_heads, 32 * sizeof(half4)]
    const void*  k_mins,          // [num_blocks, num_kv_heads, 32 * sizeof(half4)]
    const void*  v_group_meta,    // [num_body_tokens, num_kv_heads, 2 * sizeof(half4)]
    const void*  sinks_k,         // [num_sinks, num_kv_heads, 128 * sizeof(half)]
    const void*  sinks_v,         // [num_sinks, num_kv_heads, 128 * sizeof(half)]
    const void*  recent_k,        // [recent_win, num_kv_heads, 128 * sizeof(half)]
    const void*  recent_v,        // [recent_win, num_kv_heads, 128 * sizeof(half)]
    uint32_t     seq_len,
    uint32_t     num_splits,
    float        tau,
    float*       final_out,       // [num_q_heads, 128]
    uint32_t*    out_approved_blocks,
    uint32_t*    out_total_blocks,
    uint32_t     recent_win
);

const char* mzsae_metal_get_device_name(void);
uint64_t    mzsae_metal_get_recommended_max_working_set(void);

#ifdef __cplusplus
}
#endif

#endif // MZSAE_METAL_RUNTIME_H
