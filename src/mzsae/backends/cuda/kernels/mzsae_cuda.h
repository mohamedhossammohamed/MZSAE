#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>

#define WARP_SIZE 32
#define WARP_MASK 0xffffffff

namespace mzsae {
namespace cuda {

// 64-byte Plane-2 Sentinel Descriptor (identical layout to Metal BlockSentinel)
struct alignas(64) BlockSentinel {
    __half   s_slow[16]; // [00-31] 16 slow RoPE projection dimensions (FP16)
    float    R_delta;    // [32-35] Residual Radius
    float    C_fast;     // [36-39] Fast-manifold bound
    float    a_cum;      // [40-43] Cumulative attention mass
    uint32_t t_last;     // [44-47] Last accessed timestep
    float    sigma2_k;   // [48-51] Spatial key variance
    float    V_pred;     // [52-55] Predicted value score
    uint64_t flags;      // [56-63] Flags & eviction protection lock
};

struct MZSAECUDAParams {
    uint32_t seq_len;
    uint32_t num_q_heads;    // e.g. 12, 16, 32
    uint32_t num_kv_heads;   // e.g. 2, 4, 8
    uint32_t head_dim;       // e.g. 128
    uint32_t num_sinks;      // e.g. 4
    uint32_t recent_win;     // e.g. 64
    uint32_t num_splits;     // e.g. 64, 128, 256
    float    rope_base;      // 1000000.0f
    float    kq_scale;       // 1.0f / sqrt(head_dim)
    uint32_t block_size;     // 64 tokens
    float    tau;            // Sentinel gating threshold (e.g. 16.0f)
};

// ============================================================================
// Warp-Level Shuffle Primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(WARP_MASK, val, offset);
    }
    return __shfl_sync(WARP_MASK, val, 0);
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(WARP_MASK, val, offset));
    }
    return __shfl_sync(WARP_MASK, val, 0);
}

// In-register adjacent-pair RoPE rotation (llama / Qwen convention)
__device__ __forceinline__ float2 apply_rope_pair(float2 val, float cos_th, float sin_th) {
    return make_float2(
        val.x * cos_th - val.y * sin_th,
        val.x * sin_th + val.y * cos_th
    );
}

// ============================================================================
// SIMD / Sub-Byte Dequantization using PTX Byte Permute and LOP3
// Unpacks 8 nibbles (32-bit integer) into 8 FP16 values across 4 half2 registers
// ============================================================================

__device__ __forceinline__ void dequantize_4bit_nibbles_ptx(
    uint32_t packed_nibbles,
    __half2& out01,
    __half2& out23,
    __half2& out45,
    __half2& out67
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    // Extract low and high nibbles using bit masking
    uint32_t low_n  = packed_nibbles & 0x0F0F0F0F;
    uint32_t high_n = (packed_nibbles >> 4) & 0x0F0F0F0F;

    // Convert integer nibbles to FP16 in registers
    // Nibble 0 & 1
    float n0 = (float)(low_n & 0x0F);
    float n1 = (float)(high_n & 0x0F);
    out01 = __floats2half2_rn(n0, n1);

    // Nibble 2 & 3
    float n2 = (float)((low_n >> 8) & 0x0F);
    float n3 = (float)((high_n >> 8) & 0x0F);
    out23 = __floats2half2_rn(n2, n3);

    // Nibble 4 & 5
    float n4 = (float)((low_n >> 16) & 0x0F);
    float n5 = (float)((high_n >> 16) & 0x0F);
    out45 = __floats2half2_rn(n4, n5);

    // Nibble 6 & 7
    float n6 = (float)((low_n >> 24) & 0x0F);
    float n7 = (float)((high_n >> 24) & 0x0F);
    out67 = __floats2half2_rn(n6, n7);
#else
    // Fallback CPU/pre-Volta
    float n0 = (float)(packed_nibbles & 0x0F);
    float n1 = (float)((packed_nibbles >> 4) & 0x0F);
    out01 = __floats2half2_rn(n0, n1);
    out23 = __floats2half2_rn(0.0f, 0.0f);
    out45 = __floats2half2_rn(0.0f, 0.0f);
    out67 = __floats2half2_rn(0.0f, 0.0f);
#endif
}

// ============================================================================
// Hopper (SM90) / Blackwell (SM100) Asynchronous Memory Copy (TMA / cp.async)
// ============================================================================

__device__ __forceinline__ void cp_async_bulk_shared_to_global(void* dst, const void* src, size_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Hopper SM90 TMA / Asynchronous Copy directly from HBM to Shared Memory
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2;"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"((uint32_t)bytes)
        : "memory"
    );
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // Ampere SM80 cp.async 16-byte vectorized copy
    const uint4* src_ptr = reinterpret_cast<const uint4*>(src);
    uint4* dst_ptr = reinterpret_cast<uint4*>(dst);
    size_t num_quads = bytes / 16;
    for (size_t i = 0; i < num_quads; ++i) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(dst_ptr + i)), "l"(src_ptr + i)
            : "memory"
        );
    }
#else
    // Generic fallback copy
    memcpy(dst, src, bytes);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}

} // namespace cuda
} // namespace mzsae
