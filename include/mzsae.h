/*
 * MZahran Sparse Attention Engine (MZSAE)
 * Unified Driver ABI & Microarchitectural Memory Specification
 * Author: Mohammed Hossam Zahran
 */

#ifndef MZSAE_H
#define MZSAE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MZSAE_BLOCK_SIZE 32
#define MZSAE_HEAD_DIM 128
#define MZSAE_SLOW_DIMS 16
#define MZSAE_FAST_DIMS (MZSAE_HEAD_DIM - MZSAE_SLOW_DIMS)

// Plane 2: Descriptor Layout - Exactly 64 Bytes (1 Cache Line)
#pragma pack(push, 1)
typedef struct {
    uint16_t s_slow[MZSAE_SLOW_DIMS]; // [00-31] 16-element slow RoPE projection (FP16 bit patterns)
    float    R_delta;                 // [32-35] Residual Radius R_{\Delta, b} (FP32)
    float    C_fast;                  // [36-39] Fast-manifold bound / spectral scale (FP32)
    float    a_cum;                   // [40-43] Cumulative attention mass (FP32)
    uint32_t t_last;                  // [44-47] Last accessed decode timestep (UInt32)
    float    sigma2_k;                // [48-51] Spatial key variance \sigma_K^2 (FP32)
    float    V_pred;                  // [52-55] Predicted value score V_\theta from TD-Attn (FP32)
    uint64_t flags;                   // [56-63] Physical flags, pin bit, and eviction lock (UInt64)
} MZSAE_SentinelDesc;
#pragma pack(pop)

_Static_assert(sizeof(MZSAE_SentinelDesc) == 64, "MZSAE_SentinelDesc must be exactly 64 bytes (1 cache line)");

// Flags for SentinelDesc.flags
#define MZSAE_BLOCK_FLAG_ACTIVE     (1ULL << 0)
#define MZSAE_BLOCK_FLAG_PINNED     (1ULL << 1)
#define MZSAE_BLOCK_FLAG_VETOED     (1ULL << 2)
#define MZSAE_BLOCK_FLAG_SINK       (1ULL << 3)

// Plane 1: Physical Payload Layout Constants
#define MZSAE_CENTROID_OFFSET       0
#define MZSAE_CENTROID_BYTES        (MZSAE_HEAD_DIM * 2)             // 256 bytes (FP16)
#define MZSAE_RESIDUAL_OFFSET       256
#define MZSAE_RESIDUAL_BYTES        (MZSAE_BLOCK_SIZE * MZSAE_HEAD_DIM * 2 / 8) // 1024 bytes (2-bit packed)
#define MZSAE_SCALE_OFFSET          (MZSAE_RESIDUAL_OFFSET + MZSAE_RESIDUAL_BYTES) // 1280
#define MZSAE_SCALE_BYTES           8                                // FP16 scale, FP16 min, 4B pad
#define MZSAE_VALUE_OFFSET          (MZSAE_SCALE_OFFSET + MZSAE_SCALE_BYTES) // 1288
#define MZSAE_VALUE_BYTES_FP16      (MZSAE_BLOCK_SIZE * MZSAE_HEAD_DIM * 2) // 8192 bytes
#define MZSAE_VALUE_BYTES_8BIT      (MZSAE_BLOCK_SIZE * MZSAE_HEAD_DIM)     // 4096 bytes
#define MZSAE_VALUE_BYTES_COMPACT   2048                            // Compact compressed 2048 bytes

// Configurable Stride: Default aligned to 128 bytes
#define MZSAE_PAYLOAD_STRIDE_FP16   ((MZSAE_VALUE_OFFSET + MZSAE_VALUE_BYTES_FP16 + 127) & ~127)
#define MZSAE_PAYLOAD_STRIDE_2688   2688

// Core Engine Context ABI
typedef struct {
    void*     plane1_payload_ptr;     // VRAM address of CRQ blocks
    void*     plane2_metadata_ptr;    // L2 / Unified address of Sentinel descriptors
    uint32_t  total_blocks;           // Active number of physical blocks
    uint32_t  allocated_capacity;     // Max physical blocks allocated in pool
    uint32_t  payload_stride;         // Stride in bytes per Plane 1 block
    float     pruning_threshold_tau;  // Safety threshold parameter \tau (e.g. 16.0f)
    float     veto_cos_threshold;     // Directional veto threshold (default 0.4f)
    uint32_t  head_dim;               // Head dimension (default 128)
    uint32_t  num_heads;              // Number of query heads
    uint32_t  num_kv_heads;           // Number of key-value heads
} MZSAE_Context;

// Telemetry Event Structure for 128 KB In-Memory Ring Buffer
#pragma pack(push, 1)
typedef struct {
    uint32_t timestep;
    uint32_t block_id;
    float    reward;                  // Attention mass accumulated r_t
    float    state[16];               // Feature vector x_b
    float    next_state[16];          // Next state x_{b'}
    uint8_t  action;                  // 0 = Retain, 1 = Evict, 2 = Veto
    uint8_t  pad[3];
} MZSAE_TelemetryRecord;
#pragma pack(pop)

// Forward declarations of ABI functions
void mzsae_decode_forward(
    const void*    query_ptr,         // [1, H, D]
    MZSAE_Context* ctx,
    void*          output_ptr,        // [1, H, D]
    float          running_max_m,     // Running max logit from local sliding window
    void*          stream_handle      // id<MTLCommandBuffer> or NULL
);

#ifdef __cplusplus
}
#endif

#endif // MZSAE_H
