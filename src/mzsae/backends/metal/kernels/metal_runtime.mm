#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_runtime.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <mutex>

struct MZSAEParams {
    uint32_t seq_len;
    uint32_t num_q_heads;    // 12
    uint32_t num_kv_heads;   // 2
    uint32_t head_dim;       // 128
    uint32_t num_sinks;      // 4
    uint32_t recent_win;     // 64
    uint32_t num_splits;     // 16, 32, 64, or 128
    float rope_base;         // 1000000.0f
    float kq_scale;          // 1.0f / sqrt(128.0f)
    uint32_t block_size;     // 64
};

struct MZSAESelectiveParams {
    uint32_t seq_len;
    uint32_t num_q_heads;    // 12
    uint32_t num_kv_heads;   // 2
    uint32_t head_dim;       // 128
    uint32_t num_sinks;      // 4
    uint32_t recent_win;     // 64
    uint32_t num_splits;     // 16, 32, 64, or 128
    float rope_base;         // 1000000.0f
    float kq_scale;          // 1.0f / sqrt(128.0f)
    uint32_t block_size;     // 64
    float tau;               // 16.0f
};

struct MetalRuntimeState {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> fusedStage1Pipeline;
    id<MTLComputePipelineState> fusedStage2Pipeline;
    id<MTLComputePipelineState> denseStage1Pipeline;
    id<MTLComputePipelineState> selectiveStage1Pipeline;
    id<MTLComputePipelineState> computeLocalMaxPipeline;

    // Preallocated reusable scratch buffers (Zero per-step allocation)
    id<MTLBuffer> partialOutBuffer;
    id<MTLBuffer> partialMetaBuffer;
    id<MTLBuffer> localMaxBuffer;
    id<MTLBuffer> splitStatsBuffer;
    size_t maxSplits;

    char deviceName[256];
    bool initialized;
};

struct BufferEntry {
    id<MTLBuffer> buffer;
    size_t length;
    bool isNoCopy;
};

static MetalRuntimeState g_rt = {nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 128, {0}, false};
static std::map<void*, BufferEntry> g_buffer_map;
static id<MTLBuffer> g_dummy_buffer = nil;
static std::mutex g_mutex;

static id<MTLBuffer> get_or_create_buffer(const void* ptr, size_t length) {
    if (!ptr || length == 0) {
        if (!g_dummy_buffer && g_rt.device) {
            g_dummy_buffer = [g_rt.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        }
        return g_dummy_buffer;
    }
    void* non_const_ptr = const_cast<void*>(ptr);
    
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_buffer_map.find(non_const_ptr);
    if (it != g_buffer_map.end()) {
        if (it->second.length == length) {
            if (!it->second.isNoCopy) {
                // Synchronize latest host data to Metal shared buffer
                memcpy([it->second.buffer contents], non_const_ptr, length);
            }
            return it->second.buffer;
        } else {
            // Buffer size changed: invalidate stale entry (e.g. 64k pointer address recycled for 128k)
            g_buffer_map.erase(it);
        }
    }

    // Try zero-copy wrapping if 4096-byte page aligned
    uintptr_t addr = (uintptr_t)non_const_ptr;
    if ((addr & 0xFFF) == 0 && (length & 0xFFF) == 0) {
        id<MTLBuffer> buf = [g_rt.device newBufferWithBytesNoCopy:non_const_ptr
                                                           length:length
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (buf) {
            g_buffer_map[non_const_ptr] = {buf, length, true};
            return buf;
        }
    }

    // Allocate shared buffer and copy host data
    id<MTLBuffer> buf = [g_rt.device newBufferWithBytes:non_const_ptr
                                                 length:length
                                                options:MTLResourceStorageModeShared];
    g_buffer_map[non_const_ptr] = {buf, length, false};
    return buf;
}

int mzsae_metal_init(const char* shader_source) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_rt.initialized) return 0;

        g_rt.device = MTLCreateSystemDefaultDevice();
        if (!g_rt.device) {
            fprintf(stderr, "[MZSAE-Metal] Error: No Metal GPU detected.\n");
            return -1;
        }

        strncpy(g_rt.deviceName, [[g_rt.device name] UTF8String], sizeof(g_rt.deviceName) - 1);
        g_rt.commandQueue = [g_rt.device newCommandQueue];

        NSError* error = nil;
        NSString* src = shader_source ? [NSString stringWithUTF8String:shader_source] : nil;

        if (!src) {
            NSArray<NSString*>* search_paths = @[
                @"src/metal/mzsae_kernels.metal",
                @"src/mzsae/mzsae_kernels.metal",
                @"../src/metal/mzsae_kernels.metal"
            ];
            for (NSString* p in search_paths) {
                src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
                if (src) break;
            }
        }

        if (!src) {
            fprintf(stderr, "[MZSAE-Metal] Error: Could not locate mzsae_kernels.metal\n");
            return -2;
        }

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion3_1;

        id<MTLLibrary> library = [g_rt.device newLibraryWithSource:src options:options error:&error];
        if (!library) {
            fprintf(stderr, "[MZSAE-Metal] Shader Compilation Error: %s\n", [[error localizedDescription] UTF8String]);
            return -3;
        }

        id<MTLFunction> fnFused1 = [library newFunctionWithName:@"mzsae_fused_decode_stage1"];
        id<MTLFunction> fnFused2 = [library newFunctionWithName:@"mzsae_fused_decode_stage2"];
        id<MTLFunction> fnDense1 = [library newFunctionWithName:@"tuned_dense_decode_stage1"];
        id<MTLFunction> fnSelect1 = [library newFunctionWithName:@"mzsae_selective_decode_stage1"];
        id<MTLFunction> fnLocMax  = [library newFunctionWithName:@"mzsae_compute_local_max"];

        if (!fnFused1 || !fnFused2 || !fnDense1 || !fnSelect1 || !fnLocMax) {
            fprintf(stderr, "[MZSAE-Metal] Error: Required kernel functions not found.\n");
            return -4;
        }

        g_rt.fusedStage1Pipeline = [g_rt.device newComputePipelineStateWithFunction:fnFused1 error:&error];
        g_rt.fusedStage2Pipeline = [g_rt.device newComputePipelineStateWithFunction:fnFused2 error:&error];
        g_rt.denseStage1Pipeline = [g_rt.device newComputePipelineStateWithFunction:fnDense1 error:&error];
        g_rt.selectiveStage1Pipeline = [g_rt.device newComputePipelineStateWithFunction:fnSelect1 error:&error];
        g_rt.computeLocalMaxPipeline = [g_rt.device newComputePipelineStateWithFunction:fnLocMax error:&error];

        // Allocate scratch buffers for max 128 splits across 12 heads
        size_t part_out_sz = 128 * 12 * 32 * sizeof(float) * 4;
        size_t part_meta_sz = 128 * 12 * sizeof(float) * 2;
        g_rt.partialOutBuffer = [g_rt.device newBufferWithLength:part_out_sz options:MTLResourceStorageModeShared];
        g_rt.partialMetaBuffer = [g_rt.device newBufferWithLength:part_meta_sz options:MTLResourceStorageModeShared];

        size_t loc_max_sz = 12 * sizeof(float);
        size_t stats_sz = 65536 * sizeof(uint32_t);
        g_rt.localMaxBuffer = [g_rt.device newBufferWithLength:loc_max_sz options:MTLResourceStorageModeShared];
        g_rt.splitStatsBuffer = [g_rt.device newBufferWithLength:stats_sz options:MTLResourceStorageModeShared];

        g_rt.initialized = true;
        return 0;
    }
}

void mzsae_metal_shutdown(void) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_buffer_map.clear();
        g_dummy_buffer = nil;
        g_rt.partialOutBuffer = nil;
        g_rt.partialMetaBuffer = nil;
        g_rt.localMaxBuffer = nil;
        g_rt.splitStatsBuffer = nil;
        g_rt.fusedStage1Pipeline = nil;
        g_rt.fusedStage2Pipeline = nil;
        g_rt.denseStage1Pipeline = nil;
        g_rt.selectiveStage1Pipeline = nil;
        g_rt.computeLocalMaxPipeline = nil;
        g_rt.commandQueue = nil;
        g_rt.device = nil;
        g_rt.initialized = false;
    }
}

void* mzsae_metal_alloc_raw(size_t bytes) {
    @autoreleasepool {
        if (!g_rt.device) {
            if (mzsae_metal_init(nullptr) != 0) return nullptr;
        }
        id<MTLBuffer> buf = [g_rt.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) return nullptr;
        void* ptr = [buf contents];
        std::lock_guard<std::mutex> lock(g_mutex);
        g_buffer_map[ptr] = {buf, bytes, true};
        return ptr;
    }
}

void mzsae_metal_free(void* ptr) {
    @autoreleasepool {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(g_mutex);
        g_buffer_map.erase(ptr);
    }
}

void mzsae_metal_clear_cache(void) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_buffer_map.clear();
    }
}

float mzsae_metal_fused_decode(
    const float* q,
    const void*  k_payload,
    const void*  v_payload,
    const void*  k_centroids,
    const void*  k_scales,
    const void*  k_mins,
    const void*  v_group_meta,
    const void*  sinks_k,
    const void*  sinks_v,
    const void*  recent_k,
    const void*  recent_v,
    uint32_t     seq_len,
    uint32_t     num_splits,
    float*       final_out,
    uint32_t     recent_win
) {
    @autoreleasepool {
        if (!g_rt.initialized) {
            if (mzsae_metal_init(nullptr) != 0) return 0.0f;
        }

        uint32_t rec = (recent_win > 0) ? recent_win : 64;
        uint32_t nq = 12, nkv = 2, d = 128, sinks = 4, bsz = 64;
        MZSAEParams params{
            seq_len, nq, nkv, d, sinks, rec, num_splits,
            1000000.0f, 1.0f / std::sqrt(128.0f), bsz
        };

        uint32_t body_tokens = (seq_len > sinks + rec) ? (seq_len - sinks - rec) : 0;
        uint32_t num_blocks = (body_tokens + bsz - 1) / bsz;

        id<MTLBuffer> bQ = get_or_create_buffer(q, nq * d * sizeof(float));
        id<MTLBuffer> bKp = get_or_create_buffer(k_payload, body_tokens * nkv * 64);
        id<MTLBuffer> bVp = get_or_create_buffer(v_payload, body_tokens * nkv * 64);
        id<MTLBuffer> bKc = get_or_create_buffer(k_centroids, num_blocks * nkv * 32 * sizeof(uint16_t) * 4);
        id<MTLBuffer> bKs = get_or_create_buffer(k_scales, num_blocks * nkv * 32 * sizeof(uint16_t) * 4);
        id<MTLBuffer> bKm = get_or_create_buffer(k_mins, num_blocks * nkv * 32 * sizeof(uint16_t) * 4);
        id<MTLBuffer> bVm = get_or_create_buffer(v_group_meta, body_tokens * nkv * 2 * sizeof(uint16_t) * 4);
        id<MTLBuffer> bSk = get_or_create_buffer(sinks_k, sinks * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bSv = get_or_create_buffer(sinks_v, sinks * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bRk = get_or_create_buffer(recent_k, rec * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bRv = get_or_create_buffer(recent_v, rec * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bOut = get_or_create_buffer(final_out, nq * d * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_rt.commandQueue commandBuffer];

        // Stage 1
        id<MTLComputeCommandEncoder> enc1 = [cmd computeCommandEncoder];
        [enc1 setComputePipelineState:g_rt.fusedStage1Pipeline];
        [enc1 setBuffer:bQ offset:0 atIndex:0];
        [enc1 setBuffer:bKp offset:0 atIndex:1];
        [enc1 setBuffer:bVp offset:0 atIndex:2];
        [enc1 setBuffer:bKc offset:0 atIndex:3];
        [enc1 setBuffer:bKs offset:0 atIndex:4];
        [enc1 setBuffer:bKm offset:0 atIndex:5];
        [enc1 setBuffer:bVm offset:0 atIndex:6];
        [enc1 setBuffer:bSk offset:0 atIndex:7];
        [enc1 setBuffer:bSv offset:0 atIndex:8];
        [enc1 setBuffer:bRk offset:0 atIndex:9];
        [enc1 setBuffer:bRv offset:0 atIndex:10];
        [enc1 setBuffer:g_rt.partialOutBuffer offset:0 atIndex:11];
        [enc1 setBuffer:g_rt.partialMetaBuffer offset:0 atIndex:12];
        [enc1 setBytes:&params length:sizeof(params) atIndex:13];

        MTLSize grid1 = MTLSizeMake(nkv, num_splits, 1);
        MTLSize group1 = MTLSizeMake(32, 1, 1);
        [enc1 dispatchThreadgroups:grid1 threadsPerThreadgroup:group1];
        [enc1 endEncoding];

        // Stage 2
        id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
        [enc2 setComputePipelineState:g_rt.fusedStage2Pipeline];
        [enc2 setBuffer:g_rt.partialOutBuffer offset:0 atIndex:0];
        [enc2 setBuffer:g_rt.partialMetaBuffer offset:0 atIndex:1];
        [enc2 setBuffer:bOut offset:0 atIndex:2];
        [enc2 setBytes:&params length:sizeof(params) atIndex:3];

        MTLSize grid2 = MTLSizeMake(nq, 1, 1);
        MTLSize group2 = MTLSizeMake(32, 1, 1);
        [enc2 dispatchThreadgroups:grid2 threadsPerThreadgroup:group2];
        [enc2 endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([bOut contents] != final_out) {
            memcpy(final_out, [bOut contents], nq * d * sizeof(float));
        }

        CFTimeInterval start = [cmd GPUStartTime];
        CFTimeInterval end = [cmd GPUEndTime];
        float gpu_us = 0.0f;
        if (end > start) {
            gpu_us = (float)((end - start) * 1e6);
        }
        return gpu_us;
    }
}

float mzsae_metal_dense_decode(
    const float* q,
    const void*  dense_k_fp16,
    const void*  dense_v_fp16,
    uint32_t     seq_len,
    uint32_t     num_splits,
    float*       final_out
) {
    @autoreleasepool {
        if (!g_rt.initialized) {
            if (mzsae_metal_init(nullptr) != 0) return 0.0f;
        }

        uint32_t nq = 12, nkv = 2, d = 128, sinks = 4, rec = 64, bsz = 64;
        MZSAEParams params{
            seq_len, nq, nkv, d, sinks, rec, num_splits,
            1000000.0f, 1.0f / std::sqrt(128.0f), bsz
        };

        id<MTLBuffer> bQ = get_or_create_buffer(q, nq * d * sizeof(float));
        id<MTLBuffer> bK = get_or_create_buffer(dense_k_fp16, seq_len * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bV = get_or_create_buffer(dense_v_fp16, seq_len * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bOut = get_or_create_buffer(final_out, nq * d * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_rt.commandQueue commandBuffer];

        // Stage 1
        id<MTLComputeCommandEncoder> enc1 = [cmd computeCommandEncoder];
        [enc1 setComputePipelineState:g_rt.denseStage1Pipeline];
        [enc1 setBuffer:bQ offset:0 atIndex:0];
        [enc1 setBuffer:bK offset:0 atIndex:1];
        [enc1 setBuffer:bV offset:0 atIndex:2];
        [enc1 setBuffer:g_rt.partialOutBuffer offset:0 atIndex:3];
        [enc1 setBuffer:g_rt.partialMetaBuffer offset:0 atIndex:4];
        [enc1 setBytes:&params length:sizeof(params) atIndex:5];

        MTLSize grid1 = MTLSizeMake(nkv, num_splits, 1);
        MTLSize group1 = MTLSizeMake(32, 1, 1);
        [enc1 dispatchThreadgroups:grid1 threadsPerThreadgroup:group1];
        [enc1 endEncoding];

        // Stage 2
        id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
        [enc2 setComputePipelineState:g_rt.fusedStage2Pipeline];
        [enc2 setBuffer:g_rt.partialOutBuffer offset:0 atIndex:0];
        [enc2 setBuffer:g_rt.partialMetaBuffer offset:0 atIndex:1];
        [enc2 setBuffer:bOut offset:0 atIndex:2];
        [enc2 setBytes:&params length:sizeof(params) atIndex:3];

        MTLSize grid2 = MTLSizeMake(nq, 1, 1);
        MTLSize group2 = MTLSizeMake(32, 1, 1);
        [enc2 dispatchThreadgroups:grid2 threadsPerThreadgroup:group2];
        [enc2 endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([bOut contents] != final_out) {
            memcpy(final_out, [bOut contents], nq * d * sizeof(float));
        }

        CFTimeInterval start = [cmd GPUStartTime];
        CFTimeInterval end = [cmd GPUEndTime];
        float gpu_us = 0.0f;
        if (end > start) {
            gpu_us = (float)((end - start) * 1e6);
        }
        return gpu_us;
    }
}

float mzsae_metal_selective_decode(
    const float* q,
    const void*  k_payload,
    const void*  v_payload,
    const void*  sentinels,
    const void*  k_centroids,
    const void*  k_scales,
    const void*  k_mins,
    const void*  v_group_meta,
    const void*  sinks_k,
    const void*  sinks_v,
    const void*  recent_k,
    const void*  recent_v,
    uint32_t     seq_len,
    uint32_t     num_splits,
    float        tau,
    float*       final_out,
    uint32_t*    out_approved_blocks,
    uint32_t*    out_total_blocks,
    uint32_t     recent_win
) {
    @autoreleasepool {
        if (!g_rt.initialized) {
            if (mzsae_metal_init(nullptr) != 0) return 0.0f;
        }

        uint32_t rec = (recent_win > 0) ? recent_win : 64;
        uint32_t nq = 12, nkv = 2, d = 128, sinks = 4, bsz = 64;
        MZSAESelectiveParams params{
            seq_len, nq, nkv, d, sinks, rec, num_splits,
            1000000.0f, 1.0f / std::sqrt(128.0f), bsz, tau
        };

        uint32_t body_tokens = (seq_len > sinks + rec) ? (seq_len - sinks - rec) : 0;
        uint32_t num_blocks = (body_tokens + bsz - 1) / bsz;

        id<MTLBuffer> bQ = get_or_create_buffer(q, nq * d * sizeof(float));
        id<MTLBuffer> bKp = get_or_create_buffer(k_payload, body_tokens * nkv * 64);
        id<MTLBuffer> bVp = get_or_create_buffer(v_payload, body_tokens * nkv * 64);
        id<MTLBuffer> bSent = get_or_create_buffer(sentinels, num_blocks * nkv * 64);
        id<MTLBuffer> bKc = get_or_create_buffer(k_centroids, num_blocks * nkv * 32 * sizeof(uint16_t) * 4);
        id<MTLBuffer> bKs = get_or_create_buffer(k_scales, num_blocks * nkv * 32 * sizeof(uint16_t) * 4);
        id<MTLBuffer> bKm = get_or_create_buffer(k_mins, num_blocks * nkv * 32 * sizeof(uint16_t) * 4);
        id<MTLBuffer> bVm = get_or_create_buffer(v_group_meta, body_tokens * nkv * 2 * sizeof(uint16_t) * 4);
        id<MTLBuffer> bSk = get_or_create_buffer(sinks_k, sinks * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bSv = get_or_create_buffer(sinks_v, sinks * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bRk = get_or_create_buffer(recent_k, rec * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bRv = get_or_create_buffer(recent_v, rec * nkv * d * sizeof(uint16_t));
        id<MTLBuffer> bOut = get_or_create_buffer(final_out, nq * d * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_rt.commandQueue commandBuffer];

        // Pass 0: Compute local max from sinks and recent window
        id<MTLComputeCommandEncoder> enc0 = [cmd computeCommandEncoder];
        [enc0 setComputePipelineState:g_rt.computeLocalMaxPipeline];
        [enc0 setBuffer:bQ offset:0 atIndex:0];
        [enc0 setBuffer:bRk offset:0 atIndex:1];
        [enc0 setBuffer:bSk offset:0 atIndex:2];
        [enc0 setBuffer:g_rt.localMaxBuffer offset:0 atIndex:3];
        [enc0 setBytes:&params length:sizeof(params) atIndex:4];
        [enc0 dispatchThreadgroups:MTLSizeMake(nq, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc0 endEncoding];

        // Pass 1: Selective Decode Stage 1
        memset([g_rt.splitStatsBuffer contents], 0, num_blocks * nkv * sizeof(uint32_t));
        id<MTLComputeCommandEncoder> enc1 = [cmd computeCommandEncoder];
        [enc1 setComputePipelineState:g_rt.selectiveStage1Pipeline];
        [enc1 setBuffer:bQ offset:0 atIndex:0];
        [enc1 setBuffer:bKp offset:0 atIndex:1];
        [enc1 setBuffer:bVp offset:0 atIndex:2];
        [enc1 setBuffer:bSent offset:0 atIndex:3];
        [enc1 setBuffer:bKc offset:0 atIndex:4];
        [enc1 setBuffer:bKs offset:0 atIndex:5];
        [enc1 setBuffer:bKm offset:0 atIndex:6];
        [enc1 setBuffer:bVm offset:0 atIndex:7];
        [enc1 setBuffer:bSk offset:0 atIndex:8];
        [enc1 setBuffer:bSv offset:0 atIndex:9];
        [enc1 setBuffer:bRk offset:0 atIndex:10];
        [enc1 setBuffer:bRv offset:0 atIndex:11];
        [enc1 setBuffer:g_rt.localMaxBuffer offset:0 atIndex:12];
        [enc1 setBuffer:g_rt.partialOutBuffer offset:0 atIndex:13];
        [enc1 setBuffer:g_rt.partialMetaBuffer offset:0 atIndex:14];
        [enc1 setBuffer:g_rt.splitStatsBuffer offset:0 atIndex:15];
        [enc1 setBytes:&params length:sizeof(params) atIndex:16];

        [enc1 dispatchThreadgroups:MTLSizeMake(nkv, num_splits, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc1 endEncoding];

        // Pass 2: Merge Splits (Stage 2)
        id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
        [enc2 setComputePipelineState:g_rt.fusedStage2Pipeline];
        [enc2 setBuffer:g_rt.partialOutBuffer offset:0 atIndex:0];
        [enc2 setBuffer:g_rt.partialMetaBuffer offset:0 atIndex:1];
        [enc2 setBuffer:bOut offset:0 atIndex:2];
        MZSAEParams baseParams{
            seq_len, nq, nkv, d, sinks, rec, num_splits,
            1000000.0f, 1.0f / std::sqrt(128.0f), bsz
        };
        [enc2 setBytes:&baseParams length:sizeof(baseParams) atIndex:3];
        [enc2 dispatchThreadgroups:MTLSizeMake(nq, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc2 endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if ([bOut contents] != final_out) {
            memcpy(final_out, [bOut contents], nq * d * sizeof(float));
        }

        if (out_approved_blocks || out_total_blocks) {
            uint32_t tot_blocks = num_blocks * nkv;
            const uint32_t* app_ptr = (const uint32_t*)[g_rt.splitStatsBuffer contents];
            uint32_t app_sum = 0;
            for (uint32_t i = 0; i < tot_blocks; ++i) {
                app_sum += app_ptr[i];
            }
            if (out_approved_blocks) *out_approved_blocks = app_sum;
            if (out_total_blocks) *out_total_blocks = tot_blocks;
        }

        CFTimeInterval start = [cmd GPUStartTime];
        CFTimeInterval end = [cmd GPUEndTime];
        float gpu_us = 0.0f;
        if (end > start) {
            gpu_us = (float)((end - start) * 1e6);
        }
        return gpu_us;
    }
}

const char* mzsae_metal_get_device_name(void) {
    return g_rt.deviceName;
}

uint64_t mzsae_metal_get_recommended_max_working_set(void) {
    if (@available(macOS 10.15, *)) {
        if (g_rt.device) {
            return [g_rt.device recommendedMaxWorkingSetSize];
        }
    }
    return 16ULL * 1024 * 1024 * 1024;
}
