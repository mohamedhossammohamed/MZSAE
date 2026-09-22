# MZSAE CUDA Backend (Planned Architecture Roadmap)

This directory serves as the architecture roadmap and implementation placeholder for NVIDIA CUDA acceleration of the MZSAE runtime.

## Architectural Blueprint

The MZSAE CUDA backend will translate Apple Silicon Metal unified memory kernels into highly optimized CUDA/Cutlass C++ templates targeting modern NVIDIA GPUs (Ampere SM 8.0+, Ada Lovelace SM 8.9, Hopper SM 9.0, and Blackwell SM 10.0+).

### Key Kernel Primitives
1. **Plane-2 Sentinel Warp-Level Gating**:
   - Each CUDA threadblock processes one or more KV heads.
   - Warp-level reduction across slow RoPE dimensions (16 dims) computes Cauchy-Schwarz upper bounds $u_b$.
   - Blocks where $u_b < \text{local\_max} - \tau$ are pruned using `__syncthreads()` or warp shuffles before payload decompression, eliminating up to 85% of HBM traffic.

2. **Asynchronous Direct-to-SRAM Payload Streaming**:
   - On Ampere and Hopper, use `cp.async` and TMA (Tensor Memory Accelerator) to stream 4-bit quantized K/V payloads directly into Shared Memory (SRAM) without L1/RF register bloat.

3. **In-SRAM SIMD Dequantization & Tensor Core MMA**:
   - Fast sub-byte dequantization unpacks 4-bit nibbles into FP16/BF16 in SRAM.
   - Fused GEMM via `mma.sync` computes attention weights and multiplies with $V$.

4. **Multi-Split KV Reduction (FlashDecoding)**:
   - For long contexts (>16k tokens), sequence dimension is partitioned into $N$ splits across Streaming Multiprocessors (SMs), followed by a log-sum-exp reduction kernel.

## Planned Device Targets

| Architecture | Compute Capability | Memory Subsystem | Target Bandwidth |
|---|---|---|---|
| **Ampere (A100/RTX 3090)** | `sm_80`, `sm_86` | HBM2e / GDDR6X | 936 - 2,039 GB/s |
| **Ada Lovelace (RTX 4090/L40)** | `sm_89` | GDDR6X / GDDR6 | 860 - 1,008 GB/s |
| **Hopper (H100/H200)** | `sm_90a` | HBM3 / HBM3e | 3,350 - 4,800 GB/s |
| **Blackwell (B200)** | `sm_100` | HBM3e | 8,000 GB/s |

## Current Status
Metal (Apple Silicon) and CPU Reference backends are active in this release. CUDA support is slated for the upcoming v1.3.0 release.
