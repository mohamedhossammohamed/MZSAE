# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-09-23

### Added
- CUDA backend: SM90 (Hopper), GH200 (Grace Hopper), B200 (Blackwell) support
  - Real CUDA kernels (`mzsae_cuda_kernels.cu`) with warp-level RoPE
  - Triton kernel variants (`mzsae_triton_kernels.py`)
  - L2 persistence hints and DGX topology awareness
- Interactive showcase demo (`examples/demo.py`)
- Release gate script with wheel freshness hash-check

### Fixed
- **`hf_patch.py` DynamicCache NameError** when `transformers` not installed — fallback stub was missing in packaged wheel
- Dispatcher graceful CPU fallback on non-Metal platforms
- CI test collection resilient to missing optional dependencies

### Changed
- License switched to Apache 2.0
- README: DRAM-cut explicitly labeled as "estimated from kernel-observed prune counts"
- Gate script hardened: regex-based claim checks, stale-wheel detector

## [1.2.0] - 2026-09-22

### Added
- Plane-2 Sentinel Selective Fetch with Cauchy-Schwarz upper bound pruning
- Neuromorphic TD(0) eviction policy (TDAttnPolicy, 27,009-param MLP)
- Directional Veto (ACC) with cosine gate on slow manifold
- TelemetryRingBuffer for hippocampal-neocortical sleep consolidation
- 95.9% block pruning ratio at 64k-128k context
- `benchmarks/bench_selective_fetch.py`
- `docs/LIMITATIONS.md` red-team audit

### Changed
- Metal runtime upgraded with size-aware buffer tracking
- GPU timestamps via Metal command buffer completion

### Fixed
- 64k vs 128k timing anomaly (buffer pointer aliasing)

## [1.1.0] - 2026-09-15

### Added
- Fused compressed memory streaming (`mzsae_fused_decode_stage1`/`stage2`)
- GQA 6:1 threadgroup sharing with vectorized half4 loads
- Real model weights benchmark (Qwen2.5-0.5B GGUF)
- Long-context 16k token match verification

### Changed
- KV cache compression: FIX SET 1 hybrid 4-bit format

## [1.0.0] - 2026-09-08

### Added
- Initial release
- `MZSAEAttention` PyTorch module
- `MZSAEEngine` with dual-plane KV cache
- Metal MSL 3.1 backend
- CPU reference backend
- Hardware profiles: `apple_m1`, `apple_m4`, `apple_m4_max`, `nvidia_future`
- Zero-copy MLX integration
- NIAH benchmark suite
- 33 unit tests
