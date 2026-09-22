# MZSAE Configuration System

MZSAE decouples algorithmic hyper-parameters, cache sizing, and low-level kernel launch dimensions from physical hardware characteristics through a structured configuration schema.

## Configuration Schema

The root configuration container `MZSAEConfig` aggregates four sub-configs:

```python
from mzsae import MZSAEConfig

config = MZSAEConfig()
# config.hardware  -> HardwareProfile
# config.kernel    -> KernelConfig
# config.eviction  -> EvictionConfig
# config.cache     -> CacheConfig
```

### 1. `HardwareProfile`
Describes physical memory bandwidth and cache hierarchies:
- `device_type`: Target architecture (`apple_silicon`, `cpu`, `cuda`).
- `chip_name`: String descriptor (e.g. `Apple M4`, `Apple M4 Max`).
- `memory_bandwidth_gbps`: Peak DRAM bus bandwidth in GB/s.
- `l2_cache_mb`: Unified L2/SLC cache size in MB.
- `sram_size_kb`: Threadgroup / shared memory size in KB.
- `recommended_threadgroup_size`: Optimal SIMD/threadblock size.

### 2. `KernelConfig`
Low-level dispatch parameters:
- `block_size_tokens`: Temporal block chunk size (typically 32 or 64).
- `head_dim`: Key/Query head dimension (typically 128).
- `num_kv_heads`: Key/Value head count (for GQA).
- `num_query_heads`: Query head count.
- `slow_rope_dims`: Dimensions allocated to the invariant slow-RoPE subspace.
- `sentinel_bytes`: Descriptor size in Plane-2 (64 bytes).
- `threadgroup_size`: Metal threadgroup dispatch size (default 256).
- `num_splits`: FlashDecoding sequence parallel split count (32 to 128).

### 3. `EvictionConfig`
Biological eviction and gating control:
- `default_tau`: Attention gating tolerance ($u_b \ge \text{local\_max} - \tau$).
- `acc_veto_threshold`: Directional cosine similarity threshold for eviction veto.
- `ring_buffer_kb`: Capacity of in-memory telemetry buffer.
- `sleep_batch_size`: Batch size for offline TD-Attn hippocampal sleep replay.
- `sleep_steps`: Number of replay optimization iterations per sleep cycle.

### 4. `CacheConfig`
- `num_sinks`: Preserved initial attention sink tokens (default 4).
- `recent_win`: Local uncompressed sliding window tokens (default 64).
- `max_active_blocks`: RAM retention budget for blocks before eviction.

### 5. `ModelWeightConfig`
Manages model weight quantization format adaptation:
- `quantization_type`: Format string (`fp16`, `bf16`, `8bit`, `4bit`, `3bit`, `2bit`, `ternary`).
- `quantization_bits`: Number of bits per weight parameter.
- `dynamic_range_scale`: Scaling factor for centroid and residual dynamic range (e.g. 0.5 for BitNet).
- `tau_multiplier`: Scaling factor for pruning tolerance $\tau$ (e.g. 0.7 for BitNet).
- `sentinel_scale`: Scaling factor for Cauchy-Schwarz sentinel bounds.
- Helper factory: `ModelWeightConfig.from_type("ternary")`.

## Loading Configurations

### By Named Profile
```python
from mzsae import load_config

config = load_config("apple_m4")
```

### Auto-Detection
```python
config = load_config("auto")
```

### From YAML File
```python
config = load_config("path/to/custom_profile.yaml")
```
