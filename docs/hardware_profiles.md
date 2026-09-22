# MZSAE Hardware Profiles

MZSAE provides tuned hardware profiles for Apple Silicon chips and provides a blueprint for upcoming NVIDIA GPUs.

## Supported Profiles

| Profile Name | Target Device | DRAM Bandwidth | L2 / SLC | Recommended Splits | Backend |
|---|---|---|---|---|---|
| `apple_m1` | Apple M1 / M2 / M3 Base | 68.25 GB/s | 12 MB | 32 splits | Metal |
| `apple_m4` | Apple M4 Base | 120.0 GB/s | 16 MB | 64 splits | Metal |
| `apple_m4_max` | Apple M4 Max | 410.0 GB/s | 48 MB | 128 splits | Metal |
| `nvidia_future` | NVIDIA H100 (Planned) | 3,350.0 GB/s | 50 MB | 128 splits | CUDA (v1.3.0) |
| `default` | Portable Fallback | 120.0 GB/s | 16 MB | 64 splits | Metal / CPU |

## Automatic Hardware Detection (`hardware_profile="auto"`)

When `hardware_profile="auto"` is specified, MZSAE queries Darwin system parameters (`machdep.cpu.brand_string` and `hw.model` via `sysctl`).
- If an Apple Silicon chip is recognized, the matching Apple profile is applied.
- If executed on Linux / non-Apple hardware, the engine falls back to `CPUReferenceBackend` with SIMD acceleration.

## Adding a Custom Hardware Profile

Create a new YAML file in your project or under `configs/`:

```yaml
# configs/my_custom_chip.yaml
hardware:
  device_type: "apple_silicon"
  chip_name: "Custom Accelerator"
  memory_bandwidth_gbps: 200.0
  l2_cache_mb: 24.0
  sram_size_kb: 32.0
  recommended_threadgroup_size: 256

kernel:
  block_size_tokens: 64
  head_dim: 128
  num_kv_heads: 2
  num_query_heads: 12
  num_splits: 64

eviction:
  default_tau: 16.0
  acc_veto_threshold: 0.40

cache:
  num_sinks: 4
  recent_win: 64

use_metal: true
use_cpu_fallback: true
```

Load it directly in Python:
```python
from mzsae import load_config, MZSAEEngine

cfg = load_config("configs/my_custom_chip.yaml")
engine = MZSAEEngine(config=cfg)
```
