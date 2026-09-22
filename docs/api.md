# API Reference

::: mzsae.MZSAEAttention
    options:
      show_root_heading: true
      show_source: false

### Usage Example
```python
import mlx.core as mx
from mzsae import MZSAEAttention, load_config

config = load_config("configs/m4_profile.yml")
attn = MZSAEAttention(config)

q = mx.random.normal((1, 8, 128, 128))
k = mx.random.normal((1, 8, 128, 128))
v = mx.random.normal((1, 8, 128, 128))

# Forward pass
output = attn(q, k, v)
```

---

::: mzsae.MZSAEEngine
    options:
      show_root_heading: true
      show_source: false

### Usage Example
```python
from mzsae import MZSAEEngine, load_config

engine = MZSAEEngine(config=load_config())
# Lower level engine control
engine.allocate_cache(batch_size=1, max_seq_len=8192)
```

---

::: mzsae.load_config
    options:
      show_root_heading: true
      show_source: false

### Usage Example
```python
from mzsae import load_config

# Load a specific hardware profile
config = load_config("configs/m4_profile.yml")
print(f"Max retention: {config.max_retention}")
```

---

::: mzsae.MZSAENeutralWrapper
    options:
      show_root_heading: true
      show_source: false

### Usage Example
```python
from mzsae import MZSAENeutralWrapper
import torch.nn as nn

# Wrap a standard PyTorch module to ensure neutrality
wrapped_model = MZSAENeutralWrapper(my_base_model)
```

## Hardware Profiles & Telemetry

MZSAE allows tracking internal telemetry such as eviction counts, cache hits, and pruning rates.

```python
# Accessing telemetry data
telemetry = attn.get_telemetry()
print(f"Blocks pruned: {telemetry['blocks_pruned']}")
print(f"DRAM fetches saved: {telemetry['dram_fetches_saved']}")
```
