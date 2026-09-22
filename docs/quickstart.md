# MZSAE Quickstart Guide

The **MZahran Sparse Attention Engine (MZSAE)** is a hardware-aware sparse attention runtime designed to eliminate the KV-cache memory wall during long-context LLM inference.

## Installation

### From Source (Editable Mode)
```bash
git clone https://github.com/mohammedhossam/MZSAE.git
cd MZSAE
make
pip install -e .
```

### Direct Wheel Installation
```bash
pip install dist/mzsae-1.3.0-py3-none-any.whl
```

## Quick Usage

### 1. PyTorch Module (`MZSAEAttention`)

`MZSAEAttention` is a drop-in PyTorch module supporting both attention layer operations and direct functional attention:

```python
import torch
from mzsae import MZSAEAttention

# Initialize attention module with auto hardware detection
attn = MZSAEAttention(
    embed_dim=2048,
    num_heads=16,
    num_kv_heads=4,
    hardware_profile="auto"
)

# 1. Functional Attention (q, k, v)
q = torch.randn(1, 16, 16, 128)
k = torch.randn(1, 16, 4, 128)
v = torch.randn(1, 16, 4, 128)
out = attn(q, k, v, causal=True)

# 2. Layer Mode with Projections
x = torch.randn(1, 16, 2048)
out_projected = attn(x, causal=True)
```

### 2. Autoregressive Token Generation with Unified Cache

```python
import torch
from mzsae import MZSAEAttention

attn = MZSAEAttention(embed_dim=1536, num_heads=12, num_kv_heads=2)

# Prefill prompt tokens
prompt_tokens = torch.randn(1, 64, 1536)
_ = attn(prompt_tokens, causal=True, use_cache=True)

# Generate new tokens one-by-one with Plane-2 sentinel pruning
for _ in range(10):
    next_token = torch.randn(1, 1, 1536)
    out_token = attn(next_token, causal=True, use_cache=True)
```

### 3. MLX / NumPy Direct Execution

```python
import numpy as np
from mzsae import MZSAEEngine, load_config

config = load_config("apple_m4")
engine = MZSAEEngine(config=config)

# Ingest prefill
K = np.random.randn(512, 2, 128).astype(np.float16)
V = np.random.randn(512, 2, 128).astype(np.float16)
engine.cache.ingest_prefill(K, V)

# Execute single-token selective decode
q = np.random.randn(12, 128).astype(np.float32)
out, telemetry = engine.selective_decode(q, tau=16.0, return_telemetry=True)
print(f"Pruning ratio: {telemetry['pruning_ratio']*100:.1f}%")
```
