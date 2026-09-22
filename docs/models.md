# Architecture Neutrality & Model Integration Guide

MZSAE is completely model-architecture agnostic. Unlike low-level attention kernels that require compile-time hardcoding of tensor dimensions or head configurations, MZSAE operates as a biological policy and eviction layer that works with **any** LLM architecture.

---

## 1. Supported Architecture Families

MZSAE has been verified across all major transformer design patterns without requiring model downloads or weight conversions:

| Architecture Pattern | Examples | Query Heads | KV Heads | Head Dim | Special Features |
|---|---|---|---|---|---|
| **Multi-Head Attention (MHA)** | LLaMA-7B, Phi-3 | 32 | 32 | 128 / 96 | 1:1 KV ratio |
| **Grouped-Query Attention (GQA)** | LLaMA-3-8B, Qwen2.5 | 32 / 28 | 8 / 4 | 128 | 4:1 to 7:1 KV ratio |
| **Multi-Query Attention (MQA)** | Falcon-7B | 71 | 1 | 64 | 71:1 KV ratio |
| **Sliding Window Attention (SWA)** | Mistral-7B | 32 | 8 | 128 | Local attention window (e.g. 4096) |
| **Mixture of Experts (MoE)** | Mixtral-8x7B, DeepSeek-V2 | 32 / 64 | 8 | 128 | Expert-shared KV cache |
| **Large Head Dimensions** | Gemma-2-9B | 16 | 8 | 256 | $d_k = 256$ support |

---

## 2. Using MZSAE with Any Architecture

### Via `MZSAEAttention` (PyTorch Drop-in)
Pass the model's dimensions directly:

```python
import torch
from mzsae import MZSAEAttention

# Example: Configuring for Gemma-2 (Head Dim = 256)
attn = MZSAEAttention(
    embed_dim=4096,
    num_heads=16,
    num_kv_heads=8,
    head_dim=256,
    hardware_profile="auto"
)

x = torch.randn(1, 16, 4096)
out = attn(x, causal=True)
```

### Via `MZSAENeutralWrapper` (Universal Backend Wrapper)
Wrap any custom attention backend or PyTorch forward function:

```python
import numpy as np
from mzsae.neutral import MZSAENeutralWrapper, StandardAttentionWrapper

# Any shape works out of the box
q = np.random.randn(1, 1, 71, 64)   # Falcon-7B MQA
k = np.random.randn(1, 512, 1, 64)
v = np.random.randn(1, 512, 1, 64)

mzsae = MZSAENeutralWrapper(StandardAttentionWrapper())
out = mzsae.decode_step(q, k, v)
```

---

## 3. Zero-Storage Verification

All architectures can be tested and verified locally without downloading weights using the synthetic shape matrix test suite:

```bash
pytest tests/test_architecture_neutrality.py -v
pytest tests/test_moe_neutrality.py -v
pytest tests/test_property_neutrality.py -v
```
