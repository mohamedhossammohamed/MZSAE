# MZSAE: Neuromorphic Sparse Attention Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon%20%7C%20CPU-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/hardware-Metal%20MSL%203.1%20%7C%20Apple%20M--Series-purple.svg)]()

**MZSAE** is a hardware-sympathetic, biologically-inspired sparse attention engine designed to shatter the memory bandwidth wall in autoregressive LLM decoding. 

By decoupling Rotary Position Embeddings (RoPE) into fast and slow manifolds, MZSAE constructs L2-resident Cauchy-Schwarz sentinels that allow the GPU to evaluate block relevance *without* reading the payload from DRAM. Combined with a neuromorphic Temporal Difference (TD) eviction policy, MZSAE achieves infinite effective context on edge devices.

---

## 🚀 Headline Metrics (Apple Silicon M4)
* **~65 tok/s estimated E2E generation** on Qwen2.5-0.5B (663 µs **per-layer** attention latency × 24 layers; per-layer steps ≠ full-model tok/s — see `docs/LIMITATIONS.md`).
* **100.0% synthetic-needle NIAH retrieval** at 16k context under a strict 2,048-token physical RAM ceiling (magnitude-2.5 beacon; subtle background-magnitude needles characterized separately via `bench_niah.py --subtle-sweep`).
* **83.9% DRAM traffic reduction (estimated from kernel-observed prune counts)** vs. dense FP16 attention; not yet validated with hardware memory-controller counters.
* **Eviction-policy advantage**: Dense attention + FIFO truncation collapses under memory pressure, while MZSAE's biological veto protects critical semantic carriers. Note: FlashAttention-2 is an exact arithmetic kernel without native eviction — our NIAH gap measures **biological eviction vs FIFO truncation**, not attention-math quality.

> Note: MZSAE adds a Pass-0 sentinel overhead. For contexts < 2k, dense FlashAttention is faster. MZSAE's advantage emerges at 4k–16k+ context, where dense attention chokes on the memory wall. See `docs/LIMITATIONS.md` for the full red-team audit.

---

## 📦 Installation

MZSAE is distributed as a standard Python package with pre-compiled Metal kernels.

```bash
# Install from local source (development)
git clone https://github.com/mohammedhossam/MZSAE.git
cd MZSAE
make
pip install -e .

# Or install the pre-built wheel
pip install dist/mzsae-1.2.0-py3-none-any.whl
```

---

## ⚡ Quickstart

### 1. Standard PyTorch Attention Module (`MZSAEAttention`)

`MZSAEAttention` is a drop-in PyTorch module supporting both full layer forward passes with linear projections and direct functional execution on query/key/value tensors:

```python
import torch
from mzsae import MZSAEAttention

# Initialize module with auto hardware detection
attn = MZSAEAttention(
    embed_dim=2048,
    num_heads=16,
    num_kv_heads=4,
    hardware_profile="auto",
)

# 1. Functional forward pass: attn(q, k, v)
q = torch.randn(1, 16, 16, 128)
k = torch.randn(1, 16, 4, 128)
v = torch.randn(1, 16, 4, 128)
out = attn(q, k, v, causal=True)
print("Output shape:", out.shape)  # [1, 16, 16, 128]

# 2. Layer forward pass with internal linear projections
x = torch.randn(1, 16, 2048)
out_projected = attn(x, causal=True)
print("Layer output shape:", out_projected.shape)  # [1, 16, 2048]
```

### 2. Autoregressive Single-Token Decoding with Unified Cache

```python
import torch
from mzsae import MZSAEAttention

attn = MZSAEAttention(embed_dim=1536, num_heads=12, num_kv_heads=2)

# Step 1: Prefill prompt into unified dual-plane cache
prompt = torch.randn(1, 64, 1536)
_ = attn(prompt, causal=True, use_cache=True)

# Step 2: Generate tokens autoregressively with hardware selective fetch
for step in range(5):
    next_token = torch.randn(1, 1, 1536)
    out_token = attn(next_token, causal=True, use_cache=True)
    print(f"Step {step + 1} output shape:", out_token.shape)
```

### 3. Hardware Configuration & Profile Loading

```python
import mzsae

# Load specific hardware profile (e.g. Apple M4, M4 Max, M1, or auto)
config = mzsae.load_config("apple_m4")
print(f"Target Chip: {config.hardware.chip_name}")
print(f"DRAM Bandwidth: {config.hardware.memory_bandwidth_gbps} GB/s")

# Instantiate low-level execution engine
engine = mzsae.MZSAEEngine(config=config)
```

### 4. Zero-Copy MLX Integration

```python
import numpy as np
import mlx.core as mx
from mzsae import MZSAEEngine, load_config

engine = MZSAEEngine(config=load_config("auto"))

# Ingest MLX prefill into dual-plane cache
k_mlx = mx.random.normal((128, 2, 128)).astype(mx.float16)
v_mlx = mx.random.normal((128, 2, 128)).astype(mx.float16)
engine.cache.ingest_prefill(np.array(k_mlx), np.array(v_mlx))

# Decode MLX query with Metal selective fetch
q_mlx = mx.random.normal((12, 128)).astype(mx.float32)
out_np, telemetry = engine.selective_decode(np.array(q_mlx), return_telemetry=True)
print(f"Pruned blocks: {telemetry['pruning_ratio']*100:.1f}%")
```

---

## 🧠 Core Architecture

### 1. Dual-Plane Memory Hierarchy
* **Plane 1 (Payload Manifold)**: Keys and values are compressed into 4-bit affine quantized blocks aligned to 128-byte physical DRAM cache lines.
* **Plane 2 (Sentinel Descriptors)**: Compact 64-byte descriptors fitting exactly inside a single L2 cache line. Sentinels store the slow-manifold centroid $\mathbf{s}_{\text{slow}}$, maximum residual radius $R_\delta$, fast-manifold bound $C_{\text{fast}}$, accumulated attention mass $a_{\text{cum}}$, and block flags.

### 2. RoPE Manifold Decoupling & Cauchy-Schwarz Bound
Standard RoPE rotates coordinates uniformly across all frequencies. MZSAE decouples RoPE frequencies into:
* **Slow Subspace** (lowest 16 dimensions): Angular velocity $\omega \approx 0$ over long contexts, making key projections quasiconstant.
* **Fast Subspace** (remaining 112 dimensions): High angular velocities providing local token distinction.

The threadgroup evaluates the strict Cauchy-Schwarz upper bound:
$$u_b = \frac{1}{\sqrt{d}} \left( \mathbf{q}_{\text{slow}} \cdot \mathbf{s}_{\text{slow}} + \|\mathbf{q}_{\text{slow}}\| R_\delta + C_{\text{fast}} \right)$$
If $u_b < \text{local\_max} - \tau$, the block's attention contribution is mathematically bounded below numerical threshold, allowing the threadgroup to skip loading the block's Plane-1 payload entirely.

### 3. Neuromorphic TD Eviction & Counterfactual Directional Veto
* **Directional Veto**: A geometric gate intercepts every cache eviction request. If $\cos(\mathbf{q}_{\text{slow}}, \mathbf{s}_{\text{slow}}) > 0.4$, the block is classified as a critical semantic carrier and unconditionally protected from eviction.
* **TD-Attention Policy**: A 27,009-parameter MLP value network trained via online $TD(0)$ reinforcement learning during offline sleep replay cycles, mirroring hippocampal-neocortical memory consolidation.

---

## 📊 Benchmark Verification

### 1. Head-to-Head: Dense Attention (FIFO) vs MZSAE (Biological Eviction) (Apple Silicon M4)
*Task: Needle In A Haystack (NIAH) under strict 2,048-token physical memory ceiling.*
*Disclosure: FlashAttention-2 is an exact arithmetic kernel without native eviction; baseline uses FIFO truncation. Gap = eviction-policy effect.*

| Context Length | Dense+FIFO Baseline | MZSAE Metal Decode | Speedup | Dense+FIFO Acc (Budgeted) | MZSAE Acc (Budgeted) | DRAM Cut (est.) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4,096** | 1.84 ms | 1.25 ms | **1.47x** | 60.0% | **100.0%** | **83.9%** |
| **8,192** | 3.25 ms | 1.88 ms | **1.73x** | 20.0% | **100.0%** | **83.9%** |
| **16,384** | 6.12 ms | 3.10 ms | **1.97x** | 0.0% (collapse) | **100.0%** | **83.9%** |

### 2. Real Neural Model Weights (Qwen2.5-0.5B GGUF, 24 layers)
* **Metal Hardware**: Apple M4 (16 GB Unified Memory)
* **Context**: 2,048 tokens across 32 compressed blocks
* **Per-Layer Attention Latency**: **663.16 µs (~1,508 layer-steps/sec — Layer-0 only, NOT full-model tok/s)**
* **Estimated E2E Decode**: **~15.9 ms/token → ~63 tok/s** (per-layer × 24; lower bound, excludes MLP/norm/sampling)
* **Active Cache Footprint**: **585.0 KB** (Plane 1) + **3.8 KB** (Plane 2) = **588.8 KB total**
* **Memory Safety**: PASS (Zero swap, RSS << 1 GB)

---

## 🖥️ Hardware Profiles & Backend Matrix

| Profile | Hardware Architecture | DRAM Bandwidth | L2 / SLC | Backend |
|---|---|---|---|---|
| `apple_m1` | Apple M1 / M2 / M3 Base | 68.25 GB/s | 12 MB | Metal (MSL 3.1) |
| `apple_m4` | Apple M4 Base | 120.0 GB/s | 16 MB | Metal (MSL 3.1) |
| `apple_m4_max` | Apple M4 Max | 410.0 GB/s | 48 MB | Metal (MSL 3.1) |
| `default` | Portable Host Profile | 120.0 GB/s | 16 MB | Metal / CPU Reference |
| `nvidia_future` | NVIDIA H100 SXM5 | 3,350.0 GB/s | 50 MB | CUDA SM90 (Roadmap) |

---

## 📁 Repository Structure

```
mzsae/
├── pyproject.toml              # Modern build-system & packaging specification
├── Makefile                    # Developer targets (all, test, package, clean)
├── LICENSE                     # MIT License
├── README.md                   # Core documentation
├── configs/                    # Hardware configuration profiles (YAML)
│   ├── default.yaml
│   ├── apple_m1.yaml
│   ├── apple_m4.yaml
│   ├── apple_m4_max.yaml
│   └── nvidia_future.yaml
├── docs/                       # Technical specifications
│   ├── quickstart.md
│   ├── configuration.md
│   ├── LIMITATIONS.md            # Red-team audit: what benchmarks do NOT prove
│   └── hardware_profiles.md
├── examples/                   # Standalone runnable scripts
│   ├── minimal_pytorch.py
│   └── minimal_mlx.py
├── src/mzsae/
│   ├── __init__.py             # Clean public interface
│   ├── version.py              # Single source of truth (v1.2.0)
│   ├── config.py               # Dataclass schemas & YAML loader
│   ├── errors.py               # Custom exceptions
│   ├── logging.py              # Telemetry logger
│   ├── nn.py                   # MZSAEAttention(nn.Module)
│   ├── api.py                  # High-level functional facade
│   ├── core/                   # Algorithmic primitives
│   │   ├── cache.py            # Dual-plane compressed KV cache
│   │   ├── engine.py           # Core execution engine
│   │   ├── eviction.py         # Dynamic block eviction manager
│   │   ├── veto.py             # Directional veto interlock
│   │   ├── td_policy.py        # TDAttnPolicy & TelemetryRingBuffer
│   │   ├── rope.py             # Frequency spectrum decoupling & Givens rotations
│   │   └── compression.py      # CRQ 2-bit quantization routines
│   └── backends/               # Hardware abstraction layer
│       ├── base.py             # Abstract MZSAEBackend interface
│       ├── dispatcher.py       # Hardware backend selector
│       ├── cpu_reference.py    # NumPy SIMD CPU fallback backend
│       ├── metal/              # Apple Silicon Metal backend
│       │   ├── runtime.py      # C-ABI ctypes runtime bridge
│       │   └── kernels/        # MSL shaders and C++ runtime
│       └── cuda/               # NVIDIA Ampere/Hopper roadmap & stubs
└── tests/                      # 33 verified unit tests
```

---

## 🛠️ Developer Verification & Tests

```bash
# Run full test suite (33 unit tests)
make test

# Build package distribution wheel
make package

# Run benchmarks
python3 benchmarks/bench_niah.py
python3 benchmarks/bench_flash_attn_niah.py
python3 benchmarks/bench_real_weights.py
```

---

## 📜 License

MIT License. Copyright (c) 2026 Mohammed Hossam Zahran.
