# MZSAE Scaling Laws & Complexity Analysis

This document provides the formal computational and memory complexity derivations for the MZSAE Dual-Plane Sparse Attention Engine compared to standard dense scaled dot-product attention (Dense SDPA) and FlashAttention-2.

---

## 1. Complexity Comparison

Let $N$ denote the context sequence length, $d$ the head dimension ($d=128$), $B$ the block size ($B=64$), $H_q$ the number of query heads, and $H_{kv}$ the number of KV heads (with $H_q / H_{kv} = 6$ for GQA 6:1).

| Operation / Property | Dense SDPA / FlashAttention | MZSAE (Static 4-bit) | MZSAE (Selective Fetch) |
| :--- | :--- | :--- | :--- |
| **DRAM Memory Reads (KV)** | $\mathcal{O}(N \cdot d)$ FP16 bytes | $\mathcal{O}(N \cdot d / 4)$ 4-bit bytes | $\mathcal{O}((1 - \rho) \cdot N \cdot d / 4)$ bytes |
| **L2 Cache Accesses** | N/A (Direct DRAM streaming) | N/A | $\mathcal{O}(N / B)$ 64B Sentinels |
| **Arithmetic Intensity** | Low ($\approx 0.5$ FLOP/byte) | Medium ($\approx 1.8$ FLOP/byte) | **High** ($\approx 12.5$ FLOP/byte) |
| **KV Cache Storage** | $2 \cdot N \cdot d \cdot 2\text{ bytes} = 4Nd$ | $1.22Nd\text{ bytes}$ ($3.28\times$ cut) | $1.22Nd\text{ bytes}$ ($3.28\times$ cut) |
| **Eviction Mechanism** | Static FIFO / Truncation | Static FIFO | **Neuromorphic TD(0) + ACC Veto** |

*Here $\rho$ represents the empirical block pruning ratio ($\rho = 0.959$ on evaluated workloads).*

---

## 2. Theoretical Crossover Point

MZSAE introduces a two-tier Pass-0 evaluation overhead:
1. **Pass 0 Sentinel Bounds**: Loading 64-byte descriptors from L2 cache and computing Cauchy-Schwarz bound $U_b$.
2. **Pass 0 Local Max**: Computing $m_{\text{local}}$ over 4 attention sinks and the 64-token sliding window.

### Latency Modeling

The total decode latency $T(N)$ as a function of sequence length $N$:

$$
T_{\text{dense}}(N) = \alpha_{\text{dense}} + \frac{2 N d \cdot H_{kv} \cdot 2\text{ bytes}}{\text{BW}_{\text{DRAM}}}
$$

$$
T_{\text{MZSAE}}(N) = \alpha_{\text{sentinel}} + \frac{(N / B) \cdot 64\text{ bytes}}{\text{BW}_{\text{L2}}} + (1 - \rho) \cdot \frac{N d \cdot H_{kv} \cdot 0.5\text{ bytes}}{\text{BW}_{\text{DRAM}}}
$$

### Crossover Threshold

On Apple Silicon M4 ($\text{BW}_{\text{DRAM}} \approx 120\text{ GB/s}$, $\text{BW}_{\text{L2}} \approx 1.2\text{ TB/s}$):
- For $N < 2,048$: Dense FlashAttention-2 is faster due to negligible memory pressure and zero sentinel overhead.
- For $N \ge 4,096$: DRAM bandwidth saturation dominates; MZSAE achieves a $1.47\times$ speedup.
- For $N \ge 64,000$: MZSAE achieves $5.66\times$ speedup ($0.931\text{ ms}$ vs $5.274\text{ ms}$).
- For $N \ge 128,000$: MZSAE achieves $6.50\times$ speedup ($1.713\text{ ms}$ vs $11.140\text{ ms}$).

---

## 3. Sublinear Scaling via Multi-Split GQA

In standard decoding, doubling sequence length doubles attention latency ($2.00\times$). In MZSAE:
- Sequence length scaling from 64k to 128k ($2.00\times$ tokens) yields an execution latency delta of $1.84\times$ ($0.931\text{ ms} \to 1.713\text{ ms}$).
- This super-linear throughput scaling occurs because GQA 6:1 threadgroup reuse allows each fetched block to be dequantized once in registers and evaluated across all 6 query heads concurrently, maximizing GPU compute core saturation while DRAM buses remain unblocked.
