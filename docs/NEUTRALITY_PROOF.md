# Why MZSAE is More Neutral Than FlashAttention

A formal analysis of abstraction levels in LLM attention subsystems.

---

## 1. The Core Duality: Kernel vs. Policy

| Dimension | FlashAttention | MZSAE |
|---|---|---|
| **Primary Abstraction** | Arithmetic Kernel (Computes Attention) | Neuromorphic Policy (Decides What to Attend To) |
| **Operational Substrate** | Tensor contractions ($Q K^T$, $S V$) in SRAM | Attention manifold scores & Cauchy-Schwarz bounds |
| **Model Knowledge Required** | Strict: `head_dim`, `num_heads`, tiling dimensions | **None**: Decoupled from arithmetic computation |
| **Can Wrap Other Mechanisms?** | No (it is the leaf kernel) | **Yes** (wraps FlashAttention, Standard, MoE, etc.) |
| **Testing Prerequisite** | Often requires real weights / specific kernels | **Zero-Storage**: Verifiable with pure shape matrices |
| **Future Adaptability** | Must be re-engineered for non-softmax attention | **Invariant**: Operates on any scalar relevance metric |

---

## 2. FlashAttention's Structural Constraints

FlashAttention achieves its legendary speedup through fused IO-awareness: loading blocks of $Q$, $K$, and $V$ into high-speed GPU SRAM (or Apple Silicon threadgroup memory) and computing online softmax to avoid round-trips to DRAM.

However, this design binds FlashAttention to the **exact physical layout** of the attention tensors:
1. **Dimension Locking**: FlashAttention kernels are specialized for specific head dimensions (e.g. $d \in \{64, 128, 256\}$). Non-standard dimensions require padding or custom compilation.
2. **Fixed Attention Math**: FlashAttention computes standard scaled dot-product attention ($\text{softmax}(Q K^T / \sqrt{d}) V$). It cannot natively support biologically-inspired eviction vetoes or arbitrary non-standard attention formulations.
3. **No Eviction Authority**: FlashAttention computes attention over whatever tokens reside in the KV cache; it has no mechanism to determine which historical blocks should be retained or discarded under constrained memory budgets.

---

## 3. MZSAE's Architectural Neutrality

MZSAE operates at a higher level of abstraction: the **lifecycle and semantic relevance layer**.

### A. The Score-Level Decoupling
MZSAE does not dictate how attention scores are calculated. It accepts attention scores from:
- Standard scaled dot-product attention
- FlashAttention-2 / FlashAttention-3
- Sliding window attention
- MoE expert-shared caches
- Any arbitrary custom kernel

### B. Biological Policy Injection
Once scores or manifold bounds are known, MZSAE injects its neuromorphic intelligence:
1. **Cauchy-Schwarz Zero-Fetch Gating**: Evaluates Plane-2 sentinels in L2 cache, pruning low-relevance blocks before DRAM reads occur.
2. **Counterfactual Directional Veto**: Preserves critical tokens along the slow RoPE manifold ($\cos(\theta_{\text{slow}}) > 0.4$), preventing needle loss under eviction pressure.
3. **Temporal Difference (TD) Sleep Replay**: Tunes block retention priors during offline consolidation.

Because these operations are purely score- and bound-level decisions, **MZSAE can wrap FlashAttention itself**, running FlashAttention on the pruned subset of approved blocks!

---

## 4. Empirical Proof: The Zero-Storage Shape Matrix

To prove architectural neutrality without introducing storage or model-download overhead, MZSAE includes a comprehensive synthetic shape matrix covering all active LLM families:

```python
ARCHITECTURES = {
    "llama-7b":     {"heads": 32, "kv_heads": 32, "head_dim": 128}, # MHA 1:1
    "llama-3-8b":   {"heads": 32, "kv_heads": 8,  "head_dim": 128}, # GQA 4:1
    "mistral-7b":   {"heads": 32, "kv_heads": 8,  "head_dim": 128, "window": 4096}, # SWA
    "mixtral-8x7b": {"heads": 32, "kv_heads": 8,  "head_dim": 128}, # MoE
    "qwen2.5-7b":   {"heads": 28, "kv_heads": 4,  "head_dim": 128}, # GQA 7:1
    "gemma-2-9b":   {"heads": 16, "kv_heads": 8,  "head_dim": 256}, # d=256
    "phi-3-mini":   {"heads": 32, "kv_heads": 32, "head_dim": 96},  # d=96
    "falcon-7b":    {"heads": 71, "kv_heads": 1,  "head_dim": 64},  # MQA 71:1
    "deepseek-v2":  {"heads": 64, "kv_heads": 8,  "head_dim": 128},
}
```

Every architecture passes verified execution under `pytest tests/test_architecture_neutrality.py` and property-based sweeps under `pytest tests/test_property_neutrality.py`.

---

## Conclusion

**MZSAE does not compete with FlashAttention — it elevates it.**  
FlashAttention is the fast engine; MZSAE is the intelligent pilot deciding which territory is worth exploring.
