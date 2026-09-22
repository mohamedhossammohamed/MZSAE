# FlashAttention Comparison

When discussing efficient attention mechanisms, FlashAttention is frequently cited. It is crucial to understand the architectural differences and distinct use cases between FlashAttention-2 (FA-2) and MZSAE.

## What FlashAttention-2 Is
FA-2 is an exact arithmetic kernel. It computes the *exact* same mathematical output as standard dense attention but optimizes memory bandwidth via:
- Tiling (SRAM chunking)
- Online softmax (avoiding materialization of the full attention matrix)

## What FlashAttention-2 Is NOT
FA-2 is **not** an eviction policy. It still requires computing and storing the entire KV cache in DRAM.

## Architecture Comparison

| Feature | FlashAttention-2 | MZSAE |
| :--- | :--- | :--- |
| **Output** | Exact | Approximate (0.9968 cosine fidelity) |
| **Eviction Policy** | None (FIFO under constraint) | Neuromorphic TD(0) & ACC Veto |
| **Memory Footprint** | $O(N)$ KV Cache | Fixed Ceiling (e.g., 2048 tokens) |
| **Pruning** | None | 95.9% block pruning |

## Benchmark Results (Under Memory Pressure)

When constrained by a strict RAM limit (e.g., 2,048 tokens), FA-2 combined with a standard FIFO truncation policy collapses at long contexts, whereas MZSAE retains critical information.

| Metric | FA-2 + FIFO (16k context) | MZSAE (16k context) |
| :--- | :--- | :--- |
| **NIAH Retrieval** | 0% | 100% |
| **Memory Bound** | Fails early | Stays within 2048-token limit |

!!! danger "Critical Distinction"
    The performance gap in NIAH tests under memory constraints is a measurement of **eviction-policy quality**, NOT **kernel-math quality**. MZSAE succeeds because it intelligently evicts irrelevant tokens; FA-2 fails because its naive FIFO wrapper drops the needle.

## When to Use Which?

- **Use FlashAttention-2:** When you have sufficient memory to hold the entire KV cache and require 100% exact arithmetic precision (e.g., model pre-training, short-context inference).
- **Use MZSAE:** When deploying on edge hardware (Apple Silicon) with strict memory ceilings, where achieving massive context lengths (64k-128k) requires intelligent sparse eviction without catastrophic forgetting.
