# Needle In A Haystack (NIAH) Results

The Needle In A Haystack (NIAH) benchmark evaluates an attention mechanism's ability to retrieve a specific piece of information (the "needle") buried deep within a long context window (the "haystack").

## Why NIAH Matters for Long-Context

Standard dense attention mechanisms retain all KV pairs. When forced to operate under a strict memory ceiling (e.g., edge devices), they must aggressively prune tokens. If the eviction policy is naive, critical information is lost, leading to hallucination and context collapse.

## Test Setup

- **Memory Ceiling:** Strict 2,048-token active retention limit.
- **Eviction:** Active TD(0) neuromorphic eviction.
- **Task:** Retrieve a planted vector within context sizes of 4k, 8k, and 16k tokens.

## Results Table (5-Depth Sweep: 0.10, 0.30, 0.50, 0.70, 0.90)

| Context Length | Baseline (Dense + FIFO) | MZSAE |
| -------------- | ----------------------- | ----- |
| 4,096          | 60.0% (3/5 depths)      | **100.0%** (5/5 depths) |
| 8,192          | 20.0% (1/5 depths)      | **100.0%** (5/5 depths) |
| 16,384         | 0.0% (0/5 depths)       | **100.0%** (5/5 depths) |

### Why Dense + FIFO Collapses
A standard FIFO (First-In-First-Out) truncation drops the oldest tokens once the 2,048-token limit is reached. At 16k context, a needle placed at earlier or middle depths is evicted from the KV cache because the retained window only spans the most recent 2,048 tokens, resulting in context collapse.

### Why MZSAE Retains the Needle
MZSAE utilizes **Sentinel Bounds** and an **ACC Directional Veto** (cosine gate > 0.4). The neuromorphic eviction policy dynamically evaluates the utility of each block. High-magnitude/high-relevance features are protected from eviction, ensuring the needle remains in the active memory footprint regardless of its absolute position.

!!! warning "Subtle Caveat (Magnitude Beacons)"
    As detailed in [LIMITATIONS.md](LIMITATIONS.md), the current NIAH test heavily relies on synthetic magnitude beacons. Real-world semantic needles in frozen LLM embeddings may exhibit different retention profiles until v2.0.0.

### Reproduction

```bash
python benchmarks/bench_niah.py --max-retention 2048 --seq-lens 4096 8192 16384
```
