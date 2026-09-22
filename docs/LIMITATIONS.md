# MZSAE Limitations & Red-Team Disclosures (Pre-Launch Audit)

Top-tier researchers value honesty over hype. This file documents exactly what
MZSAE's benchmarks do and do **not** prove, so skeptics don't have to find out
for us on Hacker News.

---

## 1. Single-Layer vs End-to-End Throughput

`benchmarks/bench_real_weights.py` ingests tokens through **Layer-0 attention
projections only** and measures **per-layer attention decode latency**
(~638–663 µs on Apple M4, i.e. ~1,500 layer-steps/sec).

Qwen2.5-0.5B has **24 transformer layers**. Estimated end-to-end decode:

```
e2e_ms_per_token ≈ per_layer_ms × 24
e2e_tok_per_sec  ≈ 1000 / e2e_ms_per_token ≈ 50–65 tok/s
```

Audit-run example (M4, 2k context, 100 steps): per-layer 774.80 µs
(1,290.6 layer-steps/sec) → E2E ≈ 18.60 ms/token → **≈ 53.8 tok/s**
(lower bound; excludes MLP/norm/sampling, so true E2E is ≤ this).

That is still world-class for 2k–16k context on a laptop, but it is **not**
"1,500 tok/s generation." The benchmark now prints both numbers separately:

- **Per-Layer Attention Latency** (what was timed)
- **Estimated E2E Model Throughput** (layer × 24, lower bound — excludes
  MLP/norm/sampling overhead, so true E2E tok/s is ≤ the estimate)

Do not cite the per-layer figure as full-model speed.

## 2. FlashAttention Comparison Is Eviction Policy, Not Kernel Math

**COMPARISON: MZSAE (Biological Eviction) vs. Dense Attention (FIFO Truncation).**

FlashAttention-2 is an **exact arithmetic kernel** (tiling + online softmax).
It does not natively manage KV-cache eviction. Under a RAM ceiling (e.g.
2,048 tokens), the standard deployment truncates oldest tokens first (FIFO).
If the needle lies outside the retained window, dense attention drops it —
not because the math is wrong, but because the eviction policy discarded it.

MZSAE retains the needle because of its **eviction policy** (sentinel bounds +
counterfactual directional veto + TD policy), not because its attention
arithmetic differs. Accuracy gaps in `bench_flash_attn_niah.py` therefore
measure **eviction-policy quality**, not kernel-math quality.

Never claim "MZSAE attention math beats FlashAttention math."

## 3. Synthetic Needle Caveat (Magnitude Beacon vs Semantic Needles)

`run_niah_trial()` injects a synthetic beacon:

```python
needle_slow_direction[slow_idx] = 2.5
probe_query[slow_idx] = 2.5
```

This creates a large magnitude spike in exactly the slow-manifold dimensions
the sentinel evaluates. The Cauchy-Schwarz bound

```
u_b = (q_slow·s_slow + ||q_slow||·R_δ + C_fast) / √d
```

trivially catches it. Real semantic needles in natural text do **not** carry
magnitude-2.5 spikes; they show subtle vector alignment at background norms.

Consequences:

- The reported **100% NIAH = 100% on synthetic beacons**, not on natural
  semantic needles.
- Extremely subtle needles with low L2 norms **may be pruned** if τ is too
  aggressive, because `u_b` scales with magnitude even when cosine ≈ 1.0.
- The directional eviction veto (cos > 0.4) still protects aligned blocks
  from eviction, but the decode-time CS bound is a separate gate.

Stress test: `benchmarks/bench_niah.py::test_subtle_semantic_needle` injects a
needle at **background magnitude** (`0.5·√n_slow`) with **cos ≈ 1.0** to the
probe. Run it:

```bash
python3 benchmarks/bench_niah.py --subtle-sweep
```

Empirical result (M4, 4k context, 2026-09-22 audit run):

```
tau= 4.0 | cos(needle,probe)=1.000 | cos(bg,probe)=-0.143 | signal=0.341 | approved=62 | [RETAINED-ONLY]
tau=16.0 | cos(needle,probe)=1.000 | cos(bg,probe)=-0.143 | signal=0.341 | approved=62 | [RETAINED-ONLY]
```

The subtle needle is **retained** (directional veto correctly fires on
cos 1.0 > 0.4) but **not retrieved** (signal 0.34 < 1.0 threshold): without
the magnitude beacon the needle earns no excess softmax mass and is diluted
by ~62 approved background blocks. The failure mode is attention dilution,
not eviction loss — visible across all τ, so lowering τ alone does not fix
it. Proving semantic-needle retrieval requires NIAH over **real text
embeddings from a frozen LLM**, not `np.random.randn` + forced spikes
(roadmap). A dynamic-τ ACC veto and value-aware re-weighting are the
candidate mitigations.

## 4. E2E Overhead: When Dense Is Still Faster

MZSAE adds a **Pass 0 (Sentinel Eval + local-max)** overhead before selective
decode. For very short contexts (< 2k tokens) the memory wall hasn't been hit
and dense FlashAttention is still faster — the sentinel pass is pure overhead.

MZSAE's ROI begins at **~4k+ context** and grows with context length, where
dense attention chokes on DRAM bandwidth. Do not benchmark MZSAE at 512 tokens
and claim a speedup.

## 5. Random Tensor Caveat (Shape ≠ Semantics)

`tests/test_architecture_neutrality.py` uses **synthetic Gaussian tensors**.
`np.random.randn` has no attention sinks, outliers, or heavy tails, so the
CS bound is trivially loose and the test only asserts shape/finiteness.

What it proves: **Shape & Routing Invariance** (GQA/MQA broadcasting works
across MHA/GQA/MQA geometries and head dims 64–256).

What it does not prove: numerical correctness or retrieval accuracy on real
model activations. Real-world heavy-tailed distributions require the dynamic-τ
ACC veto to prevent over-pruning. Full verification needs `transformers` /
`mlx-lm` integration on real weights (roadmap).

## 6. Telemetry Trust Gap (DRAM Estimates, Not Counters)

- `approved_blocks` / `total_blocks` are **kernel-observed**: the Metal kernel
  writes a `block_approved` bitmap (lane 0, approved blocks only) into
  `splitStatsBuffer`, summed on GPU after `waitUntilCompleted`. Python does
  not invent these counts.
- **BUT** `DRAM traffic reduction %` (e.g. 83.9%) is a **Python estimate**:
  `approved_count × block_byte_size`, not Apple Silicon Memory Controller
  traffic from `powermetrics` or MPS counters.
- The kernel was red-team verified: `if (!approve) continue;` in
  `mzsae_selective_decode_stage1` branches past **all** Plane-1
  `device_load`s (centroid/scale/min fetch + k/v payload dequant loop). Only
  the 64B L2-resident sentinel + q-register reads execute for pruned blocks.
  Sinks + recent window are always fetched by design. No hidden fallback
  reloads pruned blocks (see `// RED-TEAM VERIFIED` comment in
  `src/mzsae/backends/metal/kernels/mzsae_kernels.metal`).
- Until hardware-counter validation lands, cite DRAM cuts as
  **"estimated from kernel-observed prune counts"**, not measured bus traffic.

---

*If you find a flaw not listed here, that's a bug in this file — please open
an issue. We'd rather fix the narrative now than defend hype later.*
