---
hide:
  - navigation
---

<div class="eyebrow">Neuromorphic sparse attention · v1.3.0</div>

# Attention that reads only what matters.

<p class="lede">
Autoregressive decoding is not compute-bound — it is choked by the memory bus. MZSAE evaluates 64-byte semantic sentinels in cache before a single byte crosses DRAM, and manages eviction with a biologically-inspired policy loop. The result: edge devices that remember like brains, not like FIFO queues.
</p>

<div class="hero-cta">
  <a class="md-button md-button--primary" href="quickstart.md">Quickstart Guide</a>
  <a class="md-button" href="ARCHITECTURE.md">Core Architecture</a>
  <a class="md-button" href="LIMITATIONS.md">Limitations Audit</a>
  <span class="install-badge"><b>$</b> pip install mzsae</span>
</div>

<div class="hero-meta">
  <span>Apache-2.0</span> • <span>100 tests green</span> • <span>Apple Silicon validated</span> • <span>CUDA code-complete, unvalidated</span>
</div>

<div class="metrics-band">
  <div class="metric-item">
    <div class="metric-val"><em>8.4×</em></div>
    <div class="metric-lbl">faster decode than dense SDPA at 11.6k context</div>
  </div>
  <div class="metric-item">
    <div class="metric-val">96.7<em>%</em></div>
    <div class="metric-lbl">of DRAM traffic pruned at the same context</div>
  </div>
  <div class="metric-item">
    <div class="metric-val">10<em>/</em>10</div>
    <div class="metric-lbl">retrieval scenarios bit-exact vs. dense baseline</div>
  </div>
  <div class="metric-item">
    <div class="metric-val">3.28<em>×</em></div>
    <div class="metric-lbl">smaller physical KV-cache footprint</div>
  </div>
</div>

<div class="metric-fineprint">
Measured on Apple Silicon M4 (120 GB/s), Qwen2.5-0.5B. DRAM reductions estimated from kernel-observed prune counts, not hardware counters. Conservative full-model estimate: ~63 tok/s E2E at 16k context.
</div>

---

## The Mechanism: Three Ideas, Composed

<div class="mech-cards">
  <div class="mech-card">
    <div class="n">01</div>
    <h3>Plane-2 Sentinels</h3>
    <p>RoPE is decoupled into fast positional and slow semantic manifolds. Each 64-token block leaves a 64-byte sentinel in L2. A tight Cauchy–Schwarz bound is evaluated in registers; irrelevant blocks branch past DRAM fetch entirely.</p>
    <div class="eq">U_b = (⟨q_s, s_s⟩ + ‖q_s‖·R_Δ + C_f) / √d</div>
  </div>
  <div class="mech-card">
    <div class="n">02</div>
    <h3>Dual-Plane Cache</h3>
    <p>Approved blocks live as centroid plus 2-bit residuals — the same coarse-anchor, fine-detail instinct as NVFP4 — rebuilt in registers at use. The compressed form travels; the full form exists only for a moment.</p>
    <div class="eq">K ≈ μ + Δ,  Δ ∈ 2-bit,  μ ∈ FP16</div>
  </div>
  <div class="mech-card">
    <div class="n">03</div>
    <h3>Biological Eviction</h3>
    <p>An orbitofrontal value MLP scores block survival; an ACC-style directional veto protects semantically aligned blocks from greedy eviction; a hippocampal ring buffer replays transitions offline via TD(0) during idle cycles.</p>
    <div class="eq">V(s) ← r + γ·V(s′),  veto if cos θ_slow > 0.40</div>
  </div>
</div>

---

## Benchmark Verification: Ten Scenarios

Dual-panel end-to-end generation against dense SDPA with full KV streaming. Outputs verified bit-exact across all evaluated workloads.

| Scenario | Context | Dense SDPA | MZSAE | Speedup | DRAM Pruned | Integrity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Needle-in-haystack passcode | 7,144 | 13.4 tok/s | 94.1 tok/s | **7.00×** | 97.3% | exact |
| Multi-hop cross-document | 3,382 | 16.8 tok/s | 95.1 tok/s | **5.66×** | 96.2% | exact |
| Financial ledger audit | 3,941 | 16.2 tok/s | 94.9 tok/s | **5.86×** | 96.7% | exact |
| Adversarial decoy distractors | 3,159 | 17.0 tok/s | 94.9 tok/s | **5.57×** | 95.9% | exact |
| Distant system-rule retention | 6,188 | 14.1 tok/s | 93.7 tok/s | **6.63×** | 96.9% | exact |
| Codebase API lookup | 3,044 | 17.2 tok/s | 95.3 tok/s | **5.54×** | 95.7% | exact |
| Timeline contradiction spotting | 2,564 | 17.7 tok/s | 94.7 tok/s | **5.94×** | 95.0% | exact |
| Syslog JSON incident triage | 7,212 | 13.4 tok/s | 93.9 tok/s | **7.02×** | 97.3% | exact |
| Multi-chapter narrative arc | 3,626 | 16.5 tok/s | 94.9 tok/s | **5.74×** | 96.4% | exact |
| Scaling stress test | 11,600 | 10.9 tok/s | 91.2 tok/s | **8.40×** | 96.7% | exact |

---

## 3-Line Drop-in Integration

=== "PyTorch Drop-in Module"

    ```python
    import torch
    from mzsae import MZSAEAttention

    # Drop-in multi-head / grouped-query attention
    attn = MZSAEAttention(embed_dim=2048, num_heads=16, num_kv_heads=4, hardware_profile="auto")
    out = attn(q, k, v, causal=True)
    ```

=== "Apple MLX Zero-Copy"

    ```python
    import numpy as np, mlx.core as mx
    from mzsae import MZSAEEngine, load_config

    engine = MZSAEEngine(config=load_config("auto"))
    engine.cache.ingest_prefill(np.array(k_mlx), np.array(v_mlx))
    out_np, telemetry = engine.selective_decode(np.array(q_mlx), return_telemetry=True)
    print(f"Pruned blocks: {telemetry['pruning_ratio']*100:.1f}%")
    ```

=== "Pip Installation"

    ```bash
    pip install mzsae
    ```

---

## Documentation Index

Explore the comprehensive technical documentation for MZSAE:

<div class="grid cards" markdown>

-   __Architecture Foundations__

    ---

    Dual-plane memory hierarchy, RoPE manifold decoupling, Cauchy-Schwarz sentinels, and hardware-sympathetic dispatch.

    [:material-arrow-right: Read Architecture](ARCHITECTURE.md)

-   __Neutrality Proof__

    ---

    Mathematical invariance across MHA, GQA, MQA, Sliding Window, MoE, and multi-tier quantization profiles.

    [:material-arrow-right: View Neutrality Proof](NEUTRALITY_PROOF.md)

-   __Scaling Laws__

    ---

    Memory bandwidth saturation models, arithmetic intensity crossover points, and long-context scaling bounds.

    [:material-arrow-right: Explore Scaling Laws](SCALING_LAWS.md)

-   __Quantization Compatibility__

    ---

    Outlier preservation, dynamic affine residual mapping, and BitNet b1.58 ternary support.

    [:material-arrow-right: Inspect Quantization](QUANTIZATION.md)

-   __Hardware Profiles__

    ---

    Tuned L2/SLC sentinel budgets for Apple M1 through M4 Max, plus Hopper, Grace Hopper, and Blackwell configurations.

    [:material-arrow-right: Check Profiles](hardware_profiles.md)

-   __Full Benchmark Suite__

    ---

    Per-layer attention decode benchmarks, NIAH subtle-needle sweeps, and comparative FlashAttention telemetry.

    [:material-arrow-right: Analyze Benchmarks](benchmarks.md)

</div>

---

## Known Limits & Red-Team Disclosures

<div class="honest-box">
  <h3>Read this before benchmarking us.</h3>
  <p>
  Below ~2k context, dense attention is faster — the sentinel pass is overhead until the memory wall bites. Our DRAM figures are estimates from kernel-observed prune counts, not powermetrics counters. Needles with background-level magnitude but perfect alignment dilute across approved blocks: the failure mode is attention dilution, not eviction loss. Architecture tests prove shape and routing invariance on synthetic tensors, not semantic correctness on trained weights. Every boundary above, with raw numbers, lives in <a href="LIMITATIONS.md">docs/LIMITATIONS.md</a>.
  </p>
</div>
