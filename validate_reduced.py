"""
Reduced first-pass validation of the MZSAE pruning design on REAL Qwen2.5-0.5B attention data.
No kernel changes. All results printed to stdout (caller tees to logs/mzsae_reduced_pass1.log).

Scope (per instruction):
  - Model: Qwen/Qwen2.5-0.5B-Instruct only (24 layers, 14 Q heads, 2 KV heads, head_dim=64, rope_theta=1e6)
  - 2 documents (real Wikitext-2 text), contexts 8k and 16k
  - Last 10 decode steps, 4 layers: [0, 7, 15, 23], all KV heads
  - Dense FP32 attention computed ONE LAYER AT A TIME, memory freed between layers
  - Task 1: fraction of 32-token blocks needed for 99% attention mass
  - Task 2: histogram of (U_b/sqrt(d) - true block max), prunable fraction at tau={16,8,4}
            using true local-window max m_t (last 128 tokens)
  - Watchdogs: abort + report if any phase > 300 s or macOS memory pressure turns red (>=4)
"""

import gc
import subprocess
import sys
import time

import numpy as np
import torch

LOG_TS = True


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def phase_begin(name):
    log(f"--- BEGIN phase: {name} ---")
    return time.time()


def phase_end(name, t0, mem_note=""):
    dt = time.time() - t0
    rss_gb, vm_pct, press = mem_snapshot()
    log(f"--- END phase: {name} | elapsed={dt:.1f}s | rss={rss_gb:.2f}GB "
        f"| vm_used={vm_pct:.1f}% | pressure={press} {mem_note}---")
    if dt > 300:
        log(f"WATCHDOG: phase '{name}' exceeded 300 s ({dt:.1f}s). STOPPING per instruction.")
        sys.exit(2)
    if press >= 4:
        log(f"WATCHDOG: memory pressure RED (level={press}). STOPPING per instruction.")
        sys.exit(3)
    return dt


def mem_snapshot():
    import psutil
    rss_gb = psutil.Process().memory_info().rss / 1e9
    vm_pct = psutil.virtual_memory().percent
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True, timeout=10)
        press = int(out.stdout.strip())
    except Exception:
        press = -1
    return rss_gb, vm_pct, press


# ---------------------------------------------------------------- RoPE (NeoX / Qwen2 convention)
def rope_neox(x, pos, inv_freq):
    """x (..., d), pos scalar-or-(...), inv_freq (d/2,). Matches HF apply_rotary_pos_emb."""
    x = np.asarray(x, dtype=np.float64)
    inv_freq = np.asarray(inv_freq, dtype=np.float64)
    a = np.asarray(pos, dtype=np.float64)[..., None] * inv_freq[None, :]
    c, s = np.cos(a), np.sin(a)
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return np.concatenate([o1, o2], axis=-1).astype(np.float32)


def rope_neox_inv(x, pos, inv_freq):
    return rope_neox(x, -np.asarray(pos), inv_freq)


def check_rope_against_hf():
    """Validate our NeoX rotation against transformers' own apply_rotary_pos_emb."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    rng = np.random.default_rng(0)
    d = 64
    base = 1_000_000.0
    invf = (base ** (-2.0 * np.arange(d // 2, dtype=np.float64) / d)).astype(np.float32)
    q = rng.standard_normal((2, 4, 8, d)).astype(np.float32)  # (B, H, L, d)
    k = rng.standard_normal((2, 4, 8, d)).astype(np.float32)
    pos = np.array([[3, 10, 100, 1000, 0, 1, 7, 65],
                    [9000, 5, 33, 511, 1024, 7777, 2, 40000]], dtype=np.float32)  # (B, L)
    freqs = (torch.from_numpy(pos)[:, :, None]
             * torch.from_numpy(invf.astype(np.float32))[None, None, :])  # (B, L, d/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (B, L, d), matches HF rotary forward
    cos, sin = emb.cos(), emb.sin()
    qe, ke = apply_rotary_pos_emb(torch.from_numpy(q), torch.from_numpy(k),
                                  cos, sin, unsqueeze_dim=1)
    qe = qe.numpy()
    ref = np.stack([[rope_neox(q[b, h, l], pos[b, l], invf)
                     for l in range(8)] for h in range(4)], axis=1)
    ref = np.array([[rope_neox(q[b, h, l], float(pos[b, l]), invf)
                     for l in range(8)] for b in range(2) for h in range(4)],
                   dtype=np.float32).reshape(2, 4, 8, d)
    err = np.max(np.abs(qe - ref))
    log(f"rope self-check: max|HF - ours| = {err:.3e} (must be < 1e-4)")
    assert err < 1e-4, f"RoPE convention mismatch: {err}"
    return True


# ---------------------------------------------------------------- MZSAE bound (EXACT replica of kernel_simulator.simulate_sentinel_filter)
BLOCK = 32
SLOW_DIMS = 16  # as coded in MZSAE for d=128; applied to d=64 below (last 16 coords)
MODEL_ID = "models/qwen-local"  # full-precision local weights (offline)


def mzsae_bound_terms(q_rot, k_unrot_block, slow_idx, fast_idx, d):
    """Returns (U, slow_term, resid_term, cfast_term) with U already divided by sqrt(d)."""
    centroid = k_unrot_block.mean(axis=0).astype(np.float64)
    resid = k_unrot_block.astype(np.float64) - centroid
    s_slow = centroid[slow_idx]
    r_delta = float(np.max(np.linalg.norm(resid[:, slow_idx], axis=1)))
    c_fast = float(np.max(np.linalg.norm(k_unrot_block[:, fast_idx], axis=1)))
    q = q_rot.astype(np.float64)
    q_slow = q[slow_idx]
    dot_slow = float(q_slow @ s_slow)
    nq_slow = float(np.linalg.norm(q_slow))
    inv = 1.0 / np.sqrt(d)
    slow_term = dot_slow * inv
    resid_term = (nq_slow * r_delta) * inv
    cfast_term = c_fast * inv  # BUG-REPLICA: kernel adds C_fast WITHOUT ||Q_fast|| factor
    return (slow_term + resid_term + cfast_term), slow_term, resid_term, cfast_term


# ---------------------------------------------------------------- main
def main():
    t_run0 = time.time()
    peak_rss = 0.0
    log("=" * 78)
    log("MZSAE REDUCED PASS 1: real Qwen2.5-0.5B attention data, no kernel changes")
    log("=" * 78)
    log(f"torch={torch.__version__} mps_available={torch.backends.mps.is_available()}")

    t0 = phase_begin("rope_convention_self_check")
    check_rope_against_hf()
    phase_end("rope_convention_self_check", t0)

    # ---- load tokenizer + build 2 long docs from cached wikitext-2 parquet
    t0 = phase_begin("build_documents")
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    snap = ("snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/"
            "train-00000-of-00001.parquet")
    pq = ("/Users/mohammedhossam/.cache/huggingface/hub/datasets--Salesforce--wikitext/"
          + snap)
    df = pd.read_parquet(pq)
    big_text = "\n\n".join(t for t in df["text"].tolist() if len(t.strip()) > 50)
    log(f"wikitext-2 chars for pool: {len(big_text)}")
    pool_ids = tok(big_text, add_special_tokens=False).input_ids
    log(f"token pool size: {len(pool_ids)}")
    assert len(pool_ids) > 300_000, "token pool too small"
    DOCS = {
        "docA": pool_ids[0:],
        "docB": pool_ids[200_000:],
    }
    CONTEXTS = [8192, 16384]
    for name, arr in DOCS.items():
        assert len(arr) >= 16384 + 64, f"{name} too short: {len(arr)}"
    phase_end("build_documents", t0)

    # ---- load model
    t0 = phase_begin("load_model")
    from transformers import AutoModelForCausalLM
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"loading Qwen2.5-0.5B-Instruct sdpa/bf16 -> {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16,
        attn_implementation="sdpa")
    cfg = model.config
    log(f"config: layers={cfg.num_hidden_layers} qheads={cfg.num_attention_heads} "
        f"kvheads={cfg.num_key_value_heads} hidden={cfg.hidden_size} "
        f"theta={cfg.rope_parameters}")
    NL, NH_Q, NH_KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // cfg.num_attention_heads
    GQA = NH_Q // NH_KV
    log(f"derived: head_dim={D} gqa_group={GQA}")
    assert (NL, NH_Q, NH_KV, D) == (24, 14, 2, 64), "unexpected 0.5B config"
    model.to(device)
    model.eval()
    inv_freq = model.model.rotary_emb.inv_freq.detach().to("cpu", torch.float32).numpy()
    log(f"model inv_freq[:4]={inv_freq[:4]} (theta=1/inv_freq[1]^{D/2}...)")
    SEL_LAYERS = [0, 7, 15, 23]
    log(f"selected layers (first/early-mid/late-mid/last): {SEL_LAYERS}")
    phase_end("load_model", t0)

    BIN_EDGES = np.arange(-60.5, 61.5, 1.0)  # 121 bins for gap histogram
    BIN_C = (len(BIN_EDGES) - 1)
    # accumulators per layer: over all docs x lengths x steps x qheads x blocks
    acc = {L: {"hist": np.zeros(BIN_C, dtype=np.int64),
               "under": 0, "over": 0, "n": 0,
               "viol": 0, "prune16": 0, "prune8": 0, "prune4": 0,
               "gapsum": 0.0, "gapsq": 0.0, "gapmin": 1e9, "gapmax": -1e9,
               "slow": 0.0, "resid": 0.0, "cfast": 0.0,
               "ceil99": []} for L in SEL_LAYERS}
    ceil_detail = []  # (doc, ctx, layer, kv, mean_frac, min_frac, max_frac)

    CHUNK = 2048
    NSTEP = 10
    all_phase_times = []

    with torch.no_grad():
        for doc_name, arr in DOCS.items():
            for ctx in CONTEXTS:
                ctx_ids = arr[:ctx]
                t0 = phase_begin(f"prefill+decode {doc_name} ctx={ctx}")
                input_all = torch.tensor([ctx_ids], dtype=torch.long, device=device)
                past = None
                pos = 0
                logits_last = None
                # chunked prefill
                for s in range(0, ctx, CHUNK):
                    e = min(s + CHUNK, ctx)
                    chunk = input_all[:, s:e]
                    cp = torch.arange(pos, pos + (e - s), device=device)
                    out = model(input_ids=chunk, past_key_values=past,
                                use_cache=True, cache_position=cp)
                    past = out.past_key_values
                    logits_last = out.logits[:, -1:, :]
                    pos = e
                    del out
                # decode hooks: capture UNROTATED q_proj outputs for selected layers
                saved_q = {}
                handles = []
                for li in SEL_LAYERS:
                    def _mk(lid):
                        def _h(mod, inp, outp):
                            saved_q[lid] = outp.detach().to("cpu", torch.float32).numpy()
                        return _h
                    handles.append(
                        model.model.layers[li].self_attn.q_proj.register_forward_hook(_mk(li)))
                q_unrot_steps = {li: [] for li in SEL_LAYERS}  # per layer: list of (14,64)
                for j in range(NSTEP):
                    nxt = torch.argmax(logits_last, dim=-1)  # (1,1)
                    cp = torch.tensor([pos], device=device)
                    out = model(input_ids=nxt, past_key_values=past,
                                use_cache=True, cache_position=cp)
                    logits_last = out.logits[:, -1:, :]
                    for li in SEL_LAYERS:
                        q_unrot_steps[li].append(saved_q[li].reshape(NH_Q, D).copy())
                    pos += 1
                    del out
                for h in handles:
                    h.remove()
                Ltot = pos  # ctx + 10
                log(f"{doc_name} ctx={ctx}: total cached tokens={Ltot}")
                dt = phase_end(f"prefill+decode {doc_name} ctx={ctx}", t0)
                all_phase_times.append((f"prefill+decode {doc_name}@{ctx}", dt))

                # ---- per-layer analysis, ONE LAYER AT A TIME
                for li in SEL_LAYERS:
                    t0 = phase_begin(f"analyze layer={li} {doc_name}@{ctx}")
                    Krot = past[li][0].to("cpu", torch.float32).numpy()[0]  # (nkv, L, d)
                    Vv = past[li][1]  # keep on device; not needed for this pass
                    del Vv
                    nkv, L, d = Krot.shape
                    assert (nkv, d) == (NH_KV, D) and L == Ltot
                    positions = np.arange(L)
                    # unrotate K (float32 math via float64 rope core)
                    Kun = np.stack(
                        [rope_neox_inv(Krot[g], positions, inv_freq) for g in range(nkv)],
                        axis=0)
                    del Krot
                    gc.collect()
                    Kexp = np.repeat(Kun, GQA, axis=0)  # (14, L, d) unrotated per q-head
                    Qu = np.stack(q_unrot_steps[li], axis=0)  # (10, 14, d)
                    nblocks = (L + BLOCK - 1) // BLOCK
                    slow_idx = np.arange(D - SLOW_DIMS, D)
                    fast_idx = np.arange(0, D - SLOW_DIMS)
                    inv_sqrt_d = 1.0 / np.sqrt(D)
                    # K rotation is step-independent: rotate once per layer
                    Kr = rope_neox(Kexp, positions[None, None, :], inv_freq)  # (14, L, d)
                    Kr64 = Kr.astype(np.float64)
                    del Kr
                    for j in range(NSTEP):
                        t = ctx + j
                        Qr = rope_neox(Qu[j], t, inv_freq)  # (14, d) rotated queries
                        S = np.einsum("hd,hld->hl", Qr.astype(np.float64),
                                      Kr64) * inv_sqrt_d  # (14, L)
                        # softmax (float64 for stability, values realistic in fp32 range)
                        Sm = S.max(axis=1, keepdims=True)
                        P = np.exp(S - Sm)
                        P /= P.sum(axis=1, keepdims=True)
                        # ---- Task 1: blocks for 99% mass, per q-head -> per kv-head
                        frac99_q = np.zeros(NH_Q)
                        for h in range(NH_Q):
                            masses = np.array(
                                [P[h, b * BLOCK:(b + 1) * BLOCK].sum()
                                 for b in range(nblocks)])
                            order = np.argsort(-masses)
                            cum = np.cumsum(masses[order])
                            k = int(np.searchsorted(cum, 0.99) + 1)
                            frac99_q[h] = k / nblocks
                        for g in range(NH_KV):
                            grp = frac99_q[g * GQA:(g + 1) * GQA]
                            acc[li]["ceil99"].append(float(grp.mean()))
                            ceil_detail.append(
                                (doc_name, ctx, li, g, float(grp.mean()),
                                 float(grp.min()), float(grp.max())))
                        # ---- Task 2: bound gaps + pruning
                        m_t = S[:, max(0, L - 128):].max(axis=1)  # true local-window max
                        A = acc[li]
                        for h in range(NH_Q):
                            qh = Qr[h]
                            for b in range(nblocks):
                                s0, s1 = b * BLOCK, min((b + 1) * BLOCK, L)
                                blk = Kun[h // GQA, s0:s1]
                                U, st, rt, ct = mzsae_bound_terms(
                                    qh, blk, slow_idx, fast_idx, D)
                                tru = float(S[h, s0:s1].max())
                                gap = U - tru
                                A["n"] += 1
                                A["gapsum"] += gap
                                A["gapsq"] += gap * gap
                                A["gapmin"] = min(A["gapmin"], gap)
                                A["gapmax"] = max(A["gapmax"], gap)
                                A["slow"] += st
                                A["resid"] += rt
                                A["cfast"] += ct
                                if gap < -1e-3:
                                    A["viol"] += 1
                                mt = float(m_t[h])
                                if U < mt - 16.0:
                                    A["prune16"] += 1
                                if U < mt - 8.0:
                                    A["prune8"] += 1
                                if U < mt - 4.0:
                                    A["prune4"] += 1
                                bi = int(np.searchsorted(BIN_EDGES, gap, side="right") - 1)
                                if bi < 0:
                                    A["under"] += 1
                                elif bi >= BIN_C:
                                    A["over"] += 1
                                else:
                                    A["hist"][bi] += 1
                        del S, P, Qr
                        gc.collect()
                    del Kr64, Kun, Kexp, Qu
                    gc.collect()
                    dt = phase_end(f"analyze layer={li} {doc_name}@{ctx}", t0)
                    all_phase_times.append((f"analyze L{li} {doc_name}@{ctx}", dt))
                del past
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()

    # ---------------- report
    log("=" * 78)
    log("RESULTS: Task 1 -- blocks needed for 99% attention mass (fraction of 32-tok blocks)")
    log("=" * 78)
    all99 = []
    for (doc_name, ctx) in [("docA", 8192), ("docA", 16384), ("docB", 8192), ("docB", 16384)]:
        for li in SEL_LAYERS:
            for g in range(NH_KV):
                vals = [c[4] for c in ceil_detail
                        if c[0] == doc_name and c[1] == ctx and c[2] == li and c[3] == g]
                m = float(np.mean(vals))
                all99.extend(vals)
                log(f"ceil99 {doc_name} ctx={ctx:5d} layer={li:2d} kv={g}: "
                    f"mean={m:.4f} (min={min(vals):.4f} max={max(vals):.4f} over {len(vals)} steps)")
    a99 = np.array(all99)
    log(f"CEIL99 OVERALL: n={a99.size} mean={a99.mean():.4f} p50={np.median(a99):.4f} "
        f"p90={np.quantile(a99, .9):.4f} max={a99.max():.4f}")

    log("=" * 78)
    log("RESULTS: Task 2 -- gap = U_b/sqrt(d) - true_block_max, per layer (all docs/ctx/steps/heads/blocks)")
    log("=" * 78)
    for li in SEL_LAYERS:
        A = acc[li]
        n = A["n"]
        mean = A["gapsum"] / n
        std = np.sqrt(A["gapsq"] / n - mean ** 2)
        log(f"layer {li}: n={n} gap mean={mean:+.3f} std={std:.3f} min={A['gapmin']:+.3f} "
            f"max={A['gapmax']:+.3f} violations(gap<-1e-3)={A['viol']} ({A['viol']/n*100:.2f}%)")
        log(f"  term means: slow={A['slow']/n:+.3f} resid={A['resid']/n:+.3f} "
            f"Cfast(unscaled)={A['cfast']/n:+.3f}")
        log(f"  prunable: tau=16: {A['prune16']/n*100:.2f}% | tau=8: {A['prune8']/n*100:.2f}% "
            f"| tau=4: {A['prune4']/n*100:.2f}%")
        # text histogram
        h = A["hist"]
        peak = h.max()
        log(f"  gap histogram (bin width 1.0, under={A['under']} over={A['over']}):")
        lo, hi = 0, BIN_C - 1
        while lo < hi and h[lo] == 0:
            lo += 1
        while hi > lo and h[hi] == 0:
            hi -= 1
        lo = max(0, lo - 1)
        hi = min(BIN_C - 1, hi + 1)
        for bi in range(lo, hi + 1):
            bar = "#" * int(50 * h[bi] / peak) if peak else ""
            log(f"    [{BIN_EDGES[bi]:+6.1f},{BIN_EDGES[bi+1]:+6.1f}) {h[bi]:8d} {bar}")

    log("=" * 78)
    log("TIMING + MEMORY SUMMARY")
    log("=" * 78)
    for name, dt in all_phase_times:
        rss, _, _ = mem_snapshot()
        log(f"  {name:38s} {dt:7.1f}s")
    rss_gb, vm_pct, press = mem_snapshot()
    log(f"final rss={rss_gb:.2f}GB vm_used={vm_pct:.1f}% pressure={press} "
        f"total_wall={(time.time()-t_run0)/60:.1f}min")

    # ---------------- verdict
    log("=" * 78)
    log("VERDICT")
    log("=" * 78)
    mean99 = float(a99.mean())
    p8 = {li: acc[li]["prune8"] / acc[li]["n"] for li in SEL_LAYERS}
    vrate = {li: acc[li]["viol"] / acc[li]["n"] for li in SEL_LAYERS}
    log(f"ceiling: 99%-mass needs mean {mean99*100:.1f}% of blocks "
        f"(p90 {np.quantile(a99,.9)*100:.1f}%, max {a99.max()*100:.1f}%)")
    for li in SEL_LAYERS:
        log(f"layer {li}: bound-prune@tau=8 = {p8[li]*100:.2f}%, "
            f"violation rate = {vrate[li]*100:.2f}%")
    viable = (mean99 < 0.50) and all(p > 0.50 for p in p8.values()) \
        and all(v < 0.01 for v in vrate.values())
    log(f"viability bar: ceiling<50% blocks for 99% mass AND bound prunes>50%@tau=8 "
        f"AND violations<1%: {'PASS' if viable else 'FAIL'}")
    log(f"BLOCK PRUNING VIABLE ON THIS DATA: {'YES' if viable else 'NO'}")
    log("=" * 78)


if __name__ == "__main__":
    main()
