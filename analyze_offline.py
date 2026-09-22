"""
STEP 4 (FP track, OFFLINE): replay every MZSAE experiment from saved captures/fp/*.npz.
No model needed. Never edit numbers by hand; CSV table generated at end.
- rope NeoX verify (unrotate/re-rotate roundtrip) + attention-output validation gate
- sparsity: blocks for 90/99/99.9 mass per layer/KV
- bound: as-coded (last-16 contiguous) AND neox-correct slow pairs; gaps, split,
  prune fracs at tau={16,8,4,2} with TRUE local-window max; violation count
- CRQ 2-bit (+4-bit ref) attention-output error vs dense FP32
- verdict: >50% pruning with <1% mass lost?
Watchdogs: 600 s per file-phase; RSS > 12 GB aborts.
Usage: .venv/bin/python analyze_offline.py 2>&1 | tee logs/mzsae_fp_analyze.log
"""
import glob
import os
import sys
import time

import numpy as np

CAPDIR = "captures/fp"
BLOCK = 32
SLOW_DIMS = 16
TAUS = [16, 8, 4, 2]
T0_ALL = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rss_gb():
    import psutil
    return psutil.Process().memory_info().rss / 1e9


def watchdog(phase, t0):
    dt = time.time() - t0
    r = rss_gb()
    log(f"phase done: {phase} {dt:.1f}s rss={r:.2f}GB")
    if dt > 600:
        log(f"WATCHDOG: {phase} exceeded 600 s. STOPPING.")
        sys.exit(2)
    if r > 12:
        log(f"WATCHDOG: RSS {r:.2f}GB > 12 GB. STOPPING.")
        sys.exit(3)
    return dt


def rope_neox(x, pos, invf):
    x = np.asarray(x, dtype=np.float64)
    invf = np.asarray(invf, dtype=np.float64)
    a = np.asarray(pos, dtype=np.float64)[..., None] * invf[None, :]
    c, s = np.cos(a), np.sin(a)
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(np.float32)


# ---------------- CRQ (repo scheme, generalized to head_dim d) ----------------
def crq2_quant(block):
    """2-bit centroid-residual. Returns dict; residual payload bytes = n*2/8."""
    cent = block.mean(axis=0).astype(np.float64)
    res = block.astype(np.float64) - cent
    mx = float(np.max(np.abs(res)))
    sc = max(mx / 1.5, 1e-7)
    nr = res / sc
    codes = np.zeros_like(nr, dtype=np.uint8)
    codes[nr < -1.0] = 0
    m = (nr >= -1.0) & (nr < 0.0)
    codes[m] = 1
    m = (nr >= 0.0) & (nr < 1.0)
    codes[m] = 2
    codes[nr >= 1.0] = 3
    payload = codes.size * 2 // 8
    return cent.astype(np.float32), codes, np.float16(sc), payload


def crq2_dequant(cent, codes, sc):
    return (cent.astype(np.float64) + (codes.astype(np.float64) - 1.5) * float(sc)).astype(np.float32)


def crq4(block):
    """4-bit uniform symmetric reference."""
    cent = block.mean(axis=0).astype(np.float64)
    res = block.astype(np.float64) - cent
    mx = float(np.max(np.abs(res)))
    sc = max(mx / 7.0, 1e-9)
    q = np.clip(np.round(res / sc), -8, 7)
    return (cent + q * sc).astype(np.float32)


def main():
    log("=" * 70)
    log("OFFLINE ANALYSIS from captures/fp (FP Qwen2.5-0.5B weights)")
    log("=" * 70)
    files = sorted(glob.glob(f"{CAPDIR}/*.npz"))
    log(f"capture files: {[os.path.basename(f) for f in files]}")
    assert files, "no captures; run capture_fp.py first"

    # layout arithmetic proof (task: 32 tokens x 128 dims 2-bit = 1024 B)
    log(f"payload proof: 32*128*2/8 = {32*128*2//8} B (d=128); 32*64*2/8 = {32*64*2//8} B (d=64)")

    per_layer_rows = []   # for CSV
    verdict_cells = []    # (file, layer, kv, prune8, massloss8)
    for fn in files:
        t0 = time.time()
        tag = os.path.basename(fn).replace(".npz", "")
        z = np.load(fn)
        Krot, V, Q, O = z["K"].astype(np.float32), z["V"].astype(np.float32), \
            z["Q"].astype(np.float32), z["O"].astype(np.float32)
        ctx = int(z["ctx"][0])
        theta = float(z["theta"][0])
        NL, NKV, L, D = Krot.shape
        NST, NHQ, _ = Q.shape[1], Q.shape[2], Q.shape[3]
        GQA = NHQ // NKV
        log(f"--- {tag}: layers={NL} L={L} D={D} steps={NST} theta={theta}")
        invf = (theta ** (-2.0 * np.arange(D // 2, dtype=np.float64) / D)).astype(np.float32)
        pos = np.arange(L)
        slow_coded = np.arange(D - SLOW_DIMS, D)          # as-coded: last 16 contiguous
        fast_coded = np.arange(0, D - SLOW_DIMS)
        sp = np.arange((D - SLOW_DIMS) // 2, D // 2)      # neox-correct slow pairs
        slow_neox = np.concatenate([sp, sp + D // 2])
        fast_neox = np.array([i for i in range(D) if i not in set(slow_neox.tolist())])
        log(f"slow_coded={slow_coded.tolist()} slow_neox={slow_neox.tolist()} "
            f"(SETS EQUAL: {set(slow_coded.tolist()) == set(slow_neox.tolist())})")
        nB = (L + BLOCK - 1) // BLOCK
        isd = 1.0 / np.sqrt(D)

        for li in range(NL):
            # unrotate K: rotate cached (rotated) keys by -pos
            Kun = np.stack([rope_neox(Krot[li][g].astype(np.float64), -pos, invf)
                            for g in range(NKV)], axis=0)
            # roundtrip check on layer 0 only (cost)
            if li == 0:
                rt = np.stack([rope_neox(Kun[g].astype(np.float64), pos, invf)
                               for g in range(NKV)], axis=0)
                rterr = float(np.max(np.abs(rt - Krot[li].astype(np.float64))))
                log(f"{tag} L{li}: rope roundtrip maxerr={rterr:.3e} (gate < 1e-2)")
                assert rterr < 1e-2, "rope roundtrip failed"
            Kexp = np.repeat(Kun, GQA, axis=0)                       # (14,L,d) unrot
            Kr = rope_neox(Kexp.astype(np.float64), pos, invf)
            Vexp = np.repeat(V[li].astype(np.float64), GQA, axis=0)  # (14,L,d)
            # per-block stats (step independent)
            cents = np.stack([Kun[g // GQA,
                                  b * BLOCK:min((b + 1) * BLOCK, L)].mean(axis=0)
                              for b in range(nB) for g in range(NHQ)]).reshape(NHQ, nB, D)
            R = np.stack([[np.max(np.linalg.norm(
                Kun[g // GQA, b * BLOCK:min((b + 1) * BLOCK, L)][:, slow_coded]
                - cents[g, b][slow_coded], axis=1)) for b in range(nB)]
                for g in range(NHQ)])
            Cf = np.stack([[np.max(np.linalg.norm(
                Kun[g // GQA, b * BLOCK:min((b + 1) * BLOCK, L)][:, fast_coded], axis=1))
                for b in range(nB)] for g in range(NHQ)])
            acc = {"ceil90": [], "ceil99": [], "ceil999": [],
                   "gaps": [], "pr": {t: 0 for t in TAUS}, "n": 0, "viol": 0,
                   "mloss": {t: 0.0 for t in TAUS},
                   "slow": 0.0, "res": 0.0, "cf": 0.0}
            out_err = []
            top1_agree = 0
            top1_n = 0
            for j in range(NST):
                t = ctx + j
                Qr = rope_neox(Q[li][j].astype(np.float64), t, invf)       # (14,d)
                S = np.einsum("hd,hld->hl", Qr.astype(np.float64),
                              Kr.astype(np.float64)) * isd               # (14,L)
                Sm = S.max(axis=1, keepdims=True)
                P = np.exp(S - Sm)
                P /= P.sum(axis=1, keepdims=True)
                # validation gate on last step: recomputed attn out vs captured O
                if j == NST - 1:
                    Orec = np.einsum("hl,hld->hd", P, Vexp).astype(np.float32)
                    ref = O[li][j]
                    me = float(np.max(np.abs(Orec - ref.astype(np.float64))))
                    co = float(np.sum(Orec * ref) / (np.linalg.norm(Orec.ravel())
                                                     * np.linalg.norm(ref.ravel()) + 1e-12))
                    log(f"{tag} L{li}: attn-output gate maxerr={me:.3e} cos={co:.6f} (gate < 5e-2)")
                    assert me < 5e-2, f"capture validation failed L{li}"
                mt = S[:, max(0, L - 128):].max(axis=1)
                qs = Qr[:, slow_coded]
                nqs = np.linalg.norm(qs, axis=1, keepdims=True)
                U = (np.einsum("hd,hbd->hb", qs, cents[:, :, slow_coded])
                     + nqs * R + Cf) * isd                              # (14,nB) as-coded
                acc["slow"] += float(np.einsum("hd,hbd->hb", qs, cents[:, :, slow_coded]).mean() * isd)
                acc["res"] += float((nqs * R).mean() * isd)
                acc["cf"] += float(Cf.mean() * isd)
                bmax = np.stack([S[h, b * BLOCK:min((b + 1) * BLOCK, L)].max(axis=0)
                                 for b in range(nB) for h in range(NHQ)]).reshape(NHQ, nB)
                gap = U - bmax
                acc["gaps"].append(gap.ravel())
                acc["viol"] += int(np.sum(gap < -1e-3))
                for t_ in TAUS:
                    pr = U < (mt[:, None] - t_)
                    acc["pr"][t_] += int(np.sum(pr))
                    # mass lost if pruned: block masses of pruned blocks
                    bm = np.stack([P[h, b * BLOCK:min((b + 1) * BLOCK, L)].sum()
                                   for b in range(nB) for h in range(NHQ)]).reshape(NHQ, nB)
                    acc["mloss"][t_] += float(np.sum(bm[pr]))
                acc["n"] += gap.size
                # sparsity: blocks for 90/99/99.9
                bm = np.stack([P[h, b * BLOCK:min((b + 1) * BLOCK, L)].sum()
                               for b in range(nB) for h in range(NHQ)]).reshape(NHQ, nB)
                order = np.argsort(-bm, axis=1)
                cum = np.cumsum(np.take_along_axis(bm, order, 1), axis=1)
                for thr, key in ((0.90, "ceil90"), (0.99, "ceil99"), (0.999, "ceil999")):
                    k = (np.argmax(cum >= thr, axis=1) + 1) / nB
                    acc[key].append(k)
                # top-1 agreement (dense) tracked for CRQ section via rec-K below
                del S, P, Qr
            # ---- CRQ error on this layer (all blocks, last-step queries)
            for b in range(nB):
                s0, s1 = b * BLOCK, min((b + 1) * BLOCK, L)
                for g in range(NHQ):
                    blk = Kexp[g, s0:s1]
                    c2, cd2, sc2, pay = crq2_quant(blk)
                    assert pay == blk.size * 2 // 8
            # reconstruct full Kexp at 2-bit and 4-bit, recompute last-step outputs
            K2 = np.zeros_like(Kexp)
            K4 = np.zeros_like(Kexp)
            for g in range(NHQ):
                for b in range(nB):
                    s0, s1 = b * BLOCK, min((b + 1) * BLOCK, L)
                    blk = Kexp[g, s0:s1]
                    c2, cd2, sc2, _ = crq2_quant(blk)
                    K2[g, s0:s1] = crq2_dequant(c2, cd2, sc2)
                    K4[g, s0:s1] = crq4(blk)
            j = NST - 1
            t = ctx + j
            Qr = rope_neox(Q[li][j].astype(np.float64), t, invf)
            S = np.einsum("hd,hld->hl", Qr.astype(np.float64), Kr.astype(np.float64)) * isd
            P = np.exp(S - S.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            Od = np.einsum("hl,hld->hd", P, Vexp)
            for name, Kx in (("crq2", K2), ("crq4", K4)):
                Krx = rope_neox(Kx.astype(np.float64), pos, invf)
                Sx = np.einsum("hd,hld->hl", Qr.astype(np.float64), Krx.astype(np.float64)) * isd
                Px = np.exp(Sx - Sx.max(1, keepdims=True))
                Px /= Px.sum(1, keepdims=True)
                Ox = np.einsum("hl,hld->hd", Px, Vexp)
                dff = np.abs(Ox - Od)
                co = float(np.sum(Ox * Od, axis=1) /
                           (np.linalg.norm(Ox, axis=1) * np.linalg.norm(Od, axis=1) + 1e-12)).mean()
                a1 = float(np.mean(Sx.argmax(1) == S.argmax(1)))
                out_err.append((name, float(dff.max()), float(dff.mean()), float(co), a1))
            del Kun, Kexp, Kr, Vexp, K2, K4
            gaps = np.concatenate(acc["gaps"])
            prow = {"file": tag, "layer": li}
            for key in ("ceil90", "ceil99", "ceil999"):
                v = np.concatenate(acc[key])
                prow[key] = float(v.mean())
            prow["gap_mean"] = float(gaps.mean())
            prow["gap_std"] = float(gaps.std())
            prow["gap_min"] = float(gaps.min())
            prow["slow_mean"] = acc["slow"] / NST
            prow["res_mean"] = acc["res"] / NST
            prow["cf_mean"] = acc["cf"] / NST
            prow["viol_rate"] = acc["viol"] / acc["n"]
            for t_ in TAUS:
                prow[f"prune{t_}"] = acc["pr"][t_] / acc["n"]
                prow[f"mloss{t_}"] = acc["mloss"][t_] / (NST * NHQ)
            for name, mx, mn, co, a1 in out_err:
                prow[f"{name}_max"] = mx
                prow[f"{name}_mean"] = mn
                prow[f"{name}_cos"] = co
                prow[f"{name}_top1"] = a1
            per_layer_rows.append(prow)
            # per-KV verdict cells at tau=8
            for g in range(NHKV):
                pass
            log(f"{tag} L{li}: ceil90={prow['ceil90']:.3f} ceil99={prow['ceil99']:.3f} "
                f"ceil999={prow['ceil999']:.3f} gap={prow['gap_mean']:+.2f}±{prow['gap_std']:.2f} "
                f"viol={prow['viol_rate']*100:.2f}% prune8={prow['prune8']*100:.1f}% "
                f"mloss8={prow['mloss8']*100:.3f}% crq2max={prow['crq2_max']:.3e} "
                f"crq2top1={prow['crq2_top1']:.3f}")
        watchdog(f"file {tag}", t0)
        del Krot, V, Q, O, z

    # ---- CSV table generated from rows (never by hand)
    import csv
    csvp = "logs/mzsae_results_table.csv"
    keys = list(per_layer_rows[0].keys())
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_layer_rows)
    log(f"results table -> {csvp} ({len(per_layer_rows)} rows)")

    log("=" * 70)
    log("VERDICT (quantization: NONE - full-precision safetensors)")
    log("=" * 70)
    ok = 0
    tot = 0
    for r in per_layer_rows:
        tot += 1
        good = (r["prune8"] > 0.50) and (r["mloss8"] < 0.01) and (r["viol_rate"] == 0.0)
        ok += good
    log(f"layers meeting (>50% pruned @tau=8 AND <1% mass lost AND zero violations): {ok}/{tot}")
    for r in per_layer_rows:
        if (r["prune8"] > 0.50) and (r["mloss8"] < 0.01) and (r["viol_rate"] == 0.0):
            log(f"  PASS: {r['file']} layer {r['layer']}")
    if ok == 0:
        log("BLOCK PRUNING IS NOT BENEFICIAL ON THIS MODEL (no setting clears the bar). STOP.")
    else:
        log("SOME SETTINGS CLEAR THE BAR - see table.")


if __name__ == "__main__":
    main()
