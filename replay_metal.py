"""STEP 4.5: latency/error replay on saved Q4 tensors (docA 8k, layer 0, last decode step).
- torch MPS SDPA dense vs MLX SDPA dense: 50 warmup + 300 timed runs
- MZSAE-logic replay (bound approved fraction @tau + sparse-output error, CPU, d=64 exact)
- repo Metal dense kernel best-effort LATENCY ONLY (kernel hardcodes d=128; inputs zero-padded,
  inv_sqrt_d set for 64 -- approximation, clearly labeled). Skip with note if backend missing.
Usage: .venv/bin/python replay_metal.py 2>&1 | tee logs/mzsae_replay.log
"""
import time

import numpy as np

D = 64


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bench(fn, warm=50, runs=300):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    a = np.array(ts)
    return float(a.mean()), float(np.median(a)), float(np.quantile(a, 0.9))


def main():
    import torch
    log("load tensors docA 8k L0")
    K = np.load("captures/q4_docA_8192/L00_K.npy").astype(np.float32)  # post-rope (8192,2,64)
    V = np.load("captures/q4_docA_8192/L00_V.npy").astype(np.float32)
    Q = np.load("captures/q4_docA_8192/L00_Q.npy").astype(np.float32)  # (20,14,64) post-rope
    L = K.shape[0]
    q = Q[-1]  # (14,64) last decode query
    log(f"L={L}")

    # ---- (a) torch MPS SDPA dense
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    qh = torch.from_numpy(q).unsqueeze(0).unsqueeze(2).to(dev)          # (1,14,1,64)
    kh = torch.from_numpy(K).transpose(0, 1).unsqueeze(0).to(dev)       # (1,2,L,64)
    vh = torch.from_numpy(V).transpose(0, 1).unsqueeze(0).to(dev)
    if dev == "mps":
        torch.mps.synchronize()
    # build GQA-expanded once
    kh14 = kh.repeat_interleave(7, dim=1)
    vh14 = vh.repeat_interleave(7, dim=1)
    fn_torch = lambda: torch.nn.functional.scaled_dot_product_attention(qh, kh14, vh14)
    mt, md, p9 = bench(fn_torch)
    log(f"torch {dev} SDPA dense: mean={mt:.1f}us median={md:.1f}us p90={p9:.1f}us")
    with torch.no_grad():
        ref = fn_torch().cpu().numpy()  # (1,14,1,64) dense reference

    # ---- (b) MLX SDPA dense
    try:
        import mlx.core as mx
        qm = mx.array(q[None, :, None, :])                               # (1,14,1,64)
        km = mx.array(K.transpose(1, 0, 2)[None])                         # (1,2,L,64)
        vm = mx.array(V.transpose(1, 0, 2)[None])
        km14 = mx.repeat(km, 7, axis=1)
        vm14 = mx.repeat(vm, 7, axis=1)
        import mlx.core.fast as mfast

        def fn_mlx():
            o = mfast.scaled_dot_product_attention(qm, km14, vm14, scale=1.0 / 8.0)
            mx.eval(o)
            return o
        mt, md, p9 = bench(fn_mlx)
        log(f"MLX SDPA dense: mean={mt:.1f}us median={md:.1f}us p90={p9:.1f}us")
    except Exception as e:
        log(f"MLX skipped: {type(e).__name__}: {str(e)[:120]}")

    # ---- (c) MZSAE-logic replay (exact d=64, CPU): approved frac + sparse error @tau
    Ke = np.concatenate([K[:, 0:1, :].repeat(7, 1), K[:, 1:2, :].repeat(7, 1)], 1)  # (L,14,64) post
    Ku = None
    TH = 1e6
    inv = (TH ** (-2.0 * np.arange(32) / 64)).astype(np.float64)
    pos = np.arange(L)

    def unrope(x, p):
        a = np.asarray(p, np.float64).reshape(-1, 1)[..., None] * inv[None, :]
        c, s = np.cos(a), np.sin(a)
        x1, x2 = x[..., :32], x[..., 32:]
        return np.concatenate([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

    Ku = unrope(Ke.astype(np.float64), pos).astype(np.float32)  # unrotated for sentinel
    Ve = np.concatenate([V[:, 0:1, :].repeat(7, 1), V[:, 1:2, :].repeat(7, 1)], 1).astype(np.float64)
    nB = (L + 31) // 32
    slow = np.arange(48, 64)
    fast = np.arange(0, 48)
    isd = 1.0 / np.sqrt(64)
    cents = np.stack([Ku[b * 32:min((b + 1) * 32, L)].mean(0) for b in range(nB)])
    R = np.stack([np.linalg.norm(Ku[b * 32:min((b + 1) * 32, L)][:, :, slow] - cents[b][:, slow][None],
                                 axis=2).max(0) for b in range(nB)])
    Cf = np.stack([np.linalg.norm(Ku[b * 32:min((b + 1) * 32, L)][:, :, fast], axis=2).max(0)
                   for b in range(nB)])
    S = np.einsum("hd,lhd->hl", q.astype(np.float64), Ke.astype(np.float64)) * isd
    mt_local = S[:, max(0, L - 128):].max(1)
    qs = q[:, slow]
    U = (np.einsum("hd,bhd->hb", qs, cents[:, :, slow]) + np.linalg.norm(qs, 1, keepdims=True) * R.T
         + Cf.T) * isd
    Sm = S.max(1, keepdims=True)
    P = np.exp(S - Sm)
    P /= P.sum(1, keepdims=True)
    Od = np.einsum("hl,lhd->hd", P, Ve)
    for tau in (16, 8):
        appr = U >= (mt_local[:, None] - tau)
        frac = appr.mean()
        # sparse output using approved blocks only
        Sx = np.where(np.repeat(appr, 32, axis=1)[:, :L], S, -1e30)
        Px = np.exp(Sx - Sx.max(1, keepdims=True))
        Px /= Px.sum(1, keepdims=True)
        Ox = np.einsum("hl,lhd->hd", Px, Ve)
        df = np.abs(Ox - Od)
        co = float(((Ox * Od).sum(1) / (np.linalg.norm(Ox, 2, 1) * np.linalg.norm(Od, 2, 1) + 1e-12)).mean())
        log(f"MZSAE-logic tau={tau}: approved={frac*100:.2f}% out_maxerr={df.max():.3e} cos={co:.6f}")

    # ---- (d) repo Metal dense kernel, latency only, padded d=64->128
    try:
        import sys
        sys.path.insert(0, "src")
        from mzsae.metal_backend import MetalBackend
        mb = MetalBackend()
        log(f"Metal backend: {mb.device_name}")
        qp = np.zeros(128, np.float32)
        qp[:64] = q[0]
        Kp = np.zeros((L, 128), np.float32)
        Kp[:, :64] = (K[:, 0, :]).astype(np.float32)
        Vp = np.zeros((L, 128), np.float32)
        Vp[:, :64] = (V[:, 0, :]).astype(np.float32)
        fn_metal = lambda: mb.flash_attn_dense(qp, Kp, Vp)
        mt, md, p9 = bench(fn_metal, warm=50, runs=300)
        log(f"Metal dense kernel (padded, APPROX): mean={mt:.1f}us median={md:.1f}us p90={p9:.1f}us")
    except Exception as e:
        log(f"Metal kernel skipped: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
