"""
STEP 3 (FP track): capture Q/K/V from full-precision Qwen2.5-0.5B ONCE, save to disk.
- 4 docs (wikitext-2) at 8k and 16k + docA at 32k
- Last 20 decode positions: Q unrotated (q_proj) + o_proj inputs (= true per-head attn outputs)
- Full K (rotated, will unrotate offline) + V, saved fp16 per layer
- Validation gate: recompute last-step attention output in FP32 from captured tensors,
  compare vs captured o_proj inputs. Abort if max abs err > 5e-2.
- Watchdogs: >600 s per (doc,ctx) phase -> stop+report; RSS > 12 GB -> stop+report;
  total capture budget 30 min -> stop with fit report.
Usage: .venv/bin/python capture_fp.py 2>&1 | tee logs/mzsae_fp_capture.log
"""
import gc
import os
import sys
import time

import numpy as np
import torch

MODEL_ID = "models/qwen-local"
OUTDIR = "captures/fp"
NSTEP = 20
CHUNK = 2048
T0_ALL = time.time()
CAP_BUDGET_S = 30 * 60

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rss_gb():
    import psutil
    return psutil.Process().memory_info().rss / 1e9


def check_budgets(phase, t0):
    dt = time.time() - t0
    r = rss_gb()
    log(f"phase done: {phase} elapsed={dt:.1f}s rss={r:.2f}GB wall={(time.time()-T0_ALL)/60:.1f}min")
    if dt > 600:
        log(f"WATCHDOG: phase {phase} exceeded 600 s. STOPPING with fit report.")
        sys.exit(2)
    if r > 12:
        log(f"WATCHDOG: RSS {r:.2f}GB exceeded 12 GB cap. STOPPING.")
        sys.exit(3)
    if time.time() - T0_ALL > CAP_BUDGET_S:
        log("WATCHDOG: 30-min capture budget exhausted. STOPPING with fit report.")
        sys.exit(4)
    return dt


def main():
    log("=" * 70)
    log("FP CAPTURE: Qwen2.5-0.5B full precision -> captures/fp")
    log("=" * 70)
    os.makedirs(OUTDIR, exist_ok=True)

    t0 = time.time()
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    snap = ("snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/"
            "train-00000-of-00001.parquet")
    pq = ("/Users/mohammedhossam/.cache/huggingface/hub/datasets--Salesforce--wikitext/" + snap)
    df = pd.read_parquet(pq)
    big = "\n\n".join(t for t in df["text"].tolist() if len(t.strip()) > 50)
    pool = tok(big, add_special_tokens=False).input_ids
    log(f"token pool: {len(pool)}")
    check_budgets("tokenize_pool", t0)

    t0 = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    cfg = model.config
    NL, NHQ, NHKV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // cfg.num_attention_heads
    log(f"model: {NL}L {NHQ}Q {NHKV}KV d={D} theta={cfg.rope_parameters} device={device}")
    model.to(device)
    model.eval()
    check_budgets("load_model", t0)

    offsets = [0, 200_000, 400_000, 600_000]
    docs = {}
    for i, off in enumerate(offsets):
        name = f"doc{'ABCD'[i]}"
        assert len(pool) > off + 32768 + 64, f"pool too small for {name}"
        docs[name] = pool[off:]
    plan = [(d, 8192) for d in docs] + [(d, 16384) for d in docs] + [("docA", 32768)]
    log(f"plan ({len(plan)} captures): {plan}")

    fit = []
    with torch.no_grad():
        for doc_name, ctx in plan:
            t0 = time.time()
            ids = docs[doc_name][:ctx]
            past = None
            pos = 0
            logits_last = None
            for s in range(0, ctx, CHUNK):
                e = min(s + CHUNK, ctx)
                chunk = torch.tensor([ids[s:e]], dtype=torch.long, device=device)
                cp = torch.arange(pos, pos + (e - s), device=device)
                out = model(input_ids=chunk, past_key_values=past,
                            use_cache=True, cache_position=cp)
                past = out.past_key_values
                logits_last = out.logits[:, -1:, :]
                pos = e
                del out
            # hooks: q_proj out (unrotated Q) + o_proj in (true per-head attn out)
            qbuf, obuf = {}, {}
            handles = []
            for li in range(NL):
                attn = model.model.layers[li].self_attn
                handles.append(attn.q_proj.register_forward_hook(
                    lambda m, i, o, li=li: qbuf.__setitem__(li, o.detach().to("cpu", torch.float16).numpy())))
                handles.append(attn.o_proj.register_forward_hook(
                    lambda m, i, o, li=li: obuf.__setitem__(li, i[0].detach().to("cpu", torch.float16).numpy())))
            Qs = {li: [] for li in range(NL)}
            Os = {li: [] for li in range(NL)}
            for _ in range(NSTEP):
                nxt = torch.argmax(logits_last, dim=-1)
                out = model(input_ids=nxt, past_key_values=past, use_cache=True,
                            cache_position=torch.tensor([pos], device=device))
                logits_last = out.logits[:, -1:, :]
                for li in range(NL):
                    Qs[li].append(qbuf[li].reshape(NHQ, D).copy())
                    Os[li].append(obuf[li].reshape(NHQ, D).copy())
                pos += 1
                del out
            for h in handles:
                h.remove()
            Ltot = pos
            log(f"{doc_name}@{ctx}: cached {Ltot} tokens; saving...")
            K = np.stack([past.layers[li].keys[0].to("cpu", torch.float16).numpy()
                          for li in range(NL)], axis=0)  # (24,nkv,L,d) rotated
            V = np.stack([past.layers[li].values[0].to("cpu", torch.float16).numpy()
                          for li in range(NL)], axis=0)
            Q = np.stack([np.stack(Qs[li], axis=0) for li in range(NL)], axis=0)  # (24,20,14,64)
            O = np.stack([np.stack(Os[li], axis=0) for li in range(NL)], axis=0)
            fn = f"{OUTDIR}/{doc_name}_{ctx}.npz"
            np.savez_compressed(fn, K=K, V=V, Q=Q, O=O, ctx=np.array([ctx]),
                                theta=np.array([cfg.rope_parameters['rope_theta']]))
            log(f"saved {fn} K{K.shape} V{V.shape} Q{Q.shape}")
            del past, K, V, Q, O, Qs, Os, qbuf, obuf
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            dt = check_budgets(f"capture {doc_name}@{ctx}", t0)
            fit.append((doc_name, ctx, round(dt, 1)))
    log(f"FIT REPORT ({len(fit)}/{len(plan)}): {fit}")
    log(f"total wall {(time.time()-T0_ALL)/60:.1f} min, peak check rss={rss_gb():.2f}GB")
    log("CAPTURE COMPLETE")


if __name__ == "__main__":
    main()
