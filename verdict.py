"""STEP 5: aggregate logs/rows_*.csv -> logs/mzsae_q4_table.csv + plain verdict.
Weights: Ollama qwen2.5:0.5b GGUF, Q4_K_M (file_type=15). Numbers only from CSVs."""
import csv
import glob
import numpy as np

files = sorted(glob.glob("logs/rows_*.csv"))
rows = []
for f in files:
    with open(f) as fh:
        rows.extend(list(csv.DictReader(fh)))
print(f"rows: {len(rows)} from {len(files)} files")
for r in rows:
    for k in r:
        if k not in ("run",):
            try:
                r[k] = float(r[k])
            except ValueError:
                pass

with open("logs/mzsae_q4_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("table -> logs/mzsae_q4_table.csv")

print("\n==== sparsity ceiling (fraction of 32-tok blocks) ====")
for key in ("c90", "c99", "c999"):
    v = np.array([r[key] for r in rows])
    print(f"{key}: mean={v.mean():.4f} p50={np.median(v):.4f} p90={np.quantile(v, .9):.4f} max={v.max():.4f}")

print("\n==== bound (NeoX slow band) ====")
for key in ("gap_mean", "gap_std", "gap_min", "viol_neox", "viol_coded",
            "prune16", "prune8", "prune4", "prune2",
            "mloss16", "mloss8", "mloss4", "mloss2", "slow", "resid", "cfast"):
    v = np.array([r[key] for r in rows])
    print(f"{key}: mean={v.mean():.5f} max={v.max():.5f}")

print("\n==== CRQ ====")
for key in ("crq2_max", "crq2_mean", "crq2_cos", "crq2_top1",
            "crq4_max", "crq4_mean", "crq4_cos", "crq4_top1"):
    v = np.array([r[key] for r in rows])
    print(f"{key}: mean={v.mean():.5f} worst={v.max() if 'cos' not in key and 'top1' not in key else v.min():.5f}")

print("\n==== per-layer prune8/mloss8 (averaged over runs) ====")
layers = sorted(set(int(r["layer"]) for r in rows))
for li in layers:
    sel = [r for r in rows if int(r["layer"]) == li]
    p = np.mean([r["prune8"] for r in sel])
    m = np.mean([r["mloss8"] for r in sel])
    vn = np.mean([r["viol_neox"] for r in sel])
    c99 = np.mean([r["c99"] for r in sel])
    print(f"L{li:2d}: prune8={p*100:6.2f}% mloss8={m*100:7.4f}% viol={vn*100:.3f}% ceil99={c99*100:6.2f}%")

ok = [r for r in rows if r["prune8"] > 0.50 and r["mloss8"] < 0.01 and r["viol_neox"] == 0.0]
print(f"\nsettings clearing (>50% pruned @tau=8, <1% mass lost, zero violations): {len(ok)}/{len(rows)}")
for r in ok:
    print(f"  PASS: {r['run']} layer {r['layer']}")
print("\nRESULTS FROM QUANTIZED MODEL: Ollama qwen2.5:0.5b GGUF, Q4_K_M (file_type=15).")
if not ok:
    print("VERDICT: BLOCK PRUNING IS NOT BENEFICIAL ON THIS MODEL. STOP.")
else:
    print("VERDICT: SOME SETTINGS CLEAR THE BAR - see table.")
