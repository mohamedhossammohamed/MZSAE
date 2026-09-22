#!/usr/bin/env python3
"""
fastattn_memfix/run_safe_smoke.py
Step 4: Safe Smoke Test on Apple Silicon M4 (16 GB).
Processes exactly 1 document at 8192 tokens with hard memory caps.
Measures peak RSS, vm_stat compressor & swap deltas.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import gc
import json
import time
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastattn_memfix.mem_watchdog import (
    start_memory_watchdog,
    stop_memory_watchdog,
    set_current_operation,
    assert_model_memory_safe,
    assert_context_length_safe,
    cleanup_layer_memory,
    get_peak_rss_bytes,
    get_current_rss_bytes,
)
from fastattn_memfix.gpulock import GPULock
from fastattn_memfix.replay_harness import run_eval_sequence

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_DIR, "fastattn_verify4", "model_gguf")
GGUF_FILE = "qwen2.5-1.5b.gguf"
DOC_A = os.path.join(REPO_DIR, "work_docs", "docA.txt")
OUT_JSON = os.path.join(REPO_DIR, "fastattn_memfix", "logs", "safe_smoke_result.json")

def parse_vm_stat() -> dict:
    """Parses vm_stat output into dictionary of metrics."""
    out = subprocess.check_output(["vm_stat"]).decode("utf-8")
    stats = {}
    for line in out.splitlines():
        if ":" in line:
            parts = line.split(":")
            key = parts[0].strip().strip('"')
            val_str = parts[1].strip().rstrip(".")
            try:
                stats[key] = int(val_str)
            except ValueError:
                pass
    return stats

def main():
    print("=" * 75)
    print("STEP 4: SAFE SMOKE TEST UNDER HARD MEMORY CAPS (16 GB M4)")
    print("=" * 75)

    # 1. Baseline vm_stat
    vm_before = parse_vm_stat()
    swapins_before = vm_before.get("Swapins", 0)
    swapouts_before = vm_before.get("Swapouts", 0)
    comp_pages_before = vm_before.get("Pages stored in compressor", 0)

    print(f"[smoke_test] vm_stat baseline: Swapins={swapins_before}, Swapouts={swapouts_before}, CompressorPages={comp_pages_before}")

    # 2. Start watchdog & acquire GPU lock
    start_memory_watchdog()
    print(f"[smoke_test] Milestone [AT START]: RSS = {get_current_rss_bytes() / (1024**3):.3f} GB")
    set_current_operation("acquire_gpu_lock")

    with GPULock(tag="safe_smoke_8192"):
        set_current_operation("load_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF_FILE)

        set_current_operation("load_model_float16")
        print(f"[smoke_test] Loading model {GGUF_FILE} in float16 with low_cpu_mem_usage=True...")
        t0_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            gguf_file=GGUF_FILE,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to("mps").eval()
        t_load = time.time() - t0_load
        print(f"[smoke_test] Model loaded in {t_load:.2f}s")

        # Check parameter footprint
        param_bytes = assert_model_memory_safe(model)
        rss_after_load_gb = get_current_rss_bytes() / (1024**3)
        print(f"[smoke_test] Milestone [AFTER MODEL LOAD]: RSS = {rss_after_load_gb:.3f} GB")

        # 3. Load exactly 1 document at 8192 tokens
        set_current_operation("load_docA_8192")
        text = open(DOC_A, "r", encoding="utf-8").read()
        tokens = tokenizer.encode(text, return_tensors="pt")[0]
        TARGET_LEN = 8192
        EVAL_STEPS = 256
        assert len(tokens) >= TARGET_LEN, f"Document too short: {len(tokens)} < {TARGET_LEN}"
        doc_tokens = tokens[:TARGET_LEN]

        # 4. Run eval through replay harness (FP16 baseline)
        print(f"\n[smoke_test] Processing document A ({TARGET_LEN} tokens, {EVAL_STEPS} eval steps) mode=f16...")
        t0_eval = time.time()
        res_f16 = run_eval_sequence(model, doc_tokens, mode="f16", eval_steps=EVAL_STEPS)
        t_eval = time.time() - t0_eval
        print(f"[smoke_test] f16 complete in {t_eval:.2f}s: PPL={res_f16['ppl']:.4f}, Peak RSS={res_f16['peak_rss_gb']:.2f} GB")

        # 5. Run eval through replay harness (MZSAE compressed)
        print(f"\n[smoke_test] Processing document A ({TARGET_LEN} tokens, {EVAL_STEPS} eval steps) mode=mzsae_cand_f...")
        t0_eval_c = time.time()
        res_cand_f = run_eval_sequence(model, doc_tokens, mode="mzsae_cand_f", eval_steps=EVAL_STEPS)
        t_eval_c = time.time() - t0_eval_c
        print(f"[smoke_test] mzsae_cand_f complete in {t_eval_c:.2f}s: PPL={res_cand_f['ppl']:.4f}, Peak RSS={res_cand_f['peak_rss_gb']:.2f} GB")

        delta_ppl = res_cand_f['ppl'] - res_f16['ppl']
        delta_pct = delta_ppl / res_f16['ppl'] * 100.0
        print(f"[smoke_test] Delta PPL: {delta_ppl:.4f} ({delta_pct:.2f}%)")

        # Cleanup model
        set_current_operation("cleanup_model")
        del model
        cleanup_layer_memory()

    stop_memory_watchdog()

    # 6. Post-eval vm_stat
    vm_after = parse_vm_stat()
    swapins_after = vm_after.get("Swapins", 0)
    swapouts_after = vm_after.get("Swapouts", 0)
    comp_pages_after = vm_after.get("Pages stored in compressor", 0)

    delta_swapins = swapins_after - swapins_before
    delta_swapouts = swapouts_after - swapouts_before
    delta_comp = comp_pages_after - comp_pages_before

    peak_rss_gb = get_peak_rss_bytes() / (1024**3)
    HARD_CAP_GB = 4.00

    print("\n" + "=" * 75)
    print("SMOKE TEST RESULTS & VERIFICATION")
    print("=" * 75)
    print(f"  * Peak Process RSS:         {peak_rss_gb:.2f} GB (Hard Cap: {HARD_CAP_GB:.2f} GB) -> {'PASS' if peak_rss_gb < HARD_CAP_GB else 'FAIL'}")
    print(f"  * Model Parameter Footprint:{param_bytes / (1024**3):.2f} GB (Limit: 3.50 GB) -> {'PASS' if param_bytes < 3.5 * 1024**3 else 'FAIL'}")
    print(f"  * Swapins Delta:            {delta_swapins} (Target: 0 new swapins)")
    print(f"  * Swapouts Delta:           {delta_swapouts} (Target: 0 new swapouts)")
    print(f"  * Compressor Pages Delta:   {delta_comp} pages")
    print(f"  * Replay Delta PPL:         {delta_ppl:.4f} ({delta_pct:.2f}%)")

    pass_status = (peak_rss_gb < HARD_CAP_GB) and (delta_swapins == 0) and (delta_swapouts == 0)

    report_data = {
        "status": "PASS" if pass_status else "WARNING_OR_FAIL",
        "peak_rss_gb": round(peak_rss_gb, 3),
        "param_footprint_gb": round(param_bytes / (1024**3), 3),
        "f16_ppl": round(res_f16['ppl'], 4),
        "cand_f_ppl": round(res_cand_f['ppl'], 4),
        "delta_ppl": round(delta_ppl, 4),
        "delta_ppl_pct": round(delta_pct, 2),
        "vm_stat_deltas": {
            "swapins": delta_swapins,
            "swapouts": delta_swapouts,
            "compressor_pages": delta_comp
        }
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[smoke_test] Result recorded in {OUT_JSON}")
    if not pass_status and peak_rss_gb >= HARD_CAP_GB:
        sys.exit(1)

if __name__ == "__main__":
    main()
