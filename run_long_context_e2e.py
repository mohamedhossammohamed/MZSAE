#!/usr/bin/env python3
"""
run_long_context_e2e.py
FIX 3: Real Long-Context (16,384 tokens) End-to-End Generation Test with Qwen2.5-1.5B.
- Primes cache with 16,384 tokens from work_docs/docA.txt
- Active MZSAE compression during prefill
- Generates 40 tokens autoregressively
- Compares token-for-token against uncompressed dense FP16 cache at 16k tokens
- Logs combined memory every 2s under 10 GB hard watchdog
- Reports token match count, first divergence position, tok/s, and combined peak memory
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import gc
import json
import threading
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from mzsae.hf_patch import MZSAECache
from fastattn_memfix.gpulock import GPULock

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(REPO_DIR, "fastattn_verify4", "model_gguf")
GGUF_FILE = "qwen2.5-1.5b.gguf"
DOC_FILE = os.path.join(REPO_DIR, "work_docs", "docA.txt")
TARGET_CONTEXT = 16384
GEN_TOKENS = 40
CHUNK_SIZE = 256

class CombinedMemoryWatchdog:
    def __init__(self, interval_s: float = 2.0, max_combined_gb: float = 10.0):
        self.interval_s = interval_s
        self.max_combined_bytes = max_combined_gb * (1024**3)
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self.peak_mps_curr = 0
        self.peak_mps_driver = 0
        self.peak_combined = 0
        self.stop_event = threading.Event()
        self.thread = None
        self.last_op = "init"

    def set_op(self, op: str):
        self.last_op = op

    def get_stats(self):
        rss = self.process.memory_info().rss
        mps_curr = torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0
        mps_driver = torch.mps.driver_allocated_memory() if hasattr(torch.mps, "driver_allocated_memory") else 0
        combined = rss + mps_curr + mps_driver
        return rss, mps_curr, mps_driver, combined

    def _loop(self):
        while not self.stop_event.is_set():
            rss, mps_curr, mps_driver, combined = self.get_stats()
            if rss > self.peak_rss: self.peak_rss = rss
            if mps_curr > self.peak_mps_curr: self.peak_mps_curr = mps_curr
            if mps_driver > self.peak_mps_driver: self.peak_mps_driver = mps_driver
            if combined > self.peak_combined: self.peak_combined = combined

            print(f"[watchdog 2s] RSS: {rss/(1024**3):.2f} GB | MPS Alloc: {mps_curr/(1024**3):.2f} GB | MPS Driver: {mps_driver/(1024**3):.2f} GB | Combined: {combined/(1024**3):.2f} GB", flush=True)

            if combined > self.max_combined_bytes:
                msg = (
                    f"\n[FATAL COMBINED MEMORY WATCHDOG] Combined footprint {combined/(1024**3):.2f} GB "
                    f"exceeded hard cap {self.max_combined_bytes/(1024**3):.1f} GB! "
                    f"Last operation: {self.last_op}\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                os._exit(1)

            self.stop_event.wait(self.interval_s)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        rss, mps_curr, mps_driver, combined = self.get_stats()
        if rss > self.peak_rss: self.peak_rss = rss
        if mps_curr > self.peak_mps_curr: self.peak_mps_curr = mps_curr
        if mps_driver > self.peak_mps_driver: self.peak_mps_driver = mps_driver
        if combined > self.peak_combined: self.peak_combined = combined

def main():
    print("=" * 75)
    print(f"FIX 3: REAL LONG-CONTEXT ({TARGET_CONTEXT} TOKENS) GENERATION TEST")
    print("=" * 75)

    watchdog = CombinedMemoryWatchdog(interval_s=2.0, max_combined_gb=10.0)
    watchdog.start()

    watchdog.set_op("load_tokenizer")
    print("[16k_test] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF_FILE)

    watchdog.set_op("prepare_document_tokens")
    print(f"[16k_test] Reading {DOC_FILE} and tokenizing to {TARGET_CONTEXT} tokens...")
    with open(DOC_FILE, "r", encoding="utf-8") as f:
        doc_text = f.read()

    tokens = tokenizer.encode(doc_text, return_tensors="pt")[0]
    while len(tokens) < TARGET_CONTEXT:
        doc_text += "\n" + doc_text
        tokens = tokenizer.encode(doc_text, return_tensors="pt")[0]

    tokens_16k = tokens[:TARGET_CONTEXT].unsqueeze(0).to("mps")
    print(f"[16k_test] Successfully prepared prompt of shape {tokens_16k.shape} ({tokens_16k.numel()} tokens)")

    watchdog.set_op("load_model_float16")
    print("[16k_test] Loading model in float16 with low_cpu_mem_usage=True...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        gguf_file=GGUF_FILE,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("mps").eval()

    # -------------------------------------------------------------------------
    # PART 1: UNCOMPRESSED DENSE FP16 BASELINE
    # -------------------------------------------------------------------------
    watchdog.set_op("prefill_dense_fp16")
    print(f"\n--- PART 1: Priming Uncompressed Dense FP16 Cache ({TARGET_CONTEXT} tokens in {CHUNK_SIZE}-token chunks) ---")
    dense_cache = DynamicCache()
    t0_prefill_dense = time.perf_counter()

    for s in range(0, TARGET_CONTEXT, CHUNK_SIZE):
        chunk = tokens_16k[:, s:s+CHUNK_SIZE]
        with torch.inference_mode():
            model(chunk, past_key_values=dense_cache, use_cache=True)
        torch.mps.empty_cache()
        if (s + CHUNK_SIZE) % 2048 == 0 or (s + CHUNK_SIZE) >= TARGET_CONTEXT:
            print(f"  * Dense prefilled {min(s + CHUNK_SIZE, TARGET_CONTEXT)}/{TARGET_CONTEXT} tokens...", flush=True)

    torch.mps.synchronize()
    prefill_dense_sec = time.perf_counter() - t0_prefill_dense
    print(f"[16k_test] Dense prefill complete in {prefill_dense_sec:.2f}s ({TARGET_CONTEXT/prefill_dense_sec:.1f} tok/s)")

    watchdog.set_op("generate_dense_fp16")
    print(f"[16k_test] Generating {GEN_TOKENS} tokens autoregressively from Dense FP16 cache...")
    dense_generated_ids = []
    curr_token = tokens_16k[:, -1:]
    t0_gen_dense = time.perf_counter()

    for step in range(GEN_TOKENS):
        with torch.inference_mode():
            out = model(curr_token, past_key_values=dense_cache, use_cache=True)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            dense_generated_ids.append(next_id.item())
            curr_token = next_id

    torch.mps.synchronize()
    gen_dense_sec = time.perf_counter() - t0_gen_dense
    dense_tok_s = GEN_TOKENS / gen_dense_sec
    print(f"[16k_test] Dense generation complete in {gen_dense_sec:.2f}s ({dense_tok_s:.1f} tok/s)")

    dense_text = tokenizer.decode(dense_generated_ids, skip_special_tokens=True)
    print(f"Dense FP16 Output: \"{dense_text}\"")

    # Release dense cache to ensure memory safety before MZSAE run
    watchdog.set_op("release_dense_cache")
    del dense_cache
    gc.collect()
    torch.mps.empty_cache()

    # -------------------------------------------------------------------------
    # PART 2: MZSAE ACTIVE COMPRESSED CACHE
    # -------------------------------------------------------------------------
    watchdog.set_op("prefill_mzsae_active")
    print(f"\n--- PART 2: Priming Active MZSAE Compressed Cache ({TARGET_CONTEXT} tokens in {CHUNK_SIZE}-token chunks) ---")
    mzsae_cache = MZSAECache(num_sinks=4, recent_win=64)
    t0_prefill_mzsae = time.perf_counter()

    for s in range(0, TARGET_CONTEXT, CHUNK_SIZE):
        chunk = tokens_16k[:, s:s+CHUNK_SIZE]
        with torch.inference_mode():
            model(chunk, past_key_values=mzsae_cache, use_cache=True)
        # Active MZSAE compression after each prefill chunk
        mzsae_cache.compress_all_layers()
        torch.mps.empty_cache()
        if (s + CHUNK_SIZE) % 2048 == 0 or (s + CHUNK_SIZE) >= TARGET_CONTEXT:
            print(f"  * MZSAE compressed prefill: {min(s + CHUNK_SIZE, TARGET_CONTEXT)}/{TARGET_CONTEXT} tokens...", flush=True)

    torch.mps.synchronize()
    prefill_mzsae_sec = time.perf_counter() - t0_prefill_mzsae
    print(f"[16k_test] MZSAE prefill complete in {prefill_mzsae_sec:.2f}s ({TARGET_CONTEXT/prefill_mzsae_sec:.1f} tok/s)")

    watchdog.set_op("generate_mzsae")
    print(f"[16k_test] Generating {GEN_TOKENS} tokens autoregressively from MZSAE Compressed cache...")
    mzsae_generated_ids = []
    curr_token = tokens_16k[:, -1:]
    t0_gen_mzsae = time.perf_counter()

    for step in range(GEN_TOKENS):
        with torch.inference_mode():
            out = model(curr_token, past_key_values=mzsae_cache, use_cache=True)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            mzsae_generated_ids.append(next_id.item())
            curr_token = next_id

    torch.mps.synchronize()
    gen_mzsae_sec = time.perf_counter() - t0_gen_mzsae
    mzsae_tok_s = GEN_TOKENS / gen_mzsae_sec
    print(f"[16k_test] MZSAE generation complete in {gen_mzsae_sec:.2f}s ({mzsae_tok_s:.1f} tok/s)")

    mzsae_text = tokenizer.decode(mzsae_generated_ids, skip_special_tokens=True)
    print(f"MZSAE Output:     \"{mzsae_text}\"")

    watchdog.stop()

    # -------------------------------------------------------------------------
    # PART 3: TOKEN-FOR-TOKEN COHERENCE & DIVERGENCE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TOKEN-FOR-TOKEN COHERENCE & DIVERGENCE ANALYSIS")
    print("=" * 75)

    matches = 0
    first_divergence_pos = None

    for idx, (d_tok, m_tok) in enumerate(zip(dense_generated_ids, mzsae_generated_ids)):
        if d_tok == m_tok:
            matches += 1
        elif first_divergence_pos is None:
            first_divergence_pos = idx

    match_pct = (matches / GEN_TOKENS) * 100.0

    print(f"  * Total Tokens Generated:     {GEN_TOKENS}")
    print(f"  * Matching Tokens:            {matches}/{GEN_TOKENS} ({match_pct:.1f}%)")
    if first_divergence_pos is not None:
        d_str = tokenizer.decode([dense_generated_ids[first_divergence_pos]])
        m_str = tokenizer.decode([mzsae_generated_ids[first_divergence_pos]])
        print(f"  * First Diverging Token Pos:  Position {first_divergence_pos + 1}")
        print(f"      - Dense FP16 token:       '{d_str}' (id {dense_generated_ids[first_divergence_pos]})")
        print(f"      - MZSAE token:            '{m_str}' (id {mzsae_generated_ids[first_divergence_pos]})")
    else:
        print(f"  * First Diverging Token Pos:  NONE (100% Exact Token Match across all 40 tokens!)")

    print(f"  * Dense FP16 Generation Rate: {dense_tok_s:.1f} tok/s")
    print(f"  * MZSAE Generation Rate:      {mzsae_tok_s:.1f} tok/s")

    peak_rss_gb = watchdog.peak_rss / (1024**3)
    peak_mps_curr_gb = watchdog.peak_mps_curr / (1024**3)
    peak_mps_driver_gb = watchdog.peak_mps_driver / (1024**3)
    peak_comb_gb = watchdog.peak_combined / (1024**3)

    print("\n" + "=" * 75)
    print("MEMORY FOOTPRINT TELEMETRY (PEAK DURING 16k TEST)")
    print("=" * 75)
    print(f"  * Peak psutil RSS:            {peak_rss_gb:.3f} GB")
    print(f"  * Peak torch.mps Current:     {peak_mps_curr_gb:.3f} GB")
    print(f"  * Peak torch.mps Driver:      {peak_mps_driver_gb:.3f} GB")
    print(f"  * PEAK COMBINED MEMORY:       {peak_comb_gb:.3f} GB")
    print(f"  * Safety Watchdog Ceiling:    10.000 GB")
    print("=" * 75)

    results_summary = {
        "context_tokens": TARGET_CONTEXT,
        "gen_tokens": GEN_TOKENS,
        "matches": matches,
        "match_pct": round(match_pct, 2),
        "first_divergence_pos": first_divergence_pos + 1 if first_divergence_pos is not None else None,
        "dense_tok_s": round(dense_tok_s, 2),
        "mzsae_tok_s": round(mzsae_tok_s, 2),
        "peak_rss_gb": round(peak_rss_gb, 3),
        "peak_mps_curr_gb": round(peak_mps_curr_gb, 3),
        "peak_mps_driver_gb": round(peak_mps_driver_gb, 3),
        "peak_comb_gb": round(peak_comb_gb, 3),
        "dense_text": dense_text,
        "mzsae_text": mzsae_text
    }

    out_json = os.path.join(REPO_DIR, "logs", "long_context_16k_result.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[16k_test] Saved detailed telemetry results to {out_json}")

if __name__ == "__main__":
    with GPULock(tag="run_16k_e2e"):
        main()
