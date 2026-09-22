#!/usr/bin/env python3
"""
run_generation_e2e.py
Real end-to-end text generation with Qwen2.5-1.5B using the installed mzsae package.
FIX 1: Logs combined memory every 2s (RSS + MPS current + MPS driver) and reports peak.
Enforces hard 10 GB combined memory cap with last operation logging.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import threading
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import mzsae

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(REPO_DIR, "fastattn_verify4", "model_gguf")
GGUF_FILE = "qwen2.5-1.5b.gguf"

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
    print("FIX 1: REAL MEMORY MEASUREMENT (COMBINED RSS + MPS)")
    print("=" * 75)

    watchdog = CombinedMemoryWatchdog(interval_s=2.0, max_combined_gb=10.0)
    watchdog.start()

    watchdog.set_op("load_tokenizer")
    print("[gen_e2e] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF_FILE)

    watchdog.set_op("load_model_float16")
    print("[gen_e2e] Loading model in float16 with low_cpu_mem_usage=True...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        gguf_file=GGUF_FILE,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("mps").eval()

    prompt = "The fundamental laws of classical mechanics, first established by Sir Isaac Newton,"
    print(f"\n[gen_e2e] Input prompt: \"{prompt}\"")

    watchdog.set_op("encode_prompt")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_len = input_ids.shape[1]

    # Patch model with MZSAE
    watchdog.set_op("patch_model_mzsae")
    print("\n[gen_e2e] Patching model with MZSAE Engine...")
    mzsae.patch_model(model)

    watchdog.set_op("generate_tokens")
    print("[gen_e2e] Generating 40 tokens autoregressively through MZSAE...")
    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=40,
            do_sample=False
        )

    elapsed = time.perf_counter() - t0
    gen_tokens = output_ids.shape[1] - prompt_len
    tok_per_sec = gen_tokens / elapsed

    watchdog.set_op("decode_text")
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

    watchdog.stop()

    print(f"\n[gen_e2e] Generation complete in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
    print("-" * 75)
    print("FULL GENERATED TEXT:")
    print("-" * 75)
    print(full_text)
    print("-" * 75)
    print(f"NEW GENERATED CONTINUATION:\n\"{generated_text}\"")
    print("-" * 75)

    peak_rss_gb = watchdog.peak_rss / (1024**3)
    peak_mps_curr_gb = watchdog.peak_mps_curr / (1024**3)
    peak_mps_driver_gb = watchdog.peak_mps_driver / (1024**3)
    peak_comb_gb = watchdog.peak_combined / (1024**3)

    print("\n" + "=" * 75)
    print("REAL MEMORY FOOTPRINT TELEMETRY (PEAK)")
    print("=" * 75)
    print(f"  * Peak psutil RSS:            {peak_rss_gb:.3f} GB")
    print(f"  * Peak torch.mps Current:     {peak_mps_curr_gb:.3f} GB")
    print(f"  * Peak torch.mps Driver:      {peak_mps_driver_gb:.3f} GB")
    print(f"  * PEAK COMBINED MEMORY:       {peak_comb_gb:.3f} GB")
    print(f"  * Status:                     {'UNDER 4.0 GB' if peak_comb_gb <= 4.0 else 'EXCEEDS 4.0 GB'}")
    print("=" * 75)

if __name__ == "__main__":
    main()
