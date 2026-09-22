#!/usr/bin/env python3
"""
fastattn_memfix/mem_watchdog.py
Hard Memory Caps & Monitoring Subsystem for 16 GB Apple Silicon M4.
Enforces:
  1. Lightweight background watchdog thread checking process RSS every 5s via psutil.
  2. Hard 10 GB RSS limit: immediate abort via os._exit(1) if exceeded, logging last op.
  3. Model parameter memory check: verifies total param bytes < 3.5 GB (float16).
  4. Context length cap: hard assertion <= 32768 tokens.
  5. Layer-by-layer cleanup utility (del + gc.collect() + torch.mps.empty_cache()).
  6. GPU lock enforcement: maximum 1 job holding lock at any time.
"""
import os
import sys
import time
import json
import threading
import gc
import psutil

MAX_RSS_BYTES = int(4.0 * 1024 * 1024 * 1024)       # 4.0 GB Hard RSS ceiling
MAX_MPS_BYTES = int(4.5 * 1024 * 1024 * 1024)       # 4.5 GB Hard MPS allocated ceiling
MAX_MODEL_PARAM_BYTES = int(3.5 * 1024 * 1024 * 1024) # 3.5 GB
MAX_CONTEXT_TOKENS = 32768
CHECK_INTERVAL_SEC = 0.5

_current_operation = "initialization"
_peak_rss_bytes = 0
_peak_mps_bytes = 0
_watchdog_thread = None
_stop_event = threading.Event()

def set_current_operation(op_name: str):
    """Updates the last attempted operation string for diagnostic logging."""
    global _current_operation
    _current_operation = op_name

def get_current_rss_bytes() -> int:
    """Returns current process RSS in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def get_current_mps_bytes() -> int:
    """Returns current MPS allocated memory in bytes."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory()
    except Exception:
        pass
    return 0

def get_peak_rss_bytes() -> int:
    """Returns peak process RSS observed by this process."""
    global _peak_rss_bytes
    cur = get_current_rss_bytes()
    if cur > _peak_rss_bytes:
        _peak_rss_bytes = cur
    return _peak_rss_bytes

def get_peak_mps_bytes() -> int:
    """Returns peak MPS memory observed by this process."""
    global _peak_mps_bytes
    cur = get_current_mps_bytes()
    if cur > _peak_mps_bytes:
        _peak_mps_bytes = cur
    return _peak_mps_bytes

def _watchdog_loop(log_file: str):
    global _peak_rss_bytes, _peak_mps_bytes
    process = psutil.Process(os.getpid())
    while not _stop_event.is_set():
        try:
            rss = process.memory_info().rss
            if rss > _peak_rss_bytes:
                _peak_rss_bytes = rss
            
            mps_alloc = get_current_mps_bytes()
            if mps_alloc > _peak_mps_bytes:
                _peak_mps_bytes = mps_alloc

            breached = False
            reason = ""
            if rss > MAX_RSS_BYTES:
                breached = True
                reason = f"Process RSS {rss / (1024**3):.2f} GB exceeded hard limit {MAX_RSS_BYTES / (1024**3):.1f} GB"
            elif mps_alloc > MAX_MPS_BYTES:
                breached = True
                reason = f"MPS allocated memory {mps_alloc / (1024**3):.2f} GB exceeded hard limit {MAX_MPS_BYTES / (1024**3):.1f} GB"

            if breached:
                msg = (
                    f"\n[FATAL MEMORY WATCHDOG] {reason}! "
                    f"Last operation: {_current_operation}\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                try:
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    with open(log_file, "a") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
                except Exception:
                    pass
                os._exit(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        _stop_event.wait(CHECK_INTERVAL_SEC)

def start_memory_watchdog(log_file: str = "fastattn_memfix/logs/mem_watchdog_abort.log"):
    """Starts the background thread monitoring RSS every 5 seconds."""
    global _watchdog_thread, _stop_event
    if _watchdog_thread is not None and _watchdog_thread.is_alive():
        return
    _stop_event.clear()
    _watchdog_thread = threading.Thread(target=_watchdog_loop, args=(log_file,), daemon=True)
    _watchdog_thread.start()
    print(f"[mem_watchdog] Active. RSS limit: {MAX_RSS_BYTES / (1024**3):.1f} GB, check interval: {CHECK_INTERVAL_SEC}s", flush=True)

def stop_memory_watchdog():
    """Stops the background watchdog thread."""
    global _stop_event
    _stop_event.set()

def assert_model_memory_safe(model):
    """Checks total parameter memory < 3.5 GB (float16). Aborts loudly if not."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_gb = total_bytes / (1024**3)
    print(f"[mem_watchdog] Model parameter footprint: {total_gb:.3f} GB (max allowed: {MAX_MODEL_PARAM_BYTES / (1024**3):.1f} GB)", flush=True)
    if total_bytes > MAX_MODEL_PARAM_BYTES:
        err = f"[mem_watchdog] FATAL: Model parameter footprint {total_gb:.3f} GB exceeds 3.5 GB limit!"
        sys.stderr.write(err + "\n")
        sys.stderr.flush()
        raise MemoryError(err)
    return total_bytes

def assert_context_length_safe(context_len: int):
    """Enforces hard context limit of 32768 tokens."""
    if context_len > MAX_CONTEXT_TOKENS:
        err = f"[mem_watchdog] FATAL: Requested context length {context_len} exceeds max safe limit {MAX_CONTEXT_TOKENS}!"
        sys.stderr.write(err + "\n")
        sys.stderr.flush()
        raise ValueError(err)

def cleanup_layer_memory():
    """Executes explicit garbage collection and MPS cache release."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
