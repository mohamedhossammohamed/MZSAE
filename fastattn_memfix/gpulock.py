#!/usr/bin/env python3
"""
fastattn_memfix/gpulock.py
Robust Single-Job GPU Lock with Dead-PID and Stale-Timeout Detection.
"""
import os
import sys
import time
import json

LOCK_FILE = "/tmp/fastattn_memfix_gpu.lock"

def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def clean_stale_lock():
    if not os.path.exists(LOCK_FILE):
        return
    try:
        with open(LOCK_FILE, "r") as f:
            data = json.load(f)
        pid = data.get("pid")
        start_time = data.get("start_time", 0)
        age = time.time() - start_time
        if not is_pid_alive(pid) or age > 600:
            print(f"[gpulock] Removing stale lock: pid={pid}, age={age:.1f}s", file=sys.stderr)
            try:
                os.remove(LOCK_FILE)
            except OSError:
                pass
    except Exception:
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass

class GPULock:
    def __init__(self, tag: str = "job", timeout: float = 300.0):
        self.tag = tag
        self.timeout = timeout
        self.locked = False

    def acquire(self):
        clean_stale_lock()
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            try:
                fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    json.dump({
                        "pid": os.getpid(),
                        "tag": self.tag,
                        "start_time": time.time()
                    }, f)
                self.locked = True
                print(f"[gpulock] Acquired GPU lock ({self.tag})", flush=True)
                return True
            except FileExistsError:
                clean_stale_lock()
                time.sleep(1.0)
        raise TimeoutError(f"[gpulock] Timed out waiting for GPU lock ({self.tag})")

    def release(self):
        if self.locked:
            try:
                if os.path.exists(LOCK_FILE):
                    with open(LOCK_FILE, "r") as f:
                        data = json.load(f)
                    if data.get("pid") == os.getpid():
                        os.remove(LOCK_FILE)
            except Exception:
                pass
            self.locked = False
            print(f"[gpulock] Released GPU lock ({self.tag})", flush=True)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
