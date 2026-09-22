#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "================================================================================"
echo "MZahran Sparse Attention Engine (MZSAE) - Master Verification & Benchmark Suite"
echo "Target Platform: Apple Silicon M4 (16 GB Unified Memory)"
echo "Author: Mohammed Hossam Zahran"
echo "================================================================================"

export PYTHONPATH="."
VENV_PY=".venv/bin/python3"
VENV_PYTEST=".venv/bin/pytest"

echo ""
echo "[Step 1/6] Building Metal Hardware Shaders & Runtime Dynamic Library..."
make all

echo ""
echo "[Step 2/6] Running Unit Test Suite (87 Comprehensive Tests)..."
$VENV_PYTEST -v tests/

echo ""
echo "[Step 3/6] Running Level 1: Numerical Invariance Audit..."
$VENV_PY benchmarks/bench_invariance.py

echo ""
echo "[Step 4/6] Running Level 2: Needle In A Haystack (NIAH) & FlashAttention-2 Head-to-Head..."
$VENV_PY benchmarks/bench_niah.py
$VENV_PY benchmarks/bench_flash_attn_niah.py

echo ""
echo "[Step 5/6] Running Level 3: Microarchitectural Latency & Memory Profiling..."
$VENV_PY benchmarks/bench_latency.py

echo ""
echo "[Step 6/6] Running End-to-End Model & Real Weights Benchmarks..."
$VENV_PY benchmarks/bench_e2e_model.py
$VENV_PY benchmarks/bench_real_weights.py

echo ""
echo "================================================================================"
echo "ALL MZSAE SYSTEM TESTS & BENCHMARKS COMPLETED WITH 100% PASS RATE"
echo "================================================================================"
