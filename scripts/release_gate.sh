#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
check() { if [ $1 -eq 0 ]; then echo "GATE: PASS — $2"; else echo "GATE: FAIL — $2"; FAIL=1; fi; }

VERSION=$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "Release gate for v${VERSION}"
echo

echo "=== [1/7] Unit & Neutrality & Quantization Suite ==="
./.venv/bin/python -m pytest tests/ -q --tb=short > /tmp/gate_tests.log 2>&1
check $? "Test suite green (see /tmp/gate_tests.log)"
grep -qE "1[0-9]{2} passed" /tmp/gate_tests.log; check $? "≥100 tests passing"

echo "=== [2/7] Benchmark Harness Re-Run ==="
PYTHONPATH=. ./.venv/bin/python benchmarks/bench_niah.py > /tmp/gate_niah.log 2>&1
check $? "NIAH suite (incl. subtle-needle sweep)"
PYTHONPATH=. ./.venv/bin/python benchmarks/bench_real_weights.py > /tmp/gate_weights.log 2>&1
check $? "Real-weights per-layer + E2E estimate"

echo "=== [3/7] Fresh-Venv Wheel Smoke Test ==="
rm -rf /tmp/mzsae_gate_venv
./.venv/bin/python -m venv /tmp/mzsae_gate_venv
/tmp/mzsae_gate_venv/bin/pip install -q "dist/mzsae-${VERSION}-py3-none-any.whl"
/tmp/mzsae_gate_venv/bin/python -c "import mzsae; from mzsae import MZSAEAttention, load_config; print(mzsae.__version__)" > /dev/null 2>&1
check $? "Wheel installs clean and imports in isolated venv"

echo "=== [3.5/7] Wheel Hygiene & Freshness ==="
# No bytecode caches shipped
[ "$(unzip -l "dist/mzsae-${VERSION}-py3-none-any.whl" | grep -c '__pycache__')" -eq 0 ]; check $? "Wheel contains no __pycache__"
# Wheel content matches source tree (stale-wheel detector)
SRC_HASH=$(shasum -a 256 src/mzsae/hf_patch.py | awk '{print $1}')
WHL_HASH=$(unzip -p "dist/mzsae-${VERSION}-py3-none-any.whl" "mzsae/hf_patch.py" | shasum -a 256 | awk '{print $1}')
[ "$SRC_HASH" = "$WHL_HASH" ]; check $? "Wheel hf_patch.py matches source (not stale)"
SRC_HASH2=$(shasum -a 256 src/mzsae/__init__.py | awk '{print $1}')
WHL_HASH2=$(unzip -p "dist/mzsae-${VERSION}-py3-none-any.whl" "mzsae/__init__.py" | shasum -a 256 | awk '{print $1}')
[ "$SRC_HASH2" = "$WHL_HASH2" ]; check $? "Wheel __init__.py matches source (not stale)"

echo "=== [4/7] Secrets & Credential Scan ==="
! grep -rnE "(ghp_|gho_|github_pat_|sk-[A-Za-z0-9]{20,}|api_key\s*=\s*['\"][A-Za-z0-9])" \
    src/ tests/ benchmarks/ docs/ examples/ configs/ README.md 2>/dev/null
check $? "Zero tokens/keys in tracked files"

echo "=== [5/7] No Hardcoded Local Paths ==="
! grep -rn --include='*.py' --include='*.md' --include='*.yml' --include='*.yaml' --include='*.toml' \
    "/Users/" src/ configs/ docs/ examples/ tests/ README.md mkdocs.yml 2>/dev/null
check $? "No absolute user paths shipped"

echo "=== [6/7] Claim Consistency Audit ==="
grep -q "LIMITATIONS.md" README.md; check $? "README links LIMITATIONS.md"
grep -qi "Dense+FIFO\|Dense Attention (FIFO)" README.md; check $? "Eviction-policy disclosure present"
# Conservative E2E headline present (measured range or point estimate)
grep -qE "~(50.65|63) tok/s" README.md; check $? "Conservative E2E headline present"
# DRAM cut labeled as estimate (case-insensitive, accepts 'Est.' abbreviation too)
grep -qiE "estimated|est\. dram" README.md; check $? "DRAM cut labeled as estimate"

echo "=== [7/7] Git / Release Hygiene ==="
git status --porcelain | grep -q . && echo "WARN: uncommitted changes" 
git describe --tags --exact-match HEAD > /dev/null 2>&1; check $? "HEAD is tagged (v${VERSION})"
gh release view "v${VERSION}" --json assets --jq '.assets[].name' | grep -q "whl"; check $? "Wheel attached to release"

echo
if [ $FAIL -eq 0 ]; then echo "✅ ALL GATES PASS — CLEARED TO PUBLISH"; else echo "❌ GATE FAILURE — FIX BEFORE PUBLISHING"; exit 1; fi
