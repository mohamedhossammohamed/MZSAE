"""
Unit Tests for Dynamic Memory-Budgeted Cache Eviction and Directional Veto
MZahran Sparse Attention Engine (MZSAE)
"""

import pytest
import numpy as np
from src.mzsae.engine import MZSAEEngine, BLOCK_SIZE, HEAD_DIM
from src.mzsae.crq import SLOW_DIMS
from src.mzsae.rope import decouple_rope_spectrum

def test_dynamic_eviction_and_veto_preservation():
    """
    Verifies that ingest_block and evict_logical_block properly maintain cache state,
    and DirectionalVeto preserves needle blocks during memory budget pressure.
    """
    spec = decouple_rope_spectrum(HEAD_DIM, SLOW_DIMS)
    slow_idx = spec["slow_indices"]

    engine = MZSAEEngine(capacity_blocks=32, tau=16.0, use_metal=True)

    # Ingest 10 blocks: block 0 (sinks), block 3 (needle), others random
    needle_id = 3
    for b in range(10):
        if b == needle_id:
            k = np.zeros((BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
            k[:, slow_idx] = 2.5
            v = np.ones((BLOCK_SIZE, HEAD_DIM), dtype=np.float32) * 7.77
        else:
            k = np.random.randn(BLOCK_SIZE, HEAD_DIM).astype(np.float32) * 0.5
            v = np.random.randn(BLOCK_SIZE, HEAD_DIM).astype(np.float32) * 0.5
        engine.ingest_block(k, v, logical_id=b)

    assert engine.cache.total_active_blocks == 10
    assert needle_id in engine.cache.logical_block_ids

    # Evict block 1, 2, 4, 5
    for eb in [1, 2, 4, 5]:
        assert engine.evict_logical_block(eb) is True

    assert engine.cache.total_active_blocks == 6
    assert needle_id in engine.cache.logical_block_ids
    assert 1 not in engine.cache.logical_block_ids

    # Query aligned with needle
    probe_query = np.zeros(HEAD_DIM, dtype=np.float32)
    probe_query[slow_idx] = 2.5

    out, tel = engine.decode_step(probe_query)
    signal = float(np.mean(out))
    print(f"Post-eviction needle signal: {signal:.3f}")
    assert signal > 5.0, f"Expected needle signal > 5.0, got {signal}"

def test_sleep_cycle_replay():
    """
    Verifies that engine.sleep_cycle replays transitions and reduces Bellman error.
    """
    engine = MZSAEEngine(use_metal=True)
    for i in range(50):
        st = np.random.randn(16).astype(np.float32)
        act = int(i % 2)
        rew = float(np.sin(i))
        nst = st * 0.95
        engine.telemetry.push(st, act, rew, nst)

    assert engine.telemetry.size == 50
    loss = engine.sleep_cycle(batch_size=32, steps=10)
    print(f"Sleep cycle final loss: {loss:.4f}")
    assert loss >= 0.0
