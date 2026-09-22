"""
Unit Tests for Brick 4: Autonomous Sleep & Lifecycle Engine (TD-Attn)
"""

import pytest
import numpy as np
from src.mzsae.td_attn import TDAttnPolicy, TelemetryRingBuffer, DirectionalVeto

def test_mlp_parameter_count_exact_27k():
    policy = TDAttnPolicy()
    assert policy.total_parameters == 27009, \
        f"Expected 27,009 parameters (27k), got {policy.total_parameters}"

def test_day_zero_analytic_initialization():
    policy = TDAttnPolicy()
    
    # State 1: Attention Sink token (x[7] = 1.0)
    sink_state = np.zeros(16, dtype=np.float32)
    sink_state[7] = 1.0
    val_sink, _ = policy.forward(sink_state)

    # State 2: Stale unattended token (x[1] = 100.0, x[0] = 0.0)
    stale_state = np.zeros(16, dtype=np.float32)
    stale_state[1] = 100.0
    val_stale, _ = policy.forward(stale_state)

    assert float(val_sink) > float(val_stale), f"Sink value ({val_sink}) must exceed stale token value ({val_stale})"
    print(f"Analytic Sink Value: {float(val_sink):.2f}, Stale Token Value: {float(val_stale):.2f}")

def test_online_td_update_convergence():
    np.random.seed(42)
    policy = TDAttnPolicy(lr=1e-3, gamma=0.9)
    
    # Fixed dataset of transitions to verify gradient descent convergence
    N = 32
    states = np.random.randn(N, 16).astype(np.float32) * 0.5
    rewards = np.sum(states[:, :2], axis=1).astype(np.float32)
    next_states = states * 0.95

    initial_loss = None
    for step in range(50):
        loss = policy.td_update(states, rewards, next_states)
        if initial_loss is None:
            initial_loss = loss
        final_loss = loss

    print(f"TD(0) Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "TD(0) update must reduce Bellman error"

def test_directional_veto_interlock():
    veto = DirectionalVeto(cos_threshold=0.4)

    # Case 1: Collinear vectors (cos = 1.0 > 0.4) -> Must VETO
    q_slow = np.array([1.0, 2.0, 3.0] + [0.0]*13, dtype=np.float32)
    s_slow = np.array([1.0, 2.0, 3.0] + [0.0]*13, dtype=np.float32)
    assert not veto.evaluate_eviction(block_id=10, query_slow=q_slow, sentinel_slow=s_slow, is_sink=False), \
        "Strong semantic alignment must trigger counterfactual veto"

    # Case 2: Orthogonal vectors (cos = 0.0 < 0.4) -> Must APPROVE eviction
    q_slow2 = np.array([1.0, 0.0, 0.0] + [0.0]*13, dtype=np.float32)
    s_slow2 = np.array([0.0, 1.0, 0.0] + [0.0]*13, dtype=np.float32)
    assert veto.evaluate_eviction(block_id=11, query_slow=q_slow2, sentinel_slow=s_slow2, is_sink=False), \
        "Orthogonal vector eviction must be approved"

    # Case 3: Attention sink token -> Always VETO
    assert not veto.evaluate_eviction(block_id=0, query_slow=q_slow2, sentinel_slow=s_slow2, is_sink=True), \
        "Sink tokens must be strictly pinned"
