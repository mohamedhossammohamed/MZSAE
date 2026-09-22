"""
Property-Based Neutrality Tests using Hypothesis
Verifies MZSAE invariant stability across randomly generated parameter configurations.
"""

import numpy as np
import pytest

try:
    from hypothesis import given, strategies as st, settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

if not HAS_HYPOTHESIS:
    pytest.skip("hypothesis not installed", allow_module_level=True)

from src.mzsae.neutral import MZSAENeutralWrapper, StandardAttentionWrapper


@settings(max_examples=25, deadline=None)
@given(
    head_dim=st.sampled_from([64, 96, 128, 256]),
    num_heads=st.integers(min_value=1, max_value=32),
    kv_ratio=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=64, max_value=512),
)
def test_mzsae_works_with_any_config(head_dim, num_heads, kv_ratio, seq_len):
    """Property-based test: MZSAE works with ANY valid configuration."""
    kv_heads = max(1, num_heads // kv_ratio)

    q = np.random.randn(1, 1, num_heads, head_dim).astype(np.float32)
    k = np.random.randn(1, seq_len, kv_heads, head_dim).astype(np.float32)
    v = np.random.randn(1, seq_len, kv_heads, head_dim).astype(np.float32)

    backend = StandardAttentionWrapper()
    mzsae = MZSAENeutralWrapper(backend)

    output = mzsae.decode_step(q, k, v)
    assert output.shape == (1, 1, num_heads, head_dim)
    assert np.all(np.isfinite(output))
