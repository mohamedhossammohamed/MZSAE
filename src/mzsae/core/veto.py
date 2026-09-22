"""
Directional Veto: Counterfactual Eviction Gate
MZahran Sparse Attention Engine (MZSAE)
"""

import numpy as np


class DirectionalVeto:
    """
    Hardware-Enforced Counterfactual Directional Veto.
    Intercepts eviction requests: if cos(theta_slow) > threshold, vetoes eviction.
    """

    def __init__(self, cos_threshold: float = 0.4):
        self.cos_threshold = cos_threshold
        self.total_checks = 0
        self.total_vetoes = 0

    def evaluate_eviction(
        self,
        block_id: int,
        query_slow: np.ndarray,
        sentinel_slow: np.ndarray,
        is_sink: bool = False,
    ) -> bool:
        self.total_checks += 1
        if is_sink:
            self.total_vetoes += 1
            return False

        q_norm = np.linalg.norm(query_slow)
        s_norm = np.linalg.norm(sentinel_slow)

        if q_norm < 1e-7 or s_norm < 1e-7:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(query_slow, sentinel_slow) / (q_norm * s_norm))

        if cos_sim > self.cos_threshold:
            self.total_vetoes += 1
            return False

        return True
