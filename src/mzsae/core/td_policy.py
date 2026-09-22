"""
Autonomous Sleep & Lifecycle Engine (TD-Attn)
MZahran Sparse Attention Engine (MZSAE)
"""

from typing import Tuple, Dict, Any
import numpy as np


class TelemetryRingBuffer:
    """
    128 KB In-Memory Circular Telemetry Buffer.
    Records transition tuples (state, action, reward, next_state) without disk writes.
    """

    def __init__(self, max_records: int = 1024):
        self.max_records = max_records
        self.states = np.zeros((max_records, 16), dtype=np.float32)
        self.actions = np.zeros(max_records, dtype=np.uint8)
        self.rewards = np.zeros(max_records, dtype=np.float32)
        self.next_states = np.zeros((max_records, 16), dtype=np.float32)
        self.head = 0
        self.size = 0

    def push(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ):
        idx = self.head
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.head = (self.head + 1) % self.max_records
        self.size = min(self.size + 1, self.max_records)

    def sample_batch(
        self, batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = min(batch_size, self.size)
        if n == 0:
            return (
                np.zeros((0, 16), dtype=np.float32),
                np.zeros(0, dtype=np.uint8),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 16), dtype=np.float32),
            )
        indices = np.random.choice(self.size, size=n, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
        )


class TDAttnPolicy:
    """
    27k-Parameter MLP Value Predictor.
    Architecture: 16 -> 128 -> 128 -> 64 -> 1 = exactly 27,009 parameters.
    Initialized with Day-Zero analytic weights matching StreamingLLM and H2O.
    """

    def __init__(self, lr: float = 0.001, gamma: float = 0.95):
        self.lr = lr
        self.gamma = gamma

        # Architecture weights (27,009 total parameters)
        self.W1 = np.random.randn(16, 128).astype(np.float32) * 0.05
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = np.random.randn(128, 128).astype(np.float32) * 0.05
        self.b2 = np.zeros(128, dtype=np.float32)
        self.W3 = np.random.randn(128, 64).astype(np.float32) * 0.05
        self.b3 = np.zeros(64, dtype=np.float32)
        self.W4 = np.random.randn(64, 1).astype(np.float32) * 0.05
        self.b4 = np.zeros(1, dtype=np.float32)

        self._init_day_zero_weights()

        # Adam optimizer moments
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)
        self.mW3 = np.zeros_like(self.W3)
        self.vW3 = np.zeros_like(self.W3)
        self.mb3 = np.zeros_like(self.b3)
        self.vb3 = np.zeros_like(self.b3)
        self.mW4 = np.zeros_like(self.W4)
        self.vW4 = np.zeros_like(self.W4)
        self.mb4 = np.zeros_like(self.b4)
        self.vb4 = np.zeros_like(self.b4)
        self.step_count = 0

    def _init_day_zero_weights(self):
        """Injects analytic prior: Sinks have highest survival, stale tokens penalized."""
        self.W1[0, 0] = 2.0
        self.W1[1, 0] = -1.5
        self.W1[7, 0] = 10.0  # Sinks
        self.W1[8, 0] = 8.0  # Recent window
        self.W2[0, 0] = 1.0
        self.W3[0, 0] = 1.0
        self.W4[0, 0] = 1.0

    @property
    def total_parameters(self) -> int:
        return (
            self.W1.size
            + self.b1.size
            + self.W2.size
            + self.b2.size
            + self.W3.size
            + self.b3.size
            + self.W4.size
            + self.b4.size
        )

    def forward(self, x: np.ndarray) -> Tuple[Any, Dict[str, np.ndarray]]:
        single = x.ndim == 1
        if single:
            x_mat = x.reshape(1, -1)
        else:
            x_mat = x

        z1 = np.dot(x_mat, self.W1) + self.b1
        a1 = np.maximum(0, z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)

        z3 = np.dot(a2, self.W3) + self.b3
        a3 = np.maximum(0, z3)

        z4 = np.dot(a3, self.W4) + self.b4
        cache = {
            "x": x_mat,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "z3": z3,
            "a3": a3,
        }
        out = z4.squeeze(-1)

        if single:
            return float(out[0]), cache
        return out, cache

    def td_update(
        self, states: np.ndarray, rewards: np.ndarray, next_states: np.ndarray
    ) -> float:
        v_current, cache = self.forward(states)
        v_next, _ = self.forward(next_states)

        target = rewards + self.gamma * v_next
        delta = target - v_current
        loss = float(np.mean(delta**2))

        N = states.shape[0]
        grad_out = (-2.0 * delta / N).reshape(-1, 1)

        grad_W4 = np.dot(cache["a3"].T, grad_out)
        grad_b4 = np.sum(grad_out, axis=0)

        da3 = np.dot(grad_out, self.W4.T)
        dz3 = da3 * (cache["z3"] > 0)
        grad_W3 = np.dot(cache["a2"].T, dz3)
        grad_b3 = np.sum(dz3, axis=0)

        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (cache["z2"] > 0)
        grad_W2 = np.dot(cache["a1"].T, dz2)
        grad_b2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (cache["z1"] > 0)
        grad_W1 = np.dot(cache["x"].T, dz1)
        grad_b1 = np.sum(dz1, axis=0)

        self.step_count += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        params = [
            (self.W1, grad_W1, self.mW1, self.vW1),
            (self.b1, grad_b1, self.mb1, self.vb1),
            (self.W2, grad_W2, self.mW2, self.vW2),
            (self.b2, grad_b2, self.mb2, self.vb2),
            (self.W3, grad_W3, self.mW3, self.vW3),
            (self.b3, grad_b3, self.mb3, self.vb3),
            (self.W4, grad_W4, self.mW4, self.vW4),
            (self.b4, grad_b4, self.mb4, self.vb4),
        ]

        for p, g, m, v in params:
            m[:] = beta1 * m + (1.0 - beta1) * g
            v[:] = beta2 * v + (1.0 - beta2) * (g**2)
            m_hat = m / (1.0 - beta1**self.step_count)
            v_hat = v / (1.0 - beta2**self.step_count)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        return loss
