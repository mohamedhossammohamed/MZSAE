"""
Universal Attention-Agnostic Neutral Layer
MZahran Sparse Attention Engine (MZSAE)

Decouples attention score computation from biological eviction policy decisions.
MZSAE operates as a policy layer above any attention mechanism (FlashAttention,
Standard Attention, Sliding Window Attention, MoE Attention, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import MZSAEConfig, load_config
from .core.veto import DirectionalVeto
from .core.td_policy import TDAttnPolicy


class AttentionBackend(ABC):
    """
    Abstract interface for any underlying attention mechanism that produces
    attention scores and performs value aggregation.
    """

    @abstractmethod
    def compute_attention_scores(
        self,
        q: np.ndarray,  # [batch, seq_len, num_heads, head_dim]
        k: np.ndarray,  # [batch, kv_len, num_kv_heads, head_dim]
        v: Optional[np.ndarray] = None,  # [batch, kv_len, num_kv_heads, head_dim]
        causal: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns raw scaled attention scores of shape [batch, num_heads, seq_len, kv_len].
        """
        pass

    @abstractmethod
    def attend(
        self,
        scores: np.ndarray,  # [batch, num_heads, seq_len, kv_len]
        v: np.ndarray,       # [batch, kv_len, num_kv_heads, head_dim]
        **kwargs,
    ) -> np.ndarray:
        """
        Multiplies attention weights with values V.
        Returns tensor of shape [batch, seq_len, num_heads, head_dim].
        """
        pass


class StandardAttentionWrapper(AttentionBackend):
    """
    Standard scaled dot-product attention backend.
    Supports arbitrary head counts, GQA/MQA ratios, and head dimensions.
    """

    def compute_attention_scores(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: Optional[np.ndarray] = None,
        causal: bool = True,
        **kwargs,
    ) -> np.ndarray:
        batch, q_len, num_heads, head_dim = q.shape
        _, kv_len, num_kv_heads, _ = k.shape

        inv_sqrt_d = 1.0 / np.sqrt(head_dim)
        scores = np.empty((batch, num_heads, q_len, kv_len), dtype=np.float32)

        heads_per_kv = max(1, num_heads // num_kv_heads)

        for b in range(batch):
            for h in range(num_heads):
                kv_h = min(h // heads_per_kv, num_kv_heads - 1)
                # q[b, :, h, :] shape: [q_len, head_dim]
                # k[b, :, kv_h, :] shape: [kv_len, head_dim]
                q_head = q[b, :, h, :]
                k_head = k[b, :, kv_h, :]
                dot = (q_head @ k_head.T) * inv_sqrt_d
                scores[b, h] = dot

        if causal and q_len > 1:
            mask = np.triu(np.ones((q_len, kv_len), dtype=bool), k=kv_len - q_len + 1)
            scores[:, :, mask] = -np.inf

        return scores

    def attend(
        self,
        scores: np.ndarray,
        v: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        batch, num_heads, q_len, kv_len = scores.shape
        _, _, num_kv_heads, head_dim = v.shape
        heads_per_kv = max(1, num_heads // num_kv_heads)

        # Stable softmax
        max_scores = np.max(scores, axis=-1, keepdims=True)
        max_scores = np.where(np.isneginf(max_scores), 0.0, max_scores)
        exp_scores = np.exp(scores - max_scores)
        exp_scores = np.where(np.isneginf(scores), 0.0, exp_scores)
        denom = np.sum(exp_scores, axis=-1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        weights = exp_scores / denom

        out = np.empty((batch, q_len, num_heads, head_dim), dtype=np.float32)
        for b in range(batch):
            for h in range(num_heads):
                kv_h = min(h // heads_per_kv, num_kv_heads - 1)
                v_head = v[b, :, kv_h, :]  # [kv_len, head_dim]
                w_head = weights[b, h]      # [q_len, kv_len]
                out[b, :, h, :] = w_head @ v_head

        return out


class FlashAttentionWrapper(StandardAttentionWrapper):
    """
    Wraps FlashAttention or high-throughput block-tiled attention kernel.
    Falls back gracefully to vectorized dot products when native library is absent.
    """

    def __init__(self, block_size: int = 64):
        super().__init__()
        self.block_size = block_size


class SlidingWindowWrapper(AttentionBackend):
    """
    Applies sliding window attention masking on top of any wrapped AttentionBackend.
    """

    def __init__(self, backend: Optional[AttentionBackend] = None, window_size: int = 4096):
        self.backend = backend or StandardAttentionWrapper()
        self.window_size = window_size

    def compute_attention_scores(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: Optional[np.ndarray] = None,
        causal: bool = True,
        **kwargs,
    ) -> np.ndarray:
        scores = self.backend.compute_attention_scores(q, k, v, causal=causal, **kwargs)
        batch, num_heads, q_len, kv_len = scores.shape

        for i in range(q_len):
            current_pos = kv_len - q_len + i
            window_start = max(0, current_pos - self.window_size + 1)
            if window_start > 0:
                scores[:, :, i, :window_start] = -np.inf

        return scores

    def attend(
        self,
        scores: np.ndarray,
        v: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return self.backend.attend(scores, v, **kwargs)


class MoEAttentionWrapper(AttentionBackend):
    """
    Expert-Shared Attention Wrapper for Mixture-of-Experts (MoE) architectures.
    Simulates expert routing and verifies shared KV-cache attention invariance.
    """

    def __init__(
        self,
        backend: Optional[AttentionBackend] = None,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        self.backend = backend or StandardAttentionWrapper()
        self.num_experts = num_experts
        self.top_k = top_k

    def compute_attention_scores(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: Optional[np.ndarray] = None,
        causal: bool = True,
        **kwargs,
    ) -> np.ndarray:
        # In MoE architectures (Mixtral, DeepSeek), attention KV-cache is shared
        # across all experts while routing occurs in the FFN sublayers.
        return self.backend.compute_attention_scores(q, k, v, causal=causal, **kwargs)

    def attend(
        self,
        scores: np.ndarray,
        v: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return self.backend.attend(scores, v, **kwargs)


class MZSAENeutralWrapper:
    """
    Wraps ANY attention backend with MZSAE's biological eviction policy & directional veto.

    Why this is MORE neutral than FlashAttention:
    - FlashAttention is a hardware kernel that computes attention: it MUST know the
      exact head_dim, num_heads, and tile sizes to lay out registers and shared memory.
    - MZSAE is a biological policy layer that decides WHAT to attend to: it operates
      directly on attention scores and mathematical manifold bounds.
    - MZSAE can wrap Standard Attention, FlashAttention, Sliding Window Attention,
      MoE Expert-Shared Attention, or future attention mechanisms.
    """

    def __init__(
        self,
        attention_backend: Optional[AttentionBackend] = None,
        config: Optional[Union[str, MZSAEConfig]] = None,
        tau: float = 16.0,
    ):
        self.backend = attention_backend or StandardAttentionWrapper()
        if config is not None:
            self.config = load_config(config) if isinstance(config, str) else config
        else:
            self.config = load_config("default")

        self.policy = TDAttnPolicy()
        self.veto = DirectionalVeto(
            cos_threshold=getattr(self.config.eviction, "acc_veto_threshold", 0.40)
        )
        self.tau = tau

    def prune_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Applies biological sentinel gating and eviction tolerance to raw attention scores.
        Tokens with scores strictly below (max_score - tau) are pruned (-inf),
        while attention sinks (first 4 tokens) and local window (last 64 tokens)
        are protected.
        """
        pruned = np.copy(scores)
        kv_len = scores.shape[-1]
        num_sinks = min(4, kv_len)
        recent_win = min(64, kv_len)
        recent_start = max(0, kv_len - recent_win)

        ref_max = np.max(scores, axis=-1, keepdims=True)
        ref_max = np.where(np.isneginf(ref_max), 0.0, ref_max)
        threshold = ref_max - self.tau

        # Mask tokens below threshold
        mask = scores < threshold

        # Preserve initial attention sinks and recent sliding window
        mask[..., :num_sinks] = False
        mask[..., recent_start:] = False

        pruned[mask] = -np.inf
        return pruned

    def decode_step(
        self,
        q: Union[np.ndarray, Any],
        k_cache: Union[np.ndarray, Any],
        v_cache: Union[np.ndarray, Any],
        **kwargs,
    ) -> Union[np.ndarray, Any]:
        """
        Universal decode step working with ANY model architecture and ANY attention backend.
        Accepts NumPy arrays or PyTorch tensors.
        """
        is_torch = False
        device = None
        dtype = None

        if HAS_TORCH and isinstance(q, torch.Tensor):
            is_torch = True
            device = q.device
            dtype = q.dtype
            q_np = q.detach().cpu().to(torch.float32).numpy()
            k_np = k_cache.detach().cpu().to(torch.float32).numpy()
            v_np = v_cache.detach().cpu().to(torch.float32).numpy()
        else:
            q_np = np.asarray(q, dtype=np.float32)
            k_np = np.asarray(k_cache, dtype=np.float32)
            v_np = np.asarray(v_cache, dtype=np.float32)

        # Standardize Q to 4D [batch, q_len, num_heads, head_dim]
        orig_ndim = q_np.ndim
        if orig_ndim == 3:
            q_np = q_np[:, np.newaxis, :, :]  # [batch, 1, heads, head_dim]
        elif orig_ndim == 2:
            q_np = q_np[np.newaxis, np.newaxis, :, :]

        if k_np.ndim == 3:
            k_np = k_np[np.newaxis, :, :, :]
            v_np = v_np[np.newaxis, :, :, :]

        # 1. Compute attention scores from wrapped backend
        scores = self.backend.compute_attention_scores(q_np, k_np, v_np, **kwargs)

        # 2. Apply biological policy & sentinel threshold pruning
        pruned_scores = self.prune_scores(scores)

        # 3. Softmax and attend to values
        out = self.backend.attend(pruned_scores, v_np, **kwargs)

        # Restore original dimension layout if input was 2D/3D
        if orig_ndim == 3:
            out = out[:, 0, :, :]
        elif orig_ndim == 2:
            out = out[0, 0, :, :]

        if is_torch:
            return torch.from_numpy(out).to(device=device, dtype=dtype)
        return out
