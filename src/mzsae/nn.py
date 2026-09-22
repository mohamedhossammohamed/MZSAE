"""
PyTorch-Compatible MZSAEAttention Module
MZahran Sparse Attention Engine (MZSAE)
"""

from typing import Optional, Union, Tuple, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Minimal stub base class if PyTorch is not present
    class _ModuleStub:
        def __init__(self, *args, **kwargs):
            pass
    nn = type("nn", (), {"Module": _ModuleStub})

from .config import MZSAEConfig, ModelWeightConfig, load_config
from .core.engine import MZSAEEngine
from .core.cache import MZSAEKVCache


class MZSAEAttention(nn.Module if HAS_TORCH else object):
    """
    Hardware-Aware Sparse Continual Attention Layer for PyTorch.

    Supports:
      1. Functional attention operator: attn(q, k, v, causal=True)
      2. Layer forward pass with linear projections: attn(x, causal=True)
      3. Stateful single-token autoregressive decode with cache:
           out = attn(q, k, v, use_cache=True)
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        hardware_profile: str = "auto",
        config: Optional[Union[str, MZSAEConfig]] = None,
        model_weights: Optional[Union[str, ModelWeightConfig]] = None,
        tau: float = 16.0,
        bias: bool = False,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to use MZSAEAttention. Install with: pip install torch")

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else (embed_dim // num_heads)
        self.tau = tau

        if config is not None:
            self.config = load_config(config) if isinstance(config, str) else config
        else:
            self.config = load_config(hardware_profile)

        if model_weights is not None:
            if isinstance(model_weights, str):
                self.config.model_weights = ModelWeightConfig.from_type(model_weights)
            elif isinstance(model_weights, ModelWeightConfig):
                self.config.model_weights = model_weights

        self.config.kernel.head_dim = self.head_dim
        self.config.kernel.num_query_heads = self.num_heads
        self.config.kernel.num_kv_heads = self.num_kv_heads

        # Projections for layer usage
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, embed_dim, bias=bias)

        # Underlying MZSAE execution engine
        self.engine = MZSAEEngine(
            config=self.config,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            tau=self.tau,
        )

    def reset_cache(self):
        """Clears the underlying compressed KV cache."""
        self.engine.cache.reset()

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        causal: bool = True,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward attention execution.
        Args:
          q: Query tensor of shape [batch, seq_len, embed_dim] or [batch, seq_len, num_heads, head_dim]
             or [batch, num_heads, seq_len, head_dim]
          k: Key tensor (optional if q is input embedding x)
          v: Value tensor (optional if q is input embedding x)
          causal: Enforce causal attention masking
          use_cache: Maintain stateful single-token decode cache
        Returns:
          Attention output matching input projection dimensions.
        """
        # Case A: Projection mode where q is raw hidden state x
        if k is None or v is None:
            x = q
            batch_size, seq_len, _ = x.shape
            q_proj = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k_proj = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v_proj = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            is_projection_mode = True
        else:
            q_proj = q
            k_proj = k
            v_proj = v
            is_projection_mode = False

        # Normalize shapes: standardize to [batch, seq_len, num_heads, head_dim]
        if q_proj.ndim == 4 and q_proj.shape[1] == self.num_heads and q_proj.shape[2] != self.num_heads:
            # Format was [batch, num_heads, seq_len, head_dim]
            q_proj = q_proj.transpose(1, 2)
            k_proj = k_proj.transpose(1, 2)
            v_proj = v_proj.transpose(1, 2)

        batch_size, seq_len, nq, d = q_proj.shape
        device = q_proj.device
        dtype = q_proj.dtype

        # Fast path: Single-query decode step (T = 1) with cache
        if seq_len == 1:
            q_np = q_proj[0, 0].detach().cpu().to(torch.float32).numpy()
            k_np = k_proj[0, 0].detach().cpu().to(torch.float16).numpy()
            v_np = v_proj[0, 0].detach().cpu().to(torch.float16).numpy()

            if use_cache:
                self.engine.cache.append_kv(k_np, v_np)

            out_np = self.engine.decode(q_np)
            out_tensor = torch.from_numpy(out_np).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(1)

        # Prefill / Multi-token sequence (T > 1)
        else:
            if use_cache:
                self.engine.ingest_kv_chunk(k_proj[0], v_proj[0])

            # Compute standard SDPA for prefill, returning full sequence representation
            q_sdpa = q_proj.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            k_sdpa = k_proj.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
            v_sdpa = v_proj.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]

            # GQA repeat if kv heads < query heads
            if self.num_heads != self.num_kv_heads:
                repeat_factor = self.num_heads // self.num_kv_heads
                k_sdpa = k_sdpa.repeat_interleave(repeat_factor, dim=1)
                v_sdpa = v_sdpa.repeat_interleave(repeat_factor, dim=1)

            out_sdpa = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, is_causal=causal
            )
            out_tensor = out_sdpa.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]

        if is_projection_mode:
            out_flat = out_tensor.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
            return self.out_proj(out_flat)

        return out_tensor
