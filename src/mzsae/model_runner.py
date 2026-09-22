"""
End-to-End Model Runner with MZSAE Hardware Sparse Attention
Supports loading real model weights from GGUF / SafeTensors
Author: Mohammed Hossam Zahran
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .engine import MZSAEEngine
from .rope import apply_rope_givens, compute_rope_frequencies
from .crq import BLOCK_SIZE, HEAD_DIM

class MZSAETransformerModel:
    """
    Minimal, high-efficiency autoregressive Transformer decoder
    executing real neural model weights with MZSAE attention drop-in.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        vocab_size: int,
        use_mzsae: bool = True,
        tau: float = 16.0
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.use_mzsae = use_mzsae
        self.tau = tau

        # Initialize MZSAE engines (one per KV head per layer)
        if use_mzsae:
            self.engines = [
                [MZSAEEngine(capacity_blocks=2048, tau=tau, use_metal=True) for _ in range(num_kv_heads)]
                for _ in range(num_layers)
            ]
        else:
            self.engines = None
            # Standard dense KV cache storage: (num_layers, num_kv_heads, max_len, head_dim)
            self.dense_k_cache = [ [[] for _ in range(num_kv_heads)] for _ in range(num_layers) ]
            self.dense_v_cache = [ [[] for _ in range(num_kv_heads)] for _ in range(num_layers) ]

        self.step_idx = 0

    def prefill_prompt(
        self,
        layer_idx: int,
        prompt_keys_unrot: np.ndarray, # (prompt_len, num_kv_heads, head_dim)
        prompt_values: np.ndarray       # (prompt_len, num_kv_heads, head_dim)
    ):
        """Vectorized prefill of prompt tokens into KV cache."""
        prompt_len = prompt_keys_unrot.shape[0]
        for kv_h in range(self.num_kv_heads):
            k_stream = prompt_keys_unrot[:, kv_h, :]
            v_stream = prompt_values[:, kv_h, :]
            if self.use_mzsae:
                self.engines[layer_idx][kv_h].ingest_kv_chunk(k_stream, v_stream, start_pos=0)
            else:
                for t in range(prompt_len):
                    self.dense_k_cache[layer_idx][kv_h].append(k_stream[t])
                    self.dense_v_cache[layer_idx][kv_h].append(v_stream[t])

    def attention_forward(
        self,
        layer_idx: int,
        query_states: np.ndarray, # (num_heads, head_dim)
        key_unrot: np.ndarray,    # (num_kv_heads, head_dim)
        value_states: np.ndarray, # (num_kv_heads, head_dim)
        timestep: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes a single decode step for multi-head attention.
        """
        out_heads = []
        metrics = {"approved_blocks": 0, "pruned_blocks": 0, "total_blocks": 0}

        heads_per_kv = self.num_heads // self.num_kv_heads

        for kv_h in range(self.num_kv_heads):
            k_vec = key_unrot[kv_h]
            v_vec = value_states[kv_h]

            if self.use_mzsae:
                engine = self.engines[layer_idx][kv_h]
                engine.cache.append_kv(k_vec, v_vec, timestep=timestep)
                
                # Query each attached Q head
                for q_offset in range(heads_per_kv):
                    q_idx = kv_h * heads_per_kv + q_offset
                    q_vec = query_states[q_idx]

                    # Decode through MZSAE Metal kernel
                    out_h, tel = engine.decode_step(q_vec)
                    out_heads.append(out_h)
                    metrics["approved_blocks"] += tel.get("approved_blocks", 0)
                    metrics["pruned_blocks"] += tel.get("pruned_blocks", 0)
                    metrics["total_blocks"] += tel.get("total_blocks", 0)
            else:
                # Standard dense attention
                self.dense_k_cache[layer_idx][kv_h].append(k_vec)
                self.dense_v_cache[layer_idx][kv_h].append(v_vec)

                all_k = np.stack(self.dense_k_cache[layer_idx][kv_h], axis=0) # (seq, D)
                all_v = np.stack(self.dense_v_cache[layer_idx][kv_h], axis=0) # (seq, D)
                seq_len = all_k.shape[0]
                pos = np.arange(seq_len)
                all_k_rot = apply_rope_givens(all_k, pos)

                for q_offset in range(heads_per_kv):
                    q_idx = kv_h * heads_per_kv + q_offset
                    q_vec = query_states[q_idx]
                    scores = np.dot(all_k_rot, q_vec) / np.sqrt(self.head_dim)
                    probs = np.exp(scores - np.max(scores))
                    probs = probs / np.sum(probs)
                    out_h = np.dot(probs, all_v)
                    out_heads.append(out_h)

        out_concat = np.concatenate(out_heads, axis=-1)
        return out_concat, metrics
