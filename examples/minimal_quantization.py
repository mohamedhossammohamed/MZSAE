"""
Minimal Quantization Demonstration for MZSAE
Demonstrates compatibility across 4-bit and BitNet b1.58 ternary models with zero downloads.
"""

import torch
from mzsae import MZSAEAttention, ModelWeightConfig, load_config


def demo_4bit():
    print("=== Demo 1: 4-bit Quantized Model (Q4_K_M / AWQ) ===")
    attn = MZSAEAttention(
        embed_dim=2048,
        num_heads=16,
        num_kv_heads=4,
        model_weights="4bit",
    )
    print(f"Quantization: {attn.config.model_weights.quantization_type} ({attn.config.model_weights.quantization_bits}-bit)")
    print(f"Dynamic Range Scale: {attn.config.model_weights.dynamic_range_scale}")
    print(f"Tau Multiplier: {attn.config.model_weights.tau_multiplier}")

    x = torch.randn(1, 32, 2048)
    out = attn(x, causal=True)
    print(f"Forward pass output: {out.shape}, finite: {not torch.isnan(out).any()}\n")


def demo_ternary():
    print("=== Demo 2: BitNet b1.58 Ternary Model ===")
    # Load via named profile or direct string
    attn = MZSAEAttention(
        embed_dim=2048,
        num_heads=16,
        num_kv_heads=4,
        model_weights="ternary",
    )
    print(f"Quantization: {attn.config.model_weights.quantization_type} ({attn.config.model_weights.quantization_bits}-bit container)")
    print(f"Dynamic Range Scale: {attn.config.model_weights.dynamic_range_scale} (compensates for tighter variance)")
    print(f"Tau Multiplier: {attn.config.model_weights.tau_multiplier} (prevents over-pruning on tight logits)")
    print(f"Effective Pruning Tau: {attn.engine.tau}")

    x = torch.randn(1, 32, 2048)
    out = attn(x, causal=True)
    print(f"Forward pass output: {out.shape}, finite: {not torch.isnan(out).any()}\n")


if __name__ == "__main__":
    demo_4bit()
    demo_ternary()
    print("[SUCCESS] Quantization demonstration completed cleanly.")
