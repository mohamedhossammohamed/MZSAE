"""
server_patch.py
Monkey-patching hook to equip active BonsAI models with Scheme S2 Base-3 (1.60 b/w) layers.
"""

from __future__ import annotations

import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from runtime.base3_linear import Base3Linear
from scripts.convert_to_base3 import pack_u32_to_base3


def patch_model_to_base3(model: nn.Module) -> int:
    """
    Traverses the model and converts all runtime.Packed layers to Base3Linear layers.
    Returns the count of converted layers.
    """
    converted_count = 0

    # We find all modules that are instances of Packed
    for path, module in model.named_modules():
        # Check module type or class name
        if module.__class__.__name__ == "Packed":
            # Extract weights, scales, biases, block, signs, embedding
            w_u32 = module.weight
            scales = module.scales
            biases = module.biases
            block = module.block
            signs = module.signs
            embedding = module.embedding
            dtype = module.dtype

            # Pack uint32 weight to Base-3
            w_np = np.array(w_u32)
            w_b3_np = pack_u32_to_base3(w_np)
            w_b3_mx = mx.array(w_b3_np)

            b3_layer = Base3Linear(
                w_base3=w_b3_mx,
                scales=scales,
                biases=biases,
                target_u32_shape=w_u32.shape,
                block=block,
                signs=signs,
                embedding=embedding,
                dtype=dtype,
            )

            # Replace in parent
            parts = path.split(".")
            parent = model
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)

            if parts[-1].isdigit():
                parent[int(parts[-1])] = b3_layer
            else:
                setattr(parent, parts[-1], b3_layer)

            converted_count += 1

    return converted_count
