"""
High-Level Public API Facade
MZahran Sparse Attention Engine (MZSAE)
"""

from typing import Optional, Union, Dict, Any
from .config import MZSAEConfig, load_config
from .core.engine import MZSAEEngine
from .backends.dispatcher import get_backend


def create_engine(
    config: Union[str, Dict[str, Any], MZSAEConfig] = "auto",
    **kwargs,
) -> MZSAEEngine:
    """
    Factory helper to instantiate an MZSAEEngine from a config name, path, dict, or object.
    """
    if isinstance(config, str) or isinstance(config, dict):
        cfg = load_config(config)
    else:
        cfg = config

    return MZSAEEngine(config=cfg, **kwargs)
