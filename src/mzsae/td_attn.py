"""
Backward-compatibility re-export shim for TD-Attn policy and veto.
Canonical implementation located in mzsae.core.td_policy and mzsae.core.veto.
"""

from .core.td_policy import TelemetryRingBuffer, TDAttnPolicy
from .core.veto import DirectionalVeto

__all__ = [
    "TelemetryRingBuffer",
    "TDAttnPolicy",
    "DirectionalVeto",
]
