"""
Linear Attention Mechanisms

Implements:
- Mamba-like state space models
- Performer linear attention
"""

from .linear_attention import (
    PerformerAttention,
    MambaLayer,
    LinearAttentionBlock,
)

__all__ = [
    "PerformerAttention",
    "MambaLayer",
    "LinearAttentionBlock",
]
