"""
Sparse Attention Patterns

Implements:
- Strided attention
- Local + Global attention
- Cross-attention between memory and features
"""

from .sparse_attention import (
    StridedAttention,
    LocalGlobalAttention,
    CrossMemoryAttention,
    HierarchicalAttentionBlock,
)

__all__ = [
    "StridedAttention",
    "LocalGlobalAttention",
    "CrossMemoryAttention",
    "HierarchicalAttentionBlock",
]
