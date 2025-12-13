"""
Efficient Attention Mechanisms for Long Video Understanding

Implements:
- Sparse attention patterns (strided, local + global)
- Cross-attention between memory and local features
- Linear attention variants (Mamba, Performer)
"""

from . import sparse
from . import linear

# Direct imports for convenience
from .sparse import (
    StridedAttention,
    LocalGlobalAttention,
    CrossMemoryAttention,
    HierarchicalAttentionBlock,
)

from .linear import (
    PerformerAttention,
    MambaLayer,
    LinearAttentionBlock,
)

__all__ = [
    # Modules
    "sparse",
    "linear",
    # Sparse attention
    "StridedAttention",
    "LocalGlobalAttention",
    "CrossMemoryAttention",
    "HierarchicalAttentionBlock",
    # Linear attention
    "PerformerAttention",
    "MambaLayer",
    "LinearAttentionBlock",
]
