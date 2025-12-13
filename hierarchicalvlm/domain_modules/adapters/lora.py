"""
LoRA (Low-Rank Adaptation) Adapters for Parameter-Efficient Fine-Tuning

This module implements Low-Rank Adaptation for efficient fine-tuning of large models.
LoRA decomposes weight updates into low-rank factors, reducing trainable parameters.

Key Components:
- LinearLoRA: Low-rank adaptation for linear layers
- AttentionLoRA: Low-rank adaptation for multi-head attention
- LoRAAdapter: Complete LoRA adapter module
- LoRALayerWrapper: Wrapper for adding LoRA to existing layers

Reference:
    "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    rank: int = 8
    """Low-rank dimension for LoRA decomposition."""
    
    alpha: float = 16.0
    """Scaling factor for LoRA updates."""
    
    dropout: float = 0.05
    """Dropout probability for LoRA layers."""
    
    target_modules: Optional[List[str]] = None
    """Module names to apply LoRA to."""
    
    lora_only: bool = False
    """If True, freeze base model weights and only train LoRA."""


class LinearLoRA(nn.Module):
    """Low-Rank Adaptation for Linear layers.
    
    Decomposes weight updates into low-rank factors:
        ΔW = BA where B ∈ ℝ^(d_out x rank), A ∈ ℝ^(rank x d_in)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Low-rank dimension for factorization
        alpha: Scaling factor for LoRA output
        dropout: Dropout probability
        use_bias: Whether to include bias term
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        use_bias: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Scaling factor: alpha / rank normalizes the LoRA contribution
        self.scaling = alpha / rank
        
        # Low-rank update matrices
        # A: projects from in_features to rank
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        # B: projects from rank to out_features
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Optional bias for output
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize LoRA weights using Kaiming uniform for A, zeros for B."""
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation.
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            LoRA output of shape (..., out_features)
        """
        # Apply dropout, project to rank, then back to output space
        lora_out = self.lora_b(self.dropout_layer(self.lora_a(x)))
        # Scale the LoRA contribution
        lora_out = lora_out * self.scaling
        
        # Add bias if present
        if self.bias is not None:
            lora_out = lora_out + self.bias
        
        return lora_out


class AttentionLoRA(nn.Module):
    """Low-Rank Adaptation for Multi-Head Attention layers.
    
    Applies LoRA to Query, Key, Value projections in attention.
    
    Args:
        hidden_dim: Hidden dimension of attention
        num_heads: Number of attention heads
        rank: Low-rank dimension for LoRA
        alpha: Scaling factor for LoRA
        dropout: Dropout probability
        adapt_q: Apply LoRA to query projections
        adapt_k: Apply LoRA to key projections
        adapt_v: Apply LoRA to value projections
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        adapt_q: bool = True,
        adapt_k: bool = True,
        adapt_v: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.rank = rank
        self.adapt_q = adapt_q
        self.adapt_k = adapt_k
        self.adapt_v = adapt_v
        
        # LoRA adapters for each projection
        if adapt_q:
            self.lora_q = LinearLoRA(hidden_dim, hidden_dim, rank, alpha, dropout)
        
        if adapt_k:
            self.lora_k = LinearLoRA(hidden_dim, hidden_dim, rank, alpha, dropout)
        
        if adapt_v:
            self.lora_v = LinearLoRA(hidden_dim, hidden_dim, rank, alpha, dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply LoRA to attention projections.
        
        Args:
            q: Query tensor of shape (batch, seq_len, hidden_dim)
            k: Key tensor of shape (batch, seq_len, hidden_dim)
            v: Value tensor of shape (batch, seq_len, hidden_dim)
        
        Returns:
            Tuple of (q_lora, k_lora, v_lora)
        """
        q_lora = self.lora_q(q) if self.adapt_q else q
        k_lora = self.lora_k(k) if self.adapt_k else k
        v_lora = self.lora_v(v) if self.adapt_v else v
        
        return q_lora, k_lora, v_lora


class LoRAAdapter(nn.Module):
    """Complete LoRA adapter module for model fine-tuning.
    
    Can wrap an existing model and add LoRA adapters to specified layers.
    Supports selective adaptation of query/key/value in attention layers.
    
    Args:
        dim: Model dimension (hidden size)
        rank: LoRA rank
        alpha: Scaling factor
        dropout: Dropout probability
        expansion_factor: Factor for hidden dimension in adapter
    """
    
    def __init__(
        self,
        dim: int = 768,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        expansion_factor: float = 4.0
    ):
        super().__init__()
        
        self.dim = dim
        self.rank = rank
        self.alpha = alpha
        
        # Down-projection with LoRA
        hidden_dim = int(dim * expansion_factor)
        self.down = LinearLoRA(dim, hidden_dim, rank=rank, alpha=alpha, dropout=dropout)
        
        # Activation
        self.act = nn.GELU()
        
        # Up-projection back to original dimension with LoRA
        self.up = LinearLoRA(hidden_dim, dim, rank=rank, alpha=alpha, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adapter.
        
        Args:
            x: Input tensor of shape (..., dim)
        
        Returns:
            Adapted tensor of same shape as input
        """
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x + residual
    
    def get_lora_params(self) -> List[torch.nn.Parameter]:
        """Get all LoRA parameters for training."""
        params = []
        params.extend([self.down.lora_a.weight, self.down.lora_b.weight])
        params.extend([self.up.lora_a.weight, self.up.lora_b.weight])
        if self.down.bias is not None:
            params.append(self.down.bias)
        if self.up.bias is not None:
            params.append(self.up.bias)
        return params


class LoRALayerWrapper(nn.Module):
    """Wrapper to add LoRA to existing layers without modifying them.
    
    This allows applying LoRA on top of frozen base model weights.
    
    Args:
        layer: Original layer to wrap
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05
    ):
        super().__init__()
        
        self.layer = layer
        self.rank = rank
        self.alpha = alpha
        
        # Create LoRA for this layer
        if isinstance(layer, nn.Linear):
            self.lora = LinearLoRA(
                layer.in_features,
                layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                use_bias=False
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base layer + LoRA update.
        
        Returns:
            layer_output + lora_update
        """
        # Base layer output
        base_out = self.layer(x)
        
        # LoRA update
        lora_out = self.lora(x)
        
        # Combine: original output + LoRA update
        return base_out + lora_out
