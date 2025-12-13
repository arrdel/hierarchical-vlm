"""
Sparse Attention implementations for efficient long-range modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StridedAttention(nn.Module):
    """
    Strided attention pattern for efficient processing of long sequences.
    
    Each token attends to every k-th token instead of all tokens.
    This reduces complexity from O(n²) to O(n²/stride).
    
    Args:
        stride: Stride factor for sampling keys/values (default: 4)
        dim: Model dimension (default: 768)
        num_heads: Number of attention heads (default: 12)
        dropout: Attention dropout rate (default: 0.0)
    """
    
    def __init__(self, stride: int = 4, dim: int = 768, num_heads: int = 12, 
                 dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.stride = stride
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply strided attention to input.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (batch, seq_len, seq_len_strided)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to q, k, v and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Stride the key and value tensors for efficiency
        k_strided = k[:, :, ::self.stride, :]
        v_strided = v[:, :, ::self.stride, :]
        
        # Compute attention scores
        scores = torch.matmul(q, k_strided.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Compute attention weights and apply dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v_strided)
        
        # Reshape back to (batch, seq_len, dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output


class LocalGlobalAttention(nn.Module):
    """
    Local + Global attention pattern for efficient context modeling.
    
    Combines:
    - Local attention: Each token attends to nearby tokens within a window
    - Global attention: Selected representative tokens attend globally
    
    This balances efficiency (local window) with context (global representatives).
    
    Args:
        local_window: Window size for local attention (default: 64)
        num_global_tokens: Number of global representative tokens (default: 4)
        dim: Model dimension (default: 768)
        num_heads: Number of attention heads (default: 12)
        dropout: Attention dropout rate (default: 0.0)
    """
    
    def __init__(self, local_window: int = 64, num_global_tokens: int = 4, 
                 dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.local_window = local_window
        self.num_global_tokens = num_global_tokens
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Learnable parameters for selecting global tokens
        self.global_token_indices = nn.Parameter(torch.randn(num_global_tokens))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply local + global attention to input.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute local attention
        local_output = self._compute_local_attention(q, k, v, mask)
        
        # Compute global attention
        global_output = self._compute_global_attention(q, k, v, mask)
        
        # Combine local and global (weighted average)
        output = 0.5 * local_output + 0.5 * global_output
        
        # Reshape back to (batch, seq_len, dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output
    
    def _compute_local_attention(self, q: torch.Tensor, k: torch.Tensor, 
                                  v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute local windowed attention.
        
        Each token attends to tokens within a fixed window around it.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Create a mask for local window
        # Using 1D convolution-like approach for efficiency
        local_mask = self._create_local_window_mask(seq_len, self.local_window)
        local_mask = local_mask.to(q.device)
        
        # Compute full attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply local window mask
        if local_mask is not None:
            scores = scores.masked_fill(~local_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)  # Handle -inf softmax
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def _compute_global_attention(self, q: torch.Tensor, k: torch.Tensor, 
                                   v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute global attention using selected representative tokens.
        
        Selected tokens (based on saliency) attend globally to all tokens.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Select global representative token indices
        # Use softmax of learned parameters to select top num_global_tokens
        global_indices = torch.topk(
            F.softmax(self.global_token_indices, dim=0),
            k=min(self.num_global_tokens, seq_len),
            dim=0
        )[1]
        
        # Extract global tokens
        k_global = k[:, :, global_indices, :]  # (batch, num_heads, num_global, head_dim)
        v_global = v[:, :, global_indices, :]  # (batch, num_heads, num_global, head_dim)
        
        # Compute attention to global tokens
        scores = torch.matmul(q, k_global.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided
        if mask is not None:
            # Reduce mask to global tokens
            mask_global = mask[:, :, global_indices]
            scores = scores.masked_fill(mask_global, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute global output and broadcast to full sequence
        global_out = torch.matmul(attn_weights, v_global)  # (batch, num_heads, seq_len, head_dim)
        
        return global_out
    
    @staticmethod
    def _create_local_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
        """
        Create a mask for local windowed attention.
        
        Shape: (seq_len, seq_len) where mask[i, j] = True if |i - j| <= window_size//2
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True
        return mask


class CrossMemoryAttention(nn.Module):
    """
    Cross-attention between memory tokens and local features.
    
    Instead of simple concatenation, uses cross-attention to intelligently merge
    long-term memory with current local visual features. This allows the model to
    selectively attend to relevant historical information.
    
    Args:
        dim: Model dimension (default: 768)
        num_heads: Number of attention heads (default: 12)
        dropout: Attention dropout rate (default: 0.0)
    """
    
    def __init__(self, dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5
        
    def forward(self, local_features: torch.Tensor, memory_tokens: torch.Tensor,
                residual: bool = True) -> torch.Tensor:
        """
        Apply cross-attention between local features and memory tokens.
        
        Args:
            local_features: Local visual features (batch, local_seq, dim)
            memory_tokens: Long-term memory tokens (batch, memory_seq, dim)
            residual: Whether to apply residual connection (default: True)
            
        Returns:
            Fused representation incorporating both local and memory information
            (batch, local_seq, dim)
        """
        batch_size, local_seq, _ = local_features.shape
        _, memory_seq, _ = memory_tokens.shape
        
        # Project queries from local features
        q = self.q_proj(local_features)
        q = q.view(batch_size, local_seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project keys and values from memory tokens
        k = self.k_proj(memory_tokens)
        v = self.v_proj(memory_tokens)
        k = k.view(batch_size, memory_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, memory_seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross-attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        cross_output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch, local_seq, dim)
        cross_output = cross_output.transpose(1, 2).contiguous()
        cross_output = cross_output.view(batch_size, local_seq, self.dim)
        cross_output = self.out_proj(cross_output)
        
        # Apply residual connection
        if residual:
            output = cross_output + local_features
        else:
            output = cross_output
        
        return output
    
    def fuse_with_memory(self, local_features: torch.Tensor, memory_tokens: torch.Tensor,
                        fusion_ratio: float = 0.5) -> torch.Tensor:
        """
        Fuse local features with memory using cross-attention and interpolation.
        
        Args:
            local_features: Local visual features (batch, local_seq, dim)
            memory_tokens: Long-term memory tokens (batch, memory_seq, dim)
            fusion_ratio: Ratio for blending local features and cross-attended output (0-1)
                         0.0 = pure local, 1.0 = pure memory-fused
            
        Returns:
            Fused representation (batch, local_seq, dim)
        """
        # Get cross-attended output
        cross_attended = self(local_features, memory_tokens, residual=False)
        
        # Blend with original local features
        fused = fusion_ratio * cross_attended + (1 - fusion_ratio) * local_features
        
        return fused


class HierarchicalAttentionBlock(nn.Module):
    """
    Complete attention block combining sparse, local+global, and cross-memory attention.
    
    This module provides a high-level interface for using different attention mechanisms
    in sequence or as alternatives.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        attention_type: Type of attention ('strided', 'local_global', or 'cross_memory')
        dropout: Attention dropout rate
        **kwargs: Additional arguments for specific attention types
    """
    
    def __init__(self, dim: int = 768, num_heads: int = 12, 
                 attention_type: str = 'local_global', dropout: float = 0.1, **kwargs):
        super().__init__()
        
        self.attention_type = attention_type
        
        if attention_type == 'strided':
            stride = kwargs.get('stride', 4)
            self.attention = StridedAttention(
                stride=stride, dim=dim, num_heads=num_heads, dropout=dropout
            )
        elif attention_type == 'local_global':
            local_window = kwargs.get('local_window', 64)
            num_global = kwargs.get('num_global_tokens', 4)
            self.attention = LocalGlobalAttention(
                local_window=local_window, num_global_tokens=num_global,
                dim=dim, num_heads=num_heads, dropout=dropout
            )
        elif attention_type == 'cross_memory':
            self.attention = CrossMemoryAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Feed-forward network
        hidden_dim = int(dim * 4)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention block.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            memory: Optional memory tokens for cross-attention
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Self-attention or cross-attention
        if self.attention_type == 'cross_memory' and memory is not None:
            attn_out = self.attention(x, memory)
        else:
            attn_out = self.attention(x)
        
        # Residual connection and layer norm
        x = x + attn_out
        x = self.norm1(x)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x
