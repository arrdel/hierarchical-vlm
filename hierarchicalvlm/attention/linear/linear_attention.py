"""
Linear Attention implementations with sub-quadratic complexity.

Includes:
- Performer: FAVOR+ kernel-based linear attention
- Mamba: State Space Model variant for sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PerformerAttention(nn.Module):
    """
    Performer: Fast Transformers with FAVOR+ (Fast Attention Via positive Orthogonal Random Features)
    
    Reduces attention complexity from O(n²) to O(n) using kernel methods and random features.
    
    Key idea: Approximate softmax(Q @ K^T / sqrt(d)) @ V using random feature expansion
    to achieve linear complexity in sequence length.
    
    Args:
        dim: Model dimension (default: 768)
        num_heads: Number of attention heads (default: 12)
        num_random_features: Number of random features for kernel approximation (default: 256)
        kernel_type: Type of kernel ('elu' or 'relu', default: 'elu')
        dropout: Attention dropout rate (default: 0.0)
    """
    
    def __init__(self, dim: int = 768, num_heads: int = 12, 
                 num_random_features: int = 256, kernel_type: str = 'elu',
                 dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_random_features = num_random_features
        self.kernel_type = kernel_type
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5
        
        # Register buffer for random projection matrix
        self.register_buffer(
            "random_matrix",
            torch.randn(self.head_dim, num_random_features)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Performer attention to input.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply kernel feature mapping
        q_prime = self._apply_kernel(q)  # (batch, num_heads, seq_len, num_features)
        k_prime = self._apply_kernel(k)  # (batch, num_heads, seq_len, num_features)
        
        # Linear attention computation using kernel trick
        # Instead of: softmax(Q @ K^T) @ V
        # We compute: (Q_prime @ K_prime^T) @ V / normalization
        
        # Compute K_prime^T @ V for all tokens
        kv = torch.matmul(k_prime.transpose(-2, -1), v)  # (batch, num_heads, num_features, head_dim)
        
        # Compute normalization factor
        k_sum = k_prime.sum(dim=2, keepdim=True)  # (batch, num_heads, 1, num_features)
        
        # Apply mask to k_sum if provided
        if mask is not None:
            k_prime_masked = k_prime.clone()
            k_prime_masked = k_prime_masked.masked_fill(mask.unsqueeze(1), 0.0)
            k_sum = k_prime_masked.sum(dim=2, keepdim=True)
        
        # Compute output: Q_prime @ (K_prime^T @ V) / (Q_prime @ K_sum)
        output = torch.matmul(q_prime, kv)  # (batch, num_heads, seq_len, head_dim)
        normalizer = torch.matmul(q_prime, k_sum.transpose(-2, -1))  # (batch, num_heads, seq_len, 1)
        normalizer = torch.clamp(normalizer, min=1e-6)
        
        output = output / normalizer
        
        # Reshape back to (batch, seq_len, dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output
    
    def _apply_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel feature mapping to input.
        
        Maps input using random features: phi(x) = exp(x @ w + b) or similar
        
        Args:
            x: Input tensor (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Feature-mapped tensor (batch, num_heads, seq_len, num_features)
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Project using random matrix
        x_proj = torch.matmul(x, self.random_matrix)  # (batch, num_heads, seq_len, num_features)
        
        # Apply non-linearity based on kernel type
        if self.kernel_type == 'elu':
            # ELU kernel: elu(x) + 1 to ensure positivity
            features = F.elu(x_proj) + 1.0
        elif self.kernel_type == 'relu':
            # ReLU kernel
            features = F.relu(x_proj)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Normalize features for stability
        features = features / math.sqrt(self.num_random_features)
        
        return features


class MambaLayer(nn.Module):
    """
    Mamba: Linear-Time Sequence Modeling with Selective State Space Models
    
    State space model (SSM) variant that achieves linear complexity through
    selective state updates. More efficient than standard attention for long sequences.
    
    Args:
        dim: Model dimension (default: 768)
        state_size: Dimension of the internal state (default: 16)
        expand_factor: Expansion factor for intermediate dimension (default: 2)
        dt_init: Initialization method for dt parameter ('constant' or 'random')
        dt_init_floor: Minimum value for dt initialization
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, dim: int = 768, state_size: int = 16, expand_factor: float = 2.0,
                 dt_init: str = 'constant', dt_init_floor: float = 1e-4, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.state_size = state_size
        self.expand_factor = expand_factor
        
        # Expand dimension for internal computation
        inner_dim = int(dim * expand_factor)
        
        # Input projection
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)
        
        # State space matrices
        # A: state transition matrix (initialized to provide stable dynamics)
        A = torch.arange(1, state_size + 1, dtype=torch.float32).reshape(1, 1, state_size)
        self.register_buffer('A', -A)  # Negative for stable dynamics
        self.register_buffer('A_log', torch.log(torch.abs(self.A)))
        
        # B: input projection (learned)
        self.B = nn.Parameter(torch.randn(1, 1, state_size) * 0.01)
        
        # C: output projection (learned)
        self.C = nn.Parameter(torch.randn(1, 1, state_size) * 0.01)
        
        # dt: discretization step (learned)
        dt_init_tensor = torch.empty(1, inner_dim, state_size)
        if dt_init == 'constant':
            nn.init.constant_(dt_init_tensor, 0.1)
        elif dt_init == 'random':
            nn.init.uniform_(dt_init_tensor, dt_init_floor, 0.1)
        else:
            raise ValueError(f"Unknown dt_init: {dt_init}")
        self.dt = nn.Parameter(dt_init_tensor)
        
        # Output projection
        self.out_proj = nn.Linear(inner_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Mamba layer to input.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # Project to inner dimension
        projected = self.in_proj(x)
        x_proj, gates = projected.chunk(2, dim=-1)
        
        # Initialize state
        h = torch.zeros(batch_size, x_proj.shape[-1], self.state_size, 
                       device=x.device, dtype=x.dtype)
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x_proj[:, t, :]  # (batch, inner_dim)
            
            # Compute A_tilde = exp(dt * A)
            dt_t = self.dt.expand(batch_size, -1, -1)  # (batch, inner_dim, state_size)
            A_tilde = torch.exp(dt_t * torch.exp(self.A_log))  # (batch, inner_dim, state_size)
            
            # Update state: h_new = A_tilde * h + dt * B * x_t
            # B is broadcasted and x_t is projected to state_size
            B_x = torch.matmul(
                x_t.unsqueeze(-1),  # (batch, inner_dim, 1)
                self.B  # (1, 1, state_size)
            )  # (batch, inner_dim, state_size)
            
            h = A_tilde * h + dt_t * B_x
            
            # Compute output: y = C * h + D * x
            y_t = torch.matmul(h, self.C.transpose(-2, -1)).squeeze(-1)  # (batch, inner_dim)
            
            # Apply gate
            y_t = y_t * torch.sigmoid(gates[:, t, :])
            
            outputs.append(y_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, inner_dim)
        
        # Apply activation and output projection
        output = self.act(output)
        output = self.dropout(output)
        output = self.out_proj(output)
        
        return output


class LinearAttentionBlock(nn.Module):
    """
    Complete linear attention block with layer normalization and feed-forward.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads (for Performer)
        attention_type: Type of attention ('performer' or 'mamba')
        dropout: Dropout rate
        **kwargs: Additional arguments for specific attention types
    """
    
    def __init__(self, dim: int = 768, num_heads: int = 12,
                 attention_type: str = 'performer', dropout: float = 0.1, **kwargs):
        super().__init__()
        
        self.attention_type = attention_type
        
        if attention_type == 'performer':
            num_features = kwargs.get('num_random_features', 256)
            kernel_type = kwargs.get('kernel_type', 'elu')
            self.attention = PerformerAttention(
                dim=dim, num_heads=num_heads,
                num_random_features=num_features,
                kernel_type=kernel_type,
                dropout=dropout
            )
        elif attention_type == 'mamba':
            state_size = kwargs.get('state_size', 16)
            expand_factor = kwargs.get('expand_factor', 2.0)
            self.attention = MambaLayer(
                dim=dim, state_size=state_size,
                expand_factor=expand_factor,
                dropout=dropout
            )
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear attention block.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Pre-norm architecture
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out
        
        # Feed-forward
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x
