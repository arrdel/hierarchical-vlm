"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).
    
    Reduces trainable parameters by decomposing weight updates as:
    W_new = W_original + alpha * (A @ B^T)
    
    where A and B have low rank r.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 8, alpha: float = 1.0, dropout: float = 0.1):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Original weights (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features)
        
        # LoRA decomposition: W_update = A @ B^T
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, ..., in_features)
            
        Returns:
            Output with LoRA applied
        """
        # Original linear transformation
        out = self.linear(x)
        
        # Add LoRA update
        lora_out = torch.matmul(self.dropout(x), self.lora_A)
        lora_out = torch.matmul(lora_out, self.lora_B)
        out = out + self.scaling * lora_out
        
        return out
    
    def get_trainable_parameters(self):
        """Return only LoRA parameters for training."""
        return [self.lora_A, self.lora_B]


class LoRAAdapter(nn.Module):
    """
    Adapter module with LoRA for domain-specific adaptation.
    """
    
    def __init__(self, dim: int = 768, rank: int = 8, 
                 expansion_factor: float = 4.0):
        super().__init__()
        
        self.dim = dim
        self.rank = rank
        
        # Down-projection
        hidden_dim = int(dim * expansion_factor)
        self.down = LoRALinear(dim, hidden_dim, rank=rank)
        
        # Activation
        self.act = nn.GELU()
        
        # Up-projection back to original dimension
        self.up = LoRALinear(hidden_dim, dim, rank=rank)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Adapted tensor
        """
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x + residual
