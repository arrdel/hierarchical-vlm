"""
Domain-Specific Expert Modules for Multi-Domain Video Understanding

This module implements specialized experts for different video domains:
- Sports: Action-focused, high-motion content
- Tutorials: Instructional content with step-by-step actions
- News: Caption-focused, dialogue-heavy content
- General: Balanced approach for mixed content

Each domain expert specializes in understanding domain-specific patterns
and can be combined using a router for flexible multi-domain handling.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainExpert(nn.Module):
    """Specialized expert module for a specific domain.
    
    Each expert learns domain-specific features and patterns.
    Can be combined with other experts through routing mechanisms.
    
    Args:
        input_dim: Input feature dimension
        expert_dim: Expert hidden dimension
        output_dim: Output dimension
        dropout: Dropout probability
        domain_name: Name of the domain this expert handles
        specialization_factor: How much to specialize (0=general, 1=fully specialized)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        expert_dim: int = 1024,
        output_dim: int = 768,
        dropout: float = 0.1,
        domain_name: str = "general",
        specialization_factor: float = 0.8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.output_dim = output_dim
        self.domain_name = domain_name
        self.specialization_factor = specialization_factor
        
        # Down-projection layer
        self.down_proj = nn.Linear(input_dim, expert_dim)
        self.down_bn = nn.BatchNorm1d(expert_dim)
        
        # Domain-specific processing layers
        self.expert_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, expert_dim),
        )
        
        # Up-projection layer
        self.up_proj = nn.Linear(expert_dim, output_dim)
        self.up_bn = nn.BatchNorm1d(output_dim)
        
        # Gating mechanism for specialization
        self.specialization_gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, apply_gate: bool = True) -> torch.Tensor:
        """Process input through domain expert.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)
            apply_gate: Whether to apply specialization gating
        
        Returns:
            Expert output of shape matching input
        """
        # Handle both 2D and 3D inputs
        is_3d = x.dim() == 3
        if is_3d:
            batch_size, seq_len, _ = x.shape
            x_orig = x
            x = x.view(-1, self.input_dim)
        else:
            x_orig = x
        
        # Down-projection
        z = self.down_proj(x)
        z = self.down_bn(z)
        
        # Domain-specific processing
        z = self.expert_layers(z)
        
        # Up-projection
        out = self.up_proj(z)
        out = self.up_bn(out)
        
        # Apply specialization gating
        if apply_gate:
            gate = self.specialization_gate(x)  # (batch * seq_len, 1)
            gate = gate * self.specialization_factor
            out = out * gate + x_orig.view(-1, self.input_dim) * (1 - gate)
        
        # Reshape back if needed
        if is_3d:
            out = out.view(batch_size, seq_len, self.output_dim)
        
        return out


class MultiDomainAdapter(nn.Module):
    """Adapter that routes input through multiple domain experts.
    
    Uses a learned routing mechanism to combine outputs from multiple
    domain experts based on input characteristics.
    
    Args:
        input_dim: Input dimension
        domains: List of domain names to create experts for
        expert_dim: Hidden dimension for experts
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        domains: Optional[List[str]] = None,
        expert_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.domains = domains or ['sports', 'tutorials', 'news', 'general']
        self.num_experts = len(self.domains)
        
        # Create a domain expert for each domain
        self.experts = nn.ModuleDict({
            domain: DomainExpert(
                input_dim=input_dim,
                expert_dim=expert_dim,
                output_dim=input_dim,
                dropout=dropout,
                domain_name=domain,
                specialization_factor=0.8
            )
            for domain in self.domains
        })
        
        # Router network for computing expert weights
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Route input through domain experts.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)
            return_routing: Whether to return routing weights
        
        Returns:
            Tuple of (output, routing_info)
            - output: Combined expert outputs
            - routing_info: Optional dict with expert routing weights
        """
        # Handle 3D inputs
        is_3d = x.dim() == 3
        if is_3d:
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, self.input_dim)
        else:
            x_flat = x
        
        # Compute routing weights
        router_weights = self.router(x_flat)  # (batch * seq_len, num_experts)
        
        # Process through all experts
        expert_outputs = []
        for domain in self.domains:
            expert_out = self.experts[domain](x)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs
        if is_3d:
            expert_outputs = [out.view(-1, self.input_dim) for out in expert_outputs]
        expert_stack = torch.stack(expert_outputs, dim=1)  # (batch * seq_len, num_experts, input_dim)
        
        # Weight outputs by router
        router_weights = router_weights.unsqueeze(-1)  # (batch * seq_len, num_experts, 1)
        weighted_outputs = expert_stack * router_weights
        output = weighted_outputs.sum(dim=1)  # (batch * seq_len, input_dim)
        
        # Reshape back to 3D if needed
        if is_3d:
            output = output.view(batch_size, seq_len, self.input_dim)
        
        routing_info = None
        if return_routing:
            routing_info = {
                domain: router_weights_domain
                for domain, router_weights_domain in zip(
                    self.domains,
                    router_weights.squeeze(-1).unbind(dim=1)
                )
            }
        
        return output, routing_info


class DomainRouter(nn.Module):
    """Intelligent router for selecting appropriate domain experts.
    
    Analyzes input characteristics and routes to most appropriate
    domain experts for better specialization.
    
    Args:
        input_dim: Input dimension
        domains: List of domain names
        hidden_dim: Hidden dimension for router network
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        domains: Optional[List[str]] = None,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.domains = domains or ['sports', 'tutorials', 'news', 'general']
        self.num_domains = len(self.domains)
        
        # Feature extractor for routing decisions
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Linear(hidden_dim // 2, self.num_domains)
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Domain-specific routing (soft routing)
        self.soft_router = nn.Linear(hidden_dim // 2, self.num_domains)
    
    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Route input to domain experts.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)
            return_confidence: Whether to return confidence scores
        
        Returns:
            Dictionary with:
            - 'hard_routing': One-hot routing assignment
            - 'soft_routing': Soft (weighted) routing assignment
            - 'domain_logits': Raw classification logits
            - 'confidence': Optional confidence scores
        """
        # Handle 3D inputs
        is_3d = x.dim() == 3
        if is_3d:
            x = x.mean(dim=1)  # Average over sequence
        
        # Extract features for routing
        features = self.feature_extractor(x)
        
        # Get domain logits
        domain_logits = self.domain_classifier(features)  # (batch, num_domains)
        
        # Hard routing: one-hot assignment to most likely domain
        hard_routing = F.one_hot(
            domain_logits.argmax(dim=-1),
            num_classes=self.num_domains
        ).float()  # (batch, num_domains)
        
        # Soft routing: weighted assignment
        soft_routing = F.softmax(self.soft_router(features), dim=-1)  # (batch, num_domains)
        
        routing_info = {
            'hard_routing': hard_routing,
            'soft_routing': soft_routing,
            'domain_logits': domain_logits,
            'domains': self.domains
        }
        
        # Add confidence if requested
        if return_confidence:
            confidence = self.confidence_estimator(features)
            routing_info['confidence'] = confidence
        
        return routing_info


class SpecializedTransformer(nn.Module):
    """Transformer block with domain specialization.
    
    Combines multi-head attention with domain expertise for
    specialized video understanding.
    
    Args:
        input_dim: Input dimension
        num_heads: Number of attention heads
        domains: List of domain names
        hidden_dim: Hidden dimension for domain experts
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_heads: int = 8,
        domains: Optional[List[str]] = None,
        hidden_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(input_dim)
        
        # Multi-domain adapter
        self.domain_adapter = MultiDomainAdapter(
            input_dim=input_dim,
            domains=domains,
            expert_dim=hidden_dim,
            dropout=dropout
        )
        self.domain_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention and domain specialization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.attn_norm(x)
        
        # Domain specialization
        domain_out, _ = self.domain_adapter(x, return_routing=False)
        x = x + domain_out
        x = self.domain_norm(x)
        
        return x
