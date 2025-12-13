"""
Saliency Detection for Motion-Based Token Merging

This module implements visual saliency detection for identifying important
regions in video frames. Saliency is combined with motion to determine which
tokens should be preserved during adaptive token merging.

Key Components:
- EdgeSaliency: Edge-based visual saliency using gradients
- AttentionSaliency: Saliency from transformer attention patterns
- MultiSaliencyFusion: Combines multiple saliency sources
- SaliencyDetector: Main detector combining all components

References:
    "What is and What is not a Salient Object? Learning Salient Object Detector
     by Ensembling Linear Exemplary Regressors"
    https://arxiv.org/abs/1703.02052
"""

from typing import Optional, Tuple, Dict, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeSaliency(nn.Module):
    """Compute saliency using edge/boundary detection.
    
    Identifies salient regions based on strong gradients and edges,
    assuming important objects have distinct boundaries.
    
    Args:
        edge_threshold: Minimum gradient magnitude to consider edge
        smoothing_kernel_size: Size of smoothing kernel
    """
    
    def __init__(
        self,
        edge_threshold: float = 0.1,
        smoothing_kernel_size: int = 5
    ):
        super().__init__()
        
        self.edge_threshold = edge_threshold
        self.smoothing_kernel_size = smoothing_kernel_size
        
        # Pre-define Sobel operators
        self.register_buffer('sobel_x', self._get_sobel_x())
        self.register_buffer('sobel_y', self._get_sobel_y())
        
        # Gaussian smoothing kernel
        self.register_buffer(
            'gaussian_kernel',
            self._create_gaussian_kernel(smoothing_kernel_size)
        )
    
    @staticmethod
    def _get_sobel_x() -> torch.Tensor:
        """Get Sobel X operator."""
        return torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
    
    @staticmethod
    def _get_sobel_y() -> torch.Tensor:
        """Get Sobel Y operator."""
        return torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
    
    def _create_gaussian_kernel(self, size: int) -> torch.Tensor:
        """Create Gaussian kernel for smoothing."""
        kernel = torch.zeros(1, 1, size, size)
        center = size // 2
        sigma = size / 6.0
        
        for y in range(size):
            for x in range(size):
                dy = y - center
                dx = x - center
                kernel[0, 0, y, x] = torch.exp(
                    -torch.tensor((dx*dx + dy*dy) / (2 * sigma*sigma))
                )
        
        kernel = kernel / kernel.sum()
        return kernel
    
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """Compute edge-based saliency.
        
        Args:
            frame: Input frame of shape (batch, channels, height, width)
        
        Returns:
            Edge saliency map of shape (batch, 1, height, width)
        """
        # Convert to grayscale if needed
        if frame.shape[1] == 3:
            gray = 0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3]
        else:
            gray = frame[:, 0:1]
        
        # Compute gradients
        gx = F.conv2d(gray, self.sobel_x.to(frame.device), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(frame.device), padding=1)
        
        # Compute magnitude
        edge_magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        
        # Threshold
        edge_saliency = (edge_magnitude > self.edge_threshold).float()
        
        # Smooth edges
        edge_saliency = F.conv2d(
            edge_saliency,
            self.gaussian_kernel.to(frame.device),
            padding=self.smoothing_kernel_size // 2
        )
        
        # Normalize to [0, 1]
        edge_saliency = edge_saliency / (edge_saliency.max() + 1e-6)
        
        return edge_saliency


class AttentionSaliency(nn.Module):
    """Compute saliency from transformer attention patterns.
    
    Uses attention weights to identify which regions the model focuses on,
    assuming attended regions are more important.
    
    Args:
        num_heads: Number of attention heads (if attention is provided)
        pooling_method: How to aggregate attention ('mean', 'max', 'weighted')
    """
    
    def __init__(
        self,
        num_heads: int = 12,
        pooling_method: str = 'mean'
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.pooling_method = pooling_method
        
        assert pooling_method in ['mean', 'max', 'weighted']
    
    def forward(
        self,
        features: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention-based saliency.
        
        Args:
            features: Feature tensor of shape (batch, num_tokens, feature_dim)
                     or (batch, channels, height, width)
            attention_weights: Optional attention weights of shape
                             (batch, num_heads, num_tokens, num_tokens)
        
        Returns:
            Attention saliency map of shape (batch, 1, height, width)
        """
        # If attention weights are provided, use them
        if attention_weights is not None:
            # Average attention across heads
            attention = attention_weights.mean(dim=1)  # (batch, tokens, tokens)
            
            # Use attention entropy as saliency
            # Tokens that attend to many different places are more important
            attention_entropy = -torch.sum(
                attention * torch.log(attention + 1e-8),
                dim=-1,
                keepdim=True
            )  # (batch, tokens, 1)
            
            saliency = attention_entropy
        else:
            # Use feature norm as saliency proxy
            if features.dim() == 4:
                # Image-like features (batch, channels, height, width)
                saliency = torch.norm(features, dim=1, keepdim=True)
            else:
                # Token-like features (batch, tokens, dim)
                saliency = torch.norm(features, dim=-1, keepdim=True)
        
        # Normalize
        saliency_min = saliency.min()
        saliency_max = saliency.max()
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-6)
        
        # If saliency is 1D (per-token), reshape to spatial
        if saliency.dim() == 3:
            # (batch, tokens, 1) -> reshape to spatial
            batch_size, num_tokens, _ = saliency.shape
            h = w = int(math.sqrt(num_tokens))
            saliency = saliency[:, :h*w, 0].view(batch_size, 1, h, w)
        
        return saliency


class ColorSaliency(nn.Module):
    """Compute saliency based on color distinctiveness.
    
    Identifies regions with unusual colors compared to neighborhood,
    as distinct colors often indicate objects of interest.
    
    Args:
        color_threshold: Threshold for color distinctiveness
        neighborhood_size: Size of local neighborhood for comparison
    """
    
    def __init__(
        self,
        color_threshold: float = 0.15,
        neighborhood_size: int = 5
    ):
        super().__init__()
        
        self.color_threshold = color_threshold
        self.neighborhood_size = neighborhood_size
    
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """Compute color-based saliency.
        
        Args:
            frame: Input frame of shape (batch, 3, height, width)
        
        Returns:
            Color saliency map of shape (batch, 1, height, width)
        """
        batch_size, channels, height, width = frame.shape
        
        # Ensure RGB format
        if channels != 3:
            return torch.zeros(batch_size, 1, height, width, device=frame.device)
        
        # Compute local color mean
        kernel_size = self.neighborhood_size
        pad = kernel_size // 2
        
        # Unfold to get local patches
        frame_unfold = F.unfold(
            frame,
            kernel_size=kernel_size,
            padding=pad
        )  # (batch, 3*kernel_size^2, num_patches)
        
        # Reshape to compute local mean
        frame_unfold = frame_unfold.view(
            batch_size, channels, kernel_size*kernel_size, -1
        )
        local_mean = frame_unfold.mean(dim=2, keepdim=True)  # (batch, 3, 1, num_patches)
        
        # Compute color distinctiveness
        center_color = frame.view(batch_size, channels, -1).unsqueeze(2)
        color_diff = torch.abs(center_color - local_mean).mean(dim=1)  # (batch, num_patches)
        
        # Reshape back to spatial
        color_saliency = color_diff.view(batch_size, 1, height, width)
        
        # Normalize
        color_saliency = color_saliency / (color_saliency.max() + 1e-6)
        
        return color_saliency


class MultiSaliencyFusion(nn.Module):
    """Fuse multiple saliency sources into unified importance map.
    
    Combines edge, attention, and color saliency using learnable weights.
    
    Args:
        fusion_method: How to combine saliency sources ('weighted_sum', 'product', 'max')
        edge_weight: Weight for edge saliency
        attention_weight: Weight for attention saliency
        color_weight: Weight for color saliency
    """
    
    def __init__(
        self,
        fusion_method: str = 'weighted_sum',
        edge_weight: float = 0.4,
        attention_weight: float = 0.4,
        color_weight: float = 0.2
    ):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.edge_weight = edge_weight
        self.attention_weight = attention_weight
        self.color_weight = color_weight
        
        assert fusion_method in ['weighted_sum', 'product', 'max']
        
        # Normalize weights
        total_weight = edge_weight + attention_weight + color_weight
        self.edge_weight /= total_weight
        self.attention_weight /= total_weight
        self.color_weight /= total_weight
    
    def forward(
        self,
        edge_saliency: torch.Tensor,
        attention_saliency: torch.Tensor,
        color_saliency: torch.Tensor
    ) -> torch.Tensor:
        """Fuse multiple saliency maps.
        
        Args:
            edge_saliency: Edge-based saliency (batch, 1, height, width)
            attention_saliency: Attention-based saliency (batch, 1, height, width)
            color_saliency: Color-based saliency (batch, 1, height, width)
        
        Returns:
            Fused saliency map of shape (batch, 1, height, width)
        """
        # Ensure all have same shape
        target_shape = edge_saliency.shape
        attention_saliency = F.interpolate(
            attention_saliency, size=target_shape[-2:], mode='bilinear'
        )
        color_saliency = F.interpolate(
            color_saliency, size=target_shape[-2:], mode='bilinear'
        )
        
        if self.fusion_method == 'weighted_sum':
            fused = (
                self.edge_weight * edge_saliency +
                self.attention_weight * attention_saliency +
                self.color_weight * color_saliency
            )
        
        elif self.fusion_method == 'product':
            fused = (
                edge_saliency ** self.edge_weight *
                attention_saliency ** self.attention_weight *
                color_saliency ** self.color_weight
            )
        
        else:  # max
            combined = torch.stack(
                [edge_saliency, attention_saliency, color_saliency],
                dim=-1
            )
            fused = combined.max(dim=-1)[0]
        
        # Normalize
        fused = fused / (fused.max() + 1e-6)
        
        return fused


class SaliencyDetector(nn.Module):
    """Unified saliency detection combining multiple methods.
    
    Main module that combines edge, attention, and color saliency
    to produce comprehensive importance maps for token merging.
    
    Args:
        use_edge_saliency: Whether to compute edge saliency
        use_attention_saliency: Whether to compute attention saliency
        use_color_saliency: Whether to compute color saliency
        fusion_method: How to combine saliency sources
    """
    
    def __init__(
        self,
        use_edge_saliency: bool = True,
        use_attention_saliency: bool = True,
        use_color_saliency: bool = True,
        fusion_method: str = 'weighted_sum'
    ):
        super().__init__()
        
        self.use_edge = use_edge_saliency
        self.use_attention = use_attention_saliency
        self.use_color = use_color_saliency
        
        if use_edge_saliency:
            self.edge_saliency = EdgeSaliency()
        
        if use_attention_saliency:
            self.attention_saliency = AttentionSaliency()
        
        if use_color_saliency:
            self.color_saliency = ColorSaliency()
        
        # Always use fusion
        self.fusion = MultiSaliencyFusion(fusion_method=fusion_method)
    
    def forward(
        self,
        frame: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute comprehensive saliency map.
        
        Args:
            frame: Input frame of shape (batch, channels, height, width)
            attention_weights: Optional attention weights for attention saliency
            features: Optional features for attention saliency
        
        Returns:
            Tuple of:
            - saliency_map: Combined saliency map (batch, 1, height, width)
            - saliency_components: Dictionary with individual components
        """
        saliency_components = {}
        
        # Compute edge saliency
        if self.use_edge:
            edge_sal = self.edge_saliency(frame)
            saliency_components['edge'] = edge_sal
        else:
            edge_sal = torch.zeros(
                frame.shape[0], 1, frame.shape[2], frame.shape[3],
                device=frame.device
            )
        
        # Compute attention saliency
        if self.use_attention:
            attention_sal = self.attention_saliency(
                features if features is not None else frame,
                attention_weights
            )
            saliency_components['attention'] = attention_sal
        else:
            attention_sal = torch.zeros(
                frame.shape[0], 1, frame.shape[2], frame.shape[3],
                device=frame.device
            )
        
        # Compute color saliency
        if self.use_color:
            color_sal = self.color_saliency(frame)
            saliency_components['color'] = color_sal
        else:
            color_sal = torch.zeros(
                frame.shape[0], 1, frame.shape[2], frame.shape[3],
                device=frame.device
            )
        
        # Fuse all saliency maps
        fused_saliency = self.fusion(edge_sal, attention_sal, color_sal)
        saliency_components['fused'] = fused_saliency
        
        return fused_saliency, saliency_components
    
    def get_saliency_statistics(
        self,
        saliency_map: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute statistics about saliency distribution.
        
        Args:
            saliency_map: Saliency map tensor
        
        Returns:
            Dictionary with saliency statistics
        """
        mean_saliency = saliency_map.mean()
        max_saliency = saliency_map.max()
        min_saliency = saliency_map.min()
        
        # Compute spatial concentration (how concentrated saliency is)
        saliency_std = saliency_map.std()
        
        # Compute entropy (diversity of saliency)
        saliency_flat = saliency_map.view(saliency_map.shape[0], -1)
        saliency_norm = saliency_flat / (saliency_flat.sum(dim=1, keepdim=True) + 1e-6)
        entropy = -(saliency_norm * torch.log(saliency_norm + 1e-8)).sum(dim=1)
        
        return {
            'mean': mean_saliency,
            'max': max_saliency,
            'min': min_saliency,
            'std': saliency_std,
            'entropy': entropy
        }
