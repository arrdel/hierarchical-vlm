"""
Adaptive Token Merging with Motion and Saliency Fusion

This module implements adaptive token merging that combines motion detection
(optical flow) and saliency analysis to intelligently compress video token
sequences while preserving important information.

Key Components:
- TokenSimilarity: Measures similarity between tokens for merging decisions
- AdaptiveTokenMerger: Main merger combining motion + saliency for compression
- TemporalMergeScheduler: Plans merging across time dimension

References:
    "Token Merging for Fast Stable Diffusion"
    https://arxiv.org/abs/2303.17604
"""

from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenSimilarity(nn.Module):
    """Compute similarity between tokens for merging decisions.
    
    Uses cosine similarity to identify tokens that can be safely merged
    without losing important information.
    
    Args:
        similarity_threshold: Minimum similarity to consider tokens mergeable
        aggregation_method: How to combine similar tokens ('mean', 'weighted_mean')
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.9,
        aggregation_method: str = 'weighted_mean'
    ):
        super().__init__()
        
        self.similarity_threshold = similarity_threshold
        self.aggregation_method = aggregation_method
        
        assert aggregation_method in ['mean', 'weighted_mean', 'max']
    
    def forward(
        self,
        tokens: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute token similarity and merging decisions.
        
        Args:
            tokens: Token features of shape (batch, num_tokens, feature_dim)
            importance_scores: Optional importance scores for each token
        
        Returns:
            Tuple of:
            - similarity_matrix: Pairwise token similarity (batch, num_tokens, num_tokens)
            - merge_mask: Binary mask for which tokens to merge (batch, num_tokens)
        """
        batch_size, num_tokens, feature_dim = tokens.shape
        device = tokens.device
        
        # Normalize tokens for cosine similarity
        tokens_norm = F.normalize(tokens, dim=-1)
        
        # Compute pairwise cosine similarity
        similarity = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))
        
        # Zero out self-similarity diagonal
        similarity = similarity * (1 - torch.eye(num_tokens, device=device))
        
        # Compute merge mask based on similarity
        merge_mask = (similarity > self.similarity_threshold).float()
        
        # If importance scores provided, weight them
        if importance_scores is not None:
            # Low importance tokens are more likely to be merged
            importance_expanded = importance_scores.unsqueeze(2)
            merge_mask = merge_mask * (1 - importance_expanded)
        
        return similarity, merge_mask
    
    def get_merge_groups(
        self,
        merge_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify groups of tokens to merge together.
        
        Args:
            merge_mask: Binary merge mask
        
        Returns:
            Tuple of:
            - group_ids: Group assignment for each token
            - group_sizes: Size of each group
        """
        batch_size, num_tokens = merge_mask.shape[:2]
        
        # Use connected components to find merge groups
        group_ids = torch.zeros(batch_size, num_tokens, dtype=torch.long)
        
        for b in range(batch_size):
            mask = merge_mask[b]
            group_id = 0
            assigned = torch.zeros(num_tokens, dtype=torch.bool)
            
            for i in range(num_tokens):
                if not assigned[i]:
                    # Start new group
                    group = torch.zeros(num_tokens, dtype=torch.bool)
                    group[i] = True
                    
                    # BFS to find all connected tokens
                    queue = [i]
                    while queue:
                        current = queue.pop(0)
                        for j in range(num_tokens):
                            if mask[current, j] > 0.5 and not assigned[j]:
                                group[j] = True
                                queue.append(j)
                    
                    # Assign group id
                    group_ids[b, group] = group_id
                    assigned[group] = True
                    group_id += 1
        
        # Compute group sizes
        group_sizes = torch.zeros_like(group_ids).float()
        for b in range(batch_size):
            for i in range(num_tokens):
                group_id = group_ids[b, i]
                group_sizes[b, i] = (group_ids[b] == group_id).float().sum()
        
        return group_ids, group_sizes


class AdaptiveTokenMerger(nn.Module):
    """Merge tokens adaptively using motion and saliency.
    
    Combines optical flow (motion) and saliency analysis to determine
    which tokens should be preserved and which can be merged.
    
    Args:
        height: Spatial height in patches
        width: Spatial width in patches
        num_heads: Number of attention heads
        motion_scale: Scaling factor for motion importance
        saliency_scale: Scaling factor for saliency importance
        target_compression_ratio: Target overall compression ratio
    """
    
    def __init__(
        self,
        height: int = 14,
        width: int = 14,
        num_heads: int = 12,
        motion_scale: float = 0.5,
        saliency_scale: float = 0.5,
        target_compression_ratio: float = 0.5
    ):
        super().__init__()
        
        self.height = height
        self.width = width
        self.num_heads = num_heads
        self.motion_scale = motion_scale
        self.saliency_scale = saliency_scale
        self.target_compression_ratio = target_compression_ratio
        
        self.similarity_module = TokenSimilarity()
    
    def forward(
        self,
        tokens: torch.Tensor,
        motion_magnitude: Optional[torch.Tensor] = None,
        saliency_map: Optional[torch.Tensor] = None,
        compression_ratio: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Merge tokens based on motion and saliency.
        
        Args:
            tokens: Token features (batch, num_tokens, feature_dim)
            motion_magnitude: Motion magnitude map (batch, 1, height, width)
            saliency_map: Saliency map (batch, 1, height, width)
            compression_ratio: Per-patch compression ratio (batch, height, width)
        
        Returns:
            Tuple of:
            - merged_tokens: Merged tokens (batch, merged_num_tokens, feature_dim)
            - merge_info: Dictionary with merge statistics
        """
        batch_size, num_tokens, feature_dim = tokens.shape
        device = tokens.device
        
        # Compute importance scores from motion and saliency
        importance_scores = self._compute_importance_scores(
            motion_magnitude, saliency_map, compression_ratio, num_tokens
        )
        
        # Get token similarity and merge decisions
        similarity, merge_mask = self.similarity_module(tokens, importance_scores)
        
        # Merge tokens based on importance and similarity
        merged_tokens, merge_indices = self._merge_tokens_by_importance(
            tokens, importance_scores, merge_mask
        )
        
        # Compile merge information
        merge_info = {
            'original_tokens': num_tokens,
            'merged_tokens': merged_tokens.shape[1],
            'compression_ratio': merged_tokens.shape[1] / num_tokens,
            'importance_scores': importance_scores,
            'similarity_matrix': similarity,
            'merge_indices': merge_indices
        }
        
        return merged_tokens, merge_info
    
    def _compute_importance_scores(
        self,
        motion_magnitude: Optional[torch.Tensor],
        saliency_map: Optional[torch.Tensor],
        compression_ratio: Optional[torch.Tensor],
        num_tokens: int
    ) -> torch.Tensor:
        """Compute per-token importance scores.
        
        Args:
            motion_magnitude: Motion magnitude (batch, 1, height, width)
            saliency_map: Saliency map (batch, 1, height, width)
            compression_ratio: Compression ratio (batch, height, width)
            num_tokens: Total number of tokens
        
        Returns:
            Importance scores for each token (batch, num_tokens)
        """
        batch_size = next(
            x.shape[0] for x in [motion_magnitude, saliency_map, compression_ratio]
            if x is not None
        )
        device = next(
            x.device for x in [motion_magnitude, saliency_map, compression_ratio]
            if x is not None
        )
        
        # Initialize with base importance
        importance = torch.zeros(batch_size, num_tokens, device=device)
        
        # Add motion-based importance
        if motion_magnitude is not None:
            motion_resized = F.adaptive_avg_pool2d(
                motion_magnitude, (self.height, self.width)
            )
            motion_flat = motion_resized.view(batch_size, -1)
            # Normalize to patch level
            motion_per_patch = motion_flat / (motion_flat.max() + 1e-6)
            importance += self.motion_scale * motion_per_patch
        
        # Add saliency-based importance
        if saliency_map is not None:
            saliency_resized = F.adaptive_avg_pool2d(
                saliency_map, (self.height, self.width)
            )
            saliency_flat = saliency_resized.view(batch_size, -1)
            # Normalize
            saliency_per_patch = saliency_flat / (saliency_flat.max() + 1e-6)
            importance += self.saliency_scale * saliency_per_patch
        
        # Add compression ratio guidance
        if compression_ratio is not None:
            ratio_flat = compression_ratio.view(batch_size, -1)
            # Higher compression ratio means higher importance (keep more tokens)
            importance += (1.0 - self.motion_scale - self.saliency_scale) * ratio_flat
        
        # Normalize importance to [0, 1]
        importance_min = importance.min(dim=1, keepdim=True)[0]
        importance_max = importance.max(dim=1, keepdim=True)[0]
        importance = (importance - importance_min) / (importance_max - importance_min + 1e-6)
        
        return importance
    
    def _merge_tokens_by_importance(
        self,
        tokens: torch.Tensor,
        importance_scores: torch.Tensor,
        merge_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge tokens based on importance scores and similarity.
        
        Args:
            tokens: Token features (batch, num_tokens, feature_dim)
            importance_scores: Per-token importance (batch, num_tokens)
            merge_mask: Merge decisions (batch, num_tokens, num_tokens)
        
        Returns:
            Tuple of:
            - merged_tokens: Merged tokens (batch, merged_count, feature_dim)
            - merge_indices: Mapping from merged to original tokens
        """
        batch_size, num_tokens, feature_dim = tokens.shape
        device = tokens.device
        
        # Get merge groups
        group_ids, group_sizes = self.similarity_module.get_merge_groups(merge_mask)
        
        # Merge tokens within groups using weighted averaging
        merged_tokens_list = []
        merge_indices_list = []
        
        for b in range(batch_size):
            num_groups = group_ids[b].max().item() + 1
            merged = []
            indices = []
            
            for group_id in range(num_groups):
                group_mask = group_ids[b] == group_id
                group_tokens = tokens[b, group_mask]  # (group_size, feature_dim)
                group_importance = importance_scores[b, group_mask]  # (group_size,)
                
                # Weight by importance
                weights = group_importance / (group_importance.sum() + 1e-6)
                merged_token = (group_tokens * weights.unsqueeze(1)).sum(dim=0)
                
                merged.append(merged_token)
                indices.append(group_mask.nonzero(as_tuple=True)[0].cpu().numpy())
            
            merged_tokens_list.append(torch.stack(merged) if merged else tokens[b])
            merge_indices_list.append(indices)
        
        # Pad to same length
        max_merged = max(m.shape[0] for m in merged_tokens_list)
        merged_tokens_padded = []
        
        for m in merged_tokens_list:
            if m.shape[0] < max_merged:
                pad = torch.zeros(
                    max_merged - m.shape[0], feature_dim,
                    device=device, dtype=m.dtype
                )
                m = torch.cat([m, pad], dim=0)
            merged_tokens_padded.append(m)
        
        merged_tokens = torch.stack(merged_tokens_padded)
        
        return merged_tokens, merge_indices_list
    
    def get_merge_statistics(
        self,
        merge_info: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute statistics about the merging process.
        
        Args:
            merge_info: Dictionary from forward pass
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'compression_ratio': merge_info['compression_ratio'],
            'tokens_compressed': merge_info['original_tokens'] - merge_info['merged_tokens'],
            'compression_percentage': (1.0 - merge_info['compression_ratio']) * 100
        }
        
        # Compute importance statistics
        importance = merge_info['importance_scores']
        stats['mean_importance'] = importance.mean().item()
        stats['max_importance'] = importance.max().item()
        stats['min_importance'] = importance.min().item()
        
        return stats


class TemporalMergeScheduler(nn.Module):
    """Schedule token merging across temporal dimension.
    
    Determines merging intensity for each frame based on temporal stability
    and motion patterns.
    
    Args:
        num_frames: Number of frames in sequence
        base_merge_ratio: Base merging ratio
    """
    
    def __init__(
        self,
        num_frames: int = 32,
        base_merge_ratio: float = 0.5
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.base_merge_ratio = base_merge_ratio
    
    def forward(
        self,
        motion_over_time: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-frame merge ratios based on temporal motion.
        
        Args:
            motion_over_time: Motion magnitude over time (batch, num_frames)
        
        Returns:
            Per-frame merge ratios (batch, num_frames)
        """
        batch_size, num_frames = motion_over_time.shape
        device = motion_over_time.device
        
        # Normalize motion
        motion_norm = F.normalize(motion_over_time, dim=1)
        
        # High motion frames get less merging (lower ratio)
        # Low motion frames get more merging (higher ratio)
        merge_ratios = self.base_merge_ratio + (1.0 - self.base_merge_ratio) * (1.0 - motion_norm)
        
        # Apply temporal smoothing to avoid abrupt changes
        merge_ratios_smooth = self._smooth_ratios(merge_ratios)
        
        return merge_ratios_smooth
    
    def _smooth_ratios(self, ratios: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to merge ratios.
        
        Args:
            ratios: Merge ratios over time
        
        Returns:
            Smoothed merge ratios
        """
        # Use simple moving average
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, device=ratios.device) / kernel_size
        
        # Reshape for 1D convolution
        ratios_expanded = ratios.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, num_frames)
        
        # Apply padding for same-size output
        ratios_padded = F.pad(ratios_expanded, (kernel_size//2, kernel_size//2))
        
        # Convolve
        smoothed = F.conv2d(ratios_padded, kernel)
        
        return smoothed.squeeze(1).squeeze(1)
    
    def get_schedule(self) -> torch.Tensor:
        """Get default merge schedule for num_frames.
        
        Returns:
            Default merge ratio schedule
        """
        # Create a schedule that adapts to typical video patterns
        # (e.g., more compression in middle frames)
        schedule = torch.linspace(0.3, 0.8, self.num_frames)
        
        # Apply slight bump for middle frames
        mid = self.num_frames // 2
        for i in range(max(0, mid-2), min(self.num_frames, mid+3)):
            schedule[i] = schedule[i] * 0.8
        
        return schedule
