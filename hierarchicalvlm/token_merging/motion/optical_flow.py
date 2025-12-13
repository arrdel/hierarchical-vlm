"""
Optical Flow computation for motion analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalFlowEstimator(nn.Module):
    """
    Lightweight optical flow estimation for motion quantification.
    
    Uses frame differences and gradient-based approaches for efficiency.
    """
    
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(self, frame_t, frame_t1):
        """
        Estimate optical flow between consecutive frames.
        
        Args:
            frame_t: Frame at time t (batch, 3, height, width)
            frame_t1: Frame at time t+1 (batch, 3, height, width)
            
        Returns:
            flow: Optical flow (batch, 2, height, width)
            motion_magnitude: Motion magnitude map (batch, 1, height, width)
        """
        # Convert to grayscale
        gray_t = self._to_grayscale(frame_t)
        gray_t1 = self._to_grayscale(frame_t1)
        
        # Compute frame difference
        frame_diff = gray_t1 - gray_t
        
        # Compute gradients
        grad_x = self._compute_gradient(gray_t, dim=-1)
        grad_y = self._compute_gradient(gray_t, dim=-2)
        grad_t = frame_diff
        
        # Simple Lucas-Kanade-like flow estimation
        # For efficiency, use mean shift on local patches
        motion_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_t ** 2 + 1e-6)
        
        # Normalize motion magnitude
        motion_magnitude = (motion_magnitude - motion_magnitude.min()) / \
                          (motion_magnitude.max() - motion_magnitude.min() + 1e-6)
        
        return motion_magnitude
    
    def _to_grayscale(self, img):
        """Convert RGB to grayscale."""
        if img.shape[1] == 3:
            return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        return img
    
    def _compute_gradient(self, img, dim=-1):
        """Compute spatial gradient."""
        if dim == -1:
            return img[:, :, :, 1:] - img[:, :, :, :-1]
        elif dim == -2:
            return img[:, :, 1:, :] - img[:, :, :-1, :]
        else:
            raise ValueError("dim must be -1 or -2")


class MotionAwareTokenMerge(nn.Module):
    """
    Adaptive token merging based on motion patterns.
    
    High-motion regions keep more tokens, static regions are compressed.
    """
    
    def __init__(self, base_compression_ratio: float = 0.5, 
                 motion_threshold: float = 0.1):
        super().__init__()
        self.base_compression_ratio = base_compression_ratio
        self.motion_threshold = motion_threshold
        self.flow_estimator = OpticalFlowEstimator()
        
    def forward(self, frames, tokens):
        """
        Merge tokens adaptively based on motion.
        
        Args:
            frames: Video frames (batch, num_frames, 3, height, width)
            tokens: Feature tokens (batch, num_tokens, dim)
            
        Returns:
            merged_tokens: Adaptively merged tokens (batch, merged_size, dim)
            merge_weights: Weights for each token (batch, num_tokens)
        """
        batch_size, num_frames = frames.shape[:2]
        
        # Compute motion magnitude across frames
        motion_scores = []
        for i in range(num_frames - 1):
            motion = self.flow_estimator(frames[:, i], frames[:, i + 1])
            motion_scores.append(motion)
        
        # Average motion scores across spatial dimensions
        motion_scores = torch.stack(motion_scores, dim=1)  # (batch, num_frames-1, 1, h, w)
        motion_per_frame = motion_scores.mean(dim=(2, 3, 4))  # (batch, num_frames-1)
        
        # Compute adaptive merge ratios based on motion
        merge_ratios = self._compute_merge_ratios(motion_per_frame)
        
        # Merge tokens based on ratios
        merged_tokens = self._merge_tokens(tokens, merge_ratios)
        
        return merged_tokens, merge_ratios
    
    def _compute_merge_ratios(self, motion_scores):
        """Compute token merge ratios based on motion."""
        # High motion → keep more tokens (lower merge ratio)
        # Low motion → compress more (higher merge ratio)
        merge_ratios = 1.0 - motion_scores  # Invert: high motion = low merge ratio
        merge_ratios = self.base_compression_ratio + \
                      (1.0 - self.base_compression_ratio) * merge_ratios
        
        return merge_ratios
    
    def _merge_tokens(self, tokens, merge_ratios):
        """Merge tokens based on computed ratios."""
        # TODO: Implement actual token merging/pruning based on ratios
        return tokens
