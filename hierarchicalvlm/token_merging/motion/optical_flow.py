"""
Optical Flow for Motion-Based Token Merging

This module implements dense optical flow computation for detecting motion
in video frames. The optical flow estimates are used to identify high-motion
regions that should preserve more tokens during merging.

Key Components:
- DenseOpticalFlow: Dense optical flow using Lucas-Kanade or correlation
- MotionMagnitude: Compute motion magnitude and direction
- MotionBasedCompression: Adaptive token compression based on motion

Reference:
    "FlowNet: Learning Optical Flow with Convolutional Networks"
    https://arxiv.org/abs/1504.04353
"""

from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseOpticalFlow(nn.Module):
    """Compute dense optical flow using correlation-based method.
    
    Implements a lightweight optical flow estimation using feature correlation.
    Suitable for real-time motion detection in video understanding.
    
    Args:
        window_size: Size of correlation window for flow computation
        num_scales: Number of pyramid scales for multi-scale flow
        normalize: Whether to normalize optical flow
    """
    
    def __init__(
        self,
        window_size: int = 15,
        num_scales: int = 3,
        normalize: bool = True
    ):
        super().__init__()
        
        self.window_size = window_size
        self.num_scales = num_scales
        self.normalize = normalize
        
        # Create gaussian kernels for smoothing
        self.register_buffer(
            'gaussian_kernel',
            self._create_gaussian_kernel(window_size)
        )
    
    def _create_gaussian_kernel(self, size: int) -> torch.Tensor:
        """Create 2D Gaussian kernel for flow smoothing.
        
        Args:
            size: Kernel size
        
        Returns:
            2D Gaussian kernel tensor
        """
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
        
        # Normalize
        kernel = kernel / kernel.sum()
        return kernel
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """Compute dense optical flow between two frames.
        
        Args:
            frame1: First frame of shape (batch, channels, height, width)
            frame2: Second frame of shape (batch, channels, height, width)
        
        Returns:
            Optical flow of shape (batch, 2, height, width)
            where flow[:, 0] is horizontal and flow[:, 1] is vertical
        """
        batch_size, channels, height, width = frame1.shape
        
        # Extract features using simple convolutions
        feat1 = self._extract_features(frame1)
        feat2 = self._extract_features(frame2)
        
        # Compute correlation-based flow
        flow = self._compute_correlation_flow(feat1, feat2)
        
        # Apply smoothing
        flow = self._smooth_flow(flow)
        
        # Normalize if requested
        if self.normalize:
            flow = self._normalize_flow(flow)
        
        return flow
    
    def _extract_features(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract features from frame using simple processing.
        
        Args:
            frame: Input frame tensor
        
        Returns:
            Feature tensor
        """
        # Convert to grayscale if RGB
        if frame.shape[1] == 3:
            gray = 0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3]
        else:
            gray = frame
        
        # Apply Sobel-like edge detection for feature richness
        sobelx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                             dtype=frame.dtype, device=frame.device).view(1, 1, 3, 3)
        sobely = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                             dtype=frame.dtype, device=frame.device).view(1, 1, 3, 3)
        
        fx = F.conv2d(gray, sobelx, padding=1)
        fy = F.conv2d(gray, sobely, padding=1)
        
        return torch.cat([gray, fx, fy], dim=1)  # (batch, 3, H, W)
    
    def _compute_correlation_flow(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> torch.Tensor:
        """Compute flow using feature correlation.
        
        Args:
            feat1: Features from first frame
            feat2: Features from second frame
        
        Returns:
            Optical flow tensor
        """
        batch_size, channels, height, width = feat1.shape
        device = feat1.device
        
        # Normalize features
        feat1_norm = F.normalize(feat1, dim=1)
        feat2_norm = F.normalize(feat2, dim=1)
        
        # Unfold feat2 for patch-wise comparison
        window = self.window_size
        pad = window // 2
        feat2_pad = F.pad(feat2_norm, (pad, pad, pad, pad))
        
        # Initialize flow
        flow = torch.zeros(batch_size, 2, height, width, device=device)
        
        # For efficiency, sample every 4 pixels
        step = 4
        for y in range(0, height, step):
            for x in range(0, width, step):
                # Get patch from feat1
                patch = feat1_norm[:, :, y:y+1, x:x+1]  # (batch, channels, 1, 1)
                
                # Compare with neighborhood in feat2
                search_y_start = max(0, y - window//2)
                search_y_end = min(height, y + window//2 + 1)
                search_x_start = max(0, x - window//2)
                search_x_end = min(width, x + window//2 + 1)
                
                search_region = feat2_norm[
                    :, :,
                    search_y_start:search_y_end,
                    search_x_start:search_x_end
                ]
                
                # Compute correlation
                corr = torch.sum(patch * search_region, dim=1, keepdim=True)
                
                # Find best match
                best_y = torch.argmax(corr.max(dim=3)[0], dim=2)
                best_x = torch.argmax(corr.max(dim=2)[0], dim=2)
                
                # Compute displacement
                disp_y = (best_y.float() + search_y_start - y) / float(height)
                disp_x = (best_x.float() + search_x_start - x) / float(width)
                
                flow[:, 0, y:min(y+step, height), x:min(x+step, width)] = disp_x[:, :1]
                flow[:, 1, y:min(y+step, height), x:min(x+step, width)] = disp_y[:, :1]
        
        return flow
    
    def _smooth_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """Smooth optical flow using Gaussian filtering.
        
        Args:
            flow: Optical flow tensor
        
        Returns:
            Smoothed flow tensor
        """
        # Smooth each channel
        smoothed_flow = torch.zeros_like(flow)
        for i in range(flow.shape[1]):
            smoothed_flow[:, i:i+1] = F.conv2d(
                flow[:, i:i+1],
                self.gaussian_kernel,
                padding=self.window_size // 2
            )
        
        return smoothed_flow
    
    def _normalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """Normalize optical flow to [-1, 1] range.
        
        Args:
            flow: Optical flow tensor
        
        Returns:
            Normalized flow tensor
        """
        magnitude = torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2)
        magnitude = torch.clamp(magnitude, min=1e-6)
        
        return flow / (magnitude + 1e-6)


class MotionMagnitude(nn.Module):
    """Compute motion magnitude from optical flow.
    
    Converts 2D optical flow vectors into scalar motion magnitude values
    that indicate the strength of motion at each spatial location.
    
    Args:
        motion_scale: Scale factor for motion magnitude
        motion_threshold: Minimum motion to consider significant
    """
    
    def __init__(
        self,
        motion_scale: float = 1.0,
        motion_threshold: float = 0.01
    ):
        super().__init__()
        
        self.motion_scale = motion_scale
        self.motion_threshold = motion_threshold
    
    def forward(
        self,
        optical_flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute motion magnitude from optical flow.
        
        Args:
            optical_flow: Optical flow of shape (batch, 2, height, width)
        
        Returns:
            Tuple of:
            - motion_magnitude: (batch, 1, height, width) scalar motion values
            - motion_direction: (batch, 1, height, width) motion direction angle
        """
        # Extract horizontal and vertical components
        flow_x = optical_flow[:, 0:1]  # (batch, 1, H, W)
        flow_y = optical_flow[:, 1:2]  # (batch, 1, H, W)
        
        # Compute magnitude
        magnitude = torch.sqrt(flow_x**2 + flow_y**2 + 1e-6)
        magnitude = magnitude * self.motion_scale
        
        # Clamp to threshold
        magnitude = torch.clamp(magnitude, min=self.motion_threshold)
        
        # Compute direction (angle)
        direction = torch.atan2(flow_y, flow_x)  # (batch, 1, H, W)
        
        return magnitude, direction
    
    def get_motion_statistics(
        self,
        motion_magnitude: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute statistics about motion in the frame.
        
        Args:
            motion_magnitude: Motion magnitude tensor
        
        Returns:
            Dictionary with motion statistics
        """
        # Global statistics
        mean_motion = motion_magnitude.mean()
        max_motion = motion_magnitude.max()
        min_motion = motion_magnitude.min()
        
        # Spatial statistics
        motion_per_frame = motion_magnitude.mean(dim=(2, 3))  # Average per frame
        
        return {
            'mean': mean_motion,
            'max': max_motion,
            'min': min_motion,
            'per_frame': motion_per_frame
        }


class MotionBasedCompression(nn.Module):
    """Adaptive token compression based on motion magnitude.
    
    Uses motion magnitude to determine how many tokens to keep in each
    spatial region. High-motion regions keep more tokens, static regions
    compress more aggressively.
    
    Args:
        num_frames: Number of video frames
        height: Frame height (in patches)
        width: Frame width (in patches)
        min_ratio: Minimum compression ratio (tokens kept)
        max_ratio: Maximum compression ratio (all tokens kept)
    """
    
    def __init__(
        self,
        num_frames: int = 32,
        height: int = 14,
        width: int = 14,
        min_ratio: float = 0.25,
        max_ratio: float = 1.0
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def forward(
        self,
        motion_magnitude: torch.Tensor,
        global_compression_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-patch compression ratio based on motion.
        
        Args:
            motion_magnitude: Motion magnitude from MotionMagnitude
            global_compression_ratio: Overall target compression ratio
        
        Returns:
            Tuple of:
            - compression_ratio: Per-patch compression ratio (batch, H, W)
            - importance_scores: Importance scores for each patch
        """
        batch_size = motion_magnitude.shape[0]
        
        # Resize motion magnitude to patch grid
        motion_resized = F.adaptive_avg_pool2d(
            motion_magnitude,
            (self.height, self.width)
        )  # (batch, 1, H, W)
        
        # Normalize motion to [0, 1]
        motion_min = motion_resized.min()
        motion_max = motion_resized.max()
        motion_normalized = (motion_resized - motion_min) / (motion_max - motion_min + 1e-6)
        
        # Convert motion to importance (high motion = high importance = keep more tokens)
        importance_scores = motion_normalized.squeeze(1)  # (batch, H, W)
        
        # Compute per-patch compression ratio
        # High motion regions get higher ratio (keep more tokens)
        # Low motion regions get lower ratio (compress more)
        compression_ratio = (
            self.min_ratio + 
            (self.max_ratio - self.min_ratio) * importance_scores
        )  # (batch, H, W)
        
        # Apply global compression constraint
        # Ensure average ratio matches target
        current_avg = compression_ratio.mean()
        if current_avg > 0:
            scale_factor = global_compression_ratio / (current_avg + 1e-6)
            compression_ratio = compression_ratio * scale_factor
            compression_ratio = torch.clamp(
                compression_ratio,
                min=self.min_ratio,
                max=self.max_ratio
            )
        
        return compression_ratio, importance_scores
    
    def get_merge_indices(
        self,
        compression_ratio: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        """Determine which tokens to merge based on compression ratio.
        
        Args:
            compression_ratio: Per-patch compression ratio
            num_tokens: Total number of tokens
        
        Returns:
            Binary mask indicating which tokens to keep (1) or merge (0)
        """
        batch_size, height, width = compression_ratio.shape
        
        # Flatten spatial dimensions
        ratio_flat = compression_ratio.view(batch_size, -1)  # (batch, H*W)
        
        # For each patch, determine number of tokens to keep
        tokens_per_patch = num_tokens // (height * width)
        tokens_to_keep_per_patch = (ratio_flat * tokens_per_patch).round().long()
        
        # Create merge mask
        merge_mask = torch.zeros(batch_size, num_tokens, device=compression_ratio.device)
        
        for b in range(batch_size):
            token_idx = 0
            for patch_idx in range(height * width):
                tokens_to_keep = tokens_to_keep_per_patch[b, patch_idx].item()
                tokens_to_keep = max(1, min(tokens_per_patch, tokens_to_keep))
                
                # Keep first tokens_to_keep tokens in this patch
                merge_mask[b, token_idx:token_idx+tokens_to_keep] = 1
                token_idx += tokens_per_patch
        
        return merge_mask
