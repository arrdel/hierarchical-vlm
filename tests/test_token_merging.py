"""
Comprehensive Tests for Token Merging Components

Tests cover:
- Optical flow computation and motion detection
- Saliency detection using multiple methods
- Adaptive token merging fusion
- Integration of motion + saliency + token merging
- Temporal scheduling and compression ratios
"""

import pytest
import torch
import torch.nn as nn

from hierarchicalvlm.token_merging.motion import (
    DenseOpticalFlow,
    MotionMagnitude,
    MotionBasedCompression,
)
from hierarchicalvlm.token_merging.saliency import (
    EdgeSaliency,
    AttentionSaliency,
    ColorSaliency,
    MultiSaliencyFusion,
    SaliencyDetector,
)
from hierarchicalvlm.token_merging.token_merging import (
    TokenSimilarity,
    AdaptiveTokenMerger,
    TemporalMergeScheduler,
)


# ============================================================================
# OPTICAL FLOW TESTS (Motion Detection)
# ============================================================================

class TestDenseOpticalFlow:
    """Test dense optical flow computation."""
    
    def test_output_shape(self):
        """Test output shape matches input spatial dimensions."""
        model = DenseOpticalFlow(window_size=15)
        
        batch_size, channels, height, width = 2, 3, 224, 224
        frame1 = torch.randn(batch_size, channels, height, width)
        frame2 = torch.randn(batch_size, channels, height, width)
        
        flow = model(frame1, frame2)
        
        assert flow.shape == (batch_size, 2, height, width)
    
    def test_flow_values_normalized(self):
        """Test optical flow values are normalized."""
        model = DenseOpticalFlow(normalize=True)
        
        frame1 = torch.rand(1, 3, 64, 64)
        frame2 = torch.rand(1, 3, 64, 64)
        
        flow = model(frame1, frame2)
        
        # Magnitude should be approximately normalized
        magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
        assert magnitude.max() <= 2.0  # Normalized range
    
    def test_zero_motion_identical_frames(self):
        """Test that identical frames produce near-zero optical flow."""
        model = DenseOpticalFlow()
        
        frame = torch.rand(1, 3, 64, 64)
        
        flow = model(frame, frame)
        magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
        
        # Motion should be very small
        assert magnitude.mean() < 0.3
    
    def test_large_motion_shifted_frames(self):
        """Test that shifted frames produce larger optical flow."""
        model = DenseOpticalFlow()
        
        frame1 = torch.zeros(1, 3, 64, 64)
        frame1[:, :, :, :32] = 1.0
        
        frame2 = torch.zeros(1, 3, 64, 64)
        frame2[:, :, :, 32:] = 1.0
        
        flow = model(frame1, frame2)
        magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
        
        # Motion should be detected
        assert magnitude.max() > 0.1
    
    def test_gradient_flow_backward(self):
        """Test that gradients flow backward through optical flow."""
        model = DenseOpticalFlow()
        
        frame1 = torch.randn(1, 3, 32, 32, requires_grad=True)
        frame2 = torch.randn(1, 3, 32, 32, requires_grad=True)
        
        flow = model(frame1, frame2)
        loss = flow.mean()
        loss.backward()
        
        assert frame1.grad is not None
        assert frame2.grad is not None


class TestMotionMagnitude:
    """Test motion magnitude computation."""
    
    def test_magnitude_output_shape(self):
        """Test magnitude has correct shape."""
        model = MotionMagnitude()
        
        batch_size, height, width = 2, 64, 64
        optical_flow = torch.randn(batch_size, 2, height, width)
        
        magnitude, direction = model(optical_flow)
        
        assert magnitude.shape == (batch_size, 1, height, width)
        assert direction.shape == (batch_size, 1, height, width)
    
    def test_zero_flow_zero_magnitude(self):
        """Test that zero optical flow produces near-zero magnitude."""
        model = MotionMagnitude(motion_threshold=0.0)
        
        optical_flow = torch.zeros(1, 2, 64, 64)
        magnitude, direction = model(optical_flow)
        
        assert magnitude.max() < 0.01
    
    def test_magnitude_positive(self):
        """Test that magnitude is always positive."""
        model = MotionMagnitude()
        
        optical_flow = torch.randn(2, 2, 32, 32)
        magnitude, direction = model(optical_flow)
        
        assert (magnitude >= 0).all()
    
    def test_direction_angle_range(self):
        """Test that direction angle is in valid range."""
        model = MotionMagnitude()
        
        optical_flow = torch.randn(2, 2, 32, 32)
        magnitude, direction = model(optical_flow)
        
        # Angle should be in [-pi, pi]
        assert (direction >= -torch.pi).all()
        assert (direction <= torch.pi).all()
    
    def test_motion_statistics(self):
        """Test motion statistics computation."""
        model = MotionMagnitude()
        
        optical_flow = torch.randn(2, 2, 32, 32)
        magnitude, _ = model(optical_flow)
        
        stats = model.get_motion_statistics(magnitude)
        
        assert 'mean' in stats
        assert 'max' in stats
        assert 'min' in stats
        assert 'per_frame' in stats


class TestMotionBasedCompression:
    """Test adaptive compression ratios from motion."""
    
    def test_compression_ratio_shape(self):
        """Test compression ratio output shape."""
        model = MotionBasedCompression(num_frames=32, height=14, width=14)
        
        motion_magnitude = torch.rand(2, 1, 224, 224)
        
        compression_ratio, importance = model(motion_magnitude)
        
        assert compression_ratio.shape == (2, 14, 14)
        assert importance.shape == (2, 14, 14)
    
    def test_compression_ratio_range(self):
        """Test compression ratio is in valid range."""
        model = MotionBasedCompression(min_ratio=0.25, max_ratio=1.0)
        
        motion_magnitude = torch.rand(2, 1, 224, 224)
        
        compression_ratio, _ = model(motion_magnitude)
        
        assert (compression_ratio >= 0.25).all()
        assert (compression_ratio <= 1.0).all()
    
    def test_high_motion_preserves_tokens(self):
        """Test that high motion regions keep more tokens."""
        model = MotionBasedCompression()
        
        # Create motion with strong center signal
        motion = torch.zeros(1, 1, 128, 128)
        motion[:, :, 32:96, 32:96] = 1.0  # High motion center
        
        compression_ratio, importance = model(motion, global_compression_ratio=0.5)
        
        # Center should have higher compression ratio (keep more tokens)
        center_ratio = compression_ratio[:, 7, 7]
        edge_ratio = compression_ratio[:, 0, 0]
        
        assert center_ratio > edge_ratio
    
    def test_merge_indices_generation(self):
        """Test merge indices generation."""
        model = MotionBasedCompression(height=4, width=4)
        
        compression_ratio = torch.ones(2, 4, 4) * 0.5
        merge_mask = model.get_merge_indices(compression_ratio, num_tokens=128)
        
        assert merge_mask.shape == (2, 128)
        assert (merge_mask >= 0).all()
        assert (merge_mask <= 1).all()


# ============================================================================
# SALIENCY DETECTION TESTS
# ============================================================================

class TestEdgeSaliency:
    """Test edge-based saliency detection."""
    
    def test_output_shape(self):
        """Test edge saliency output shape."""
        model = EdgeSaliency()
        
        frame = torch.randn(2, 3, 224, 224)
        saliency = model(frame)
        
        assert saliency.shape == (2, 1, 224, 224)
    
    def test_saliency_range(self):
        """Test edge saliency is normalized to [0, 1]."""
        model = EdgeSaliency()
        
        frame = torch.rand(2, 3, 128, 128)
        saliency = model(frame)
        
        assert (saliency >= 0).all()
        assert (saliency <= 1).all()
    
    def test_edge_detection_strong_edges(self):
        """Test edge saliency detects strong edges."""
        model = EdgeSaliency()
        
        # Create frame with strong edges
        frame = torch.zeros(1, 3, 64, 64)
        frame[:, :, :, 32:] = 1.0  # Hard edge in middle
        
        saliency = model(frame)
        
        # Center region should have higher saliency
        center_saliency = saliency[:, :, :, 32].mean()
        edge_saliency = saliency[:, :, :, 0].mean()
        
        assert center_saliency > edge_saliency


class TestAttentionSaliency:
    """Test attention-based saliency detection."""
    
    def test_output_shape_without_attention(self):
        """Test attention saliency shape without attention weights."""
        model = AttentionSaliency()
        
        features = torch.randn(2, 196, 768)  # 14x14 patches, 768-dim
        saliency = model(features)
        
        assert saliency.shape == (2, 1, 14, 14)
    
    def test_output_shape_with_attention(self):
        """Test attention saliency shape with attention weights."""
        model = AttentionSaliency(num_heads=12)
        
        attention_weights = torch.randn(2, 12, 196, 196)
        saliency = model(torch.zeros(2, 196, 768), attention_weights)
        
        assert saliency.ndim >= 2
    
    def test_saliency_range(self):
        """Test attention saliency is normalized."""
        model = AttentionSaliency()
        
        features = torch.randn(2, 196, 768)
        saliency = model(features)
        
        assert (saliency >= 0).all()
        assert (saliency <= 1).all()


class TestColorSaliency:
    """Test color-based saliency detection."""
    
    def test_output_shape(self):
        """Test color saliency output shape."""
        model = ColorSaliency()
        
        frame = torch.randn(2, 3, 224, 224)
        saliency = model(frame)
        
        assert saliency.shape == (2, 1, 224, 224)
    
    def test_color_distinctiveness_detection(self):
        """Test color saliency detects distinctive colors."""
        model = ColorSaliency()
        
        # Create frame with distinct color patch
        frame = torch.ones(1, 3, 64, 64) * 0.5  # Gray background
        frame[:, :, 20:44, 20:44] = torch.tensor([0.9, 0.1, 0.1]).view(1, 3, 1, 1)  # Red patch
        
        saliency = model(frame)
        
        # Red patch region should have higher saliency
        patch_saliency = saliency[:, :, 20:44, 20:44].mean()
        bg_saliency = saliency[:, :, :10, :10].mean()
        
        assert patch_saliency > bg_saliency


class TestMultiSaliencyFusion:
    """Test multi-source saliency fusion."""
    
    def test_fusion_output_shape(self):
        """Test fused saliency shape."""
        model = MultiSaliencyFusion()
        
        edge_sal = torch.rand(2, 1, 64, 64)
        att_sal = torch.rand(2, 1, 64, 64)
        color_sal = torch.rand(2, 1, 64, 64)
        
        fused = model(edge_sal, att_sal, color_sal)
        
        assert fused.shape == (2, 1, 64, 64)
    
    def test_fusion_different_methods(self):
        """Test different fusion methods produce different results."""
        edge_sal = torch.rand(2, 1, 32, 32)
        att_sal = torch.rand(2, 1, 32, 32)
        color_sal = torch.rand(2, 1, 32, 32)
        
        model_sum = MultiSaliencyFusion(fusion_method='weighted_sum')
        model_prod = MultiSaliencyFusion(fusion_method='product')
        
        fused_sum = model_sum(edge_sal, att_sal, color_sal)
        fused_prod = model_prod(edge_sal, att_sal, color_sal)
        
        # Methods should produce different results
        assert not torch.allclose(fused_sum, fused_prod, atol=1e-5)
    
    def test_fusion_normalized(self):
        """Test fused saliency is normalized."""
        model = MultiSaliencyFusion()
        
        edge_sal = torch.rand(2, 1, 64, 64)
        att_sal = torch.rand(2, 1, 64, 64)
        color_sal = torch.rand(2, 1, 64, 64)
        
        fused = model(edge_sal, att_sal, color_sal)
        
        assert (fused >= 0).all()
        assert (fused <= 1).all()


class TestSaliencyDetector:
    """Test unified saliency detection."""
    
    def test_saliency_detector_output(self):
        """Test saliency detector returns map and components."""
        model = SaliencyDetector()
        
        frame = torch.randn(2, 3, 224, 224)
        saliency_map, components = model(frame)
        
        assert saliency_map.shape == (2, 1, 224, 224)
        assert isinstance(components, dict)
        assert 'fused' in components
    
    def test_individual_components(self):
        """Test individual saliency components are computed."""
        model = SaliencyDetector(
            use_edge_saliency=True,
            use_attention_saliency=True,
            use_color_saliency=True
        )
        
        frame = torch.randn(2, 3, 128, 128)
        _, components = model(frame)
        
        assert 'edge' in components
        assert 'attention' in components
        assert 'color' in components
    
    def test_saliency_statistics(self):
        """Test saliency statistics computation."""
        model = SaliencyDetector()
        
        saliency_map = torch.rand(2, 1, 64, 64)
        stats = model.get_saliency_statistics(saliency_map)
        
        assert 'mean' in stats
        assert 'max' in stats
        assert 'min' in stats
        assert 'std' in stats
        assert 'entropy' in stats


# ============================================================================
# TOKEN MERGING TESTS
# ============================================================================

class TestTokenSimilarity:
    """Test token similarity computation."""
    
    def test_similarity_matrix_shape(self):
        """Test similarity matrix shape."""
        model = TokenSimilarity()
        
        tokens = torch.randn(2, 196, 768)
        similarity, merge_mask = model(tokens)
        
        assert similarity.shape == (2, 196, 196)
        assert merge_mask.shape == (2, 196)
    
    def test_similarity_matrix_symmetric(self):
        """Test similarity matrix is symmetric."""
        model = TokenSimilarity()
        
        tokens = torch.randn(1, 64, 256)
        similarity, _ = model(tokens)
        
        # Check approximate symmetry
        assert torch.allclose(similarity, similarity.transpose(1, 2), atol=1e-5)
    
    def test_merge_groups_generation(self):
        """Test merge group generation."""
        model = TokenSimilarity()
        
        merge_mask = torch.zeros(2, 16, 16)
        # Create some merge groups
        merge_mask[0, 0, 1] = 1.0
        merge_mask[0, 1, 0] = 1.0
        
        group_ids, group_sizes = model.get_merge_groups(merge_mask)
        
        assert group_ids.shape == (2, 16)
        assert group_sizes.shape == (2, 16)
    
    def test_similar_tokens_identified(self):
        """Test similar tokens are identified for merging."""
        model = TokenSimilarity(similarity_threshold=0.9)
        
        # Create similar tokens
        base_token = torch.randn(1, 1, 768)
        tokens = torch.cat([base_token, base_token + 0.01 * torch.randn(1, 1, 768)] * 8, dim=1)
        
        similarity, merge_mask = model(tokens)
        
        # Should find merging candidates
        assert merge_mask.sum() > 0


class TestAdaptiveTokenMerger:
    """Test adaptive token merging."""
    
    def test_merger_output_shape(self):
        """Test merged tokens have correct shape."""
        model = AdaptiveTokenMerger(height=14, width=14)
        
        tokens = torch.randn(2, 196, 768)
        merged_tokens, merge_info = model(tokens)
        
        assert merged_tokens.shape[0] == 2
        assert merged_tokens.shape[2] == 768
        assert merged_tokens.shape[1] <= 196  # Compressed
    
    def test_merge_with_motion_and_saliency(self):
        """Test merging with motion and saliency inputs."""
        model = AdaptiveTokenMerger()
        
        tokens = torch.randn(2, 196, 768)
        motion_mag = torch.rand(2, 1, 224, 224)
        saliency = torch.rand(2, 1, 224, 224)
        
        merged_tokens, merge_info = model(tokens, motion_mag, saliency)
        
        assert 'compression_ratio' in merge_info
        assert merge_info['compression_ratio'] < 1.0
    
    def test_compression_statistics(self):
        """Test merge statistics computation."""
        model = AdaptiveTokenMerger()
        
        tokens = torch.randn(2, 196, 768)
        merged_tokens, merge_info = model(tokens)
        
        stats = model.get_merge_statistics(merge_info)
        
        assert 'compression_ratio' in stats
        assert 'compression_percentage' in stats
        assert 'mean_importance' in stats
    
    def test_importance_score_computation(self):
        """Test importance score computation."""
        model = AdaptiveTokenMerger(height=14, width=14)
        
        # Test with motion only
        motion_mag = torch.rand(2, 1, 224, 224)
        importance = model._compute_importance_scores(motion_mag, None, None, 196)
        
        assert importance.shape == (2, 196)
        assert (importance >= 0).all()
        assert (importance <= 1).all()


class TestTemporalMergeScheduler:
    """Test temporal merge scheduling."""
    
    def test_schedule_output_shape(self):
        """Test schedule output shape."""
        model = TemporalMergeScheduler(num_frames=32)
        
        motion_over_time = torch.rand(2, 32)
        schedule = model(motion_over_time)
        
        assert schedule.shape == (2, 32)
    
    def test_schedule_values_valid(self):
        """Test schedule values are valid."""
        model = TemporalMergeScheduler()
        
        motion_over_time = torch.rand(2, 32)
        schedule = model(motion_over_time)
        
        # Should be between base ratio and 1.0
        assert (schedule >= model.base_merge_ratio).all()
        assert (schedule <= 1.0).all()
    
    def test_high_motion_less_merging(self):
        """Test high motion frames get less merging."""
        model = TemporalMergeScheduler(base_merge_ratio=0.5)
        
        motion_over_time = torch.zeros(1, 32)
        motion_over_time[:, :16] = 0.9  # High motion first half
        
        schedule = model(motion_over_time)
        
        # First half should have lower merge ratio (higher values)
        assert schedule[:, :16].mean() > schedule[:, 16:].mean()
    
    def test_default_schedule_generation(self):
        """Test default schedule generation."""
        model = TemporalMergeScheduler(num_frames=32)
        schedule = model.get_schedule()
        
        assert schedule.shape == (32,)
        assert (schedule >= 0).all()
        assert (schedule <= 1).all()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTokenMergingIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline(self):
        """Test complete token merging pipeline."""
        # Setup components
        optical_flow = DenseOpticalFlow()
        motion_magnitude = MotionMagnitude()
        motion_compression = MotionBasedCompression()
        
        saliency_detector = SaliencyDetector()
        
        adaptive_merger = AdaptiveTokenMerger()
        
        # Generate synthetic data
        frame1 = torch.randn(1, 3, 224, 224)
        frame2 = torch.randn(1, 3, 224, 224)
        tokens = torch.randn(1, 196, 768)
        
        # Run pipeline
        flow = optical_flow(frame1, frame2)
        motion_mag, _ = motion_magnitude(flow)
        compress_ratio, _ = motion_compression(motion_mag)
        
        saliency_map, _ = saliency_detector(frame1)
        
        merged_tokens, merge_info = adaptive_merger(tokens, motion_mag, saliency_map)
        
        # Verify outputs
        assert merged_tokens.shape[0] == 1
        assert merged_tokens.shape[2] == 768
        assert 'compression_ratio' in merge_info
    
    def test_batch_processing(self):
        """Test batch processing across multiple sequences."""
        model = AdaptiveTokenMerger()
        
        batch_size = 4
        tokens = torch.randn(batch_size, 196, 768)
        motion_mag = torch.rand(batch_size, 1, 224, 224)
        saliency = torch.rand(batch_size, 1, 224, 224)
        
        merged_tokens, merge_info = model(tokens, motion_mag, saliency)
        
        assert merged_tokens.shape[0] == batch_size
        assert isinstance(merge_info, dict)
    
    def test_gradient_flow_full_pipeline(self):
        """Test gradient flow through full pipeline."""
        optical_flow = DenseOpticalFlow()
        motion_magnitude = MotionMagnitude()
        saliency_detector = SaliencyDetector()
        adaptive_merger = AdaptiveTokenMerger()
        
        frame1 = torch.randn(1, 3, 128, 128, requires_grad=True)
        frame2 = torch.randn(1, 3, 128, 128, requires_grad=True)
        tokens = torch.randn(1, 64, 768, requires_grad=True)
        
        # Forward pass
        flow = optical_flow(frame1, frame2)
        motion_mag, _ = motion_magnitude(flow)
        saliency_map, _ = saliency_detector(frame1)
        merged_tokens, _ = adaptive_merger(tokens, motion_mag, saliency_map)
        
        # Backward pass
        loss = merged_tokens.mean()
        loss.backward()
        
        assert frame1.grad is not None
        assert frame2.grad is not None
        assert tokens.grad is not None


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_token_merging(self):
        """Test merging with minimal tokens."""
        model = AdaptiveTokenMerger()
        
        tokens = torch.randn(1, 4, 768)
        merged_tokens, _ = model(tokens)
        
        assert merged_tokens.shape[0] == 1
        assert merged_tokens.shape[1] <= 4
    
    def test_all_zero_motion(self):
        """Test with all-zero motion magnitude."""
        model = MotionBasedCompression()
        
        motion_magnitude = torch.zeros(1, 1, 128, 128)
        compression_ratio, _ = model(motion_magnitude, 0.5)
        
        assert compression_ratio.shape == (1, 14, 14)
        assert (compression_ratio >= 0.25).all()
    
    def test_uniform_saliency(self):
        """Test with uniform saliency everywhere."""
        model = SaliencyDetector()
        
        frame = torch.ones(1, 3, 128, 128) * 0.5
        saliency_map, _ = model(frame)
        
        # Should produce a valid output
        assert saliency_map.shape == (1, 1, 128, 128)
    
    def test_very_small_frames(self):
        """Test with very small input frames."""
        model = DenseOpticalFlow()
        
        frame1 = torch.randn(1, 3, 16, 16)
        frame2 = torch.randn(1, 3, 16, 16)
        
        flow = model(frame1, frame2)
        
        assert flow.shape == (1, 2, 16, 16)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
