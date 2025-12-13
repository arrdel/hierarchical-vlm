"""
Comprehensive tests for attention mechanisms.

Tests:
- Output shape correctness
- Gradient flow
- Computational efficiency
- Different input sizes
"""

import torch
import torch.nn as nn
import pytest
from hierarchicalvlm.attention.sparse import (
    StridedAttention,
    LocalGlobalAttention,
    CrossMemoryAttention,
    HierarchicalAttentionBlock,
)
from hierarchicalvlm.attention.linear import (
    PerformerAttention,
    MambaLayer,
    LinearAttentionBlock,
)


class TestStridedAttention:
    """Test StridedAttention mechanism."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 2, 256, 768
        stride = 4
        
        attn = StridedAttention(stride=stride, dim=dim, num_heads=12)
        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        batch_size, seq_len, dim = 2, 64, 768
        
        attn = StridedAttention(dim=dim, num_heads=12)
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
    
    def test_different_strides(self):
        """Test attention with different stride values."""
        x = torch.randn(2, 128, 768)
        
        for stride in [1, 2, 4, 8]:
            attn = StridedAttention(stride=stride, dim=768, num_heads=12)
            output = attn(x)
            assert output.shape == x.shape
    
    def test_dropout(self):
        """Test dropout functionality."""
        attn_train = StridedAttention(dim=768, num_heads=12, dropout=0.5)
        attn_eval = StridedAttention(dim=768, num_heads=12, dropout=0.5)
        attn_eval.eval()
        
        x = torch.randn(2, 64, 768)
        
        # Training mode should have variance in outputs (due to dropout)
        attn_train.train()
        out1 = attn_train(x)
        out2 = attn_train(x)
        
        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)
        
        # Eval mode should be deterministic
        out3 = attn_eval(x)
        out4 = attn_eval(x)
        assert torch.allclose(out3, out4)


class TestLocalGlobalAttention:
    """Test LocalGlobalAttention mechanism."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 2, 256, 768
        
        attn = LocalGlobalAttention(
            local_window=64,
            num_global_tokens=4,
            dim=dim,
            num_heads=12
        )
        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_gradient_flow(self):
        """Test gradient flow through the module."""
        x = torch.randn(2, 128, 768, requires_grad=True)
        
        attn = LocalGlobalAttention(dim=768, num_heads=12)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_different_window_sizes(self):
        """Test with different local window sizes."""
        x = torch.randn(2, 256, 768)
        
        for window_size in [32, 64, 128]:
            attn = LocalGlobalAttention(
                local_window=window_size,
                dim=768,
                num_heads=12
            )
            output = attn(x)
            assert output.shape == x.shape
    
    def test_global_token_selection(self):
        """Test that global token selection works."""
        attn = LocalGlobalAttention(
            num_global_tokens=4,
            dim=768,
            num_heads=12
        )
        x = torch.randn(2, 128, 768)
        output = attn(x)
        
        # Should not raise any errors
        assert output.shape == x.shape


class TestCrossMemoryAttention:
    """Test CrossMemoryAttention mechanism."""
    
    def test_basic_forward(self):
        """Test basic cross-attention forward pass."""
        batch_size = 2
        local_seq = 64
        memory_seq = 32
        dim = 768
        
        attn = CrossMemoryAttention(dim=dim, num_heads=12)
        local_features = torch.randn(batch_size, local_seq, dim)
        memory_tokens = torch.randn(batch_size, memory_seq, dim)
        
        output = attn(local_features, memory_tokens)
        
        # Output should have same shape as local features
        assert output.shape == (batch_size, local_seq, dim)
    
    def test_without_residual(self):
        """Test cross-attention without residual connection."""
        attn = CrossMemoryAttention(dim=768, num_heads=12)
        local = torch.randn(2, 64, 768)
        memory = torch.randn(2, 32, 768)
        
        output_with_residual = attn(local, memory, residual=True)
        output_without_residual = attn(local, memory, residual=False)
        
        # They should be different due to residual connection
        assert not torch.allclose(output_with_residual, output_without_residual)
    
    def test_fusion_ratio(self):
        """Test fusion ratio effect."""
        attn = CrossMemoryAttention(dim=768, num_heads=12)
        local = torch.randn(2, 64, 768)
        memory = torch.randn(2, 32, 768)
        
        # Different fusion ratios should give different results
        out_0 = attn.fuse_with_memory(local, memory, fusion_ratio=0.0)
        out_1 = attn.fuse_with_memory(local, memory, fusion_ratio=1.0)
        
        # At ratio 0, should be close to local features
        assert torch.allclose(out_0, local, atol=1e-5)
        
        # At ratio 1, should be the cross-attended output
        cross = attn(local, memory, residual=False)
        assert torch.allclose(out_1, cross, atol=1e-5)
    
    def test_gradient_flow(self):
        """Test gradient flow."""
        attn = CrossMemoryAttention(dim=768, num_heads=12)
        local = torch.randn(2, 64, 768, requires_grad=True)
        memory = torch.randn(2, 32, 768, requires_grad=True)
        
        output = attn(local, memory)
        loss = output.sum()
        loss.backward()
        
        assert local.grad is not None
        assert memory.grad is not None


class TestHierarchicalAttentionBlock:
    """Test HierarchicalAttentionBlock."""
    
    def test_strided_attention_block(self):
        """Test attention block with strided attention."""
        block = HierarchicalAttentionBlock(
            dim=768,
            num_heads=12,
            attention_type='strided',
            stride=4
        )
        x = torch.randn(2, 128, 768)
        output = block(x)
        assert output.shape == x.shape
    
    def test_local_global_block(self):
        """Test attention block with local+global attention."""
        block = HierarchicalAttentionBlock(
            dim=768,
            num_heads=12,
            attention_type='local_global'
        )
        x = torch.randn(2, 256, 768)
        output = block(x)
        assert output.shape == x.shape
    
    def test_with_memory(self):
        """Test attention block with memory tokens."""
        block = HierarchicalAttentionBlock(
            dim=768,
            num_heads=12,
            attention_type='cross_memory'
        )
        x = torch.randn(2, 64, 768)
        memory = torch.randn(2, 32, 768)
        output = block(x, memory=memory)
        assert output.shape == x.shape


class TestPerformerAttention:
    """Test PerformerAttention mechanism."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 2, 512, 768
        
        attn = PerformerAttention(dim=dim, num_heads=12, num_random_features=256)
        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_long_sequence(self):
        """Test with very long sequences (where Performer shines)."""
        batch_size = 1
        seq_len = 4096  # Very long sequence
        dim = 768
        
        attn = PerformerAttention(dim=dim, num_heads=12)
        x = torch.randn(batch_size, seq_len, dim)
        
        # Should not run out of memory with linear attention
        output = attn(x)
        assert output.shape == x.shape
    
    def test_different_kernels(self):
        """Test different kernel types."""
        x = torch.randn(2, 128, 768)
        
        for kernel_type in ['elu', 'relu']:
            attn = PerformerAttention(
                dim=768,
                num_heads=12,
                kernel_type=kernel_type
            )
            output = attn(x)
            assert output.shape == x.shape
    
    def test_gradient_flow(self):
        """Test gradient flow."""
        x = torch.randn(2, 128, 768, requires_grad=True)
        
        attn = PerformerAttention(dim=768, num_heads=12)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestMambaLayer:
    """Test MambaLayer mechanism."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        batch_size, seq_len, dim = 2, 256, 768
        
        mamba = MambaLayer(dim=dim, state_size=16)
        x = torch.randn(batch_size, seq_len, dim)
        output = mamba(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_long_sequence(self):
        """Test with long sequences."""
        batch_size = 1
        seq_len = 8192
        dim = 768
        
        mamba = MambaLayer(dim=dim, state_size=16)
        x = torch.randn(batch_size, seq_len, dim)
        output = mamba(x)
        
        assert output.shape == x.shape
    
    def test_different_state_sizes(self):
        """Test with different state sizes."""
        x = torch.randn(2, 128, 768)
        
        for state_size in [8, 16, 32]:
            mamba = MambaLayer(dim=768, state_size=state_size)
            output = mamba(x)
            assert output.shape == x.shape
    
    def test_gradient_flow(self):
        """Test gradient flow."""
        x = torch.randn(2, 64, 768, requires_grad=True)
        
        mamba = MambaLayer(dim=768, state_size=16)
        output = mamba(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestLinearAttentionBlock:
    """Test LinearAttentionBlock."""
    
    def test_performer_block(self):
        """Test linear attention block with Performer."""
        block = LinearAttentionBlock(
            dim=768,
            num_heads=12,
            attention_type='performer'
        )
        x = torch.randn(2, 256, 768)
        output = block(x)
        assert output.shape == x.shape
    
    def test_mamba_block(self):
        """Test linear attention block with Mamba."""
        block = LinearAttentionBlock(
            dim=768,
            attention_type='mamba',
            state_size=16
        )
        x = torch.randn(2, 256, 768)
        output = block(x)
        assert output.shape == x.shape


class TestEfficiency:
    """Test computational efficiency of attention mechanisms."""
    
    def test_performer_efficiency(self):
        """Compare Performer vs standard attention memory usage."""
        import time
        
        # Standard attention approximation (using strided)
        strided = StridedAttention(stride=4, dim=768, num_heads=12)
        
        # Performer
        performer = PerformerAttention(dim=768, num_heads=12)
        
        x = torch.randn(1, 1024, 768)
        
        # Time strided attention
        start = time.time()
        _ = strided(x)
        strided_time = time.time() - start
        
        # Time performer
        start = time.time()
        _ = performer(x)
        performer_time = time.time() - start
        
        # Both should complete without OOM
        assert strided_time > 0
        assert performer_time > 0
    
    def test_mamba_efficiency(self):
        """Test Mamba efficiency on long sequences."""
        mamba = MambaLayer(dim=768, state_size=16)
        
        # Very long sequence
        x = torch.randn(1, 4096, 768)
        
        import time
        start = time.time()
        output = mamba(x)
        elapsed = time.time() - start
        
        # Should be relatively fast
        assert elapsed < 10.0  # Reasonable time bound
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
