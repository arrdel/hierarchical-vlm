"""
Quick start examples for efficient attention mechanisms.

This script demonstrates how to use the different attention types
in HierarchicalVLM.
"""

import torch
from hierarchicalvlm.attention import (
    # Sparse attention
    StridedAttention,
    LocalGlobalAttention,
    CrossMemoryAttention,
    HierarchicalAttentionBlock,
    # Linear attention
    PerformerAttention,
    MambaLayer,
    LinearAttentionBlock,
)


def example_strided_attention():
    """Example: Strided Attention for long sequences."""
    print("=" * 60)
    print("Example 1: Strided Attention")
    print("=" * 60)
    
    # Create strided attention
    attn = StridedAttention(stride=4, dim=768, num_heads=12, dropout=0.1)
    
    # Create input
    batch_size, seq_len, dim = 2, 1024, 768
    x = torch.randn(batch_size, seq_len, dim)
    
    # Forward pass
    output = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Stride: 4 (attends to every 4th token)")
    print(f"Complexity reduction: 4x memory, 4x faster")
    print()


def example_local_global_attention():
    """Example: Local+Global Attention for balanced efficiency."""
    print("=" * 60)
    print("Example 2: Local+Global Attention")
    print("=" * 60)
    
    # Create local+global attention
    attn = LocalGlobalAttention(
        local_window=64,
        num_global_tokens=4,
        dim=768,
        num_heads=12,
        dropout=0.1
    )
    
    # Create input
    x = torch.randn(2, 512, 768)
    
    # Forward pass
    output = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Local window: 64 tokens (efficient local context)")
    print(f"Global tokens: 4 (maintains global information)")
    print()


def example_cross_memory_attention():
    """Example: Cross-Memory Attention for temporal modeling."""
    print("=" * 60)
    print("Example 3: Cross-Memory Attention")
    print("=" * 60)
    
    # Create cross-memory attention
    cross_attn = CrossMemoryAttention(dim=768, num_heads=12, dropout=0.1)
    
    # Local features from current frames
    local_features = torch.randn(2, 64, 768)
    
    # Memory tokens from previous video segments
    memory_tokens = torch.randn(2, 32, 768)
    
    # Fuse with residual connection
    output = cross_attn(local_features, memory_tokens, residual=True)
    
    print(f"Local features shape: {local_features.shape}")
    print(f"Memory tokens shape: {memory_tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Method: Cross-attention fusion + residual connection")
    
    # Custom fusion ratio
    fused = cross_attn.fuse_with_memory(local_features, memory_tokens, fusion_ratio=0.5)
    print(f"Custom fusion (ratio=0.5) shape: {fused.shape}")
    print()


def example_performer_attention():
    """Example: Performer for linear-time attention on very long sequences."""
    print("=" * 60)
    print("Example 4: Performer Attention (Linear Complexity)")
    print("=" * 60)
    
    # Create Performer attention
    attn = PerformerAttention(
        dim=768,
        num_heads=12,
        num_random_features=256,
        kernel_type='elu',
        dropout=0.1
    )
    
    # Very long sequence (where Performer excels)
    batch_size, seq_len, dim = 1, 8192, 768
    x = torch.randn(batch_size, seq_len, dim)
    
    # Forward pass
    output = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Complexity: O(n) instead of O(n²)")
    print(f"Method: FAVOR+ kernel approximation")
    print(f"Random features: 256 (for kernel approximation)")
    print()


def example_mamba_layer():
    """Example: Mamba for state-space sequence modeling."""
    print("=" * 60)
    print("Example 5: Mamba Layer (State Space Model)")
    print("=" * 60)
    
    # Create Mamba layer
    mamba = MambaLayer(
        dim=768,
        state_size=16,
        expand_factor=2.0,
        dropout=0.1
    )
    
    # Long sequence
    x = torch.randn(2, 4096, 768)
    
    # Forward pass
    output = mamba(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"State size: 16 (internal state dimension)")
    print(f"Complexity: O(n) with efficient computation")
    print(f"Advantage: Better for very long sequences")
    print()


def example_hierarchical_attention_block():
    """Example: Using HierarchicalAttentionBlock wrapper."""
    print("=" * 60)
    print("Example 6: Hierarchical Attention Block")
    print("=" * 60)
    
    # Type 1: Strided attention block
    block_strided = HierarchicalAttentionBlock(
        dim=768,
        num_heads=12,
        attention_type='strided',
        stride=4,
        dropout=0.1
    )
    
    # Type 2: Local+global attention block
    block_local_global = HierarchicalAttentionBlock(
        dim=768,
        num_heads=12,
        attention_type='local_global',
        local_window=64,
        num_global_tokens=4,
        dropout=0.1
    )
    
    # Type 3: Cross-memory attention block
    block_cross_memory = HierarchicalAttentionBlock(
        dim=768,
        num_heads=12,
        attention_type='cross_memory',
        dropout=0.1
    )
    
    x = torch.randn(2, 256, 768)
    
    # Process with each block
    out1 = block_strided(x)
    out2 = block_local_global(x)
    out3 = block_cross_memory(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Strided output shape: {out1.shape}")
    print(f"Local+Global output shape: {out2.shape}")
    print(f"Cross-Memory output shape: {out3.shape}")
    print()


def example_linear_attention_block():
    """Example: Using LinearAttentionBlock wrapper."""
    print("=" * 60)
    print("Example 7: Linear Attention Block")
    print("=" * 60)
    
    # Performer block
    block_performer = LinearAttentionBlock(
        dim=768,
        num_heads=12,
        attention_type='performer',
        num_random_features=256,
        dropout=0.1
    )
    
    # Mamba block
    block_mamba = LinearAttentionBlock(
        dim=768,
        attention_type='mamba',
        state_size=16,
        dropout=0.1
    )
    
    x = torch.randn(1, 2048, 768)
    
    out_performer = block_performer(x)
    out_mamba = block_mamba(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Performer output shape: {out_performer.shape}")
    print(f"Mamba output shape: {out_mamba.shape}")
    print(f"Both achieve O(n) complexity!")
    print()


def example_stacking_attention():
    """Example: Stacking multiple attention layers."""
    print("=" * 60)
    print("Example 8: Stacking Multiple Attention Layers")
    print("=" * 60)
    
    import torch.nn as nn
    
    # Create a sequence of attention layers
    layers = nn.ModuleList([
        HierarchicalAttentionBlock(attention_type='strided', stride=4),
        HierarchicalAttentionBlock(attention_type='local_global'),
        LinearAttentionBlock(attention_type='performer'),
    ])
    
    # Forward pass through all layers
    x = torch.randn(1, 1024, 768)
    
    for i, layer in enumerate(layers):
        x = layer(x)
        print(f"After layer {i+1}: shape {x.shape}")
    
    print(f"\nFinal output shape: {x.shape}")
    print()


def compare_attention_types():
    """Compare different attention types."""
    print("=" * 60)
    print("Attention Types Comparison")
    print("=" * 60)
    
    seq_len = 2048
    x = torch.randn(1, seq_len, 768)
    
    import time
    
    # Strided
    attn_strided = StridedAttention(stride=4, dim=768, num_heads=12)
    start = time.time()
    _ = attn_strided(x)
    strided_time = time.time() - start
    
    # Local+Global
    attn_local_global = LocalGlobalAttention(dim=768, num_heads=12)
    start = time.time()
    _ = attn_local_global(x)
    local_global_time = time.time() - start
    
    # Performer
    attn_performer = PerformerAttention(dim=768, num_heads=12)
    start = time.time()
    _ = attn_performer(x)
    performer_time = time.time() - start
    
    # Mamba
    mamba = MambaLayer(dim=768, state_size=16)
    start = time.time()
    _ = mamba(x)
    mamba_time = time.time() - start
    
    print(f"Sequence length: {seq_len}")
    print(f"\nTiming (seconds):")
    print(f"  Strided:       {strided_time:.4f}s")
    print(f"  Local+Global:  {local_global_time:.4f}s")
    print(f"  Performer:     {performer_time:.4f}s")
    print(f"  Mamba:         {mamba_time:.4f}s")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HierarchicalVLM: Efficient Attention Examples")
    print("=" * 60 + "\n")
    
    example_strided_attention()
    example_local_global_attention()
    example_cross_memory_attention()
    example_performer_attention()
    example_mamba_layer()
    example_hierarchical_attention_block()
    example_linear_attention_block()
    example_stacking_attention()
    compare_attention_types()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
