# Efficient Attention Mechanisms

This document provides a comprehensive guide to the efficient attention mechanisms implemented in HierarchicalVLM.

## Overview

The efficient attention module provides three key innovation areas:

1. **Sparse Attention Patterns** - Reduce complexity from O(n²) to O(n²/k)
2. **Linear Attention** - Achieve O(n) complexity using kernel methods or state space models
3. **Cross-Memory Attention** - Intelligently fuse historical memory with current features

## Sparse Attention Patterns

### Strided Attention

Strided attention reduces computational cost by having each token attend only to every k-th token.

**Complexity:** O(n²/stride)

**Use Cases:**
- Long sequences where full attention is prohibitive
- When approximate attention is acceptable
- Combined with other attention patterns

**Example:**
```python
from hierarchicalvlm.attention.sparse import StridedAttention

# Create strided attention with stride=4
attn = StridedAttention(stride=4, dim=768, num_heads=12, dropout=0.1)

# Forward pass
x = torch.randn(batch_size=2, seq_len=1024, dim=768)
output = attn(x)  # (2, 1024, 768)
```

**Parameters:**
- `stride` (int): How many tokens to skip. Larger stride = more efficiency but less accuracy
- `dim` (int): Model dimension
- `num_heads` (int): Number of attention heads
- `dropout` (float): Attention dropout rate

### Local+Global Attention

Combines local windowed attention (efficient) with global representative tokens (context-aware).

**Complexity:** O(n * window_size) + O(n * num_global_tokens)

**Key Features:**
- Each token attends to tokens within a local window
- Selected representative tokens attend globally
- Balances efficiency and context modeling

**Example:**
```python
from hierarchicalvlm.attention.sparse import LocalGlobalAttention

# Create local+global attention
attn = LocalGlobalAttention(
    local_window=64,           # Window size for local attention
    num_global_tokens=4,       # Number of global representatives
    dim=768,
    num_heads=12,
    dropout=0.1
)

x = torch.randn(2, 512, 768)
output = attn(x)  # (2, 512, 768)
```

**Parameters:**
- `local_window` (int): Size of local attention window. Typical values: 32-128
- `num_global_tokens` (int): Number of representative tokens. Typical values: 2-8
- `dim` (int): Model dimension
- `num_heads` (int): Number of attention heads

### Cross-Memory Attention

Uses cross-attention to intelligently merge long-term memory with current local features, replacing naive concatenation.

**Complexity:** O(n * m) where n=local_seq_len, m=memory_seq_len

**Key Features:**
- Selective attention to relevant historical information
- Learned fusion strategy
- Supports residual connections

**Example:**
```python
from hierarchicalvlm.attention.sparse import CrossMemoryAttention

# Create cross-memory attention
cross_attn = CrossMemoryAttention(dim=768, num_heads=12, dropout=0.1)

# Local features from current frames
local_features = torch.randn(2, 64, 768)

# Memory tokens from previous video segments
memory_tokens = torch.randn(2, 32, 768)

# Fuse with residual connection
output = cross_attn(local_features, memory_tokens, residual=True)

# Or fuse without residual
output = cross_attn(local_features, memory_tokens, residual=False)

# Custom fusion ratio (0=local only, 1=memory-only)
fused = cross_attn.fuse_with_memory(local_features, memory_tokens, fusion_ratio=0.5)
```

**Parameters:**
- `dim` (int): Model dimension
- `num_heads` (int): Number of attention heads
- `dropout` (float): Attention dropout rate

## Linear Attention

### Performer Attention

Fast transformers with FAVOR+ (Fast Attention Via positive Orthogonal Random Features) achieving O(n) complexity.

**Key Ideas:**
- Approximates softmax(QK^T)V using random features
- Achieves linear complexity in sequence length
- No accuracy loss for long sequences

**Complexity:** O(n * num_features)

**Example:**
```python
from hierarchicalvlm.attention.linear import PerformerAttention

# Create Performer attention
attn = PerformerAttention(
    dim=768,
    num_heads=12,
    num_random_features=256,  # More features = more accurate
    kernel_type='elu',        # 'elu' or 'relu'
    dropout=0.1
)

# Process very long sequences efficiently
x = torch.randn(batch_size=1, seq_len=8192, dim=768)
output = attn(x)

# Memory usage is O(n) instead of O(n²)
```

**Parameters:**
- `dim` (int): Model dimension
- `num_heads` (int): Number of attention heads
- `num_random_features` (int): Number of random features. Typical values: 64-512
- `kernel_type` (str): Kernel type ('elu' or 'relu')
- `dropout` (float): Attention dropout rate

**When to use:**
- Very long sequences (>2048 tokens)
- Memory-constrained environments
- When linear complexity is critical

### Mamba Layer

Linear-time sequence modeling using selective state space models.

**Key Ideas:**
- State space model (SSM) variant
- Selective state updates based on input
- Linear complexity with efficient computation

**Complexity:** O(n)

**Example:**
```python
from hierarchicalvlm.attention.linear import MambaLayer

# Create Mamba layer
mamba = MambaLayer(
    dim=768,
    state_size=16,           # Internal state dimension
    expand_factor=2.0,       # Expansion factor for intermediate dim
    dropout=0.1
)

# Process sequences
x = torch.randn(batch_size=2, seq_len=4096, dim=768)
output = mamba(x)

# Efficient even for very long sequences
```

**Parameters:**
- `dim` (int): Model dimension
- `state_size` (int): Dimension of internal state. Typical values: 8-32
- `expand_factor` (float): Expansion for intermediate dimension
- `dropout` (float): Dropout rate
- `dt_init` (str): Initialization method for discretization step ('constant' or 'random')

**When to use:**
- Very long sequences (>4096 tokens)
- When efficiency is paramount
- As alternative to attention for certain tasks

## Hierarchical Attention Block

Complete attention block with layer normalization and feed-forward network.

**Example:**
```python
from hierarchicalvlm.attention.sparse import HierarchicalAttentionBlock

# Create a strided attention block with FFN
block = HierarchicalAttentionBlock(
    dim=768,
    num_heads=12,
    attention_type='strided',
    stride=4,
    dropout=0.1
)

x = torch.randn(2, 512, 768)
output = block(x)  # (2, 512, 768)

# With memory for cross-attention
if using_cross_memory:
    block = HierarchicalAttentionBlock(
        attention_type='cross_memory',
        dim=768,
        num_heads=12
    )
    memory = torch.randn(2, 64, 768)
    output = block(x, memory=memory)
```

**Attention Types:**
- `'strided'` - Strided attention
- `'local_global'` - Local+Global attention
- `'cross_memory'` - Cross-Memory attention

## Efficiency Comparison

| Attention Type | Complexity | Memory | Speed | Context |
|---|---|---|---|---|
| Standard | O(n²) | O(n²) | Slow | Full |
| Strided | O(n²/k) | O(n²/k) | Medium | Limited |
| Local+Global | O(n*w) | O(n*w) | Fast | Local+Global |
| Performer | O(n) | O(n) | Very Fast | Full |
| Mamba | O(n) | O(n) | Very Fast | State |

## Best Practices

### Choosing an Attention Type

1. **Short sequences (<512)**: Use standard attention (LocalGlobalAttention)
2. **Medium sequences (512-2048)**: Use StridedAttention or LocalGlobalAttention
3. **Long sequences (>2048)**: Use PerformerAttention or MambaLayer
4. **With historical context**: Use CrossMemoryAttention
5. **Memory-critical**: Always use linear attention (Performer/Mamba)

### Configuration Guidelines

```python
# For short sequences with full context
block = HierarchicalAttentionBlock(
    attention_type='local_global',
    local_window=128,
    num_global_tokens=8
)

# For long sequences, efficient
block = HierarchicalAttentionBlock(
    attention_type='strided',
    stride=8
)

# For very long sequences, linear
block = LinearAttentionBlock(
    attention_type='performer',
    num_random_features=512
)

# For extremely long sequences, state-space
block = LinearAttentionBlock(
    attention_type='mamba',
    state_size=32
)
```

## Advanced Usage

### Combining Multiple Attention Types

```python
# Alternating attention layers
layers = nn.ModuleList([
    HierarchicalAttentionBlock(attention_type='strided', stride=4),
    HierarchicalAttentionBlock(attention_type='local_global'),
    LinearAttentionBlock(attention_type='performer'),
])

# Forward pass through all layers
x = input_tokens
for layer in layers:
    x = layer(x)
```

### Dynamic Attention Selection

```python
def get_attention(seq_len, dim=768):
    if seq_len < 512:
        return HierarchicalAttentionBlock(attention_type='local_global')
    elif seq_len < 4096:
        return HierarchicalAttentionBlock(attention_type='strided', stride=4)
    else:
        return LinearAttentionBlock(attention_type='performer')
```

## Benchmarks

### Memory Usage (per layer, batch_size=1)
- Standard (n=1024): ~1.5 GB
- Strided (stride=4): ~0.4 GB
- Performer: ~0.2 GB
- Mamba: ~0.15 GB

### Speed (sequence length, batch_size=1)
- Standard (n=512): ~50ms
- Strided (n=512): ~15ms
- Performer (n=8192): ~80ms
- Mamba (n=8192): ~60ms

*Note: Benchmarks are approximate and vary by hardware*

## References

- **Strided Attention**: [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)
- **Local+Global**: [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- **Performer**: [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
- **Mamba**: [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.08129)

## Testing

Run the comprehensive test suite:

```bash
cd tests
pytest test_attention.py -v

# Run specific test
pytest test_attention.py::TestPerformerAttention -v

# Run with coverage
pytest test_attention.py --cov=hierarchicalvlm.attention
```

## Troubleshooting

### High Memory Usage
- Reduce `num_global_tokens` in LocalGlobalAttention
- Use smaller `num_random_features` in Performer
- Switch to Mamba for very long sequences

### Slow Training
- Use Performer or Mamba for long sequences
- Reduce `num_heads` (though this may hurt accuracy)
- Use mixed precision training

### Poor Convergence
- Ensure proper scaling: `scaling = head_dim ** -0.5`
- Check gradient flow with `register_forward_hook`
- Consider longer warmup period for Performer

## Contributing

To add new attention mechanisms:

1. Extend appropriate base class
2. Implement `forward()` method
3. Add tests in `test_attention.py`
4. Update documentation
5. Submit PR

