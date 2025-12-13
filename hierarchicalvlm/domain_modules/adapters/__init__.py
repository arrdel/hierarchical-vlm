"""
Domain Adapters - LoRA and Parameter-Efficient Fine-Tuning

This module provides parameter-efficient adaptation methods:
- LinearLoRA: Low-rank adaptation for linear layers
- AttentionLoRA: Low-rank adaptation for attention layers
- LoRAAdapter: Complete adapter module with fusion
- LoRALayerWrapper: Wrapper for existing layers
"""

from .lora import (
    LinearLoRA,
    AttentionLoRA,
    LoRAAdapter,
    LoRALayerWrapper,
    LoRAConfig
)

__all__ = [
    'LinearLoRA',
    'AttentionLoRA',
    'LoRAAdapter',
    'LoRALayerWrapper',
    'LoRAConfig'
]
