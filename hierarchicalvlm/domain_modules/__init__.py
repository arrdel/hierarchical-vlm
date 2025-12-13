"""
Domain-Specific Fine-Tuning Modules for HierarchicalVLM

This package provides comprehensive domain specialization and parameter-efficient
fine-tuning for long video understanding tasks.

Core Components:
- Adapters: LoRA-based parameter-efficient adaptation
- Heads: Task-specific prediction heads (action, QA, captioning)
- Domain Experts: Multi-domain specialization and routing

Example:
    >>> from hierarchicalvlm.domain_modules.adapters import LoRAAdapter
    >>> from hierarchicalvlm.domain_modules.heads import ActionDetectionHead
    >>> 
    >>> lora = LoRAAdapter(dim=768, rank=8)
    >>> head = ActionDetectionHead(num_classes=150)
    >>> 
    >>> video_features = torch.randn(2, 32, 768)
    >>> adapted = lora(video_features)
    >>> output = head(adapted)
"""

from . import adapters
from . import heads
from . import domain_experts

__all__ = [
    'adapters',
    'heads',
    'domain_experts',
]
