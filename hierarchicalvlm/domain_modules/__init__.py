"""
Domain-Specific Fine-Tuning Modules

Implements:
- Modular adapters for specific domains (sports, tutorials, news, etc.)
- LoRA and parameter-efficient fine-tuning
- Task-specific heads (action detection, QA, captioning)
"""

from . import adapters
from . import heads

__all__ = ["adapters", "heads"]
