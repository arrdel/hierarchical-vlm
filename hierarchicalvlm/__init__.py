"""
HierarchicalVLM: Adaptive Long Video Understanding with Efficient Attention and Domain-Specific Adaptation

See more, understand better, compute less.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import attention
from . import domain_modules
from . import token_merging
from . import model
from . import utils

__all__ = [
    "attention",
    "domain_modules",
    "token_merging",
    "model",
    "utils",
]
