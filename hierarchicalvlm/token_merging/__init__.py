"""
Adaptive Token Merging Strategy

Implements:
- Content-aware merging based on motion/saliency
- Optical flow and frame difference analysis
- Dynamic compression based on scene properties
"""

from . import motion
from . import saliency

__all__ = ["motion", "saliency"]
