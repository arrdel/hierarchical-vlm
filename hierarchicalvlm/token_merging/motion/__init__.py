"""
Motion Analysis for Adaptive Token Merging

Implements optical flow and motion-based token importance estimation.
"""

from .optical_flow import (
    DenseOpticalFlow,
    MotionMagnitude,
    MotionBasedCompression,
)

__all__ = [
    'DenseOpticalFlow',
    'MotionMagnitude',
    'MotionBasedCompression',
]
