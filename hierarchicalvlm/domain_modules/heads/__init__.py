"""
Task-Specific Heads - Output Layers for Different Tasks

This module provides specialized prediction heads:
- ActionDetectionHead: Temporal action localization
- VideoQAHead: Video question answering
- VideoCaptioningHead: Dense video captioning
- MultiTaskHead: Unified multi-task interface
"""

from .task_heads import (
    ActionDetectionHead,
    VideoQAHead,
    VideoCaptioningHead,
    MultiTaskHead
)

__all__ = [
    'ActionDetectionHead',
    'VideoQAHead',
    'VideoCaptioningHead',
    'MultiTaskHead'
]
