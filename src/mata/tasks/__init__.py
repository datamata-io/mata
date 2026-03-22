"""Task protocols initialization."""

from .base import (
    ClassifyAdapter,
    DepthAdapter,
    DetectAdapter,
    EmbedAdapter,
    SegmentAdapter,
    TaskAdapter,
    TrackAdapter,
)

__all__ = [
    "TaskAdapter",
    "DetectAdapter",
    "SegmentAdapter",
    "ClassifyAdapter",
    "DepthAdapter",
    "TrackAdapter",
    "EmbedAdapter",
]
