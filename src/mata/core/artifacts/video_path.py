"""VideoPath artifact — thin wrapper for a video file path.

Enables graph pipelines to receive a video file location as a typed artifact
via ``input.video``, keeping the graph runtime agnostic of how the video
will be consumed (frame extraction, metadata lookup, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mata.core.artifacts.base import Artifact


@dataclass(frozen=True)
class VideoPath(Artifact):
    """Artifact carrying a video file path through graph pipelines.

    Attributes:
        path: Absolute or project-relative path to the video file.

    Examples:
        >>> vp = VideoPath(path="videos/dashcam.mp4")
        >>> vp.path
        'videos/dashcam.mp4'
    """

    path: str

    def validate(self) -> None:
        if not self.path:
            raise ValueError("VideoPath.path must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoPath:
        return cls(path=data["path"])
