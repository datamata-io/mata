"""VideoIndexData artifact — wraps a VideoIndex for graph pipelines.

Produced by the IndexVideo node; consumed by EmbeddingSearch and any
other node that needs to search an indexed video by embedding similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class VideoIndexData(Artifact):
    """Artifact wrapping a :class:`~mata.recognition.VideoIndex`.

    Carries the full gallery + timing metadata produced by ``index_video()``
    through the graph so downstream nodes (e.g. EmbeddingSearch) can search
    it by cosine similarity.

    Attributes:
        index: The VideoIndex instance (gallery + frame_map + timing).
        meta: Optional provenance metadata.

    Examples:
        >>> from mata.recognition import index_video
        >>> vi = index_video("video.mp4", model="openai/clip-vit-base-patch32")
        >>> artifact = VideoIndexData(index=vi)
        >>> len(artifact.index.gallery)
        120
    """

    index: Any  # VideoIndex — typed as Any to avoid circular imports
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.index is None:
            raise ValueError("VideoIndexData.index must not be None.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict (delegates to VideoIndex.to_dict)."""
        return {
            "index": self.index.to_dict(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoIndexData:
        from mata.recognition.video_index import VideoIndex

        return cls(
            index=VideoIndex.from_dict(data["index"]),
            meta=data.get("meta", {}),
        )
