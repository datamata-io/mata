"""IndexVideo node — build a searchable VideoIndex from a video file."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.video_index_data import VideoIndexData
from mata.core.artifacts.video_path import VideoPath
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class IndexVideo(Node):
    """Index a video file into a searchable embedding store.

    Samples frames from the video, embeds them with the specified provider,
    and stores the result as a :class:`VideoIndexData` artifact for downstream
    :class:`EmbeddingSearch` nodes.

    Args:
        using: Provider name (resolved at runtime from providers dict).
        mode: Sampling mode — ``"frame"`` (one embed per frame) or
              ``"clip"`` (one embed per clip segment).  Defaults to
              ``"frame"``.
        sample_fps: Frames per second to sample from the video.  Defaults
              to ``1.0``.
        out: Name of the output artifact key.  Defaults to
             ``"video_index"``.
        name: Optional human-readable node name.
        **embed_kwargs: Extra keyword arguments forwarded to
              :func:`mata.recognition.index_video`.

    Inputs:
        video (VideoPath): Path to the video file to index.

    Outputs:
        video_index (VideoIndexData): Searchable index of frame embeddings.

    Example:
        ```python
        from mata.nodes import IndexVideo, EmbeddingSearch
        from mata.core.graph import Graph

        graph = (
            Graph("urban_search")
            .then(IndexVideo(using="embedder", sample_fps=1.0))
            .then(EmbeddingSearch(using="embedder", text=["red car", "pedestrian"]))
        )
        result = graph.run(video="traffic.mp4", providers={"embedder": embed_model})
        ```
    """

    inputs: dict[str, type[Artifact]] = {"video": VideoPath}
    outputs: dict[str, type[Artifact]] = {"video_index": VideoIndexData}

    def __init__(
        self,
        using: str,
        mode: str = "frame",
        sample_fps: float = 1.0,
        out: str = "video_index",
        name: str | None = None,
        **embed_kwargs,
    ) -> None:
        super().__init__(name=name)
        self.using = using
        self.mode = mode
        self.sample_fps = sample_fps
        self.out = out
        self.embed_kwargs = embed_kwargs
        self.inputs = {"video": VideoPath}
        self.outputs = {out: VideoIndexData}

    def run(self, ctx: ExecutionContext, **inputs: VideoPath) -> dict[str, Artifact]:
        """Build a VideoIndex from the supplied video path.

        Args:
            ctx: Execution context (provides access to providers and metrics).
            **inputs: Single :class:`VideoPath` artifact keyed by ``"video"``.

        Returns:
            Dict with key ``self.out`` mapping to a :class:`VideoIndexData`.
        """
        from mata.recognition import index_video

        video_path: VideoPath = next(iter(inputs.values()))
        adapter = ctx.get_provider("embed", self.using)

        vi = index_video(
            video_path.path,
            adapter=adapter,
            mode=self.mode,
            sample_fps=self.sample_fps,
            **self.embed_kwargs,
        )

        ctx.record_metric(self.name, "indexed_frames", len(vi.frame_map))

        return {
            self.out: VideoIndexData(
                index=vi,
                meta={"model": self.using, "mode": self.mode, "sample_fps": self.sample_fps},
            )
        }
