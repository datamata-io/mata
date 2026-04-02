"""EmbeddingSearch node — query a VideoIndex with natural-language text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.search_results import QueryResult, SearchResults
from mata.core.artifacts.video_index_data import VideoIndexData
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class EmbeddingSearch(Node):
    """Search a VideoIndex with one or more text queries.

    Embeds each query string using the specified provider and performs a
    nearest-neighbour search against the :class:`VideoIndexData` artifact
    produced by :class:`IndexVideo`.

    Args:
        using: Provider name (resolved at runtime from providers dict).
        text: A single query string or a list of query strings.
        src: Name of the input :class:`VideoIndexData` artifact key.
              Defaults to ``"video_index"``.
        out: Name of the output :class:`SearchResults` artifact key.
              Defaults to ``"search_results"``.
        top_k: Number of top matches to return per query.  Defaults to ``5``.
        threshold: Minimum cosine-similarity threshold; matches below this
              score are discarded.  ``None`` disables filtering.
        name: Optional human-readable node name.
        **embed_kwargs: Reserved for future use.

    Inputs:
        video_index (VideoIndexData): Pre-built index from :class:`IndexVideo`.

    Outputs:
        search_results (SearchResults): Per-query match results.

    Example:
        ```python
        graph = (
            Graph("urban_search")
            .then(IndexVideo(using="embedder", sample_fps=1.0))
            .then(EmbeddingSearch(
                using="embedder",
                text=["red car", "pedestrian crossing"],
                top_k=5,
            ))
        )
        result = graph.run(video="traffic.mp4", providers={"embedder": embed_model})
        for qr in result["search_results"].results:
            print(qr.query, qr.matches)
        ```
    """

    inputs: dict[str, type[Artifact]] = {"video_index": VideoIndexData}
    outputs: dict[str, type[Artifact]] = {"search_results": SearchResults}

    def __init__(
        self,
        using: str,
        text: str | list[str],
        src: str = "video_index",
        out: str = "search_results",
        top_k: int = 5,
        threshold: float | None = None,
        name: str | None = None,
        **embed_kwargs,
    ) -> None:
        super().__init__(name=name)
        self.using = using
        self.text = [text] if isinstance(text, str) else list(text)
        self.top_k = top_k
        self.threshold = threshold
        self.embed_kwargs = embed_kwargs
        self.out = out
        self.inputs = {src: VideoIndexData}
        self.outputs = {out: SearchResults}

    def run(self, ctx: ExecutionContext, **inputs: VideoIndexData) -> dict[str, Artifact]:
        """Execute all queries against the provided VideoIndex.

        Args:
            ctx: Execution context (provides access to providers and metrics).
            **inputs: Single :class:`VideoIndexData` artifact.

        Returns:
            Dict with key ``self.out`` mapping to a :class:`SearchResults`.
        """
        import numpy as np

        vid_data: VideoIndexData = next(iter(inputs.values()))
        adapter = ctx.get_provider("embed", self.using)

        query_results: list[QueryResult] = []
        for query in self.text:
            vec = adapter.embed(query)  # (1, D) float32
            vec = np.asarray(vec, dtype=np.float32).ravel()  # (D,)
            matches = vid_data.index.search(vec, top_k=self.top_k, threshold=self.threshold)
            query_results.append(QueryResult(query=query, matches=tuple(matches)))

        ctx.record_metric(self.name, "num_queries", len(self.text))

        return {
            self.out: SearchResults(
                results=tuple(query_results),
                meta={"model": self.using, "top_k": self.top_k},
            )
        }
