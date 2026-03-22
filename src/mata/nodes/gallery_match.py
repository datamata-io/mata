"""GalleryMatchNode — graph node that matches embeddings against a Gallery.

Consumes an Embeddings artifact and produces a Matches artifact by running
cosine similarity search against a pre-populated Gallery.

Typical usage in a recognition pipeline:

    Detect >> ExtractROIs >> Embed >> GalleryMatchNode(gallery=gallery)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mata.core.artifacts.embeddings import Embeddings
    from mata.core.artifacts.matches import Matches
    from mata.recognition.gallery import Gallery


class GalleryMatchNode:
    """Graph node: Embeddings → Matches via cosine similarity search.

    Wraps a :class:`~mata.recognition.Gallery` as a graph-composable node.
    Accepts an Embeddings artifact and returns a Matches artifact, preserving
    per-instance ID linkage.

    Args:
        gallery: Pre-populated Gallery instance.
        top_k: Maximum number of gallery matches to return per embedding.
        threshold: Minimum cosine similarity (overrides gallery default if set).
        src: Name of the input Embeddings artifact key. Default: "embeddings".
        out: Name of the output Matches artifact key. Default: "matches".
        name: Optional human-readable node name for graph display.

    Example:
        >>> from mata import Gallery
        >>> from mata.nodes import GalleryMatchNode
        >>>
        >>> gallery = Gallery()
        >>> gallery.add("alice", alice_embedding)
        >>>
        >>> node = GalleryMatchNode(gallery=gallery, top_k=1, threshold=0.6)
        >>> # Use in graph:
        >>> graph = (
        ...     Graph("recognition")
        ...     .then(Detect(using="detector"))
        ...     .then(ExtractROIs(src_dets="dets"))
        ...     .then(Embed(using="encoder"))
        ...     .then(GalleryMatchNode(gallery=gallery, top_k=1))
        ... )
    """

    def __init__(
        self,
        gallery: Any,
        top_k: int = 1,
        threshold: float | None = None,
        src: str = "embeddings",
        out: str = "matches",
        name: str | None = None,
    ) -> None:
        self._gallery = gallery
        self._top_k = top_k
        self._threshold = threshold
        self._src = src
        self._out = out
        self.name = name or "GalleryMatchNode"

    @property
    def inputs(self) -> dict[str, Any]:
        from mata.core.artifacts.embeddings import Embeddings

        return {self._src: Embeddings}

    @property
    def outputs(self) -> dict[str, Any]:
        from mata.core.artifacts.matches import Matches

        return {self._out: Matches}

    def run(self, ctx: Any, **artifacts: Any) -> dict[str, Any]:
        """Match embeddings against the gallery.

        Args:
            ctx: ExecutionContext (unused; gallery is injected at construction).
            **artifacts: Must contain the Embeddings artifact keyed by self._src.

        Returns:
            Dict mapping self._out → Matches artifact.

        Raises:
            ValueError: If the required embeddings artifact is missing.
        """
        from mata.core.artifacts.matches import MatchEntry, Matches

        emb_artifact = artifacts.get(self._src)
        if emb_artifact is None:
            raise ValueError(
                f"GalleryMatchNode: missing input artifact '{self._src}'. "
                f"Available keys: {list(artifacts.keys())}"
            )

        vectors = emb_artifact.vectors  # (N, D) float32
        instance_ids = emb_artifact.instance_ids  # tuple[str, ...]

        all_batch_matches = self._gallery.search_batch(
            vectors, top_k=self._top_k, threshold=self._threshold
        )

        entries: list[MatchEntry] = []
        for i, matches in enumerate(all_batch_matches):
            iid = instance_ids[i] if i < len(instance_ids) else f"emb_{i:04d}"
            best = matches[0] if matches else None
            entries.append(
                MatchEntry(
                    instance_id=iid,
                    label=best.label if best is not None else "unknown",
                    similarity=best.similarity if best is not None else 0.0,
                    all_matches=[m.to_dict() for m in matches],
                )
            )

        return {self._out: Matches(entries=entries, meta={})}

    def __repr__(self) -> str:
        return (
            f"GalleryMatchNode(gallery_size={self._gallery.size}, "
            f"top_k={self._top_k}, threshold={self._threshold})"
        )
