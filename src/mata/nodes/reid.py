"""ReID node for cross-camera re-identification in graph pipelines.

Consumes Tracks and Embeddings artifacts, publishes per-track embeddings
to Valkey via ReIDBridge, and queries for cross-camera matches, producing
a CrossMatches artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.cross_matches import CrossMatch, CrossMatches
from mata.core.artifacts.embeddings import Embeddings
from mata.core.artifacts.tracks import Tracks
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class ReID(Node):
    """Cross-camera re-identification node.

    For each active track with a corresponding embedding, publishes the
    embedding to Valkey via a :class:`~mata.trackers.ReIDBridge` provider
    and queries for matches from other cameras. Produces a
    :class:`~mata.core.artifacts.CrossMatches` artifact carrying all
    cross-camera matches found in the current frame.

    Args:
        using: Provider name (resolved to a ``ReIDBridge`` instance at runtime).
        tracks_src: Input artifact key for the Tracks artifact (default: ``"tracks"``).
        embeddings_src: Input artifact key for the Embeddings artifact
            (default: ``"embeddings"``).
        out: Output artifact key for the CrossMatches artifact
            (default: ``"cross_matches"``).
        top_k: Maximum number of cross-camera matches to return per track
            (default: ``1``).
        name: Optional human-readable node name.

    Inputs:
        tracks (Tracks): Active track objects from current frame.
        embeddings (Embeddings): Feature vectors aligned to the active tracks.

    Outputs:
        cross_matches (CrossMatches): Cross-camera re-identification results.

    Example:
        ```python
        from mata.nodes import Track, ReID
        from mata.nodes.embed import Embed
        from mata.trackers import ReIDBridge

        bridge = ReIDBridge("valkey://localhost:6379", camera_id="cam-1")

        graph = (Graph("reid_pipeline")
            .then(Track(using="tracker", out="tracks"))
            .then(Embed(using="encoder", out="embeddings"))
            .then(ReID(using="bridge", out="cross_matches"))
        )

        results = mata.infer(graph, image="frame.jpg",
                             providers={"tracker": tracker,
                                        "encoder": encoder,
                                        "bridge": bridge})
        ```
    """

    inputs: dict[str, type[Artifact]] = {"tracks": Tracks, "embeddings": Embeddings}
    outputs: dict[str, type[Artifact]] = {"cross_matches": CrossMatches}

    def __init__(
        self,
        using: str,
        tracks_src: str = "tracks",
        embeddings_src: str = "embeddings",
        out: str = "cross_matches",
        top_k: int = 1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.using = using
        self.tracks_src = tracks_src
        self.embeddings_src = embeddings_src
        self.out = out
        self.top_k = top_k
        # Dynamic artifact mapping — configurable input/output keys
        self.inputs = {tracks_src: Tracks, embeddings_src: Embeddings}
        self.outputs = {out: CrossMatches}

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        """Publish embeddings and query for cross-camera matches.

        Args:
            ctx: Execution context (provides access to the ReIDBridge provider).
            **inputs: Input artifacts keyed by name (tracks + embeddings).

        Returns:
            Dict with single key (``self.out``) mapping to CrossMatches artifact.
            Returns empty CrossMatches if no tracks or no embeddings are provided.
        """
        bridge = ctx.get_provider("reid", self.using)
        tracks_art: Tracks = inputs[self.tracks_src]  # type: ignore[assignment]
        emb_art: Embeddings = inputs[self.embeddings_src]  # type: ignore[assignment]

        active = tracks_art.get_active_tracks().tracks
        vecs = emb_art.vectors
        n = vecs.shape[0] if vecs.ndim == 2 else 0

        match_list: list[CrossMatch] = []
        for idx, track in enumerate(active):
            if idx >= n:
                break
            emb = vecs[idx]
            bridge.publish(
                track_id=track.track_id,
                embedding=emb,
                bbox=track.bbox,
                label=track.label_id or 0,
            )
            results = bridge.query(
                emb,
                exclude_camera=bridge.camera_id,
                top_k=self.top_k,
            )
            for m in results:
                match_list.append(
                    CrossMatch(
                        local_track_id=track.track_id,
                        remote_camera_id=m["camera_id"],
                        remote_track_id=m["track_id"],
                        similarity=m["similarity"],
                        remote_bbox=tuple(m["bbox"]) if m.get("bbox") else None,
                    )
                )

        ctx.record_metric(self.name, "num_tracks_published", min(n, len(active)))
        ctx.record_metric(self.name, "num_cross_matches", len(match_list))

        return {
            self.out: CrossMatches(
                matches=match_list,
                camera_id=bridge.camera_id,
            )
        }
