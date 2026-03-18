"""Embed node for feature extraction in graph pipelines.

Consumes ROIs (cropped image regions) and produces Embeddings artifact
using any Embedder-conforming provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.embeddings import Embeddings
from mata.core.artifacts.rois import ROIs
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class Embed(Node):
    """Extract feature embeddings from ROIs using an embedding provider.

    Takes cropped image regions (from ExtractROIs) and runs them through
    an embedding model to produce a fixed-dimensional feature vector per
    region. Output embeddings are L2-normalized by default.

    Args:
        using: Provider name (resolved from providers dict at runtime).
        src: Name of the input ROIs artifact key (default: "rois").
        out: Name of the output Embeddings artifact key (default: "embeddings").
        normalize: Whether to L2-normalize embeddings (default: True).
        name: Optional human-readable node name.

    Inputs:
        rois (ROIs): Cropped image regions to embed.

    Outputs:
        embeddings (Embeddings): Feature vectors with source ROI mapping.

    Example:
        ```python
        from mata.nodes import Detect, ExtractROIs, Embed, Filter

        graph = (Graph("embed_pipeline")
            .then(Detect(using="detector", out="dets"))
            .then(Filter(src="dets", score_gt=0.3, out="filtered"))
            .then(ExtractROIs(src_dets="filtered", out="rois"))
            .then(Embed(using="encoder", src="rois", out="embeddings"))
        )
        ```
    """

    inputs: dict[str, type[Artifact]] = {"rois": ROIs}
    outputs: dict[str, type[Artifact]] = {"embeddings": Embeddings}

    def __init__(
        self,
        using: str,
        src: str = "rois",
        out: str = "embeddings",
        normalize: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.using = using
        self.src = src
        self.out = out
        self.normalize = normalize
        # Dynamic artifact mapping (follows Filter pattern)
        self.inputs = {src: ROIs}
        self.outputs = {out: Embeddings}

    def run(self, ctx: ExecutionContext, **inputs: ROIs) -> dict[str, Artifact]:
        """Extract embeddings from ROIs.

        Args:
            ctx: Execution context (provides access to providers).
            **inputs: The single input ROIs artifact, keyed by src name.

        Returns:
            Dict with single key (``self.out``) mapping to Embeddings artifact.
            Returns empty Embeddings if no ROIs provided.
        """
        import numpy as np

        rois: ROIs = next(iter(inputs.values()))
        provider = ctx.get_provider("embed", self.using)

        if len(rois.roi_images) == 0:
            empty = np.empty((0, 0), dtype=np.float32)
            return {self.out: Embeddings(vectors=empty, meta={"model": self.using})}

        # Delegate to provider — accepts ROIs artifact via Embedder protocol
        vectors = provider.embed(rois, normalize=self.normalize)

        ctx.record_metric(self.name, "num_embeddings", vectors.shape[0])
        if vectors.ndim == 2 and vectors.shape[1] > 0:
            ctx.record_metric(self.name, "embedding_dim", vectors.shape[1])

        return {
            self.out: Embeddings(
                vectors=vectors,
                instance_ids=rois.instance_ids,
                normalized=self.normalize,
                meta={"model": self.using},
            )
        }
