"""Embeddings artifact for graph system.

Carries feature embedding vectors through graph pipelines, mapping each
embedding back to its source ROI or detection via instance_ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mata.core.artifacts.base import Artifact


def _generate_embedding_id(index: int) -> str:
    """Generate stable embedding instance ID."""
    return f"emb_{index:04d}"


@dataclass(frozen=True)
class Embeddings(Artifact):
    """Feature embedding vectors artifact.

    Attributes:
        vectors: (N, D) float32 array of embedding vectors.
        instance_ids: Stable string identifiers mapping to source ROIs/detections.
        embedding_dim: Dimensionality of each embedding vector (D).
        normalized: Whether vectors are L2-normalized.
        meta: Optional metadata (model name, extraction layer, etc.).

    Examples:
        >>> import numpy as np
        >>> embs = Embeddings(vectors=np.random.randn(5, 512).astype(np.float32))
        >>> len(embs)
        5
        >>> embs[0].shape
        (512,)
        >>> embs.instance_ids
        ('emb_0000', 'emb_0001', 'emb_0002', 'emb_0003', 'emb_0004')
    """

    vectors: np.ndarray  # (N, D) float32
    instance_ids: tuple[str, ...] = ()
    embedding_dim: int = 0
    normalized: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim == 1:
            object.__setattr__(self, "vectors", self.vectors.reshape(1, -1))
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D (N, D), got shape {self.vectors.shape}")
        n, d = self.vectors.shape
        if not self.instance_ids:
            ids = tuple(_generate_embedding_id(i) for i in range(n))
            object.__setattr__(self, "instance_ids", ids)
        if self.embedding_dim == 0 and d > 0:
            object.__setattr__(self, "embedding_dim", d)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.vectors[idx]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vectors": self.vectors.tolist(),
            "instance_ids": list(self.instance_ids),
            "embedding_dim": self.embedding_dim,
            "normalized": self.normalized,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Embeddings:
        """Construct from dictionary representation.

        Args:
            data: Dictionary with vectors, instance_ids, embedding_dim,
                  normalized, and meta fields.

        Returns:
            Embeddings artifact.
        """
        vectors = np.array(data["vectors"], dtype=np.float32)
        return cls(
            vectors=vectors,
            instance_ids=tuple(data.get("instance_ids", ())),
            embedding_dim=data.get("embedding_dim", 0),
            normalized=data.get("normalized", True),
            meta=data.get("meta", {}),
        )

    def _repr_html_(self) -> str | None:
        """Rich HTML display for Jupyter notebooks."""
        try:
            from mata.notebook import render_embeddings_html

            return render_embeddings_html(self)
        except Exception:
            return None
