"""Gallery — In-memory embedding store with cosine similarity search.

Provides the Gallery class for identity enrollment and matching, and the
GalleryMatch dataclass representing a single search result.

Design decisions:
- Zero external dependencies (numpy only)
- L2-normalises all stored embeddings on insertion
- Brute-force cosine similarity via matrix multiplication (numpy)
- Suitable for galleries up to ~50 000 entries
- .npz persistence with allow_pickle=False for security
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GalleryMatch:
    """Single match result from a gallery search.

    Attributes:
        label: Identity label of the matched entry (e.g. "alice").
        similarity: Cosine similarity in [-1, 1]; higher means more similar.
        index: Position of the matched entry in the gallery.

    Examples:
        >>> match = GalleryMatch(label="alice", similarity=0.92, index=0)
        >>> match.to_dict()
        {'label': 'alice', 'similarity': 0.92, 'index': 0}
    """

    label: str
    similarity: float
    index: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {"label": self.label, "similarity": float(self.similarity), "index": self.index}

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)


class Gallery:
    """In-memory embedding store with cosine similarity search.

    Stores labeled embeddings and supports efficient nearest-neighbour
    matching via brute-force cosine similarity (numpy only, no external deps).

    All embeddings are L2-normalised on insertion so that dot products equal
    cosine similarity.

    Attributes:
        similarity_thresh: Default minimum cosine similarity for search results.

    Examples:
        >>> gallery = Gallery(similarity_thresh=0.5)
        >>> gallery.add("alice", alice_embedding)  # returns insertion index
        0
        >>> gallery.add_many(["bob", "carol"], embeddings_matrix)
        [1, 2]
        >>> matches = gallery.search(query, top_k=1)
        >>> matches[0].label
        'alice'
        >>> gallery.save("gallery.npz")
        >>> gallery2 = Gallery.load("gallery.npz")
    """

    def __init__(self, similarity_thresh: float = 0.5) -> None:
        self._similarity_thresh = similarity_thresh
        self._embeddings: list[np.ndarray] = []  # list of (D,) L2-normalised vectors
        self._labels: list[str] = []
        self._matrix: np.ndarray | None = None  # lazy (N, D) cache
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def add(self, label: str, embedding: np.ndarray) -> int:
        """Add a single embedding with an identity label.

        Args:
            label: Identity label (e.g. "alice").
            embedding: 1-D or 2-D float array; L2-normalised on insertion.

        Returns:
            Index of the inserted entry.
        """
        vec = self._normalize(np.asarray(embedding, dtype=np.float32).ravel())
        self._embeddings.append(vec)
        self._labels.append(label)
        self._dirty = True
        return len(self._embeddings) - 1

    def add_many(self, labels: list[str], embeddings: np.ndarray) -> list[int]:
        """Add multiple embeddings at once.

        Args:
            labels: Identity labels aligned with embedding rows.
            embeddings: (N, D) float array or (D,) for a single vector.

        Returns:
            List of insertion indices.

        Raises:
            ValueError: If len(labels) != number of embedding rows.
        """
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if len(labels) != arr.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} does not match embeddings rows {arr.shape[0]}"
            )
        return [self.add(label, arr[i]) for i, label in enumerate(labels)]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[GalleryMatch]:
        """Search for nearest neighbours in the gallery.

        Args:
            query: 1-D float array of the query embedding.
            top_k: Maximum number of results to return.
            threshold: Minimum cosine similarity; overrides the instance default
                when provided.

        Returns:
            List of GalleryMatch sorted by descending similarity.
            Empty list if gallery is empty or no matches exceed threshold.
        """
        if self.size == 0:
            return []
        thresh = threshold if threshold is not None else self._similarity_thresh
        q = self._normalize(np.asarray(query, dtype=np.float32).ravel())
        matrix = self._get_matrix()
        similarities = matrix @ q  # (N,) cosine similarities
        order = np.argsort(similarities)[::-1]
        results: list[GalleryMatch] = []
        for idx in order[:top_k]:
            sim = float(similarities[idx])
            if sim < thresh:
                break
            results.append(
                GalleryMatch(label=self._labels[idx], similarity=sim, index=int(idx))
            )
        return results

    def search_batch(
        self,
        queries: np.ndarray,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[list[GalleryMatch]]:
        """Batch search for multiple query embeddings.

        Args:
            queries: (N, D) float array or single (D,) vector.
            top_k: Maximum results per query.
            threshold: Minimum similarity threshold.

        Returns:
            List of match lists, one per query row.
        """
        arr = np.asarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return [self.search(arr[i], top_k=top_k, threshold=threshold) for i in range(len(arr))]

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, label: str) -> int:
        """Remove all entries with the given label.

        Args:
            label: Identity label to remove.

        Returns:
            Number of entries removed.
        """
        indices = [i for i, lbl in enumerate(self._labels) if lbl == label]
        for i in reversed(indices):
            del self._embeddings[i]
            del self._labels[i]
        if indices:
            self._dirty = True
        return len(indices)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of stored embeddings."""
        return len(self._embeddings)

    @property
    def labels(self) -> list[str]:
        """All labels in insertion order (may contain duplicates)."""
        return list(self._labels)

    @property
    def unique_labels(self) -> list[str]:
        """Deduplicated list of labels preserving insertion order."""
        seen: set[str] = set()
        result: list[str] = []
        for label in self._labels:
            if label not in seen:
                seen.add(label)
                result.append(label)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist gallery to a .npz file.

        Uses ``allow_pickle=False`` for security.

        Args:
            path: Destination file path (should end with .npz).
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if self.size > 0:
            matrix = self._get_matrix()
            np.savez(
                path,
                embeddings=matrix,
                labels=np.array(self._labels, dtype=str),
                similarity_thresh=np.array([self._similarity_thresh]),
            )
        else:
            np.savez(
                path,
                embeddings=np.empty((0,), dtype=np.float32),
                labels=np.array([], dtype=str),
                similarity_thresh=np.array([self._similarity_thresh]),
            )

    @classmethod
    def load(cls, path: str) -> Gallery:
        """Load gallery from a .npz file.

        Args:
            path: Path to a .npz file previously saved via :meth:`save`.

        Returns:
            Populated Gallery instance.
        """
        data = np.load(path, allow_pickle=False)
        gallery = cls(similarity_thresh=float(data["similarity_thresh"][0]))
        embeddings = data["embeddings"]
        labels = data["labels"].tolist()
        if embeddings.ndim == 2 and len(labels) > 0:
            gallery.add_many(labels, embeddings)
        return gallery

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize gallery to a JSON-compatible dict.

        Returns:
            Dict with keys: embeddings, labels, similarity_thresh, size.
        """
        if self.size > 0:
            matrix = self._get_matrix()
            emb_list: list[Any] = matrix.tolist()
        else:
            emb_list = []
        return {
            "embeddings": emb_list,
            "labels": list(self._labels),
            "similarity_thresh": self._similarity_thresh,
            "size": self.size,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize gallery to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Gallery:
        """Create gallery from a dict (output of :meth:`to_dict`).

        Args:
            data: Dict with embeddings, labels, and optional similarity_thresh.

        Returns:
            Populated Gallery instance.
        """
        gallery = cls(similarity_thresh=data.get("similarity_thresh", 0.5))
        embeddings = data.get("embeddings", [])
        labels = data.get("labels", [])
        if embeddings and labels:
            gallery.add_many(labels, np.array(embeddings, dtype=np.float32))
        return gallery

    @classmethod
    def from_json(cls, json_str: str) -> Gallery:
        """Create gallery from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_matrix(self) -> np.ndarray:
        """Return cached (N, D) embedding matrix, rebuilding if dirty."""
        if self._matrix is None or self._dirty:
            self._matrix = np.stack(self._embeddings, axis=0)
            self._dirty = False
        return self._matrix

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """L2-normalize a 1-D vector; returns v unchanged if norm < 1e-8."""
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-8 else v

    # ------------------------------------------------------------------
    # Python dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"Gallery(size={self.size}, unique_labels={len(self.unique_labels)}, "
            f"thresh={self._similarity_thresh})"
        )
