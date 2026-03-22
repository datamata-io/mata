"""Matches artifact — results from gallery similarity search.

Produced by GalleryMatchNode; stores per-instance match results
from cosine similarity search against a Gallery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mata.core.artifacts.base import Artifact


@dataclass(frozen=True)
class MatchEntry:
    """Single gallery search result for one query embedding.

    Attributes:
        instance_id: Instance ID from the source Embeddings artifact.
        label: Best-matching gallery label, or "unknown" when no match found.
        similarity: Cosine similarity of the best match in [-1, 1].
        all_matches: Serialised list of all top-k GalleryMatch results.
    """

    instance_id: str
    label: str
    similarity: float
    all_matches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "similarity": self.similarity,
            "all_matches": self.all_matches,
        }


@dataclass(frozen=True)
class Matches(Artifact):
    """Collection of gallery match results, one per query embedding.

    Each entry corresponds to one vector from the source Embeddings
    artifact, preserving instance_id linkage back to the upstream
    detection/track.

    Attributes:
        entries: List of MatchEntry, one per query vector.
        meta: Optional provenance metadata.
    """

    entries: list[MatchEntry]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Matches:
        entries = [
            MatchEntry(
                instance_id=e["instance_id"],
                label=e["label"],
                similarity=e["similarity"],
                all_matches=e.get("all_matches", []),
            )
            for e in data.get("entries", [])
        ]
        return cls(entries=entries, meta=data.get("meta", {}))

    def _repr_html_(self) -> str | None:
        """Rich HTML display for Jupyter notebooks."""
        try:
            from mata.notebook import render_matches_html

            return render_matches_html(self)
        except Exception:
            return None
