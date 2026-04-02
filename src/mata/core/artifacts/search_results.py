"""SearchResults artifact — per-query video search results.

Produced by the EmbeddingSearch node; groups VideoMatch results by
their originating text query for easy iteration in downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from mata.core.artifacts.base import Artifact

if TYPE_CHECKING:
    from mata.recognition.video_index import VideoMatch


@dataclass(frozen=True)
class QueryResult:
    """Search results for a single text query.

    Attributes:
        query: The natural-language text query string.
        matches: Top-K VideoMatch results ordered by descending similarity.
    """

    query: str
    matches: tuple  # tuple[VideoMatch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "matches": [m.to_dict() for m in self.matches],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryResult:
        from mata.recognition.video_index import VideoMatch

        return cls(
            query=data["query"],
            matches=tuple(
                VideoMatch(
                    label=m["label"],
                    similarity=float(m["similarity"]),
                    index=int(m["index"]),
                    start_s=float(m["start_s"]),
                    end_s=float(m["end_s"]),
                )
                for m in data.get("matches", [])
            ),
        )


@dataclass(frozen=True)
class SearchResults(Artifact):
    """Collection of per-query video search results.

    Produced by :class:`~mata.nodes.EmbeddingSearch`; one QueryResult entry
    per query string passed to that node.

    Attributes:
        results: Tuple of QueryResult, one per query.
        meta: Optional provenance metadata (model, threshold, top_k, etc.).

    Examples:
        >>> for qr in search_results.results:
        ...     print(f'"{qr.query}"')
        ...     for m in qr.matches:
        ...         print(f"  sim={m.similarity:.4f}  @ {m.start_s:.0f}s")

        >>> # Works with zip(queries, search_results) after unpacking .results
        >>> for qr in result["search_results"].results:
        ...     for m in qr.matches:
        ...         mm, ss = int(m.start_s) // 60, int(m.start_s) % 60
        ...         print(f'{qr.query} → {mm:02d}m{ss:02d}s')
    """

    results: tuple  # tuple[QueryResult, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[QueryResult]:
        return iter(self.results)

    def __getitem__(self, idx: int) -> QueryResult:
        return self.results[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResults:
        return cls(
            results=tuple(QueryResult.from_dict(r) for r in data.get("results", [])),
            meta=data.get("meta", {}),
        )
