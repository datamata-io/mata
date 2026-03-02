"""OCRText artifact for graph system.

Wraps OCRResult for typed graph wiring, providing immutable OCR results
with instance-ID correlation for ROI pipelines and serialization helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact

if TYPE_CHECKING:
    from mata.core.types import OCRResult


@dataclass(frozen=True)
class TextBlock:
    """A single recognized text block within the graph artifact.

    Attributes:
        text: The recognized text string.
        confidence: Confidence score in [0.0, 1.0].
        bbox: Optional bounding box in xyxy absolute pixel coords (x1, y1, x2, y2).
        language: Optional language code or modality tag (e.g. ``"en"``, ``"printed"``).
    """

    text: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None
    language: str | None = None


@dataclass(frozen=True)
class OCRText(Artifact):
    """OCR results artifact for graph wiring.

    Carries recognized text blocks so OCR output can participate in the
    strongly-typed graph system.  When OCR is run on individual ROI crops,
    ``instance_ids`` maps each ``TextBlock`` back to its source detection /
    ROI so downstream ``Fuse`` nodes can correlate results.

    Attributes:
        text_blocks: Immutable tuple of :class:`TextBlock` items.
        full_text: Pre-joined concatenation of all block texts (newline-separated).
        instance_ids: Tuple of instance IDs, one per ``text_blocks`` entry.
            Empty when OCR is run on a whole image rather than individual ROIs.
        meta: Arbitrary metadata (engine, model info, timing, etc.).

    Example:
        ```python
        from mata.core.artifacts.ocr_text import OCRText, TextBlock

        blocks = (TextBlock(text="Hello", confidence=0.98),)
        artifact = OCRText(text_blocks=blocks, full_text="Hello")
        artifact.validate()
        ```
    """

    text_blocks: tuple[TextBlock, ...] = ()
    full_text: str = ""
    instance_ids: tuple[str, ...] = ()  # index N → text_blocks[N]
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate artifact consistency."""
        if not isinstance(self.text_blocks, tuple):
            raise ValueError("OCRText.text_blocks must be a tuple")
        if not isinstance(self.instance_ids, tuple):
            raise ValueError("OCRText.instance_ids must be a tuple")

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_ocr_result(
        cls,
        result: OCRResult,
        instance_ids: tuple[str, ...] = (),
    ) -> OCRText:
        """Create from a public :class:`~mata.core.types.OCRResult`.

        Args:
            result: OCRResult returned by an OCR adapter.
            instance_ids: Correlation IDs to attach — one per region in
                ``result.regions``.  Pass an empty tuple when processing
                a whole image (not individual ROI crops).

        Returns:
            OCRText artifact with ``text_blocks`` mirroring ``result.regions``.
        """
        blocks = tuple(
            TextBlock(
                text=r.text,
                confidence=r.score,
                bbox=r.bbox,
                language=r.label,
            )
            for r in result.regions
        )
        return cls(
            text_blocks=blocks,
            full_text=result.full_text,
            instance_ids=instance_ids,
            meta=dict(result.meta) if result.meta else {},
        )

    def to_ocr_result(self) -> OCRResult:
        """Convert back to :class:`~mata.core.types.OCRResult` for adapter compatibility.

        Returns:
            OCRResult with regions and meta derived from this artifact.
        """
        from mata.core.types import OCRResult, TextRegion

        regions = [
            TextRegion(
                text=b.text,
                score=b.confidence,
                bbox=b.bbox,
                label=b.language,
            )
            for b in self.text_blocks
        ]
        return OCRResult(regions=regions, meta=dict(self.meta) if self.meta else {})

    # ------------------------------------------------------------------
    # Serialization (implements Artifact ABC)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "text_blocks": [
                {
                    "text": b.text,
                    "confidence": b.confidence,
                    "bbox": list(b.bbox) if b.bbox else None,
                    "language": b.language,
                }
                for b in self.text_blocks
            ],
            "full_text": self.full_text,
            "instance_ids": list(self.instance_ids),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRText:
        """Construct from a serialised dictionary (round-trip with :meth:`to_dict`)."""
        blocks = tuple(
            TextBlock(
                text=b["text"],
                confidence=b["confidence"],
                bbox=tuple(b["bbox"]) if b.get("bbox") else None,  # type: ignore[arg-type]
                language=b.get("language"),
            )
            for b in data.get("text_blocks", [])
        )
        return cls(
            text_blocks=blocks,
            full_text=data.get("full_text", ""),
            instance_ids=tuple(data.get("instance_ids", [])),
            meta=data.get("meta", {}),
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> OCRText:
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        """Number of recognized text blocks."""
        return len(self.text_blocks)

    @property
    def is_empty(self) -> bool:
        """True when no text was recognized."""
        return len(self.text_blocks) == 0
