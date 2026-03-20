"""BarcodeData artifact for graph system.

Wraps BarcodeResult for typed graph wiring, providing immutable barcode
results with instance-ID correlation for ROI pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact

if TYPE_CHECKING:
    from mata.core.types import BarcodeResult


@dataclass(frozen=True)
class BarcodeEntry:
    """A single decoded barcode within the graph artifact.

    Attributes:
        data: Decoded payload string.
        type: Barcode symbology (e.g. "QR_CODE", "EAN_13").
        confidence: Confidence score in [0.0, 1.0].
        bbox: Optional bounding box in xyxy format (x1, y1, x2, y2).
    """

    data: str
    type: str
    confidence: float = 1.0
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class BarcodeData(Artifact):
    """Barcode/QR results artifact for graph wiring.

    Carries decoded barcode entries so barcode output can participate in
    the strongly-typed graph system. When barcodes are read from ROI crops,
    ``instance_ids`` maps each ``BarcodeEntry`` back to its source detection
    so downstream ``Fuse`` nodes can correlate results.

    Attributes:
        entries: Immutable tuple of BarcodeEntry items.
        instance_ids: Tuple of instance IDs, one per entry. Empty when
            processing a whole image rather than individual ROI crops.
        meta: Arbitrary metadata (engine, timing, etc.).

    Example:
        ```python
        from mata.core.artifacts.barcode_data import BarcodeData, BarcodeEntry

        entries = (BarcodeEntry(data="https://example.com", type="QR_CODE"),)
        artifact = BarcodeData(entries=entries)
        artifact.validate()
        ```
    """

    entries: tuple[BarcodeEntry, ...] = ()
    instance_ids: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate artifact consistency."""
        if not isinstance(self.entries, tuple):
            raise ValueError("BarcodeData.entries must be a tuple")
        if not isinstance(self.instance_ids, tuple):
            raise ValueError("BarcodeData.instance_ids must be a tuple")

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_barcode_result(
        cls,
        result: BarcodeResult,
        instance_ids: tuple[str, ...] = (),
    ) -> BarcodeData:
        """Create from a public :class:`~mata.core.types.BarcodeResult`.

        Args:
            result: BarcodeResult returned by a barcode adapter.
            instance_ids: Correlation IDs to attach — one per barcode in
                ``result.barcodes``. Pass an empty tuple when processing
                a whole image (not individual ROI crops).

        Returns:
            BarcodeData artifact with ``entries`` mirroring ``result.barcodes``.
        """
        entries = tuple(
            BarcodeEntry(
                data=b.data,
                type=b.type,
                confidence=b.score,
                bbox=b.bbox,
            )
            for b in result.barcodes
        )
        return cls(
            entries=entries,
            instance_ids=instance_ids,
            meta=dict(result.meta) if result.meta else {},
        )

    # ------------------------------------------------------------------
    # Serialization (implements Artifact ABC)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "entries": [
                {
                    "data": e.data,
                    "type": e.type,
                    "confidence": e.confidence,
                    "bbox": list(e.bbox) if e.bbox else None,
                }
                for e in self.entries
            ],
            "instance_ids": list(self.instance_ids),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BarcodeData:
        """Construct from a serialised dictionary (round-trip with :meth:`to_dict`)."""
        entries = tuple(
            BarcodeEntry(
                data=e["data"],
                type=e["type"],
                confidence=e.get("confidence", 1.0),
                bbox=tuple(e["bbox"]) if e.get("bbox") else None,  # type: ignore[arg-type]
            )
            for e in data.get("entries", [])
        )
        return cls(
            entries=entries,
            instance_ids=tuple(data.get("instance_ids", [])),
            meta=data.get("meta", {}),
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> BarcodeData:
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_barcodes(self) -> int:
        """Number of decoded barcodes."""
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        """True when no barcodes were decoded."""
        return len(self.entries) == 0
