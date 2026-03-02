"""Classifications artifact for graph system.

Wraps ClassifyResult for typed graph wiring, providing immutable
classification results with serialization and convenience accessors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mata.core.artifacts.base import Artifact
from mata.core.types import Classification, ClassifyResult


@dataclass(frozen=True)
class Classifications(Artifact):
    """Classification results artifact for graph wiring.

    Wraps a list of :class:`Classification` predictions with stable IDs
    and convenience accessors so classification output can participate in
    the strongly-typed graph system.

    Attributes:
        predictions: Sorted list of Classification predictions (descending score).
        meta: Arbitrary metadata (model info, timing, etc.).

    Example:
        ```python
        from mata.core.artifacts.classifications import Classifications
        from mata.core.types import Classification

        preds = [
            Classification(label=0, score=0.92, label_name="cat"),
            Classification(label=1, score=0.06, label_name="dog"),
        ]
        cls = Classifications(predictions=preds)
        cls.top1.label_name  # "cat"
        ```
    """

    predictions: tuple = ()  # Tuple[Classification, ...] for frozen compat
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_classify_result(cls, result: ClassifyResult) -> Classifications:
        """Create from an existing ClassifyResult.

        Args:
            result: ClassifyResult from a classification adapter.

        Returns:
            Classifications artifact.
        """
        return cls(
            predictions=tuple(result.predictions),
            meta=dict(result.meta) if result.meta else {},
        )

    def to_classify_result(self) -> ClassifyResult:
        """Convert back to ClassifyResult for adapter compatibility.

        Returns:
            ClassifyResult with predictions and meta.
        """
        return ClassifyResult(
            predictions=list(self.predictions),
            meta=dict(self.meta) if self.meta else None,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def top1(self) -> Classification | None:
        """Return the highest-confidence prediction, or None if empty."""
        return self.predictions[0] if self.predictions else None

    @property
    def top5(self) -> list[Classification]:
        """Return top-5 predictions (or fewer if less available)."""
        return list(self.predictions[:5])

    @property
    def labels(self) -> list[str]:
        """Return label names for all predictions."""
        return [p.label_name or str(p.label) for p in self.predictions]

    @property
    def scores(self) -> list[float]:
        """Return scores for all predictions."""
        return [p.score for p in self.predictions]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Classifications:
        """Deserialize from dictionary."""
        predictions = tuple(
            Classification(
                label=p["label"],
                score=p["score"],
                label_name=p.get("label_name"),
            )
            for p in data.get("predictions", [])
        )
        return cls(predictions=predictions, meta=data.get("meta", {}))

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    def validate(self) -> None:
        """Validate predictions are properly structured."""
        for pred in self.predictions:
            if not isinstance(pred, Classification):
                raise ValueError(f"All predictions must be Classification instances, " f"got {type(pred).__name__}")

    def __len__(self) -> int:
        return len(self.predictions)

    def __repr__(self) -> str:
        top = self.top1
        top_str = f"{top.label_name}={top.score:.3f}" if top else "empty"
        return f"Classifications(n={len(self)}, top={top_str})"
