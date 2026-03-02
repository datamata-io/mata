"""Plain-text exporter for MATA OCR result types.

Exports OCRResult.full_text (or region text) to a UTF-8 .txt file — the
simplest extract-then-save workflow for OCR output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mata.core.logging import get_logger

logger = get_logger(__name__)


def export_text(result: Any, output_path: str | Path, encoding: str = "utf-8") -> None:
    """Write OCRResult.full_text to a plain-text .txt file.

    Falls back to joining region text when ``full_text`` is absent, so the
    function works with any result that exposes a ``regions`` attribute
    (list of objects with a ``.text`` attribute).

    Args:
        result: ``OCRResult`` instance (or any object with ``full_text`` /
            ``regions``).
        output_path: Destination ``.txt`` file path.  Parent directories are
            created automatically.
        encoding: File encoding used when writing (default ``"utf-8"``).

    Raises:
        TypeError: When *result* has neither ``full_text`` nor ``regions``.
        IOError: When the file cannot be written.

    Examples:
        >>> from mata.core.exporters import export_text
        >>> export_text(ocr_result, "out/receipt.txt")

        # Dispatched automatically via OCRResult.save():
        >>> ocr_result.save("out/receipt.txt")
    """
    if hasattr(result, "full_text"):
        text: str = result.full_text
    elif hasattr(result, "regions"):
        text = "\n".join(r.text for r in result.regions)
    else:
        raise TypeError(
            f"export_text() requires an OCRResult (or an object with 'full_text' / "
            f"'regions'), got {type(result).__name__}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding=encoding) as fh:
        fh.write(text)

    logger.debug("Wrote %d chars to %s", len(text), output_path)
