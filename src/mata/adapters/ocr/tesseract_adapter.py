"""Tesseract OCR adapter for MATA framework."""

from __future__ import annotations

from typing import Any

from mata.adapters.base import BaseAdapter
from mata.core.logging import get_logger
from mata.core.types import OCRResult, TextRegion

logger = get_logger(__name__)

_pytesseract = None


def _ensure_tesseract() -> Any:
    """Ensure pytesseract is imported (lazy loading).

    Returns:
        pytesseract module

    Raises:
        ImportError: If pytesseract cannot be imported
    """
    global _pytesseract
    if _pytesseract is None:
        try:
            import pytesseract

            _pytesseract = pytesseract
            logger.debug("pytesseract loaded successfully")
        except ImportError as exc:
            raise ImportError(
                "pytesseract is required for TesseractAdapter.\n"
                "Install Python package: pip install pytesseract\n"
                "or: pip install mata[ocr-tesseract]\n"
                "Also install the Tesseract binary:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "  macOS: brew install tesseract\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
            ) from exc
    return _pytesseract


class TesseractAdapter(BaseAdapter):
    """OCR adapter wrapping Tesseract via the pytesseract Python package.

    Tesseract is a classic open-source OCR engine that produces word-level
    bounding boxes with confidence scores. It requires both the ``pytesseract``
    Python package **and** the Tesseract binary to be installed on the system.

    Confidence values from Tesseract are integers in the range 0–100; this
    adapter normalizes them to ``[0.0, 1.0]``. Entries with ``conf == -1``
    (Tesseract found no text in that region) and entries with empty text are
    filtered out automatically.

    Note:
        Tesseract bbox format is ``(x, y, width, height)``; this adapter
        converts to xyxy ``(x, y, x+w, y+h)`` for consistency with MATA's
        ``TextRegion.bbox`` contract.

    Note:
        The Tesseract binary must be installed separately from the Python
        package. See the import error message for OS-specific instructions.

    Args:
        lang: Tesseract language code(s) (default: ``"eng"``).
            Use ``+``-separated codes for multiple languages, e.g. ``"eng+fra"``.
        config: Additional Tesseract configuration string passed directly to
            ``pytesseract`` (default: ``""``). Example: ``"--oem 3 --psm 6"``.
        **kwargs: Additional keyword arguments (reserved for future use).

    Example:
        >>> adapter = TesseractAdapter(lang="eng")
        >>> result = adapter.predict("image.jpg")
        >>> print(result.full_text)
    """

    name = "tesseract"
    task = "ocr"

    def __init__(
        self,
        lang: str = "eng",
        config: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lang = lang
        self.config = config
        _ensure_tesseract()
        logger.info(f"TesseractAdapter initialized (lang={self.lang!r}, config={self.config!r})")

    def predict(self, image: Any, **kwargs: Any) -> OCRResult:
        """Run OCR on an image using Tesseract.

        Args:
            image: Input image — file path, URL, PIL Image, numpy array,
                or MATA ``Image`` artifact.
            **kwargs: Additional keyword arguments forwarded to
                ``pytesseract.image_to_data()``.

        Returns:
            :class:`~mata.core.types.OCRResult` with one
            :class:`~mata.core.types.TextRegion` per recognized word.
            ``bbox`` is in ``(x_min, y_min, x_max, y_max)`` xyxy format.
            Confidence scores are normalized to ``[0.0, 1.0]``.

        Raises:
            ImportError: If pytesseract or the Tesseract binary is not installed.
        """
        import pytesseract
        from pytesseract import Output

        pil_image, _ = self._load_image(image)

        # Allow per-call overrides of lang/config while avoiding duplicate kwargs
        lang = kwargs.pop("lang", self.lang)
        config = kwargs.pop("config", self.config)

        logger.debug(f"Running pytesseract.image_to_data (lang={lang!r})")
        data = pytesseract.image_to_data(
            pil_image,
            lang=lang,
            config=config,
            output_type=Output.DICT,
            **kwargs,
        )

        regions: list[TextRegion] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            # conf == -1 means no text was recognized in this region
            if not text or conf < 0:
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            regions.append(
                TextRegion(
                    text=text,
                    score=conf / 100.0,
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                )
            )

        logger.debug(f"Tesseract detected {len(regions)} text region(s)")
        return OCRResult(
            regions=regions,
            meta={"engine": "tesseract", "lang": self.lang},
        )

    def info(self) -> dict[str, Any]:
        """Return adapter metadata.

        Returns:
            Dict with ``name``, ``task``, ``lang``, and ``config`` keys.
        """
        return {
            "name": self.name,
            "task": self.task,
            "lang": self.lang,
            "config": self.config,
        }
