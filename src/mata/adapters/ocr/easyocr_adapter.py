"""EasyOCR adapter for MATA framework."""

from __future__ import annotations

from typing import Any

from mata.adapters.base import BaseAdapter
from mata.core.logging import get_logger
from mata.core.types import OCRResult, TextRegion

logger = get_logger(__name__)

_easyocr = None
_EASYOCR_AVAILABLE = None


def _ensure_easyocr() -> Any:
    """Ensure easyocr is imported (lazy loading).

    Returns:
        easyocr module

    Raises:
        ImportError: If easyocr cannot be imported
    """
    global _easyocr, _EASYOCR_AVAILABLE
    if _easyocr is None:
        try:
            import easyocr

            _easyocr = easyocr
            _EASYOCR_AVAILABLE = True
            logger.debug("EasyOCR loaded successfully")
        except ImportError as exc:
            _EASYOCR_AVAILABLE = False
            raise ImportError(
                "easyocr is required for EasyOCRAdapter. "
                "Install with: pip install easyocr\n"
                "or: pip install datamata[ocr]"
            ) from exc
    return _easyocr


def _polygon_to_xyxy(polygon: list) -> tuple[float, float, float, float]:
    """Convert EasyOCR 4-point polygon [[x,y],...] to xyxy bbox.

    EasyOCR returns bounding boxes as a list of 4 corner points
    (clockwise from top-left). This converts to axis-aligned xyxy format by
    taking the min/max of all point coordinates.

    Args:
        polygon: List of [x, y] points (4 points, may also be numpy arrays)

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in absolute pixel coordinates

    Example:
        >>> _polygon_to_xyxy([[10, 20], [100, 20], [100, 50], [10, 50]])
        (10, 20, 100, 50)
    """
    xs = [float(pt[0]) for pt in polygon]
    ys = [float(pt[1]) for pt in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


class EasyOCRAdapter(BaseAdapter):
    """OCR adapter wrapping the EasyOCR engine.

    EasyOCR is a ready-to-use OCR engine supporting 80+ languages. It returns
    word/line-level text with polygon bounding boxes, which are converted to
    xyxy format by this adapter for consistency with MATA's ``TextRegion.bbox``
    contract.

    Note:
        EasyOCR manages its own GPU device internally via the ``gpu`` flag.
        Unlike PyTorch-based adapters, this adapter does not accept a ``device``
        kwarg — use ``gpu=True/False`` instead.

    Args:
        languages: List of language codes to recognize (default: ``["en"]``).
            Examples: ``["en"]``, ``["en", "fr"]``, ``["ch_sim", "en"]``.
        gpu: Whether to use GPU acceleration (default: ``True``).
        **kwargs: Additional keyword arguments forwarded to ``easyocr.Reader``.

    Example:
        >>> adapter = EasyOCRAdapter(["en"], gpu=False)
        >>> result = adapter.predict("image.jpg")
        >>> print(result.full_text)
    """

    name = "easyocr"
    task = "ocr"

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader_kwargs = kwargs

        easyocr = _ensure_easyocr()
        logger.info(f"Initializing EasyOCR reader for languages: {self.languages}, gpu={self.gpu}")
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu, **self._reader_kwargs)
        logger.info("EasyOCR reader initialized successfully")

    def predict(self, image: Any, detail: int = 1, **kwargs: Any) -> OCRResult:
        """Run OCR on an image.

        Args:
            image: Input image — file path, URL, PIL Image, numpy array,
                or MATA ``Image`` artifact.
            detail: EasyOCR detail level. ``1`` (default) returns full results
                including bboxes; ``0`` returns text-only strings (bboxes will
                be ``None``).
            **kwargs: Additional keyword arguments forwarded to
                ``reader.readtext()``.

        Returns:
            :class:`~mata.core.types.OCRResult` with one
            :class:`~mata.core.types.TextRegion` per recognized text span.
            ``bbox`` is in ``(x_min, y_min, x_max, y_max)`` xyxy format.

        Raises:
            ImportError: If easyocr is not installed.
        """
        import numpy as np

        pil_image, _ = self._load_image(image)
        img_array = np.array(pil_image)

        logger.debug(f"Running EasyOCR readtext (detail={detail})")
        raw_results = self._reader.readtext(img_array, detail=detail, **kwargs)

        if detail == 0:
            # detail=0: returns list of plain strings, no spatial info
            regions = [TextRegion(text=str(text), score=1.0, bbox=None) for text in raw_results if str(text).strip()]
        else:
            # detail=1 (default): returns list of (bbox_polygon, text, confidence)
            regions = [
                TextRegion(
                    text=str(text),
                    score=float(confidence),
                    bbox=_polygon_to_xyxy(bbox_polygon),
                )
                for bbox_polygon, text, confidence in raw_results
                if str(text).strip()
            ]

        logger.debug(f"EasyOCR detected {len(regions)} text region(s)")
        return OCRResult(
            regions=regions,
            meta={"engine": "easyocr", "languages": self.languages},
        )

    def info(self) -> dict[str, Any]:
        """Return adapter metadata.

        Returns:
            Dict with ``name``, ``task``, ``languages``, and ``gpu`` keys.
        """
        return {
            "name": self.name,
            "task": self.task,
            "languages": self.languages,
            "gpu": self.gpu,
        }
