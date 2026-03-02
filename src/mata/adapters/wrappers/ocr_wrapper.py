"""OCR adapter wrapper for the MATA graph system.

Bridges raw OCR adapters (HuggingFaceOCRAdapter, EasyOCRAdapter,
PaddleOCRAdapter, TesseractAdapter) to the graph provider protocol.

The OCR graph node calls ``recognizer.recognize(image)``; this wrapper
implements that contract, converting the ``Image`` artifact to PIL/path,
delegating to ``adapter.predict()``, and returning an ``OCRText`` artifact.

It also exposes a ``predict()`` method for VLM tool dispatch compatibility.
"""

from __future__ import annotations

import logging
from typing import Any

from mata.core.artifacts.image import Image
from mata.core.artifacts.ocr_text import OCRText

logger = logging.getLogger(__name__)


class OCRWrapper:
    """Wraps OCR adapters as graph-compatible ``recognize()`` providers.

    Bridges the graph system's ``Image`` artifact to the adapter's
    ``predict()`` interface, and converts ``OCRResult`` → ``OCRText``
    artifact for the strongly-typed graph wire.

    Implements the graph provider protocol:
        ``recognize(image: Image, **kwargs) -> OCRText``

    Also exposes ``predict()`` for VLM tool-dispatch compatibility.

    Supported adapters:
    - HuggingFaceOCRAdapter (GOT-OCR2, TrOCR)
    - EasyOCRAdapter
    - PaddleOCRAdapter
    - TesseractAdapter
    - Any adapter with ``predict(image, **kwargs) -> OCRResult``

    Example:
        >>> from mata.adapters.wrappers import wrap_ocr
        >>> adapter = mata.load("ocr", "easyocr")
        >>> recognizer = wrap_ocr(adapter)
        >>> img = Image.from_path("document.jpg")
        >>> ocr_text = recognizer.recognize(img)
        >>> print(ocr_text.full_text)
    """

    def __init__(self, adapter: Any) -> None:
        """Initialize wrapper with an existing OCR adapter.

        Args:
            adapter: Any OCR adapter with a ``predict()`` method that
                accepts image input and returns ``OCRResult``.

        Raises:
            TypeError: If adapter does not have a ``predict`` method.
        """
        if not hasattr(adapter, "predict"):
            raise TypeError(f"OCRWrapper requires an adapter with a 'predict' method, " f"got {type(adapter).__name__}")
        self.adapter = adapter

    # ------------------------------------------------------------------
    # Image conversion
    # ------------------------------------------------------------------

    def _convert_image(self, image: Image) -> Any:
        """Convert Image artifact to adapter-compatible format.

        Prefers ``source_path`` for disk-backed images (avoids an
        unnecessary decode/re-encode round-trip); falls back to PIL for
        in-memory images.

        Args:
            image: Image artifact from the graph system.

        Returns:
            File path (str) or PIL Image.
        """
        if image.source_path:
            return image.source_path
        return image.to_pil()

    # ------------------------------------------------------------------
    # Graph-protocol method
    # ------------------------------------------------------------------

    def recognize(self, image: Image, **kwargs: Any) -> OCRText:
        """Run OCR on an Image artifact, returning an OCRText artifact.

        This is the primary graph-provider method consumed by the ``OCR``
        graph node.

        Args:
            image: Image artifact from the graph system.
            **kwargs: Passed through to ``adapter.predict()``.

        Returns:
            OCRText artifact containing recognized text blocks and metadata.
        """
        img_input = self._convert_image(image)
        raw_result = self.adapter.predict(img_input, **kwargs)
        return OCRText.from_ocr_result(raw_result)

    # ------------------------------------------------------------------
    # VLM tool-dispatch compatibility
    # ------------------------------------------------------------------

    def predict(self, image: Any, **kwargs: Any) -> Any:
        """Fallback ``predict()`` for VLM tool-dispatch compatibility.

        When used as a VLM tool, the tool registry may call ``predict()``
        directly with either an ``Image`` artifact or a raw PIL/path input.
        This method handles both forms.

        Args:
            image: ``Image`` artifact **or** PIL Image / file path.
            **kwargs: Passed through to the underlying adapter.

        Returns:
            ``OCRText`` artifact when given an ``Image`` artifact;
            raw ``OCRResult`` otherwise.
        """
        if isinstance(image, Image):
            return self.recognize(image, **kwargs)
        return self.adapter.predict(image, **kwargs)


def wrap_ocr(adapter: Any) -> OCRWrapper:
    """Factory function: wrap an OCR adapter for use as a graph provider.

    Args:
        adapter: An OCR adapter with ``predict(image, **kwargs) -> OCRResult``.

    Returns:
        OCRWrapper ready to use as a ``recognize()`` graph provider.

    Example:
        >>> ocr_provider = wrap_ocr(mata.load("ocr", "easyocr"))
        >>> result = mata.infer(graph, image="doc.jpg",
        ...     providers={"my_ocr": ocr_provider})
    """
    return OCRWrapper(adapter)
