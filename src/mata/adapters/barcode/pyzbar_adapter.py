"""Pyzbar adapter for MATA framework."""

from __future__ import annotations

from typing import Any

from mata.adapters.base import BaseAdapter
from mata.core.logging import get_logger
from mata.core.types import BarcodeRegion, BarcodeResult

logger = get_logger(__name__)

_pyzbar = None
_PYZBAR_AVAILABLE = None


def _ensure_pyzbar() -> Any:
    """Lazy-load pyzbar module.

    Raises:
        ImportError: If pyzbar is not installed.
    """
    global _pyzbar, _PYZBAR_AVAILABLE
    if _pyzbar is None:
        try:
            from pyzbar import pyzbar as pz

            _pyzbar = pz
            _PYZBAR_AVAILABLE = True
            logger.debug("pyzbar loaded successfully")
        except ImportError as exc:
            _PYZBAR_AVAILABLE = False
            raise ImportError(
                "pyzbar is required for PyzbarAdapter. "
                "Install with: pip install pyzbar\n"
                "or: pip install datamata[barcode]\n"
                "On Linux, also install libzbar0: sudo apt-get install libzbar0"
            ) from exc
    return _pyzbar


# pyzbar type string → MATA normalized type name
_PYZBAR_TYPE_MAP = {
    "QRCODE": "QR_CODE",
    "EAN13": "EAN_13",
    "EAN8": "EAN_8",
    "UPCA": "UPC_A",
    "UPCE": "UPC_E",
    "CODE128": "CODE_128",
    "CODE39": "CODE_39",
    "CODE93": "CODE_93",
    "I25": "ITF",
    "DATABAR": "DATABAR",
    "DATABAR_EXP": "DATABAR_EXP",
    "CODABAR": "CODABAR",
    "PDF417": "PDF_417",
}


class PyzbarAdapter(BaseAdapter):
    """Barcode/QR adapter wrapping the pyzbar (libzbar) engine.

    pyzbar supports QR codes plus 12+ 1D barcode symbologies via the
    underlying C library libzbar. Decode latency is typically 1-5ms.

    Args:
        symbols: Optional list of symbology types to restrict detection
            (e.g. ``["QR_CODE", "EAN_13"]``). None means detect all types.
        **kwargs: Reserved for future options.

    Example:
        >>> adapter = PyzbarAdapter()
        >>> result = adapter.predict("shelf_image.jpg")
        >>> for bc in result.barcodes:
        ...     print(bc.data, bc.type)
    """

    name = "pyzbar"
    task = "barcode"

    def __init__(
        self,
        symbols: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.symbols = symbols
        _ensure_pyzbar()

    def predict(self, image: Any, **kwargs: Any) -> BarcodeResult:
        """Detect and decode barcodes/QR codes in an image.

        Args:
            image: Input image — file path, PIL Image, numpy array,
                or MATA Image artifact.
            **kwargs: Additional keyword arguments (reserved).

        Returns:
            BarcodeResult with one BarcodeRegion per decoded barcode.
        """
        import numpy as np

        pz = _ensure_pyzbar()
        pil_image, _ = self._load_image(image)
        img_array = np.array(pil_image)

        decoded = pz.decode(img_array)

        barcodes: list[BarcodeRegion] = []
        for obj in decoded:
            # Normalize type name
            raw_type = obj.type
            barcode_type = _PYZBAR_TYPE_MAP.get(raw_type, raw_type)

            # Filter by requested symbology if set
            if self.symbols and barcode_type not in self.symbols:
                continue

            # Extract bbox from pyzbar Rect (left, top, width, height) → xyxy
            rect = obj.rect
            bbox = (
                float(rect.left),
                float(rect.top),
                float(rect.left + rect.width),
                float(rect.top + rect.height),
            )

            barcodes.append(
                BarcodeRegion(
                    data=obj.data.decode("utf-8", errors="replace"),
                    type=barcode_type,
                    bbox=bbox,
                    score=1.0,  # algorithmic decoder — exact result
                    raw_bytes=bytes(obj.data),
                )
            )

        logger.debug(f"pyzbar decoded {len(barcodes)} barcode(s)")
        return BarcodeResult(
            barcodes=barcodes,
            meta={"engine": "pyzbar"},
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "symbols": self.symbols,
        }
