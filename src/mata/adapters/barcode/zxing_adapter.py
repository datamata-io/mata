"""Zxing-cpp adapter for MATA framework."""

from __future__ import annotations

from typing import Any

from mata.adapters.base import BaseAdapter
from mata.core.logging import get_logger
from mata.core.types import BarcodeRegion, BarcodeResult

logger = get_logger(__name__)

_zxingcpp = None
_ZXING_AVAILABLE = None


def _ensure_zxing() -> Any:
    """Lazy-load zxingcpp module.

    Raises:
        ImportError: If zxingcpp is not installed.
    """
    global _zxingcpp, _ZXING_AVAILABLE
    if _zxingcpp is None:
        try:
            import zxingcpp

            _zxingcpp = zxingcpp
            _ZXING_AVAILABLE = True
            logger.debug("zxingcpp loaded successfully")
        except ImportError as exc:
            _ZXING_AVAILABLE = False
            raise ImportError(
                "zxingcpp is required for ZxingAdapter. "
                "Install with: pip install zxing-cpp\n"
                "or: pip install datamata[barcode-zxing]"
            ) from exc
    return _zxingcpp


class ZxingAdapter(BaseAdapter):
    """Barcode/QR adapter wrapping the zxing-cpp engine.

    zxing-cpp supports QR, Aztec, DataMatrix, MaxiCode, PDF417, and
    all major 1D barcode formats. Broader symbology coverage than pyzbar.

    Args:
        formats: Optional list of barcode format names to restrict detection.
            None means detect all formats.
        **kwargs: Reserved for future options.

    Example:
        >>> adapter = ZxingAdapter()
        >>> result = adapter.predict("image.jpg")
    """

    name = "zxing"
    task = "barcode"

    def __init__(
        self,
        formats: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.formats = formats
        _ensure_zxing()

    def predict(self, image: Any, **kwargs: Any) -> BarcodeResult:
        """Detect and decode barcodes/QR codes in an image.

        Args:
            image: Input image — file path, PIL Image, numpy array,
                or MATA Image artifact.

        Returns:
            BarcodeResult with one BarcodeRegion per decoded barcode.
        """
        zxingcpp = _ensure_zxing()
        pil_image, _ = self._load_image(image)

        results = zxingcpp.read_barcodes(pil_image)

        barcodes: list[BarcodeRegion] = []
        for r in results:
            barcode_type = r.format.name if hasattr(r.format, "name") else str(r.format)

            if self.formats and barcode_type not in self.formats:
                continue

            # zxingcpp position → xyxy bbox
            pos = r.position
            xs = [pos.top_left.x, pos.top_right.x, pos.bottom_left.x, pos.bottom_right.x]
            ys = [pos.top_left.y, pos.top_right.y, pos.bottom_left.y, pos.bottom_right.y]
            bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

            barcodes.append(
                BarcodeRegion(
                    data=r.text,
                    type=barcode_type,
                    bbox=bbox,
                    score=1.0,
                    raw_bytes=r.bytes if hasattr(r, "bytes") else None,
                )
            )

        logger.debug(f"zxing decoded {len(barcodes)} barcode(s)")
        return BarcodeResult(
            barcodes=barcodes,
            meta={"engine": "zxing"},
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "formats": self.formats,
        }
