"""OCR (text extraction) adapters for MATA framework."""

from .easyocr_adapter import EasyOCRAdapter
from .huggingface_ocr_adapter import HuggingFaceOCRAdapter
from .paddleocr_adapter import PaddleOCRAdapter
from .tesseract_adapter import TesseractAdapter

__all__ = [
    "EasyOCRAdapter",
    "HuggingFaceOCRAdapter",
    "PaddleOCRAdapter",
    "TesseractAdapter",
]
