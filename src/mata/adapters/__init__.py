"""Universal adapters for model-agnostic loading."""

from .base import BaseAdapter
from .huggingface_adapter import HuggingFaceDetectAdapter
from .huggingface_classify_adapter import HuggingFaceClassifyAdapter
from .huggingface_depth_adapter import HuggingFaceDepthAdapter
from .huggingface_segment_adapter import HuggingFaceSegmentAdapter
from .huggingface_zeroshot_segment_adapter import HuggingFaceZeroShotSegmentAdapter
from .ocr.easyocr_adapter import EasyOCRAdapter
from .ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter
from .ocr.paddleocr_adapter import PaddleOCRAdapter
from .ocr.tesseract_adapter import TesseractAdapter
from .onnx_adapter import ONNXDetectAdapter
from .onnx_base import ONNXBaseAdapter
from .onnx_classify_adapter import ONNXClassifyAdapter
from .pytorch_adapter import PyTorchDetectAdapter
from .pytorch_base import PyTorchBaseAdapter
from .pytorch_classify_adapter import PyTorchClassifyAdapter
from .torchscript_adapter import TorchScriptDetectAdapter
from .torchscript_classify_adapter import TorchScriptClassifyAdapter
from .torchvision_detect_adapter import TorchvisionDetectAdapter

__all__ = [
    # Base classes
    "BaseAdapter",
    "PyTorchBaseAdapter",
    "ONNXBaseAdapter",
    # Detection adapters
    "HuggingFaceDetectAdapter",
    "HuggingFaceDepthAdapter",
    "PyTorchDetectAdapter",
    "ONNXDetectAdapter",
    "TorchScriptDetectAdapter",
    "TorchvisionDetectAdapter",
    # Segmentation adapters
    "HuggingFaceSegmentAdapter",
    "HuggingFaceZeroShotSegmentAdapter",
    # Classification adapters
    "HuggingFaceClassifyAdapter",
    "PyTorchClassifyAdapter",
    "ONNXClassifyAdapter",
    "TorchScriptClassifyAdapter",
    # OCR adapters
    "HuggingFaceOCRAdapter",
    "EasyOCRAdapter",
    "PaddleOCRAdapter",
    "TesseractAdapter",
]
