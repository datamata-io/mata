"""ONNX Runtime implementation."""

from typing import Any

import numpy as np

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXRuntime:
    """ONNX Runtime for optimized inference.

    Executes ONNX models with CPU or GPU acceleration.
    """

    name: str = "onnx"

    def __init__(self, device: str = "auto", **session_options: Any) -> None:
        """Initialize ONNX Runtime.

        Args:
            device: Device to use ("cuda", "cpu", or "auto")
            **session_options: Additional ONNX Runtime session options
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime is not installed. "
                "Install with: pip install datamata[onnx]  (CPU)  or  pip install datamata[onnx-gpu]  (CUDA)"
            )

        self.device = device
        self.session_options = session_options

        # Determine providers
        if device == "auto":
            self.providers = ort.get_available_providers()
        elif device == "cuda":
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

    def load_model(self, model_path: str, **kwargs: Any) -> ort.InferenceSession:
        """Load an ONNX model from path.

        Args:
            model_path: Path to .onnx file
            **kwargs: Additional session options

        Returns:
            ONNX Runtime InferenceSession
        """
        session = ort.InferenceSession(model_path, providers=self.providers, **kwargs)
        return session

    def infer(self, model: ort.InferenceSession, inputs: dict[str, np.ndarray], **kwargs: Any) -> dict[str, np.ndarray]:
        """Run inference with ONNX model.

        Args:
            model: ONNX Runtime InferenceSession
            inputs: Dictionary of input arrays (must match model input names)
            **kwargs: Additional inference options

        Returns:
            Dictionary of output arrays
        """
        # Run inference
        outputs = model.run(None, inputs)

        # Map outputs to names
        output_names = [o.name for o in model.get_outputs()]
        return {name: output for name, output in zip(output_names, outputs)}

    def get_device(self) -> str:
        """Get the device this runtime uses.

        Returns:
            Device string indicating providers
        """
        return f"onnx({', '.join(self.providers)})"
