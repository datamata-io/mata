"""ONNX Runtime adapter for image classification.

Supports ONNX classification models with automatic I/O tensor detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.adapters.onnx_base import ONNXBaseAdapter
from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import Classification, ClassifyResult

logger = get_logger(__name__)


class ONNXClassifyAdapter(ONNXBaseAdapter):
    """ONNX Runtime classification adapter.

    Loads and runs ONNX models for image classification. Provides optimized
    inference with automatic I/O tensor detection and configurable top-k predictions.

    Supports execution providers:
    - CUDA (GPU acceleration)
    - CPU (cross-platform)
    - TensorRT (advanced GPU optimization - requires TensorRT)

    Examples:
        >>> # Load ONNX model
        >>> classifier = ONNXClassifyAdapter("model.onnx", top_k=5)
        >>> # Run inference
        >>> result = classifier.predict("image.jpg", top_k=10)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        top_k: int = 5,
        threshold: float = 0.0,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize ONNX classification adapter.

        Args:
            model_path: Path to ONNX model file (.onnx)
            device: Device ("cuda", "cpu", or "auto")
            top_k: Number of top predictions to return (default: 5)
            threshold: Minimum confidence threshold for predictions (default: 0.0)
            id2label: Optional custom label mapping

        Raises:
            ImportError: If onnxruntime is not installed
            ModelLoadError: If model loading fails
            FileNotFoundError: If model file not found
        """
        # Initialize base adapter (handles onnxruntime import, device, and threshold)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.top_k = top_k

        # Load ONNX session
        self._load_session()

    def _load_session(self) -> None:
        """Load ONNX Runtime session."""
        try:
            logger.info(f"Loading ONNX classification model: {self.model_path}")

            # Create session with optimizations
            sess_options = self.ort.SessionOptions()
            sess_options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

            self.session = self.ort.InferenceSession(
                str(self.model_path), sess_options=sess_options, providers=self.providers
            )

            # Extract I/O metadata
            model_inputs = self.session.get_inputs()
            self.input_names = [inp.name for inp in model_inputs]
            self.input_shapes = {inp.name: inp.shape for inp in model_inputs}

            # Primary input (usually 'input' or 'images' for vision models)
            self.input_name = model_inputs[0].name
            self.input_shape = model_inputs[0].shape

            # Output names (usually 'output' or 'logits')
            model_outputs = self.session.get_outputs()
            self.output_names = [output.name for output in model_outputs]
            self.output_name = model_outputs[0].name  # Primary output

            # Try to infer number of classes from output shape
            output_shape = model_outputs[0].shape
            if len(output_shape) >= 2:
                num_classes = output_shape[-1]
                if isinstance(num_classes, int) and not self.id2label:
                    # Generate generic labels if not provided
                    self.id2label = {i: f"class_{i}" for i in range(num_classes)}
                    logger.info(f"Generated {num_classes} generic class labels")

            logger.info(
                f"ONNX classification session ready. "
                f"Inputs: {self.input_names} {self.input_shapes}, "
                f"Outputs: {self.output_names}"
            )

        except Exception as e:
            raise ModelLoadError(
                str(self.model_path), f"Failed to load ONNX classification model: {type(e).__name__}: {str(e)}"
            )

    def classify(
        self,
        image: str | Path | Image.Image | np.ndarray,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> ClassifyResult:
        """Classify image (Classify node interface).

        This method provides compatibility with the Classify graph node,
        which expects a classify() method. It wraps predict() with the
        appropriate parameters.

        Args:
            image: Input image (path, PIL Image, numpy array, or MATA Image artifact)
            top_k: Number of top predictions to return
            **kwargs: Additional arguments passed to predict()

        Returns:
            ClassifyResult with predictions
        """
        return self.predict(image=image, top_k=top_k, **kwargs)

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "ONNXClassifyAdapter",
            "task": "classify",
            "model_path": str(self.model_path),
            "providers": self.providers,
            "top_k": self.top_k,
            "num_classes": len(self.id2label) if self.id2label else "unknown",
            "input_shape": self.input_shape,
            "backend": "onnxruntime",
        }

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for ONNX classification model.

        Args:
            image: PIL Image in RGB

        Returns:
            Preprocessed numpy array ready for inference [1, C, H, W]
        """
        # Get target size from model input shape
        # Shape is typically [batch, channels, height, width]
        if len(self.input_shape) == 4:
            _, channels, target_h, target_w = self.input_shape
            # Handle dynamic shapes
            if isinstance(target_h, str) or target_h <= 0:
                target_h = 224  # Default for classification
            if isinstance(target_w, str) or target_w <= 0:
                target_w = 224
        else:
            # Default to 224x224 if shape is unclear (standard for classification)
            target_h, target_w = 224, 224

        # Resize image
        resized = image.resize((target_w, target_h), Image.BILINEAR)

        # Convert to numpy array and normalize
        img_array = np.array(resized, dtype=np.float32)

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Apply ImageNet normalization (standard for classification)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        # Convert HWC to CHW
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(
        self, image: str | Path | Image.Image | np.ndarray, top_k: int | None = None, **kwargs: Any
    ) -> ClassifyResult:
        """Perform image classification on input image.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            top_k: Number of top predictions to return (overrides instance setting)
            **kwargs: Additional inference parameters

        Returns:
            ClassifyResult with top-k predictions sorted by confidence (descending)

        Raises:
            InvalidInputError: If image is invalid or cannot be loaded
            RuntimeError: If inference fails
        """
        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)

        # Determine top_k (runtime override or instance default)
        k = top_k if top_k is not None else self.top_k

        try:
            # Preprocess image
            input_tensor = self._preprocess(pil_image)

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # Get logits (first output)
            logits = outputs[0][0]  # Shape: (num_classes,)

            # Convert to probabilities using softmax
            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            probs = exp_logits / np.sum(exp_logits)

            # Get top-k predictions
            top_k_indices = np.argsort(probs)[::-1][:k]
            top_k_probs = probs[top_k_indices]

            # Convert to Classification objects
            predictions = [
                Classification(
                    label=int(idx), score=float(prob), label_name=self.id2label.get(int(idx), f"class_{int(idx)}")
                )
                for idx, prob in zip(top_k_indices, top_k_probs)
            ]

            # Build metadata
            meta = {
                "model_path": str(self.model_path),
                "providers": self.providers,
                "backend": "onnxruntime",
                "image_size": pil_image.size,
                "top_k": len(predictions),
                "input_path": input_path,
            }

            return ClassifyResult(predictions=predictions, meta=meta)

        except Exception as e:
            raise RuntimeError(f"ONNX classification inference failed: {type(e).__name__}: {str(e)}")
