"""ONNX Runtime adapter for optimized inference.

Supports ONNX models with automatic I/O tensor detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.adapters.onnx_base import ONNXBaseAdapter
from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import Detection, DetectResult

logger = get_logger(__name__)


class ONNXDetectAdapter(ONNXBaseAdapter):
    """ONNX Runtime detection adapter.

    Loads and runs ONNX models for object detection. Provides optimized
    inference with automatic I/O tensor detection.

    Supports execution providers:
    - CUDA (GPU acceleration)
    - CPU (cross-platform)
    - TensorRT (advanced GPU optimization - requires TensorRT)

    Examples:
        >>> # Load ONNX model
        >>> detector = ONNXDetectAdapter("model.onnx")
        >>> # Run inference
        >>> result = detector.predict("image.jpg", threshold=0.5)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        threshold: float = 0.3,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize ONNX detection adapter.

        Args:
            model_path: Path to ONNX model file (.onnx)
            device: Device ("cuda", "cpu", or "auto")
            threshold: Detection confidence threshold [0.0, 1.0]
            id2label: Optional custom label mapping

        Raises:
            ImportError: If onnxruntime is not installed
            ModelLoadError: If model loading fails
            FileNotFoundError: If model file not found
        """
        # Initialize base adapter (handles onnxruntime import, device, threshold, id2label)
        super().__init__(device=device, threshold=threshold, id2label=id2label)

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Load ONNX session
        self._load_session()

    def _load_session(self) -> None:
        """Load ONNX Runtime session."""
        try:
            logger.info(f"Loading ONNX model: {self.model_path}")

            # Create session with optimizations
            sess_options = self.ort.SessionOptions()
            # Reduce optimization level to speed up loading
            # ORT_ENABLE_ALL can be slow on first load
            sess_options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

            self.session = self.ort.InferenceSession(
                str(self.model_path), sess_options=sess_options, providers=self.providers
            )

            # Extract I/O metadata - support multiple inputs
            model_inputs = self.session.get_inputs()
            self.input_names = [inp.name for inp in model_inputs]
            self.input_shapes = {inp.name: inp.shape for inp in model_inputs}

            # Primary input (usually 'images' for vision models)
            self.input_name = model_inputs[0].name
            self.input_shape = model_inputs[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]

            logger.info(
                f"ONNX session ready. "
                f"Inputs: {self.input_names} {self.input_shapes}, "
                f"Outputs: {len(self.output_names)}"
            )

        except Exception as e:
            raise ModelLoadError(str(self.model_path), f"Failed to load ONNX model: {type(e).__name__}: {str(e)}")

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "onnx",
            "task": "detect",
            "model_path": str(self.model_path),
            "providers": self.providers,
            "threshold": self.threshold,
            "input_shape": self.input_shape,
            "backend": "onnxruntime",
        }

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for ONNX model.

        Args:
            image: PIL Image in RGB

        Returns:
            Preprocessed numpy array ready for inference
        """
        # Get target size from model input shape
        # Shape is typically [batch, channels, height, width]
        if len(self.input_shape) == 4:
            _, _, target_h, target_w = self.input_shape
            # Handle dynamic shapes
            if isinstance(target_h, str) or target_h <= 0:
                target_h = 640
            if isinstance(target_w, str) or target_w <= 0:
                target_w = 640
        else:
            # Default to 640x640 if shape is unclear
            target_h, target_w = 640, 640

        # Resize image
        resized = image.resize((target_w, target_h), Image.BILINEAR)

        # Convert to numpy array and normalize
        img_array = np.array(resized, dtype=np.float32)

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Apply ImageNet normalization (required for most detection models)
        # Mean and std values from ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img_array = (img_array - mean) / std

        # Transpose to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def _postprocess_rtdetr(self, outputs: list[np.ndarray], threshold: float) -> list[Detection]:
        """Postprocess RT-DETR ONNX model outputs.

        RT-DETR ONNX models output 3 tensors directly:
        - labels: [batch, num_queries] - class IDs
        - boxes: [batch, num_queries, 4] - xyxy format (already scaled)
        - scores: [batch, num_queries] - confidence scores

        Args:
            outputs: Model outputs [labels, boxes, scores]
            threshold: Confidence threshold

        Returns:
            List of Detection objects
        """
        detections = []

        if len(outputs) != 3:
            logger.error(f"RT-DETR expects 3 outputs (labels, boxes, scores), got {len(outputs)}")
            return detections

        # Extract outputs - RT-DETR format
        labels = outputs[0]  # [batch, num_queries]
        boxes = outputs[1]  # [batch, num_queries, 4]
        scores = outputs[2]  # [batch, num_queries]

        # Remove batch dimension
        if len(labels.shape) == 2:
            labels = labels[0]
        if len(boxes.shape) == 3:
            boxes = boxes[0]
        if len(scores.shape) == 2:
            scores = scores[0]

        logger.debug(f"RT-DETR outputs - Labels: {labels.shape}, Boxes: {boxes.shape}, Scores: {scores.shape}")

        # Filter by threshold
        mask = scores > threshold

        # Process filtered detections
        for idx in np.where(mask)[0]:
            label_id = int(labels[idx])
            score = float(scores[idx])
            box = boxes[idx]

            # Box is already in xyxy format (absolute coordinates)
            x1, y1, x2, y2 = box

            # Get label name
            label_name = self.id2label.get(label_id, f"class_{label_id}")

            detections.append(
                Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    score=score,
                    label=label_id,
                    label_name=label_name,
                )
            )

        return detections

    def _postprocess_detr(
        self, outputs: list[np.ndarray], orig_width: int, orig_height: int, threshold: float
    ) -> list[Detection]:
        """Postprocess DETR-family model outputs.

        Args:
            outputs: Model outputs (can be [logits, boxes] or [boxes, logits])
            orig_width: Original image width
            orig_height: Original image height
            threshold: Confidence threshold

        Returns:
            List of Detection objects
        """
        detections = []

        # DETR-family models typically output 2 tensors
        if len(outputs) != 2:
            logger.warning(
                f"Expected 2 outputs for DETR model, got {len(outputs)}. " f"Attempting to process anyway..."
            )
            if len(outputs) < 2:
                logger.error("Insufficient outputs for DETR postprocessing")
                return detections

        # Detect which output is logits vs boxes based on shape
        # Logits: [batch, num_queries, num_classes] - last dim is large (80-91)
        # Boxes: [batch, num_queries, 4] - last dim is 4
        output_0 = outputs[0]
        output_1 = outputs[1]

        # Remove batch dimension first
        if len(output_0.shape) == 3:
            output_0 = output_0[0]
        if len(output_1.shape) == 3:
            output_1 = output_1[0]

        # Determine which is which based on last dimension
        if output_0.shape[-1] == 4 and output_1.shape[-1] > 4:
            # output_0 is boxes, output_1 is logits
            boxes = output_0
            logits = output_1
            logger.debug("Detected boxes in output[0], logits in output[1]")
        elif output_1.shape[-1] == 4 and output_0.shape[-1] > 4:
            # output_0 is logits, output_1 is boxes
            logits = output_0
            boxes = output_1
            logger.debug("Detected logits in output[0], boxes in output[1]")
        else:
            logger.error(f"Cannot determine output format. Shapes: {output_0.shape}, {output_1.shape}")
            return detections

        # Log shapes for debugging
        logger.debug(f"Logits shape: {logits.shape}, Boxes shape: {boxes.shape}")

        # Get class probabilities (softmax)
        # DETR uses num_classes+1 (extra "no object" class)
        # We want to exclude the last class
        if logits.shape[-1] > len(self.id2label):
            # Has "no object" class - use softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            # Exclude "no object" class (last one)
            probs = probs[:, :-1]
        else:
            # Standard softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Get best class and score for each query
        scores = np.max(probs, axis=-1)
        labels = np.argmax(probs, axis=-1)

        # Filter by threshold
        mask = scores > threshold

        # Process filtered detections
        for idx in np.where(mask)[0]:
            label_id = int(labels[idx])
            score = float(scores[idx])
            box = boxes[idx]

            # Handle different box formats
            # Ensure box is 1D array with 4 elements
            box = np.array(box).flatten()

            if len(box) != 4:
                logger.warning(
                    f"Expected 4 values for bounding box, got {len(box)}. "
                    f"Shape: {box.shape}. Skipping this detection."
                )
                continue

            # Convert from cxcywh (normalized) to xyxy (absolute)
            cx, cy, w, h = box

            # Denormalize
            cx *= orig_width
            cy *= orig_height
            w *= orig_width
            h *= orig_height

            # Convert to xyxy
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Clip to image bounds
            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))

            # Get label name
            label_name = self.id2label.get(label_id, f"class_{label_id}")

            detections.append(Detection(bbox=[x1, y1, x2, y2], score=score, label=label_id, label_name=label_name))

        return detections

    def predict(
        self, image: str | Path | Image.Image | np.ndarray, threshold: float | None = None, **kwargs: Any
    ) -> DetectResult:
        """Run object detection on an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Optional threshold override
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            DetectResult with detections

        Raises:
            InvalidInputError: If image is invalid
        """
        # Load and validate image (capture original path if from file)
        pil_image, input_path = self._load_image(image)
        orig_width, orig_height = pil_image.size

        # Use provided threshold or default
        conf_threshold = threshold if threshold is not None else self.threshold

        # Preprocess
        input_tensor = self._preprocess(pil_image)

        # Prepare input feed dict - handle multiple inputs
        input_feed = {self.input_name: input_tensor}

        # Check if model requires orig_target_sizes (RT-DETR models)
        if "orig_target_sizes" in self.input_names:
            # Provide original image size as [batch, 2] tensor [height, width]
            orig_target_sizes = np.array([[orig_height, orig_width]], dtype=np.int64)
            input_feed["orig_target_sizes"] = orig_target_sizes
            logger.debug(f"Added orig_target_sizes input: {orig_target_sizes.shape}")

        # Run inference
        logger.info(f"Running ONNX inference on {orig_width}x{orig_height} image")
        outputs = self.session.run(self.output_names, input_feed)

        # Detect model type and postprocess accordingly
        logger.info(f"Postprocessing {len(outputs)} output tensors")

        # RT-DETR ONNX models output 3 tensors: labels, boxes, scores
        # Standard DETR models output 2 tensors: logits, boxes
        if len(outputs) == 3 and "orig_target_sizes" in self.input_names:
            # RT-DETR format - outputs already processed
            logger.debug("Detected RT-DETR ONNX model format")
            detections = self._postprocess_rtdetr(outputs, conf_threshold)
        else:
            # Standard DETR format - need to process logits
            logger.debug("Using standard DETR postprocessing")
            detections = self._postprocess_detr(outputs, orig_width, orig_height, conf_threshold)

        # Create result
        result = DetectResult(
            detections=detections,
            meta={
                "model_path": str(self.model_path),
                "threshold": conf_threshold,
                "image_size": [orig_width, orig_height],
                "backend": "onnxruntime",
                "providers": self.providers,
                "input_path": input_path,
            },
        )

        logger.info(f"Found {len(detections)} detections above threshold {conf_threshold}")
        return result
