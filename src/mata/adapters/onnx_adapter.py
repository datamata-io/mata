"""ONNX Runtime adapter for optimized inference.

Supports ONNX models with automatic I/O tensor detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from mata.adapters.onnx_base import ONNXBaseAdapter
from mata.core.exceptions import InvalidInputError, ModelLoadError
from mata.core.logging import get_logger
from mata.core.preprocess import LetterboxTransform, letterbox
from mata.core.types import Detection, Instance, VisionResult

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
        preprocess: str = "letterbox",
        iou_threshold: float = 0.45,
        max_det: int | None = 300,
        channel_order: str = "rgb",
        session_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ONNX detection adapter.

        Args:
            model_path: Path to ONNX model file (.onnx)
            device: Device ("cuda", "cuda:0", "cpu", or "auto")
            threshold: Detection confidence threshold [0.0, 1.0]
            id2label: Optional custom label mapping
            preprocess: YOLO preprocessing mode, "letterbox" (default) or "stretch"
            iou_threshold: NMS IoU threshold for YOLO outputs
            max_det: Maximum detections per image after NMS (None disables the cap)
            channel_order: Channel order of numpy inputs ("rgb" or "bgr")
            session_options: Optional ONNX Runtime SessionOptions overrides

        Raises:
            ImportError: If onnxruntime is not installed
            ValueError: If an argument is invalid
            ModelLoadError: If model loading fails
            FileNotFoundError: If model file not found
        """
        if preprocess not in {"letterbox", "stretch"}:
            raise ValueError("preprocess must be 'letterbox' or 'stretch'")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_det is not None and max_det <= 0:
            raise ValueError("max_det must be positive or None")

        # Initialize base adapter (onnxruntime import, device, threshold, labels, channels)
        super().__init__(
            device=device,
            threshold=threshold,
            id2label=id2label,
            channel_order=channel_order,
            session_options=session_options,
        )

        self.preprocess = preprocess
        self.iou_threshold = float(iou_threshold)
        self.max_det = max_det

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Load ONNX session
        self._load_session()

    def _load_session(self) -> None:
        """Load ONNX Runtime session."""
        try:
            logger.info(f"Loading ONNX model: {self.model_path}")

            # Create session with optimizations (defaults + caller overrides)
            sess_options = self._make_session_options()

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

            # Report the providers the session actually got, not the ones we asked for.
            active = getattr(self.session, "get_providers", None)
            active = active() if callable(active) else None
            if isinstance(active, (list, tuple)) and active:
                self.active_providers = [str(p) for p in active]
            else:  # mocks and exotic runtimes
                self.active_providers = list(self.providers)

            if "CUDAExecutionProvider" in self.providers and "CUDAExecutionProvider" not in self.active_providers:
                raise ModelLoadError(
                    str(self.model_path),
                    f"CUDAExecutionProvider was requested but the session fell back to {self.active_providers}",
                )

            logger.info(
                f"ONNX session ready. "
                f"Inputs: {self.input_names} {self.input_shapes}, "
                f"Outputs: {len(self.output_names)}"
            )

            # Detect YOLO ONNX format: single output, input named 'images'
            # YOLO v5/v7 old: [1, anchors, 5+cls]  (last dim >= 5, 3D)
            # YOLO v8/v10+  new: [1, 4+cls, anchors] (3D, 2nd dim > 4th dim)
            self._is_yolo = self._detect_yolo_format()
            if self._is_yolo:
                self._maybe_set_yolo_labels()

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(str(self.model_path), f"Failed to load ONNX model: {type(e).__name__}: {str(e)}")

    def _detect_yolo_format(self) -> bool:
        """Heuristically detect whether the loaded ONNX model uses YOLO output format.

        YOLO ONNX models (v5, v7, v8, v10, v11, v12) share these traits:
        - Single output tensor
        - Input named 'images'
        - Output is 3-dimensional: [batch, ?, ?]

        Returns:
            True if the model appears to be a YOLO-family ONNX model.
        """
        if len(self.output_names) != 1:
            return False
        if self.input_name != "images":
            return False
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) != 3:
            return False
        return True

    def _maybe_set_yolo_labels(self) -> None:
        """Auto-populate id2label when the caller did not supply a custom label map.

        Three YOLO ONNX export layouts are handled:

          end-to-end (NMS baked in): output shape [1, N, 6]
            Class IDs are embedded in column 5; COCO 80-class map applied by default.

          new layout (YOLO v8/v10+): output shape [1, 4+nc, anchors]
            num_classes = second_dim - 4

          old layout (YOLO v5/v7): output shape [1, anchors, 5+nc]
            num_classes = third_dim - 5

        Callers who need a different label set should pass id2label= to mata.load().
        """
        if self.id2label:
            return  # caller-supplied map wins

        out_shape = self.session.get_outputs()[0].shape  # [batch, A, B]
        # Guard against symbolic/dynamic dimensions
        try:
            a, b = int(out_shape[1]), int(out_shape[2])
        except (TypeError, ValueError):
            self.id2label = self._get_coco_labels()
            logger.info("Auto-applied COCO 80-class label map to YOLO model (dynamic shape)")
            return

        # End-to-end (post-NMS) export: [batch, max_dets, 6]
        if b == 6:
            self.id2label = self._get_coco_labels()
            logger.info("Auto-applied COCO 80-class label map to end-to-end YOLO model")
            return

        # Anchor-based layouts
        num_classes = (a - 4) if a < b else (b - 5)
        if num_classes == 80:
            self.id2label = self._get_coco_labels()
            logger.info("Auto-applied COCO 80-class label map to YOLO model")
        else:
            logger.info(
                f"YOLO model has {num_classes} classes; " "pass id2label= to mata.load() for custom class names"
            )

    def info(self) -> dict[str, Any]:
        """Get adapter information.

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "name": "onnx",
            "task": "detect",
            "model_path": str(self.model_path),
            "providers": getattr(self, "active_providers", self.providers),
            "threshold": self.threshold,
            "input_shape": self.input_shape,
            "backend": "onnxruntime",
        }

    def _get_input_size(self) -> tuple[int, int]:
        """Return (target_h, target_w) from model input shape, defaulting to 640x640."""
        if len(self.input_shape) == 4:
            _, _, target_h, target_w = self.input_shape
            if isinstance(target_h, str) or target_h <= 0:
                target_h = 640
            if isinstance(target_w, str) or target_w <= 0:
                target_w = 640
        else:
            target_h, target_w = 640, 640
        return int(target_h), int(target_w)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for ONNX model (DETR-family: /255 + ImageNet norm).

        Args:
            image: PIL Image in RGB

        Returns:
            Preprocessed numpy array ready for inference
        """
        target_h, target_w = self._get_input_size()

        # Resize image
        resized = image.resize((target_w, target_h), Image.BILINEAR)

        # Normalize to [0, 1]
        img_array = np.array(resized, dtype=np.float32) / 255.0

        # Apply ImageNet normalization (required for DETR-family models)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img_array = (img_array - mean) / std

        # Transpose to CHW and add batch dimension
        return np.expand_dims(np.transpose(img_array, (2, 0, 1)), axis=0)

    def _preprocess_yolo(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for YOLO ONNX models (divide-by-255 only, no ImageNet norm).

        YOLO models are trained with pixel values scaled to [0, 1] without
        channel-wise mean/std subtraction.  This follows the normalization
        described in the original YOLO papers (Redmon et al., 2016 onward)
        and the ONNX export specification used by YOLO-family architectures.

        Args:
            image: PIL Image in RGB

        Returns:
            Preprocessed float32 array of shape [1, 3, H, W]
        """
        target_h, target_w = self._get_input_size()
        resized = image.resize((target_w, target_h), Image.BILINEAR)
        img_array = np.array(resized, dtype=np.float32) / 255.0
        return np.expand_dims(np.transpose(img_array, (2, 0, 1)), axis=0)

    def _as_rgb_array(self, image: np.ndarray) -> np.ndarray:
        """Normalize a numpy image to HWC RGB uint8 without a PIL round-trip."""
        array = image
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise InvalidInputError(
                f"Unsupported array shape for detection: {array.shape}. " "Expected HxW, HxWx3, or HxWx4."
            )
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        if self.channel_order == "bgr":
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array if array.flags.c_contiguous else np.ascontiguousarray(array)

    def _preprocess_yolo_letterbox(self, image: Image.Image | np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        """Letterbox to the model input size; return the tensor and its geometry."""
        target_h, target_w = self._get_input_size()
        canvas, transform = letterbox(image, (target_w, target_h))
        tensor = np.empty((1, 3, target_h, target_w), dtype=np.float32)
        np.copyto(tensor[0], canvas.transpose(2, 0, 1))
        tensor /= 255.0
        return tensor, transform

    def _to_original_xyxy(
        self,
        cx: np.ndarray,
        cy: np.ndarray,
        w: np.ndarray,
        h: np.ndarray,
        transform: LetterboxTransform | None,
        orig_width: int,
        orig_height: int,
    ) -> np.ndarray:
        """Map center/size boxes from model input space to original pixels."""
        if transform is None:  # legacy stretch path
            target_h, target_w = self._get_input_size()
            sx, sy = orig_width / target_w, orig_height / target_h
        else:
            sx = sy = 1.0 / transform.ratio
            cx = cx - transform.pad_left
            cy = cy - transform.pad_top
        x1 = np.clip((cx - w / 2) * sx, 0, orig_width)
        y1 = np.clip((cy - h / 2) * sy, 0, orig_height)
        x2 = np.clip((cx + w / 2) * sx, 0, orig_width)
        y2 = np.clip((cy + h / 2) * sy, 0, orig_height)
        return np.stack([x1, y1, x2, y2], axis=1)

    def _limit_detections(self, detections: list[Detection]) -> list[Detection]:
        """Apply the max_det cap, keeping the highest-scoring detections."""
        if self.max_det is None or len(detections) <= self.max_det:
            return detections
        return sorted(detections, key=lambda d: d.score, reverse=True)[: self.max_det]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        """Vectorised Non-Maximum Suppression.

        Standard greedy NMS as described in Neubeck & Van Gool (2006),
        "Efficient Non-Maximum Suppression", ICPR 2006.
        No third-party library required.

        Args:
            boxes: [N, 4] float array in xyxy format.
            scores: [N] float array of confidence scores.
            iou_threshold: IoU overlap threshold for suppression.

        Returns:
            Integer indices of kept boxes, sorted by descending score.
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_threshold]
        return np.array(keep, dtype=np.int64)

    def _postprocess_yolo(
        self,
        outputs: list[np.ndarray],
        orig_width: int,
        orig_height: int,
        threshold: float,
        iou_threshold: float = 0.45,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Postprocess YOLO ONNX model outputs.

        Dispatches to the appropriate sub-method based on the output tensor shape:

        - End-to-end / post-NMS (YOLO v8 export with ``--nms``):
            output shape [1, max_dets, 6]
            Cols 0-3: x1, y1, x2, y2 in model-input pixel space
            Col  4:   confidence score [0, 1]
            Col  5:   class ID (integer encoded as float)
            NMS is already applied inside the ONNX graph.

        - New layout (YOLO v8 / v10 / v11 / v12, raw anchors):
            output shape [1, 4+num_classes, num_anchors]
            Rows 0-3: cx, cy, w, h (normalised to input size)
            Rows 4+:  per-class scores (no separate objectness)

        - Old layout (YOLO v5 / v7):
            output shape [1, num_anchors, 5+num_classes]
            Cols 0-3: cx, cy, w, h (normalised to input size)
            Col  4:   objectness confidence
            Cols 5+:  per-class scores

        Args:
            outputs: List containing the single YOLO output tensor.
            orig_width: Original image width (pixels).
            orig_height: Original image height (pixels).
            threshold: Confidence threshold for filtering detections.
            iou_threshold: IoU threshold for NMS (anchor-based layouts only).
            transform: Letterbox geometry, or None for the legacy stretch path.

        Returns:
            List of Detection objects in xyxy absolute-pixel coordinates.
        """
        raw = outputs[0]  # [1, ?, ?]
        if raw.ndim == 3:
            raw = raw[0]  # remove batch dim → [A, B]

        target_h, target_w = self._get_input_size()
        scale_x = orig_width / target_w
        scale_y = orig_height / target_h

        # End-to-end export: last dim == 6 means [x1,y1,x2,y2,conf,class_id]
        if raw.shape[1] == 6:
            logger.debug("YOLO end-to-end (post-NMS) format detected")
            return self._postprocess_yolo_e2e(raw, scale_x, scale_y, orig_width, orig_height, threshold, transform)

        # Anchor-based layouts -----------------------------------------------
        # New layout: shape [4+cls, anchors]  (first dim < second dim)
        # Old layout: shape [anchors, 5+cls]  (first dim > second dim)
        if raw.shape[0] < raw.shape[1]:
            # New layout: [4+cls, anchors] — transpose to [anchors, 4+cls]
            raw = raw.T
            cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
            class_scores = raw[:, 4:]  # [anchors, num_classes]
            scores = class_scores.max(axis=1)
            labels = class_scores.argmax(axis=1)
            logger.debug(f"YOLO new layout: {raw.shape[1]-4} classes, {raw.shape[0]} anchors")
        else:
            # Old layout: [anchors, 5+cls]
            cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
            objectness = raw[:, 4]
            class_scores = raw[:, 5:]  # [anchors, num_classes]
            scores = objectness * class_scores.max(axis=1)
            labels = class_scores.argmax(axis=1)
            logger.debug(f"YOLO old layout: {raw.shape[1]-5} classes, {raw.shape[0]} anchors")

        # Filter by confidence before NMS ------------------------------------
        mask = scores >= threshold
        if not mask.any():
            return []

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        scores = scores[mask]
        labels = labels[mask]

        # cxcywh (input space) → xyxy (original image pixels)
        boxes_xyxy = self._to_original_xyxy(cx, cy, w, h, transform, orig_width, orig_height)

        # Class-aware NMS ----------------------------------------------------
        detections: list[Detection] = []
        for cls_id in np.unique(labels):
            cls_mask = labels == cls_id
            kept = self._nms(boxes_xyxy[cls_mask], scores[cls_mask], iou_threshold)
            cls_boxes = boxes_xyxy[cls_mask][kept]
            cls_scores = scores[cls_mask][kept]
            for box, score in zip(cls_boxes, cls_scores):
                detections.append(
                    Detection(
                        bbox=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        score=float(score),
                        label=int(cls_id),
                        label_name=self.id2label.get(int(cls_id), f"class_{cls_id}"),
                    )
                )
        return self._limit_detections(detections)

    def _postprocess_yolo_e2e(
        self,
        raw: np.ndarray,
        scale_x: float,
        scale_y: float,
        orig_width: int,
        orig_height: int,
        threshold: float,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Postprocess end-to-end YOLO ONNX output (NMS baked into the graph).

        Expected layout after batch removal: [max_dets, 6]
          col 0-3: x1, y1, x2, y2 in model-input pixel coordinates
          col 4:   confidence score [0, 1]
          col 5:   class ID (integer, stored as float)

        Args:
            raw: Array of shape [max_dets, 6].
            scale_x: Horizontal scale factor (orig_width / model_input_width).
            scale_y: Vertical scale factor (orig_height / model_input_height).
            orig_width: Original image width for coordinate clipping.
            orig_height: Original image height for coordinate clipping.
            threshold: Confidence score threshold.
            transform: Letterbox geometry, or None for the legacy stretch path.

        Returns:
            List of Detection objects.
        """
        conf = raw[:, 4]
        mask = conf >= threshold
        if not mask.any():
            return []

        raw = raw[mask]
        if transform is None:
            x1 = np.clip(raw[:, 0] * scale_x, 0, orig_width)
            y1 = np.clip(raw[:, 1] * scale_y, 0, orig_height)
            x2 = np.clip(raw[:, 2] * scale_x, 0, orig_width)
            y2 = np.clip(raw[:, 3] * scale_y, 0, orig_height)
        else:
            ratio = transform.ratio
            x1 = np.clip((raw[:, 0] - transform.pad_left) / ratio, 0, orig_width)
            y1 = np.clip((raw[:, 1] - transform.pad_top) / ratio, 0, orig_height)
            x2 = np.clip((raw[:, 2] - transform.pad_left) / ratio, 0, orig_width)
            y2 = np.clip((raw[:, 3] - transform.pad_top) / ratio, 0, orig_height)
        scores = raw[:, 4]
        class_ids = raw[:, 5].astype(np.int32)

        detections: list[Detection] = []
        for box_coords, score, cls_id in zip(zip(x1, y1, x2, y2), scores, class_ids):
            detections.append(
                Detection(
                    bbox=[float(c) for c in box_coords],
                    score=float(score),
                    label=int(cls_id),
                    label_name=self.id2label.get(int(cls_id), f"class_{cls_id}"),
                )
            )
        return self._limit_detections(detections)

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
    ) -> VisionResult:
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
        # Load and validate image (capture original path if from file).
        # Numpy inputs on the default YOLO letterbox path stay as arrays: the
        # PIL round-trip costs more than resize + pad + normalize combined.
        source: Image.Image | np.ndarray
        if (
            isinstance(image, np.ndarray)
            and image.dtype == np.uint8
            and self._is_yolo
            and self.preprocess == "letterbox"
        ):
            source = self._as_rgb_array(image)
            orig_height, orig_width = source.shape[:2]
            input_path: str | None = None
        else:
            pil_image, input_path = self._load_image(image)
            orig_width, orig_height = pil_image.size
            source = pil_image

        # Use provided threshold or default
        conf_threshold = threshold if threshold is not None else self.threshold

        # Preprocess — YOLO models use /255 only; DETR-family use ImageNet norm
        transform: LetterboxTransform | None = None
        if self._is_yolo:
            if self.preprocess == "letterbox":
                input_tensor, transform = self._preprocess_yolo_letterbox(source)
            else:
                input_tensor = self._preprocess_yolo(source)  # type: ignore[arg-type]
        else:
            input_tensor = self._preprocess(source)  # type: ignore[arg-type]

        # Prepare input feed dict - handle multiple inputs
        input_feed = {self.input_name: input_tensor}

        # Check if model requires orig_target_sizes (RT-DETR models)
        if "orig_target_sizes" in self.input_names:
            # Provide original image size as [batch, 2] tensor [height, width]
            orig_target_sizes = np.array([[orig_height, orig_width]], dtype=np.int64)
            input_feed["orig_target_sizes"] = orig_target_sizes
            logger.debug(f"Added orig_target_sizes input: {orig_target_sizes.shape}")

        # Run inference
        logger.debug(f"Running ONNX inference on {orig_width}x{orig_height} image")
        outputs = self.session.run(self.output_names, input_feed)

        # Detect model type and postprocess accordingly
        logger.debug(f"Postprocessing {len(outputs)} output tensors")

        # RT-DETR ONNX models output 3 tensors: labels, boxes, scores
        # YOLO ONNX models output 1 tensor (auto-detected at load time)
        # Standard DETR models output 2 tensors: logits, boxes
        if len(outputs) == 3 and "orig_target_sizes" in self.input_names:
            logger.debug("Detected RT-DETR ONNX model format")
            detections = self._postprocess_rtdetr(outputs, conf_threshold)
        elif self._is_yolo:
            logger.debug("Detected YOLO ONNX model format")
            detections = self._postprocess_yolo(
                outputs,
                orig_width,
                orig_height,
                conf_threshold,
                iou_threshold=self.iou_threshold,
                transform=transform,
            )
        else:
            logger.debug("Using standard DETR postprocessing")
            detections = self._postprocess_detr(outputs, orig_width, orig_height, conf_threshold)

        # Convert to Instance list (VisionResult is the universal type expected by
        # tracking, graph nodes, and the rest of the framework).
        instances = [
            Instance(
                bbox=tuple(d.bbox),
                score=d.score,
                label=d.label,
                label_name=d.label_name,
            )
            for d in detections
        ]
        result = VisionResult(
            instances=instances,
            meta={
                "model_path": str(self.model_path),
                "threshold": conf_threshold,
                "image_size": [orig_width, orig_height],
                "backend": "onnxruntime",
                "providers": getattr(self, "active_providers", self.providers),
                "input_path": input_path,
            },
        )

        logger.debug(f"Found {len(instances)} detections above threshold {conf_threshold}")
        return result
