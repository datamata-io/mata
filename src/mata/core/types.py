"""Core data types for MATA framework.

This module defines immutable dataclasses for task results and common types.
All coordinate systems use xyxy format (x1, y1, x2, y2) in absolute pixels.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Type aliases
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in absolute pixels


@dataclass(frozen=True)
class Instance:
    """Universal instance with optional bbox, mask, and keypoints.

    This class represents a single detected/segmented instance in an image,
    supporting multi-modal vision tasks by combining bounding boxes, masks,
    and other attributes in a unified structure.

    Attributes:
        bbox: Optional bounding box in xyxy format (x1, y1, x2, y2) absolute pixels
        mask: Optional mask representation
            - RLE format: {"size": [height, width], "counts": bytes or str}
            - Binary format: np.ndarray of shape (H, W) with dtype bool
            - Polygon format: List of [x1, y1, x2, y2, ...] coordinates
        score: Confidence score [0.0, 1.0]
        label: Integer class label (0-indexed)
        label_name: Optional human-readable class name
        area: Optional mask/bbox area in pixels
        is_stuff: Optional flag for panoptic segmentation
            - False: Instance (countable object)
            - True: Stuff (uncountable region)
            - None: Not specified
        embedding: Optional feature embedding vector
        track_id: Optional tracking identifier
        keypoints: Optional keypoints (future support)

    Examples:
        >>> # Detection-only instance
        >>> inst = Instance(
        ...     bbox=(50, 50, 300, 300),
        ...     score=0.95,
        ...     label=0,
        ...     label_name="cat"
        ... )
        >>>
        >>> # Segmentation-only instance
        >>> inst = Instance(
        ...     mask=binary_mask,
        ...     score=0.87,
        ...     label=0,
        ...     label_name="cat"
        ... )
        >>>
        >>> # Combined detection + segmentation
        >>> inst = Instance(
        ...     bbox=(50, 50, 300, 300),
        ...     mask=rle_mask,
        ...     score=0.92,
        ...     label=0,
        ...     label_name="cat",
        ...     area=12500
        ... )
    """

    score: float
    label: int
    bbox: BBox | None = None
    mask: dict[str, Any] | np.ndarray | list[float] | None = None
    label_name: str | None = None
    area: int | None = None
    is_stuff: bool | None = None
    embedding: np.ndarray | None = None
    track_id: int | None = None
    keypoints: Any | None = None  # Future support

    def __post_init__(self):
        """Validate instance has at least bbox or mask."""
        if self.bbox is None and self.mask is None:
            raise ValueError("Instance must have at least one of: bbox, mask. " "Both cannot be None.")

        # Validate mask format if present
        if self.mask is not None:
            if NUMPY_AVAILABLE and isinstance(self.mask, np.ndarray):
                if self.mask.ndim != 2:
                    raise ValueError(f"Binary mask must be 2D array, got shape {self.mask.shape}")
            elif isinstance(self.mask, dict):
                if "size" not in self.mask or "counts" not in self.mask:
                    raise ValueError("RLE mask dict must contain 'size' and 'counts' keys")
            elif isinstance(self.mask, list):
                if len(self.mask) % 2 != 0:
                    raise ValueError(f"Polygon must have even number of coordinates, got {len(self.mask)}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "score": self.score,
            "label": self.label,
            "bbox": list(self.bbox) if self.bbox else None,
            "label_name": self.label_name,
            "area": self.area,
            "is_stuff": self.is_stuff,
            "track_id": self.track_id,
        }

        # Handle mask serialization
        if self.mask is not None:
            if NUMPY_AVAILABLE and isinstance(self.mask, np.ndarray):
                result["mask"] = {
                    "format": "binary",
                    "data": self.mask.tolist(),
                    "shape": list(self.mask.shape),
                    "dtype": str(self.mask.dtype),
                }
            elif isinstance(self.mask, dict):
                result["mask"] = {
                    "format": "rle",
                    "data": {
                        "size": self.mask["size"],
                        "counts": (
                            self.mask["counts"].decode("utf-8")
                            if isinstance(self.mask["counts"], bytes)
                            else self.mask["counts"]
                        ),
                    },
                }
            elif isinstance(self.mask, list):
                result["mask"] = {"format": "polygon", "data": self.mask}
            else:
                result["mask"] = {"format": "unknown", "data": self.mask}
        else:
            result["mask"] = None

        # Handle embedding serialization
        if self.embedding is not None:
            if NUMPY_AVAILABLE and isinstance(self.embedding, np.ndarray):
                result["embedding"] = {
                    "data": self.embedding.tolist(),
                    "shape": list(self.embedding.shape),
                    "dtype": str(self.embedding.dtype),
                }
            else:
                result["embedding"] = self.embedding
        else:
            result["embedding"] = None

        return result

    def is_rle(self) -> bool:
        """Check if mask is in RLE format."""
        return isinstance(self.mask, dict) and "counts" in self.mask

    def is_binary(self) -> bool:
        """Check if mask is a binary numpy array."""
        return NUMPY_AVAILABLE and isinstance(self.mask, np.ndarray)

    def is_polygon(self) -> bool:
        """Check if mask is in polygon format."""
        return isinstance(self.mask, list) and len(self.mask) > 0


@dataclass(frozen=True)
class Entity:
    """Semantic entity from VLM structured output (stage 1).

    Represents a concept/object identified by a VLM in text output,
    without spatial grounding (no bbox or mask). Can be promoted to
    a full Instance once spatial data is provided by a downstream
    adapter (e.g., GroundingDINO for bbox, SAM for mask).

    This is the stage-1 type in the 2-stage contract:
      Stage 1: VLM → Entity (label + score + attributes)
      Stage 2: Spatial adapter → Entity.promote() → Instance (label + score + bbox/mask)

    Attributes:
        label: Human-readable label/class name
        score: Confidence score [0.0, 1.0], defaults to 1.0
        attributes: Optional dictionary of additional attributes
            parsed from VLM output (e.g., color, size, count, description)
    """

    label: str
    score: float = 1.0
    attributes: dict[str, Any] | None = field(default_factory=dict)

    def promote(
        self,
        bbox: BBox | None = None,
        mask: dict[str, Any] | np.ndarray | list[float] | None = None,
        label_id: int = 0,
        **kwargs: Any,
    ) -> Instance:
        """Promote entity to a full Instance with spatial data (stage 2).

        Args:
            bbox: Bounding box in xyxy format
            mask: Mask in RLE, binary, or polygon format
            label_id: Integer class label (default: 0)
            **kwargs: Additional Instance fields (area, is_stuff, embedding, etc.)

        Returns:
            Instance with spatial data + entity metadata

        Raises:
            ValueError: If neither bbox nor mask is provided
        """
        return Instance(
            bbox=bbox,
            mask=mask,
            score=self.score,
            label=label_id,
            label_name=self.label,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary representation.

        Returns:
            Dictionary with label, score, and attributes
        """
        return {"label": self.label, "score": self.score, "attributes": self.attributes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Create entity from dictionary representation.

        Args:
            data: Dictionary with label (required), score (optional), attributes (optional)

        Returns:
            Entity instance
        """
        return cls(
            label=data["label"],
            score=data.get("score", 1.0),
            attributes=data.get("attributes", {}),
        )


@dataclass(frozen=True)
class VisionResult:
    """Universal result for all vision tasks.

    This unified result type supports multiple vision tasks:
    - Object detection (instances with bboxes)
    - Instance/panoptic segmentation (instances with masks)
    - Multi-modal pipelines (instances with both bbox and mask)
    - Tracking (instances with track_id)
    - Future: VQA, captioning (text field)

    Provides backward compatibility with DetectResult and SegmentResult
    through type aliases and filtering properties.

    Attributes:
        instances: List of detected/segmented instances
        meta: Optional metadata dictionary
        text: Optional text output (for VQA, captioning tasks)
        prompt: Optional input prompt/query (for VQA tasks)

    Examples:
        >>> # Detection result (backward compatible with DetectResult)
        >>> result = VisionResult(
        ...     instances=[
        ...         Instance(bbox=(10, 20, 100, 200), score=0.95, label=0, label_name="cat")
        ...     ]
        ... )
        >>> detections = result.detections  # Filters instances with bbox
        >>>
        >>> # Segmentation result (backward compatible with SegmentResult)
        >>> result = VisionResult(
        ...     instances=[
        ...         Instance(mask=rle_mask, score=0.87, label=0, label_name="cat")
        ...     ]
        ... )
        >>> masks = result.masks  # Filters instances with mask
        >>>
        >>> # Multi-modal result (detection + segmentation)
        >>> result = VisionResult(
        ...     instances=[
        ...         Instance(
        ...             bbox=(10, 20, 100, 200),
        ...             mask=rle_mask,
        ...             score=0.92,
        ...             label=0,
        ...             label_name="cat"
        ...         )
        ...     ],
        ...     meta={"pipeline": "grounding_sam"}
        ... )
    """

    instances: list[Instance]
    meta: dict[str, Any] = field(default_factory=dict)
    text: str | None = None
    prompt: str | None = None
    entities: list[Entity] = field(default_factory=list)

    # Backward compatibility properties
    @property
    def detections(self) -> list[Instance]:
        """Get instances with bboxes (DetectResult compatibility).

        Returns:
            List of instances that have bbox attribute set
        """
        return [inst for inst in self.instances if inst.bbox is not None]

    @property
    def masks(self) -> list[Instance]:
        """Get instances with masks (SegmentResult compatibility).

        Returns:
            List of instances that have mask attribute set
        """
        return [inst for inst in self.instances if inst.mask is not None]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "instances": [inst.to_dict() for inst in self.instances],
            "entities": [ent.to_dict() for ent in self.entities],
            "meta": self.meta,
            "text": self.text,
            "prompt": self.prompt,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string.

        Args:
            **kwargs: Additional arguments passed to json.dumps()

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionResult:
        """Create from dictionary representation.

        Args:
            data: Dictionary with instances and optional meta/text/prompt

        Returns:
            VisionResult instance
        """
        instances = []
        for inst_data in data["instances"]:
            # Reconstruct mask if present
            mask = None
            mask_info = inst_data.get("mask")
            if mask_info:
                if mask_info["format"] == "rle":
                    mask = mask_info["data"]
                elif mask_info["format"] == "binary" and NUMPY_AVAILABLE:
                    mask = np.array(mask_info["data"], dtype=mask_info.get("dtype", "bool"))
                elif mask_info["format"] == "polygon":
                    mask = mask_info["data"]
                else:
                    mask = mask_info["data"]

            # Reconstruct embedding if present
            embedding = None
            emb_info = inst_data.get("embedding")
            if emb_info and NUMPY_AVAILABLE:
                if isinstance(emb_info, dict):
                    embedding = np.array(emb_info["data"], dtype=emb_info.get("dtype", "float32"))
                else:
                    embedding = emb_info

            instances.append(
                Instance(
                    bbox=tuple(inst_data["bbox"]) if inst_data.get("bbox") else None,
                    mask=mask,
                    score=inst_data["score"],
                    label=inst_data["label"],
                    label_name=inst_data.get("label_name"),
                    area=inst_data.get("area"),
                    is_stuff=inst_data.get("is_stuff"),
                    embedding=embedding,
                    track_id=inst_data.get("track_id"),
                )
            )

        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        return cls(
            instances=instances,
            entities=entities,
            meta=data.get("meta", {}),
            text=data.get("text"),
            prompt=data.get("prompt"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> VisionResult:
        """Deserialize from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            VisionResult instance
        """
        return cls.from_dict(json.loads(json_str))

    def filter_by_score(self, threshold: float) -> VisionResult:
        """Filter instances and entities by confidence threshold.

        Args:
            threshold: Minimum confidence score [0.0, 1.0]

        Returns:
            New VisionResult with filtered instances and entities
        """
        filtered_instances = [inst for inst in self.instances if inst.score >= threshold]
        filtered_entities = [ent for ent in self.entities if ent.score >= threshold]
        return VisionResult(
            instances=filtered_instances, entities=filtered_entities, meta=self.meta, text=self.text, prompt=self.prompt
        )

    def get_instances(self) -> list[Instance]:
        """Get only instance objects (is_stuff=False).

        Returns:
            List of instance objects (countable)
        """
        return [inst for inst in self.instances if inst.is_stuff is False or inst.is_stuff is None]

    def get_stuff(self) -> list[Instance]:
        """Get only stuff regions (is_stuff=True).

        Returns:
            List of stuff regions (uncountable)
        """
        return [inst for inst in self.instances if inst.is_stuff is True]

    def get_input_path(self) -> str | None:
        """Get original input image path if available.

        Returns path string if image was loaded from file, None otherwise.
        When image is provided as PIL.Image or numpy array, no path is available.

        Returns:
            Path string if available, None otherwise

        Examples:
            >>> result = mata.run("detect", "image.jpg")
            >>> result.get_input_path()
            'image.jpg'
            >>>
            >>> # PIL Image input - no path
            >>> from PIL import Image
            >>> img = Image.open("test.jpg")
            >>> result = mata.run("detect", img)
            >>> result.get_input_path()
            None
        """
        return self.meta.get("input_path")

    def save(
        self,
        output_path: str | Path,
        image: str | Path | Image.Image | np.ndarray | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save result to file in various formats.

        Format is auto-detected from file extension:
        - .json: JSON export
        - .csv: CSV tabular export (detections/classifications)
        - .png/.jpg/.jpeg: Image overlay with bboxes/masks
        - Special prefix 'crops_' or crop_dir parameter: Extract detection crops

        Args:
            output_path: Path to save result
            image: Original image for overlay/crop export.
                If None, uses result.meta['input_path'] (from mata.run())
            format: Override format detection ('json', 'csv', 'image', 'crops')
            **kwargs: Additional parameters for specific exporters:
                - JSON: indent (default: 2)
                - CSV: (reserved)
                - Image: show_boxes, show_labels, show_scores, show_masks, alpha
                - Crops: crop_dir, padding

        Raises:
            InvalidInputError: If image required but not available
            ValueError: If unsupported file format
            IOError: If file save fails

        Examples:
            >>> # Auto-detect format from extension
            >>> result = mata.run("detect", "image.jpg")
            >>> result.save("detections.json")    # JSON export
            >>> result.save("detections.csv")     # CSV export
            >>> result.save("overlay.png")        # Image overlay
            >>>
            >>> # Extract detection crops
            >>> result.save("crops.png", crop_dir="my_crops")
            >>>
            >>> # Customize image overlay
            >>> result.save(
            ...     "output.png",
            ...     show_boxes=True,
            ...     show_labels=True,
            ...     alpha=0.5
            ... )
            >>>
            >>> # PIL Image input - provide image explicitly
            >>> from PIL import Image
            >>> pil_img = Image.open("test.jpg")
            >>> result = mata.run("detect", pil_img)
            >>> result.save("output.png", image="test.jpg")
        """
        from pathlib import Path

        from mata.core.exporters import export_crops, export_csv, export_image, export_json

        output_path = Path(output_path)

        # Auto-detect format from extension if not specified
        if format is None:
            ext = output_path.suffix.lower()
            if ext == ".json":
                format = "json"
            elif ext == ".csv":
                format = "csv"
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                # Check if crops are requested
                if "crop_dir" in kwargs or output_path.stem.startswith("crops_"):
                    format = "crops"
                else:
                    format = "image"
            else:
                raise ValueError(f"Unsupported file format: '{ext}'. " f"Supported: .json, .csv, .png, .jpg, .jpeg")

        # Route to appropriate exporter
        if format == "json":
            export_json(self, output_path, **kwargs)
        elif format == "csv":
            export_csv(self, output_path, **kwargs)
        elif format == "image":
            export_image(self, output_path, image=image, **kwargs)
        elif format == "crops":
            export_crops(self, output_path, image=image, **kwargs)
        else:
            raise ValueError(f"Unknown format: '{format}'")


@dataclass(frozen=True)
class DepthResult:
    """Depth estimation task result.

    Attributes:
        depth: Raw depth map as float array (H, W)
        normalized: Optional normalized depth map in [0, 1]
        meta: Optional metadata (model info, input path, etc.)
    """

    depth: np.ndarray
    normalized: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate depth map shape."""
        if NUMPY_AVAILABLE and isinstance(self.depth, np.ndarray):
            if self.depth.ndim != 2:
                raise ValueError(f"Depth map must be 2D array, got shape {self.depth.shape}")
        if self.normalized is not None and NUMPY_AVAILABLE and isinstance(self.normalized, np.ndarray):
            if self.normalized.ndim != 2:
                raise ValueError(f"Normalized depth must be 2D array, got shape {self.normalized.shape}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""

        def _serialize_array(arr: Any) -> dict[str, Any]:
            if NUMPY_AVAILABLE and isinstance(arr, np.ndarray):
                return {"data": arr.tolist(), "shape": list(arr.shape), "dtype": str(arr.dtype)}
            return {"data": arr}

        return {
            "depth": _serialize_array(self.depth),
            "normalized": _serialize_array(self.normalized) if self.normalized is not None else None,
            "meta": self.meta,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepthResult:
        """Create from dictionary representation."""

        def _deserialize_array(arr_info: Any) -> Any:
            if not isinstance(arr_info, dict):
                return arr_info
            if "data" not in arr_info:
                return arr_info
            if NUMPY_AVAILABLE and "shape" in arr_info:
                return np.array(arr_info["data"], dtype=arr_info.get("dtype", "float32")).reshape(arr_info["shape"])
            return arr_info["data"]

        depth = _deserialize_array(data.get("depth"))
        normalized = _deserialize_array(data.get("normalized")) if data.get("normalized") is not None else None
        return cls(depth=depth, normalized=normalized, meta=data.get("meta", {}))

    @classmethod
    def from_json(cls, json_str: str) -> DepthResult:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @property
    def depth_map(self) -> np.ndarray:
        """Return the depth map: normalized ([0, 1]) if available, otherwise raw depth."""
        return self.normalized if self.normalized is not None else self.depth

    def save(
        self,
        output_path: str | Path,
        image: str | Path | Image.Image | np.ndarray | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save depth result to file.

        Format auto-detected from extension (.json, .png, .jpg, etc.).
        """
        from pathlib import Path

        from mata.core.exporters import export_image, export_json

        output_path = Path(output_path)

        if format is None:
            ext = output_path.suffix.lower()
            if ext == ".json":
                format = "json"
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                format = "image"
            else:
                raise ValueError(f"Unsupported file format: '{ext}'")

        if format == "json":
            export_json(self, output_path, **kwargs)
        elif format == "image":
            export_image(self, output_path, image=image, **kwargs)
        else:
            raise ValueError(f"Unknown format: '{format}'")


@dataclass(frozen=True)
class TextRegion:
    """A single recognized text span, optionally with spatial location.

    Attributes:
        text: The recognized text string
        score: Confidence score in [0.0, 1.0]
        bbox: Optional bounding box in xyxy absolute pixel coords. None for
              whole-image OCR models (e.g. TrOCR) that produce no spatial info.
        label: Optional label, e.g. language code "en"/"zh" or modality
               "printed"/"handwritten".
    """

    text: str
    score: float
    bbox: tuple[float, float, float, float] | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "score": self.score,
            "bbox": list(self.bbox) if self.bbox else None,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextRegion:
        """Create from dictionary representation."""
        bbox = tuple(data["bbox"]) if data.get("bbox") else None
        return cls(
            text=data["text"],
            score=data["score"],
            bbox=bbox,  # type: ignore[arg-type]
            label=data.get("label"),
        )


@dataclass(frozen=True)
class OCRResult:
    """Result of an OCR (text extraction) operation on an image.

    Attributes:
        regions: List of recognized text regions.
        meta: Optional metadata (model info, input path, engine, etc.)

    Examples:
        >>> result = OCRResult(regions=[TextRegion(text="hello", score=0.9)])
        >>> result.full_text
        'hello'
        >>> filtered = result.filter_by_score(0.95)
    """

    regions: list[TextRegion]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all region texts with newlines."""
        return "\n".join(r.text for r in self.regions)

    def filter_by_score(self, threshold: float) -> OCRResult:
        """Return new OCRResult containing only regions with score >= threshold.

        Args:
            threshold: Minimum confidence score [0.0, 1.0]

        Returns:
            New OCRResult with filtered regions
        """
        return OCRResult(
            regions=[r for r in self.regions if r.score >= threshold],
            meta=self.meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "regions": [r.to_dict() for r in self.regions],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRResult:
        """Create from dictionary representation."""
        return cls(
            regions=[TextRegion.from_dict(r) for r in data.get("regions", [])],
            meta=data.get("meta", {}),
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> OCRResult:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, output_path: str, **kwargs: Any) -> None:
        """Save OCR result to file.

        Format is auto-detected from the file extension:
          - .json  → structured JSON
          - .csv   → CSV with columns: text, score, x1, y1, x2, y2
          - .txt   → plain concatenated text
          - .png / .jpg / .jpeg → image with text-region overlays

        Args:
            output_path: Destination file path.
            **kwargs: Forwarded to the underlying exporter.
        """
        from pathlib import Path

        suffix = Path(output_path).suffix.lower()
        if suffix == ".json":
            from mata.core.exporters import export_json

            export_json(self, output_path, **kwargs)
        elif suffix == ".csv":
            from mata.core.exporters import export_ocr_csv

            export_ocr_csv(self, output_path, **kwargs)
        elif suffix == ".txt":
            from mata.core.exporters import export_text

            export_text(self, output_path, **kwargs)
        elif suffix in (".png", ".jpg", ".jpeg"):
            from mata.core.exporters import export_ocr_image

            export_ocr_image(self, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported OCR save format: '{suffix}'. " "Use .json, .csv, .txt, or .png/.jpg")


class ModelType(str, Enum):
    """Supported model source types for explicit loading.

    This enum allows users to explicitly specify the model format when loading,
    bypassing auto-detection. This is particularly useful for:
    - Disambiguating .pt files (TorchScript vs PyTorch checkpoint)
    - Ensuring correct adapter selection
    - Improving load performance (skip detection probes)

    Attributes:
        AUTO: Auto-detect model type from source (default behavior)
            - Uses file extension, path patterns, and probing
            - Supported for all source types

        HUGGINGFACE: Load from HuggingFace Hub
            - Source format: "org/model-id" or full URL
            - Valid kwargs: threshold, device, id2label
            - Example: "facebook/detr-resnet-50"

        PYTORCH_CHECKPOINT: PyTorch state dict checkpoint (.pth, .pt, .bin)
            - Source: Local file path to checkpoint
            - Valid kwargs: checkpoint_path, config, device, threshold, id2label
            - Requires: Optional config YAML for architecture
            - Note: Use this for .pt files that are NOT TorchScript

        TORCHSCRIPT: TorchScript serialized model (.pt)
            - Source: Local file path to TorchScript archive
            - Valid kwargs: model_path, device, threshold, input_size, id2label
            - Required: input_size parameter for preprocessing
            - Note: Use this for .pt files created with torch.jit.save()

        ONNX: ONNX model file (.onnx)
            - Source: Local file path to ONNX model
            - Valid kwargs: model_path, device, threshold, id2label
            - Requires: onnxruntime or onnxruntime-gpu installed

        TENSORRT: TensorRT engine (.trt, .engine)
            - Source: Local file path to TensorRT engine
            - Valid kwargs: engine_path, device, threshold
            - Requires: tensorrt and cuda libraries installed
            - Status: Implementation in progress

        CONFIG_ALIAS: Load from config file alias
            - Source: Alias name defined in models.yaml
            - Inherits kwargs from config (can be overridden)
            - Config locations: ~/.mata/models.yaml or .mata/models.yaml

        LEGACY_PLUGIN: Legacy plugin system (deprecated)
            - Will be removed in MATA 2.0
            - Use explicit model types instead

    Examples:
        >>> from mata import load
        >>> from mata.core.types import ModelType
        >>>
        >>> # Explicit TorchScript (avoids .pt ambiguity)
        >>> detector = load("detect", "model.pt",
        ...                 model_type=ModelType.TORCHSCRIPT,
        ...                 input_size=640)
        >>>
        >>> # Explicit PyTorch checkpoint
        >>> detector = load("detect", "checkpoint.pt",
        ...                 model_type=ModelType.PYTORCH_CHECKPOINT,
        ...                 config="config.yaml")
        >>>
        >>> # String format (soft deprecated, use enum)
        >>> detector = load("detect", "model.onnx",
        ...                 model_type="onnx")  # Warns, but works
        >>>
        >>> # Auto-detection (default)
        >>> detector = load("detect", "model.onnx")  # AUTO mode
    """

    AUTO = "auto"
    HUGGINGFACE = "huggingface"
    PYTORCH_CHECKPOINT = "pytorch_checkpoint"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    CONFIG_ALIAS = "config_alias"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"

    @classmethod
    def normalize(cls, value: str | ModelType | None) -> ModelType | None:
        """Normalize string or enum to ModelType enum.

        Provides soft deprecation for string literals while maintaining
        backward compatibility. Accepts case-insensitive strings.

        Args:
            value: String, ModelType enum, or None

        Returns:
            ModelType enum or None (for AUTO mode)

        Examples:
            >>> ModelType.normalize("torchscript")
            <ModelType.TORCHSCRIPT: 'torchscript'>
            >>> ModelType.normalize(ModelType.ONNX)
            <ModelType.ONNX: 'onnx'>
            >>> ModelType.normalize(None)
            None
        """
        if value is None:
            return None

        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            # Soft deprecation warning for string literals
            warnings.warn(
                f"Passing model_type as string ('{value}') is deprecated. "
                f"Use ModelType enum instead: model_type=ModelType.{value.upper()}",
                DeprecationWarning,
                stacklevel=3,
            )

            try:
                # Case-insensitive lookup
                normalized = value.lower().strip()
                return cls(normalized)
            except ValueError:
                # Invalid string - return None and let validation handle it
                valid_values = [t.value for t in cls]
                warnings.warn(
                    f"Unknown model_type '{value}'. " f"Valid values: {valid_values}. " f"Using AUTO detection.",
                    UserWarning,
                    stacklevel=3,
                )
                return None

        raise TypeError(f"model_type must be str, ModelType, or None; got {type(value).__name__}")


@dataclass(frozen=True)
class Detection:
    """Single object detection result.

    DEPRECATED: Use Instance from VisionResult instead.
    Maintained for backward compatibility only.

    Attributes:
        bbox: Bounding box in xyxy format (x1, y1, x2, y2) absolute pixels
        score: Confidence score [0.0, 1.0]
        label: Integer class label
        label_name: Optional human-readable class name
    """

    bbox: BBox
    score: float
    label: int
    label_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "bbox": list(self.bbox),
            "score": self.score,
            "label": self.label,
            "label_name": self.label_name,
        }

    def to_instance(self) -> Instance:
        """Convert to unified Instance representation.

        Returns:
            Instance object with bbox field populated
        """
        return Instance(bbox=self.bbox, score=self.score, label=self.label, label_name=self.label_name)


@dataclass(frozen=True)
class DetectResult:
    """Object detection task result.

    Attributes:
        detections: List of detected objects
        meta: Optional metadata (model info, timing, etc.)
    """

    detections: list[Detection]
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "detections": [d.to_dict() for d in self.detections],
            "meta": self.meta,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string.

        Args:
            **kwargs: Additional arguments passed to json.dumps()

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectResult:
        """Create from dictionary representation.

        Args:
            data: Dictionary with detections and optional meta

        Returns:
            DetectResult instance
        """
        detections = [
            Detection(
                bbox=tuple(d["bbox"]),
                score=d["score"],
                label=d["label"],
                label_name=d.get("label_name"),
            )
            for d in data["detections"]
        ]
        return cls(detections=detections, meta=data.get("meta"))

    @classmethod
    def from_json(cls, json_str: str) -> DetectResult:
        """Deserialize from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            DetectResult instance
        """
        return cls.from_dict(json.loads(json_str))

    def save(
        self,
        output_path: str | Path,
        image: str | Path | Image.Image | np.ndarray | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save detection result to file.

        Format auto-detected from extension (.json, .csv, .png, etc.).
        See VisionResult.save() for full documentation.

        Args:
            output_path: Path to save result
            image: Original image for overlay/crop export
            format: Override format detection
            **kwargs: Additional exporter parameters
        """
        from pathlib import Path

        from mata.core.exporters import export_crops, export_csv, export_image, export_json

        output_path = Path(output_path)

        # Auto-detect format
        if format is None:
            ext = output_path.suffix.lower()
            if ext == ".json":
                format = "json"
            elif ext == ".csv":
                format = "csv"
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                if "crop_dir" in kwargs or output_path.stem.startswith("crops_"):
                    format = "crops"
                else:
                    format = "image"
            else:
                raise ValueError(f"Unsupported file format: '{ext}'")

        # Route to exporter
        if format == "json":
            export_json(self, output_path, **kwargs)
        elif format == "csv":
            export_csv(self, output_path, **kwargs)
        elif format == "image":
            export_image(self, output_path, image=image, **kwargs)
        elif format == "crops":
            export_crops(self, output_path, image=image, **kwargs)
        else:
            raise ValueError(f"Unknown format: '{format}'")


@dataclass(frozen=True)
class SegmentMask:
    """Single segmentation mask (instance or panoptic).

    Supports multiple mask formats:
    - RLE (Run-Length Encoding): Dict with 'size' and 'counts' from pycocotools
    - Binary mask: numpy array (H, W) with boolean values
    - Polygon: List of polygon coordinates (future support)

    For panoptic segmentation, the `is_stuff` field distinguishes between:
    - Instance masks (is_stuff=False): countable objects (person, car, etc.)
    - Stuff masks (is_stuff=True): uncountable regions (sky, road, etc.)

    Attributes:
        mask: Mask representation
            - RLE format: {"size": [height, width], "counts": bytes or str}
            - Binary format: np.ndarray of shape (H, W) with dtype bool
            - Polygon format: List of [x1, y1, x2, y2, ...] coordinates (future)
        score: Confidence score [0.0, 1.0]
        label: Integer class label (0-indexed)
        label_name: Optional human-readable class name
        bbox: Optional bounding box in xyxy format (x1, y1, x2, y2)
        is_stuff: Optional flag for panoptic segmentation
            - False: Instance (countable object)
            - True: Stuff (uncountable region)
            - None: Not specified (instance segmentation mode)
        area: Optional mask area in pixels (computed from RLE or binary mask)

    Examples:
        >>> # RLE mask (from pycocotools)
        >>> from pycocotools import mask as mask_utils
        >>> rle = mask_utils.encode(np.asfortranarray(binary_mask))
        >>> seg_mask = SegmentMask(
        ...     mask=rle,
        ...     score=0.95,
        ...     label=0,
        ...     label_name="person",
        ...     is_stuff=False
        ... )
        >>>
        >>> # Binary mask (numpy array)
        >>> binary_mask = np.array([[0, 0, 1], [0, 1, 1]], dtype=bool)
        >>> seg_mask = SegmentMask(
        ...     mask=binary_mask,
        ...     score=0.87,
        ...     label=15,
        ...     label_name="cat"
        ... )
    """

    mask: dict[str, Any] | np.ndarray | list[float]  # RLE, binary array, or polygon
    score: float
    label: int
    label_name: str | None = None
    bbox: BBox | None = None
    is_stuff: bool | None = None  # For panoptic segmentation
    area: int | None = None  # Mask area in pixels

    def __post_init__(self):
        """Validate mask format and compute area if needed."""
        # Validate mask type
        if NUMPY_AVAILABLE and isinstance(self.mask, np.ndarray):
            if self.mask.ndim != 2:
                raise ValueError(f"Binary mask must be 2D array, got shape {self.mask.shape}")
        elif isinstance(self.mask, dict):
            # Validate RLE format
            if "size" not in self.mask or "counts" not in self.mask:
                raise ValueError(
                    "RLE mask dict must contain 'size' and 'counts' keys. "
                    "Use pycocotools.mask.encode() to create RLE masks."
                )
        elif isinstance(self.mask, list):
            # Polygon format - validate even number of coordinates
            if len(self.mask) % 2 != 0:
                raise ValueError(f"Polygon must have even number of coordinates, got {len(self.mask)}")
        else:
            # Allow other formats for extensibility, but warn
            warnings.warn(
                f"Mask type {type(self.mask)} is not explicitly supported. "
                "Supported types: dict (RLE), np.ndarray (binary), list (polygon)",
                UserWarning,
                stacklevel=2,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Note: Binary masks (numpy arrays) are converted to nested lists for JSON serialization.
        For compact storage, consider converting to RLE format using pycocotools before serialization.

        Returns:
            Dictionary with mask data and metadata
        """
        # Handle mask serialization based on type
        if NUMPY_AVAILABLE and isinstance(self.mask, np.ndarray):
            # Convert numpy array to nested list for JSON compatibility
            mask_data = {
                "format": "binary",
                "data": self.mask.tolist(),
                "shape": list(self.mask.shape),
                "dtype": str(self.mask.dtype),
            }
        elif isinstance(self.mask, dict):
            # RLE format - handle bytes encoding
            mask_data = {
                "format": "rle",
                "data": {
                    "size": self.mask["size"],
                    "counts": (
                        self.mask["counts"].decode("utf-8")
                        if isinstance(self.mask["counts"], bytes)
                        else self.mask["counts"]
                    ),
                },
            }
        elif isinstance(self.mask, list):
            # Polygon format
            mask_data = {"format": "polygon", "data": self.mask}
        else:
            # Fallback - store as-is (may not be JSON serializable)
            mask_data = {"format": "unknown", "data": self.mask}

        return {
            "mask": mask_data,
            "score": self.score,
            "label": self.label,
            "label_name": self.label_name,
            "bbox": list(self.bbox) if self.bbox else None,
            "is_stuff": self.is_stuff,
            "area": self.area,
        }

    def is_rle(self) -> bool:
        """Check if mask is in RLE format."""
        return isinstance(self.mask, dict) and "counts" in self.mask

    def is_binary(self) -> bool:
        """Check if mask is a binary numpy array."""
        return NUMPY_AVAILABLE and isinstance(self.mask, np.ndarray)

    def is_polygon(self) -> bool:
        """Check if mask is in polygon format."""
        return isinstance(self.mask, list) and len(self.mask) > 0

    def to_instance(self) -> Instance:
        """Convert to unified Instance representation.

        Returns:
            Instance object with mask field populated
        """
        return Instance(
            mask=self.mask,
            score=self.score,
            label=self.label,
            label_name=self.label_name,
            bbox=self.bbox,
            area=self.area,
            is_stuff=self.is_stuff,
        )


@dataclass(frozen=True)
class SegmentResult:
    """Segmentation task result (instance or panoptic).

    Supports two segmentation modes:
    - Instance segmentation: Multiple instances of countable objects
    - Panoptic segmentation: Combination of instance and stuff masks

    For panoptic mode, use the `is_stuff` field in each SegmentMask to distinguish
    between instance masks (is_stuff=False) and stuff masks (is_stuff=True).

    Attributes:
        masks: List of segmentation masks (instances and/or stuff regions)
        meta: Optional metadata including:
            - "mode": "instance" or "panoptic"
            - "image_size": [width, height]
            - "model_id": Model identifier
            - "threshold": Confidence threshold used

    Examples:
        >>> # Instance segmentation result
        >>> result = SegmentResult(
        ...     masks=[
        ...         SegmentMask(mask=rle1, score=0.95, label=0, label_name="person"),
        ...         SegmentMask(mask=rle2, score=0.87, label=0, label_name="person")
        ...     ],
        ...     meta={"mode": "instance", "threshold": 0.5}
        ... )
        >>>
        >>> # Panoptic segmentation result
        >>> result = SegmentResult(
        ...     masks=[
        ...         SegmentMask(..., label=0, is_stuff=False),  # Instance: person
        ...         SegmentMask(..., label=10, is_stuff=True),  # Stuff: sky
        ...     ],
        ...     meta={"mode": "panoptic"}
        ... )
    """

    masks: list[SegmentMask]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with masks and metadata
        """
        return {
            "masks": [mask.to_dict() for mask in self.masks],
            "meta": self.meta,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string.

        Args:
            **kwargs: Additional arguments passed to json.dumps()
                     (e.g., indent=2 for pretty printing)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentResult:
        """Create from dictionary representation.

        Args:
            data: Dictionary with masks and optional meta

        Returns:
            SegmentResult instance
        """
        masks = []
        for m in data["masks"]:
            mask_info = m["mask"]

            # Reconstruct mask based on format
            if mask_info["format"] == "rle":
                mask_data = mask_info["data"]
            elif mask_info["format"] == "binary" and NUMPY_AVAILABLE:
                mask_data = np.array(mask_info["data"], dtype=mask_info.get("dtype", "bool"))
            elif mask_info["format"] == "polygon":
                mask_data = mask_info["data"]
            else:
                mask_data = mask_info["data"]

            masks.append(
                SegmentMask(
                    mask=mask_data,
                    score=m["score"],
                    label=m["label"],
                    label_name=m.get("label_name"),
                    bbox=tuple(m["bbox"]) if m.get("bbox") else None,
                    is_stuff=m.get("is_stuff"),
                    area=m.get("area"),
                )
            )

        return cls(masks=masks, meta=data.get("meta", {}))

    @classmethod
    def from_json(cls, json_str: str) -> SegmentResult:
        """Deserialize from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            SegmentResult instance
        """
        return cls.from_dict(json.loads(json_str))

    def save(
        self,
        output_path: str | Path,
        image: str | Path | Image.Image | np.ndarray | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save segmentation result to file.

        Format auto-detected from extension (.json, .csv, .png, etc.).
        See VisionResult.save() for full documentation.

        Args:
            output_path: Path to save result
            image: Original image for mask overlay
            format: Override format detection
            **kwargs: Additional exporter parameters
        """
        from pathlib import Path

        from mata.core.exporters import export_csv, export_image, export_json

        output_path = Path(output_path)

        # Auto-detect format
        if format is None:
            ext = output_path.suffix.lower()
            if ext == ".json":
                format = "json"
            elif ext == ".csv":
                format = "csv"
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                format = "image"
            else:
                raise ValueError(f"Unsupported file format: '{ext}'")

        # Route to exporter
        if format == "json":
            export_json(self, output_path, **kwargs)
        elif format == "csv":
            export_csv(self, output_path, **kwargs)
        elif format == "image":
            export_image(self, output_path, image=image, **kwargs)
        else:
            raise ValueError(f"Unknown format: '{format}'")

    def filter_by_score(self, threshold: float) -> SegmentResult:
        """Filter masks by confidence threshold.

        Args:
            threshold: Minimum confidence score [0.0, 1.0]

        Returns:
            New SegmentResult with filtered masks
        """
        filtered_masks = [m for m in self.masks if m.score >= threshold]
        return SegmentResult(masks=filtered_masks, meta=self.meta)

    def get_instances(self) -> list[SegmentMask]:
        """Get only instance masks (is_stuff=False).

        Returns:
            List of instance masks
        """
        return [m for m in self.masks if m.is_stuff is False or m.is_stuff is None]

    def get_stuff(self) -> list[SegmentMask]:
        """Get only stuff masks (is_stuff=True).

        Returns:
            List of stuff masks
        """
        return [m for m in self.masks if m.is_stuff is True]


@dataclass(frozen=True)
class Classification:
    """Single classification prediction.

    Attributes:
        label: Integer class label
        score: Confidence score [0.0, 1.0]
        label_name: Optional human-readable class name
    """

    label: int
    score: float
    label_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "label": self.label,
            "score": self.score,
            "label_name": self.label_name,
        }


@dataclass(frozen=True)
class ClassifyResult:
    """Image classification task result.

    Attributes:
        predictions: List of classification predictions sorted by score (descending)
        meta: Optional metadata (model info, timing, etc.)
    """

    predictions: list[Classification]
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "meta": self.meta,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string.

        Args:
            **kwargs: Additional arguments passed to json.dumps()

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassifyResult:
        """Create from dictionary representation.

        Args:
            data: Dictionary with predictions and optional meta

        Returns:
            ClassifyResult instance
        """
        predictions = [
            Classification(
                label=p["label"],
                score=p["score"],
                label_name=p.get("label_name"),
            )
            for p in data["predictions"]
        ]
        return cls(predictions=predictions, meta=data.get("meta"))

    @classmethod
    def from_json(cls, json_str: str) -> ClassifyResult:
        """Deserialize from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            ClassifyResult instance
        """
        return cls.from_dict(json.loads(json_str))

    def save(
        self,
        output_path: str | Path,
        image: str | Path | Image.Image | np.ndarray | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save classification result to file.

        Format auto-detected from extension (.json, .csv, .png for bar chart).
        See VisionResult.save() for full documentation.

        Args:
            output_path: Path to save result
            image: Original image (for bar chart visualization)
            format: Override format detection
            **kwargs: Additional exporter parameters (e.g., top_k for charts)
        """
        from pathlib import Path

        from mata.core.exporters import export_csv, export_image, export_json

        output_path = Path(output_path)

        # Auto-detect format
        if format is None:
            ext = output_path.suffix.lower()
            if ext == ".json":
                format = "json"
            elif ext == ".csv":
                format = "csv"
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                format = "image"  # Bar chart
            else:
                raise ValueError(f"Unsupported file format: '{ext}'")

        # Route to exporter
        if format == "json":
            export_json(self, output_path, **kwargs)
        elif format == "csv":
            export_csv(self, output_path, **kwargs)
        elif format == "image":
            export_image(self, output_path, image=image, **kwargs)
        else:
            raise ValueError(f"Unknown format: '{format}'")

    @property
    def top1(self) -> Classification | None:
        """Return the highest-confidence prediction, or None if empty."""
        return self.predictions[0] if self.predictions else None

    @property
    def top5(self) -> list[Classification]:
        """Return top-5 predictions (or fewer if less are available)."""
        return list(self.predictions[:5])

    def get_top1(self) -> Classification | None:
        """Get top-1 (highest confidence) prediction.

        Returns:
            Top prediction or None if no predictions
        """
        return self.predictions[0] if self.predictions else None

    def filter_by_score(self, threshold: float) -> ClassifyResult:
        """Filter predictions by minimum confidence score.

        Args:
            threshold: Minimum score threshold [0.0, 1.0]

        Returns:
            New ClassifyResult with filtered predictions
        """
        filtered = [p for p in self.predictions if p.score >= threshold]
        return ClassifyResult(predictions=filtered, meta=self.meta)


@dataclass(frozen=True)
class Track:
    """Single tracked object.

    Attributes:
        track_id: Unique track identifier
        bbox: Bounding box in xyxy format
        score: Detection confidence score
        label: Integer class label
        age: Number of frames this track has existed
    """

    track_id: int
    bbox: BBox
    score: float
    label: int
    age: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "score": self.score,
            "label": self.label,
            "age": self.age,
        }


@dataclass(frozen=True)
class TrackResult:
    """Object tracking task result.

    Attributes:
        tracks: List of tracked objects
        meta: Optional metadata
    """

    tracks: list[Track]
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tracks": [t.to_dict() for t in self.tracks],
            "meta": self.meta,
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)
