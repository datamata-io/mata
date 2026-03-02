"""Universal model loader with auto-detection capabilities.

This module provides llama.cpp-style model loading where users can load any model
by path, ID, or alias without needing to know plugin names.
"""

import zipfile
from pathlib import Path
from typing import Any

from mata.core.exceptions import ModelNotFoundError, UnsupportedModelError
from mata.core.logging import get_logger
from mata.core.types import ModelType

logger = get_logger(__name__)

_EXTERNAL_OCR_ENGINES = {"easyocr", "paddleocr", "tesseract"}


class UniversalLoader:
    """Universal model loader with automatic source detection.

    Supports loading models from:
    - Config aliases (from ~/.mata/models.yaml or .mata/models.yaml)
    - Local files (.pth, .onnx, .trt, .engine)
    - HuggingFace model IDs (format: "org/model-id")
    - Default models for each task

    Examples:
        >>> loader = UniversalLoader()
        >>> # Load from HuggingFace
        >>> detector = loader.load("detect", "facebook/detr-resnet-50")
        >>> # Load from local file
        >>> detector = loader.load("detect", "model.onnx")
        >>> # Load from config alias
        >>> detector = loader.load("detect", "rtdetr-fast")
        >>> # Load default
        >>> detector = loader.load("detect")
    """

    def __init__(self, model_registry=None):
        """Initialize universal loader.

        Args:
            model_registry: Optional ModelRegistry instance. If None, creates one.
        """
        # Lazy import to avoid circular dependencies
        from mata.core.model_registry import ModelRegistry

        self.registry = model_registry or ModelRegistry()
        self._adapter_cache: dict[str, Any] = {}
        self._probe_cache: dict[str, tuple[str, float]] = {}  # path -> (type, mtime)

    def load(self, task: str, source: str | None = None, model_type: str | ModelType | None = None, **kwargs) -> Any:
        """Load a model adapter using universal auto-detection or explicit type.

        Auto-detection priority (when model_type is None or AUTO):
        1. Check if source is a config alias
        2. Check if source is a local file (exists on disk)
        3. Check if source is a HuggingFace ID (contains '/')
        4. Fall back to default model for task

        Explicit routing (when model_type is specified):
        - Bypasses auto-detection for performance
        - Ensures correct adapter selection for ambiguous files (.pt)
        - Validates source compatibility with specified type

        Args:
            task: Task type ("detect", "segment", "classify", "track")
            source: Model source (alias, path, HF ID, or None for default)
            model_type: Optional explicit model type (ModelType enum or string)
                - ModelType.AUTO or None: Use auto-detection (default)
                - ModelType.HUGGINGFACE: Load from HuggingFace Hub
                - ModelType.PYTORCH_CHECKPOINT: PyTorch state dict (.pth/.pt)
                - ModelType.TORCHSCRIPT: TorchScript model (.pt)
                - ModelType.ONNX: ONNX model (.onnx)
                - ModelType.TENSORRT: TensorRT engine (.trt/.engine)
                - ModelType.CONFIG_ALIAS: Config file alias
                - String values accepted but deprecated (use enum)
            **kwargs: Additional arguments passed to adapter constructor
                - threshold: Detection confidence threshold
                - device: Device to run on ("cpu", "cuda", "auto")
                - config: Optional config file path for PyTorch checkpoints
                - input_size: Input size for TorchScript models

        Returns:
            Model adapter instance implementing the task protocol

        Raises:
            ModelNotFoundError: If source cannot be found or resolved
            UnsupportedModelError: If model format is not supported

        Examples:
            >>> from mata.core.types import ModelType
            >>> loader = UniversalLoader()
            >>>
            >>> # Auto-detection (default)
            >>> detector = loader.load("detect", "model.onnx")
            >>>
            >>> # Explicit TorchScript (avoids .pt ambiguity)
            >>> detector = loader.load("detect", "model.pt",
            ...                        model_type=ModelType.TORCHSCRIPT,
            ...                        input_size=640)
            >>>
            >>> # Explicit PyTorch checkpoint
            >>> detector = loader.load("detect", "checkpoint.pt",
            ...                        model_type=ModelType.PYTORCH_CHECKPOINT,
            ...                        config="config.yaml")
            >>>
            >>> # String format (deprecated but supported)
            >>> detector = loader.load("detect", "model.onnx",
            ...                        model_type="onnx")  # Warns
        """
        task_str = task

        # Normalize model_type (handles string deprecation and validation)
        normalized_type = ModelType.normalize(model_type) if model_type is not None else None

        # Validate kwargs for the specified model type (lint-level warnings)
        if normalized_type and normalized_type != ModelType.AUTO:
            self._validate_adapter_kwargs(normalized_type, **kwargs)

        # If explicit model type provided (and not AUTO), route directly
        if normalized_type and normalized_type != ModelType.AUTO:
            logger.info(f"Loading {task_str} model with explicit type: {normalized_type.value}")
            result = self._load_with_explicit_type(task_str, source, normalized_type, **kwargs)
            return result

        # Special case: pipeline task supports detector_model_id kwarg in place of positional source
        if task_str == "pipeline" and source is None and "detector_model_id" in kwargs:
            source = kwargs.pop("detector_model_id")

        # Auto-detection path
        # Detect source type and route to appropriate loader
        source_type, resolved_source = self._detect_source_type(task_str, source)

        logger.info(f"Loading {task_str} model from {source_type}: {resolved_source}")

        # Route to appropriate adapter based on source type
        if source_type == "config_alias":
            result = self._load_from_config(task_str, resolved_source, **kwargs)
        elif source_type == "torchvision":
            result = self._load_from_torchvision(task_str, resolved_source, **kwargs)
        elif source_type == "huggingface":
            result = self._load_from_huggingface(task_str, resolved_source, **kwargs)
        elif source_type == "local_file":
            result = self._load_from_file(task_str, resolved_source, **kwargs)
        elif source_type == "default":
            result = self._load_default(task_str, **kwargs)
        elif source_type == "legacy_plugin":
            result = self._load_legacy_plugin(task_str, resolved_source, **kwargs)
        elif source_type == "external_engine":
            result = self._load_from_external_engine(task_str, resolved_source, **kwargs)
        else:
            raise UnsupportedModelError(
                f"Unknown source type: {source_type}. "
                f"Supported: config alias, local file, torchvision, HuggingFace ID, or legacy plugin name."
            )

        return result

    def _validate_adapter_kwargs(self, model_type: ModelType, **kwargs) -> None:
        """Validate kwargs for specified model type (lint-level warnings).

        Provides helpful warnings for potentially incorrect kwargs combinations
        without blocking model loading. This is intentionally non-blocking to
        allow flexibility while guiding users toward best practices.

        Args:
            model_type: The explicit model type being loaded
            **kwargs: User-provided kwargs to validate
        """
        # Define valid kwargs for each adapter type
        ADAPTER_KWARGS = {  # noqa: N806
            ModelType.HUGGINGFACE: {"model_id", "threshold", "device", "id2label"},
            ModelType.PYTORCH_CHECKPOINT: {"checkpoint_path", "config", "device", "threshold", "id2label"},
            ModelType.TORCHSCRIPT: {"model_path", "device", "threshold", "input_size", "id2label"},
            ModelType.ONNX: {"model_path", "device", "threshold", "id2label"},
            ModelType.TENSORRT: {"engine_path", "device", "threshold"},
        }

        if model_type not in ADAPTER_KWARGS:
            return  # No validation for config_alias, legacy_plugin, etc.

        valid_kwargs = ADAPTER_KWARGS[model_type]
        provided_kwargs = set(kwargs.keys())

        # Check for unexpected kwargs
        unexpected = provided_kwargs - valid_kwargs
        if unexpected:
            logger.warning(
                f"Unexpected kwargs for {model_type.value}: {unexpected}. "
                f"Valid kwargs: {sorted(valid_kwargs)}. "
                f"These will be ignored by the adapter."
            )

        # Check for common mistakes
        if model_type == ModelType.PYTORCH_CHECKPOINT and "input_size" in provided_kwargs:
            logger.warning(
                "'input_size' is for TorchScript models, not PyTorch checkpoints. "
                "Use model_type=ModelType.TORCHSCRIPT if this is a TorchScript model."
            )

        if model_type == ModelType.TORCHSCRIPT:
            if "config" in provided_kwargs:
                logger.warning(
                    "'config' is for PyTorch checkpoints, not TorchScript models. "
                    "TorchScript models are self-contained."
                )
            if "input_size" not in provided_kwargs:
                logger.warning(
                    "TorchScript models typically require 'input_size' parameter. "
                    "Consider adding: input_size=640 (or your model's input size)"
                )

    def _load_with_explicit_type(self, task: str, source: str | None, model_type: ModelType, **kwargs) -> Any:
        """Load model with explicitly specified type (bypass auto-detection).

        Args:
            task: Task type
            source: Model source (path, ID, alias, or None)
            model_type: Explicit model type enum
            **kwargs: Adapter-specific kwargs

        Returns:
            Model adapter instance

        Raises:
            ModelNotFoundError: If source is invalid for the specified type
            UnsupportedModelError: If type/task combination not supported
        """
        if task == "track":
            tracker_config, frame_rate = self._resolve_tracker_kwargs(kwargs)
            detect_adapter = self._load_with_explicit_type("detect", source, model_type, **kwargs)
            return self._wrap_with_tracking(detect_adapter, tracker_config, frame_rate)

        if model_type == ModelType.HUGGINGFACE:
            if not source:
                raise ModelNotFoundError("HuggingFace model ID required when model_type=HUGGINGFACE")
            return self._load_from_huggingface(task, source, **kwargs)

        elif model_type == ModelType.PYTORCH_CHECKPOINT:
            if not source or not self._is_local_file(source):
                raise ModelNotFoundError(
                    f"Valid PyTorch checkpoint file required when model_type=PYTORCH_CHECKPOINT. " f"Got: {source}"
                )
            # Directly load as PyTorch checkpoint (skip probe)
            if task == "detect":
                from mata.adapters.pytorch_adapter import PyTorchDetectAdapter

                return PyTorchDetectAdapter(checkpoint_path=source, **kwargs)
            elif task == "classify":
                from mata.adapters.pytorch_classify_adapter import PyTorchClassifyAdapter

                return PyTorchClassifyAdapter(checkpoint_path=source, **kwargs)
            else:
                raise UnsupportedModelError(f"PyTorch adapter not yet implemented for task '{task}'")

        elif model_type == ModelType.TORCHSCRIPT:
            if not source or not self._is_local_file(source):
                raise ModelNotFoundError(
                    f"Valid TorchScript file required when model_type=TORCHSCRIPT. " f"Got: {source}"
                )
            # Directly load as TorchScript (skip probe)
            if task == "detect":
                from mata.adapters.torchscript_adapter import TorchScriptDetectAdapter

                return TorchScriptDetectAdapter(model_path=source, **kwargs)
            elif task == "classify":
                from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

                return TorchScriptClassifyAdapter(model_path=source, **kwargs)
            else:
                raise UnsupportedModelError(f"TorchScript adapter not yet implemented for task '{task}'")

        elif model_type == ModelType.ONNX:
            if not source or not self._is_local_file(source):
                raise ModelNotFoundError(f"Valid ONNX file required when model_type=ONNX. " f"Got: {source}")
            if task == "detect":
                from mata.adapters.onnx_adapter import ONNXDetectAdapter

                return ONNXDetectAdapter(model_path=source, **kwargs)
            elif task == "classify":
                from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

                return ONNXClassifyAdapter(model_path=source, **kwargs)
            else:
                raise UnsupportedModelError(f"ONNX adapter not yet implemented for task '{task}'")

        elif model_type == ModelType.TENSORRT:
            if not source or not self._is_local_file(source):
                raise ModelNotFoundError(f"Valid TensorRT engine required when model_type=TENSORRT. " f"Got: {source}")
            from mata.adapters.tensorrt_adapter import TensorRTDetectAdapter

            if task == "detect":
                return TensorRTDetectAdapter(engine_path=source, **kwargs)
            else:
                raise UnsupportedModelError(f"TensorRT adapter not yet implemented for task '{task}'")

        elif model_type == ModelType.CONFIG_ALIAS:
            if not source:
                raise ModelNotFoundError("Config alias name required when model_type=CONFIG_ALIAS")
            return self._load_from_config(task, source, **kwargs)

        elif model_type == ModelType.EASYOCR:
            from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

            return EasyOCRAdapter(**kwargs)

        elif model_type == ModelType.PADDLEOCR:
            from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

            return PaddleOCRAdapter(**kwargs)

        elif model_type == ModelType.TESSERACT:
            from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

            return TesseractAdapter(**kwargs)

        else:
            raise UnsupportedModelError(f"Unknown model type: {model_type}")

    def _detect_source_type(self, task: str, source: str | None) -> tuple[str, str]:
        """Detect the type of model source and resolve it.

        Args:
            task: Task type string
            source: Model source string or None

        Returns:
            Tuple of (source_type, resolved_source)

        Detection logic:
        1. None → "default"
        2. Config alias (exists in registry) → "config_alias"
        3. Local file (exists on disk) → "local_file"
        4. Torchvision model (starts with "torchvision/") → "torchvision"
        5. Contains '/' → "huggingface"
        6. Otherwise → "config_alias" (will fail in _load_from_config)
        """
        if source is None:
            return "default", ""

        # Check if it's a config alias
        if self.registry.has_alias(task, source):
            return "config_alias", source

        # Check if it's a local file
        if self._is_local_file(source):
            return "local_file", source

        # Check if it looks like a file path (has extension) even if file doesn't exist yet
        # This handles relative paths that might be valid from different working directories
        path = Path(source)
        if path.suffix.lower() in [".pt", ".pth", ".onnx", ".bin", ".trt", ".engine"]:
            logger.debug(f"Treating '{source}' as local file path based on extension")
            return "local_file", source

        # Check for torchvision models
        if source.startswith("torchvision/"):
            return "torchvision", source

        # Check for external OCR engine names (bare strings, no '/')
        # Must be checked before HuggingFace slash-check to avoid intercepting HF model IDs
        if source.lower() in _EXTERNAL_OCR_ENGINES:
            return "external_engine", source.lower()

        # Check if it's a HuggingFace ID (contains '/')
        if "/" in source:
            return "huggingface", source

        # Assume it's a config alias (will fail with helpful error if not found)
        return "config_alias", source

    def _is_local_file(self, source: str) -> bool:
        """Check if source is a local file path.

        Args:
            source: Source string

        Returns:
            True if source is an existing file
        """
        try:
            path = Path(source)
            return path.exists() and path.is_file()
        except (OSError, ValueError):
            return False

    def clear_cache(self) -> None:
        """Clear probe cache and adapter cache.

        Useful for:
        - Testing scenarios where files change
        - Forcing re-detection of model types
        - Memory cleanup in long-running processes

        Example:
            >>> loader = UniversalLoader()
            >>> loader.clear_cache()
        """
        self._probe_cache.clear()
        self._adapter_cache.clear()
        logger.debug("Cleared model loader caches")

    def _probe_pt_file(self, file_path: str) -> str:
        """Probe .pt file to determine if it's TorchScript or PyTorch checkpoint.

        Uses lightweight file structure detection instead of full model loading:
        - TorchScript: ZIP archive with specific structure (constants.pkl, code/)
        - PyTorch checkpoint: Pickle format (state dict)

        Results are cached with file modification time for performance.

        Args:
            file_path: Path to .pt file

        Returns:
            "torchscript" or "pytorch_checkpoint"

        Performance:
            - Probe: <5ms (ZIP header check)
            - Full load (old method): 50-200ms
            - Cache hit: <1ms
        """
        path = Path(file_path)

        # Check cache first (with mtime validation)
        if file_path in self._probe_cache:
            cached_type, cached_mtime = self._probe_cache[file_path]
            try:
                current_mtime = path.stat().st_mtime
                if abs(current_mtime - cached_mtime) < 0.001:  # Same file
                    logger.debug(f"Probe cache hit: {file_path} -> {cached_type}")
                    return cached_type
            except OSError:
                # File disappeared, clear cache entry
                del self._probe_cache[file_path]

        # Perform lightweight probe
        detected_type = "pytorch_checkpoint"  # Default assumption

        try:
            # TorchScript models are ZIP archives with specific structure
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path, "r") as z:
                    names = z.namelist()
                    # TorchScript markers: constants.pkl, code/ directory, or version file
                    has_constants = "constants.pkl" in names
                    has_code_dir = any("code/" in n for n in names)
                    has_version = "version" in names

                    if has_constants or has_code_dir or has_version:
                        detected_type = "torchscript"
                        logger.debug(f"Detected TorchScript file: {file_path} (ZIP structure)")
                    else:
                        logger.debug(f"ZIP file but not TorchScript: {file_path}")
            else:
                logger.debug(f"Not a ZIP file, assuming PyTorch checkpoint: {file_path}")

        except Exception as e:
            # If probe fails, fall back to checkpoint (safer default)
            logger.debug(f"Probe failed for {file_path}: {e}. Assuming PyTorch checkpoint.")

        # Cache result with current mtime
        try:
            mtime = path.stat().st_mtime
            self._probe_cache[file_path] = (detected_type, mtime)
        except OSError:
            pass  # Don't cache if we can't get mtime

        return detected_type

    def _load_from_config(self, task: str, alias: str, **kwargs) -> Any:
        """Load model from config alias.

        Args:
            task: Task type
            alias: Config alias name
            **kwargs: Override parameters

        Returns:
            Model adapter instance
        """
        config = self.registry.get_config(task, alias)

        # Merge config with kwargs (kwargs take precedence)
        merged_kwargs = {**config, **kwargs}

        # Extract source from config
        source = merged_kwargs.pop("source", None)
        if not source:
            raise ModelNotFoundError(f"Config alias '{alias}' for task '{task}' has no 'source' field")

        # Recursively load from the actual source
        return self.load(task, source, **merged_kwargs)

    def _load_from_huggingface(self, task: str, model_id: str, **kwargs) -> Any:
        """Load model from HuggingFace Hub.

        Args:
            task: Task type ("detect", "segment", "pipeline", etc.)
            model_id: HuggingFace model ID
                Examples:
                - detect: "facebook/detr-resnet-50"
                - detect (zero-shot): "IDEA-Research/grounding-dino-tiny"
                - segment: "facebook/mask2former-swin-tiny-coco-instance"
                - segment (SAM): "facebook/sam-vit-base"
                - pipeline: "grounding_sam" (detector + SAM)
            **kwargs: Additional arguments including:
                - detect_mode: "supervised" (default) or "zeroshot"
                - pipeline: "grounding_sam" for multi-modal chaining
                - text_prompts: Zero-shot prompts (prediction-time only)

        Returns:
            HuggingFace adapter or pipeline instance for the task
        """
        if task == "track":
            tracker_config, frame_rate = self._resolve_tracker_kwargs(kwargs)
            detect_adapter = self._load_from_huggingface("detect", model_id, **kwargs)
            return self._wrap_with_tracking(detect_adapter, tracker_config, frame_rate)

        if task == "detect":
            # Check for zero-shot detection mode
            detect_mode = kwargs.get("detect_mode", "auto")
            is_zeroshot_mode = detect_mode == "zeroshot"

            # Check model ID for zero-shot indicators
            model_id_lower = model_id.lower()
            is_zeroshot_id = any(
                x in model_id_lower for x in ["grounding", "groundingdino", "owlvit", "owl-vit", "owlv2"]
            )

            if is_zeroshot_mode or is_zeroshot_id:
                # Use zero-shot detection adapter
                # Filter prediction-time parameters (text_prompts)
                prediction_params = {"detect_mode", "text_prompts"}
                zs_kwargs = {k: v for k, v in kwargs.items() if k not in prediction_params}
                from mata.adapters.huggingface_zeroshot_detect_adapter import HuggingFaceZeroShotDetectAdapter

                return HuggingFaceZeroShotDetectAdapter(model_id=model_id, **zs_kwargs)
            else:
                # Use standard supervised detection adapter
                from mata.adapters.huggingface_adapter import HuggingFaceDetectAdapter

                return HuggingFaceDetectAdapter(model_id=model_id, **kwargs)

        elif task == "pipeline":
            # Multi-modal pipeline (e.g., GroundingDINO→SAM)
            pipeline_type = kwargs.get("pipeline", "grounding_sam")

            if pipeline_type == "grounding_sam":
                # Extract pipeline-specific kwargs
                # Accept "detector" or "detector_model_id" as the detector source
                detector_model = kwargs.get("detector") or kwargs.get("detector_model_id") or model_id
                sam_model = kwargs.get("sam_model_id", "facebook/sam-vit-base")

                # Filter prediction-time parameters
                prediction_params = {"pipeline", "text_prompts", "detector", "detector_model_id", "sam_model_id"}
                pipeline_kwargs = {k: v for k, v in kwargs.items() if k not in prediction_params}

                from mata.adapters.pipeline_adapter import GroundingDINOSAMPipeline

                return GroundingDINOSAMPipeline(
                    detector_model_id=detector_model, sam_model_id=sam_model, **pipeline_kwargs
                )
            else:
                raise UnsupportedModelError(
                    f"Pipeline type '{pipeline_type}' not supported. " f"Currently supported: grounding_sam"
                )

        elif task == "segment":
            # Check model ID patterns to route to the right adapter
            model_id_lower = model_id.lower()
            segment_mode = kwargs.get("segment_mode", "auto")

            # 1. Zero-shot semantic segmentation (CLIPSeg, etc.)
            #    Lightweight text-to-mask models (~150M params)
            is_zeroshot_segment = any(x in model_id_lower for x in ["clipseg", "clip-seg"])
            if is_zeroshot_segment:
                prediction_params = {"segment_mode", "text_prompts"}
                zs_kwargs = {k: v for k, v in kwargs.items() if k not in prediction_params}
                from mata.adapters.huggingface_zeroshot_segment_adapter import (
                    HuggingFaceZeroShotSegmentAdapter,
                )

                return HuggingFaceZeroShotSegmentAdapter(model_id=model_id, **zs_kwargs)

            # 2. SAM prompt-based segmentation (SAM, SAM3)
            #    Heavy prompt-based mask generators (375M–641M params)
            is_sam_mode = segment_mode == "zeroshot"
            is_sam_id = "sam" in model_id_lower

            if is_sam_mode or is_sam_id:
                # Use SAM adapter for prompt-based segmentation
                # Filter out segment_mode and prediction-time parameters from kwargs
                # Prediction parameters (text_prompts, point_prompts, box_prompts, box_labels)
                # should only be passed to predict(), not __init__()
                prediction_params = {"segment_mode", "text_prompts", "point_prompts", "box_prompts", "box_labels"}
                sam_kwargs = {k: v for k, v in kwargs.items() if k not in prediction_params}
                from mata.adapters.huggingface_sam_adapter import HuggingFaceSAMAdapter

                return HuggingFaceSAMAdapter(model_id=model_id, **sam_kwargs)
            else:
                # 3. Standard segmentation adapter (instance/panoptic/semantic)
                from mata.adapters.huggingface_segment_adapter import HuggingFaceSegmentAdapter

                return HuggingFaceSegmentAdapter(model_id=model_id, **kwargs)

        elif task == "classify":
            from mata.adapters.huggingface_classify_adapter import HuggingFaceClassifyAdapter

            return HuggingFaceClassifyAdapter(model_id=model_id, **kwargs)

        elif task == "depth":
            prediction_params = {"target_size"}
            depth_kwargs = {k: v for k, v in kwargs.items() if k not in prediction_params}
            from mata.adapters.huggingface_depth_adapter import HuggingFaceDepthAdapter

            return HuggingFaceDepthAdapter(model_id=model_id, **depth_kwargs)

        elif task == "vlm":
            from mata.adapters.huggingface_vlm_adapter import HuggingFaceVLMAdapter
            from mata.adapters.wrappers.vlm_wrapper import VLMWrapper

            prediction_params = {
                "prompt",
                "system_prompt",
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "output_mode",
                "images",
            }
            vlm_kwargs = {k: v for k, v in kwargs.items() if k not in prediction_params}
            adapter = HuggingFaceVLMAdapter(model_id=model_id, **vlm_kwargs)
            return VLMWrapper(adapter)

        elif task == "ocr":
            from mata.adapters.ocr.huggingface_ocr_adapter import HuggingFaceOCRAdapter

            return HuggingFaceOCRAdapter(model_id=model_id, **kwargs)

        else:
            raise UnsupportedModelError(
                f"HuggingFace adapter not yet implemented for task '{task}'. "
                f"Currently supported: detect, segment, classify, depth, vlm, track, ocr"
            )

    def _load_from_external_engine(self, task: str, engine_name: str, **kwargs) -> Any:
        """Load an external OCR engine adapter by name.

        Args:
            task: Task type (must be "ocr")
            engine_name: Engine name, one of "easyocr", "paddleocr", "tesseract"
            **kwargs: Additional arguments forwarded to the adapter constructor

        Returns:
            Adapter instance for the requested engine

        Raises:
            UnsupportedModelError: If task is not "ocr" or engine_name is unknown
        """
        if task != "ocr":
            raise UnsupportedModelError(
                f"External engine '{engine_name}' is only supported for task='ocr', got task='{task}'"
            )
        if engine_name == "easyocr":
            from mata.adapters.ocr.easyocr_adapter import EasyOCRAdapter

            return EasyOCRAdapter(**kwargs)
        elif engine_name == "paddleocr":
            from mata.adapters.ocr.paddleocr_adapter import PaddleOCRAdapter

            return PaddleOCRAdapter(**kwargs)
        elif engine_name == "tesseract":
            from mata.adapters.ocr.tesseract_adapter import TesseractAdapter

            return TesseractAdapter(**kwargs)
        else:
            raise UnsupportedModelError(f"Unknown external OCR engine: '{engine_name}'")

    def _load_from_torchvision(self, task: str, model_name: str, **kwargs) -> Any:
        """Load torchvision detection model.

        Args:
            task: Task type (currently only "detect" supported)
            model_name: Full model name (e.g., "torchvision/retinanet_resnet50_fpn")
            **kwargs: Additional arguments passed to adapter

        Returns:
            TorchvisionDetectAdapter instance

        Raises:
            UnsupportedModelError: If task is not "detect"
        """
        if task == "detect":
            from mata.adapters.torchvision_detect_adapter import TorchvisionDetectAdapter

            logger.info(f"Loading torchvision detection model: {model_name}")
            return TorchvisionDetectAdapter(model_name=model_name, **kwargs)
        else:
            raise UnsupportedModelError(
                f"Torchvision adapter not yet implemented for task '{task}'. " f"Supported tasks: detect"
            )

    def _load_from_file(self, task: str, file_path: str, **kwargs) -> Any:
        """Load model from local file based on extension.

        Uses two-stage loading for .pt files:
        1. Probe stage: Lightweight file structure detection (<5ms)
        2. Load stage: Single load with correct adapter (no retry)

        Args:
            task: Task type
            file_path: Path to model file
            **kwargs: Additional arguments (e.g., config file path)

        Returns:
            Appropriate adapter instance based on file extension
        """
        if task == "track":
            tracker_config, frame_rate = self._resolve_tracker_kwargs(kwargs)
            detect_adapter = self._load_from_file("detect", file_path, **kwargs)
            return self._wrap_with_tracking(detect_adapter, tracker_config, frame_rate)

        path = Path(file_path)
        extension = path.suffix.lower()

        if extension in [".onnx"]:
            if task == "detect":
                from mata.adapters.onnx_adapter import ONNXDetectAdapter

                return ONNXDetectAdapter(model_path=file_path, **kwargs)
            elif task == "classify":
                from mata.adapters.onnx_classify_adapter import ONNXClassifyAdapter

                return ONNXClassifyAdapter(model_path=file_path, **kwargs)
            elif task == "ocr":
                raise UnsupportedModelError(
                    "ONNX OCR models are not yet supported. "
                    "Use a HuggingFace model ID (e.g., 'microsoft/trocr-base-handwritten') "
                    "or an external engine ('easyocr', 'paddleocr', 'tesseract')."
                )
            else:
                raise UnsupportedModelError(f"ONNX adapter not yet implemented for task '{task}'")

        elif extension in [".pth", ".pt", ".bin"]:
            # Use two-stage probe for .pt files to avoid ambiguity
            if extension == ".pt":
                # Probe stage: Fast detection without loading model
                detected_type = self._probe_pt_file(file_path)

                if detected_type == "torchscript":
                    logger.info(f"Detected TorchScript model via probe: {file_path}")
                    if task == "detect":
                        from mata.adapters.torchscript_adapter import TorchScriptDetectAdapter

                        return TorchScriptDetectAdapter(model_path=file_path, **kwargs)
                    elif task == "classify":
                        from mata.adapters.torchscript_classify_adapter import TorchScriptClassifyAdapter

                        return TorchScriptClassifyAdapter(model_path=file_path, **kwargs)
                    else:
                        raise UnsupportedModelError(f"TorchScript adapter not yet implemented for task '{task}'")
                else:
                    # Load as PyTorch checkpoint
                    logger.debug(f"Detected PyTorch checkpoint via probe: {file_path}")
                    # Filter out TorchScript-specific parameters
                    checkpoint_kwargs = {k: v for k, v in kwargs.items() if k not in ["input_size", "id2label"]}
                    if task == "detect":
                        from mata.adapters.pytorch_adapter import PyTorchDetectAdapter

                        return PyTorchDetectAdapter(checkpoint_path=file_path, **checkpoint_kwargs)
                    elif task == "classify":
                        from mata.adapters.pytorch_classify_adapter import PyTorchClassifyAdapter

                        return PyTorchClassifyAdapter(checkpoint_path=file_path, **kwargs)
                    else:
                        raise UnsupportedModelError(f"PyTorch adapter not yet implemented for task '{task}'")
            else:
                # .pth and .bin files are always PyTorch checkpoints
                if task == "detect":
                    from mata.adapters.pytorch_adapter import PyTorchDetectAdapter

                    return PyTorchDetectAdapter(checkpoint_path=file_path, **kwargs)
                elif task == "classify":
                    from mata.adapters.pytorch_classify_adapter import PyTorchClassifyAdapter

                    return PyTorchClassifyAdapter(checkpoint_path=file_path, **kwargs)
                else:
                    raise UnsupportedModelError(f"PyTorch adapter not yet implemented for task '{task}'")

        elif extension in [".trt", ".engine"]:
            from mata.adapters.tensorrt_adapter import TensorRTDetectAdapter

            if task == "detect":
                return TensorRTDetectAdapter(engine_path=file_path, **kwargs)
            else:
                raise UnsupportedModelError(f"TensorRT adapter not yet implemented for task '{task}'")
        else:
            raise UnsupportedModelError(
                f"Unsupported file extension: {extension}. " f"Supported: .onnx, .pth, .pt, .bin, .trt, .engine"
            )

    def _resolve_tracker_kwargs(self, kwargs: dict) -> tuple:
        """Pop tracker-related kwargs and return ``(tracker_config, frame_rate)``.

        Handles the case where the registry config supplies both a ``tracker``
        name (string) **and** a ``tracker_config`` override dict.  The two are
        merged into a single dict so that :class:`TrackerConfig.from_dict`
        picks up the full set of settings:

        .. code-block:: yaml

            track:
              highway-tracker:
                source: facebook/detr-resnet-50
                tracker: botsort
                tracker_config:
                  track_buffer: 60
                  match_thresh: 0.7

        When *tracker* is a plain string and *tracker_config* is a dict the
        result is ``{"tracker_type": tracker, **tracker_config}``.  In all
        other cases *tracker* is returned unchanged (it may already be a
        dict, a YAML path, or a :class:`TrackerConfig` instance).

        Args:
            kwargs: Mutable kwargs dict — ``tracker``, ``tracker_config``, and
                ``frame_rate`` are **popped** in place.

        Returns:
            ``(resolved_tracker_config, frame_rate)`` tuple.
        """
        tracker = kwargs.pop("tracker", "botsort")
        tracker_config_overrides = kwargs.pop("tracker_config", None)
        frame_rate = kwargs.pop("frame_rate", 30)

        if tracker_config_overrides is not None and isinstance(tracker_config_overrides, dict):
            if isinstance(tracker, str):
                # Merge the tracker name (as tracker_type) with the override dict.
                # TrackerConfig.from_dict() uses dataclass defaults for any
                # fields not present, so the user only needs to specify overrides.
                resolved: Any = {"tracker_type": tracker, **tracker_config_overrides}
            else:
                # tracker is already a dict / TrackerConfig — overrides ignored.
                resolved = tracker
        else:
            resolved = tracker

        return resolved, frame_rate

    def _wrap_with_tracking(self, detect_adapter: Any, tracker_config: Any, frame_rate: int) -> Any:
        """Wrap a detection adapter with a :class:`TrackingAdapter`.

        This is a shared helper called by all source-type loaders when
        ``task == "track"``.

        Args:
            detect_adapter: A detection adapter (any adapter with ``predict()``).
            tracker_config: Tracker selection — built-in name ``"bytetrack"`` /
                ``"botsort"``, a path to a YAML file, a :class:`TrackerConfig`
                instance, a plain dict of tracker parameters, or ``None`` for
                the BotSort default.
            frame_rate: Video frame rate used to derive ``max_time_lost``.

        Returns:
            :class:`TrackingAdapter` composing *detect_adapter* with the
            configured tracker.
        """
        from mata.adapters.tracking_adapter import TrackingAdapter

        return TrackingAdapter(detect_adapter, tracker_config, frame_rate)

    def _load_default(self, task: str, **kwargs) -> Any:
        """Load default model for task.

        Args:
            task: Task type
            **kwargs: Additional arguments

        Returns:
            Default model adapter for task
        """
        # Check if user has configured a default in config file
        default_alias = self.registry.get_default(task)
        if default_alias:
            logger.info(f"Loading user-configured default '{default_alias}' for task '{task}'")
            return self._load_from_config(task, default_alias, **kwargs)

        # No default configured - raise helpful error
        if task == "track":
            example = "mata.load('track', 'facebook/detr-resnet-50')"
        else:
            example = f"mata.load('{task}', 'PekingU/rtdetr_v2_r18vd')"
        raise ModelNotFoundError(
            f"No default model configured for task '{task}'. "
            f"Please specify a model explicitly or configure a default in .mata/models.yaml. "
            f"Example: {example}"
        )
