"""MATA public API - YOLO-like UX for model-agnostic tasks.

This module provides the main entry points for using MATA:
- load(): Load a task adapter
- run(): One-shot inference on an input
- infer(): Execute a multi-task graph on an image
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .core.exceptions import TaskNotSupportedError
from .core.model_loader import UniversalLoader
from .core.types import ClassifyResult, DepthResult, DetectResult, ModelType, SegmentResult, VisionResult

if TYPE_CHECKING:
    from .core.artifacts.result import MultiResult
    from .core.graph.graph import Graph
    from .core.graph.node import Node

# Singleton universal loader instance
_universal_loader: UniversalLoader | None = None


def _get_universal_loader() -> UniversalLoader:
    """Get or create the singleton UniversalLoader instance."""
    global _universal_loader
    if _universal_loader is None:
        _universal_loader = UniversalLoader()
    return _universal_loader


def load(task: str, model: str | None = None, model_type: str | ModelType | None = None, **kwargs: Any) -> Any:
    """Load a task adapter using universal model loading.

    This is the primary way to instantiate adapters in MATA.
    Supports multiple model sources:
    - HuggingFace model IDs (e.g., "facebook/detr-resnet-50")
    - Local model files (e.g., "model.onnx", "checkpoint.pth")
    - Config aliases (e.g., "rtdetr-fast" from ~/.mata/models.yaml)
    - Legacy plugin names (e.g., "rtdetr", "dino") - deprecated
    - Default model if None specified

    Args:
        task: Task type ("detect", "segment", "classify", "depth", "track")
        model: Model source (HF ID, file path, alias, or legacy plugin name)
            If None, uses default model for task
        model_type: Optional explicit model type specification (v1.5.2+)
            - None or ModelType.AUTO: Auto-detect (default)
            - ModelType.TORCHSCRIPT: TorchScript model (.pt)
            - ModelType.PYTORCH_CHECKPOINT: PyTorch state dict (.pth/.pt)
            - ModelType.ONNX: ONNX model (.onnx)
            - ModelType.HUGGINGFACE: HuggingFace Hub model
            - ModelType.TENSORRT: TensorRT engine (.trt/.engine)
            - String values deprecated (use enum from mata.core.types)
        **kwargs: Arguments passed to adapter constructor
                (threshold, device, config, input_size, etc.)

    Returns:
        Adapter instance implementing the task protocol

    Raises:
        TaskNotSupportedError: If task is not supported
        ModelNotFoundError: If specified model is not found
        UnsupportedModelError: If model format is not supported

    Examples:
        >>> # Load from HuggingFace
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")

        >>> # Load from local ONNX file
        >>> detector = mata.load("detect", "model.onnx", threshold=0.4)

        >>> # Load from PyTorch checkpoint
        >>> detector = mata.load("detect", "checkpoint.pth", config="config.yaml")

        >>> # Load from config alias
        >>> detector = mata.load("detect", "rtdetr-fast")

        >>> # Load default model
        >>> detector = mata.load("detect")

        >>> # Legacy plugin name (deprecated, will warn)
        >>> detector = mata.load("detect", "rtdetr", model_id="PekingU/rtdetr_v2_r50vd")

        >>> # Explicit type specification (v1.5.2+)
        >>> from mata.core.types import ModelType
        >>>
        >>> # TorchScript with explicit type (avoids .pt ambiguity)
        >>> detector = mata.load("detect", "model.pt",
        ...                      model_type=ModelType.TORCHSCRIPT,
        ...                      input_size=640)
        >>>
        >>> # PyTorch checkpoint with explicit type
        >>> detector = mata.load("detect", "checkpoint.pt",
        ...                      model_type=ModelType.PYTORCH_CHECKPOINT,
        ...                      config="config.yaml")
    """
    # Use universal loader for all model loading
    loader = _get_universal_loader()
    return loader.load(task=task, source=model, model_type=model_type, **kwargs)


def run(
    task: str,
    input: str | Path | Image.Image | np.ndarray,
    model: str | None = None,
    model_type: str | ModelType | None = None,
    **kwargs: Any,
) -> DetectResult | SegmentResult | ClassifyResult | DepthResult | VisionResult:
    """One-shot inference on an input.

    Provides YOLO-like UX for quick inference without manually
    creating adapters. For repeated inference, use load() instead.

    Args:
        task: Task type ("detect", "segment", "classify", "depth", "vlm")
        input: Input image (path, PIL Image, or numpy array)
        model: Optional model source (path, HF ID, or alias)
        model_type: Optional explicit model type (see load() for details)
        **kwargs: Additional arguments for adapter creation and inference.
            For "vlm" task:
                - prompt (str, required): Text prompt for vision-language model
                - system_prompt (str, optional): System prompt to guide model behavior
                - max_new_tokens (int, optional): Maximum tokens to generate (default: 512)
                - temperature (float, optional): Sampling temperature (default: 0.7)
                - top_p (float, optional): Nucleus sampling threshold (default: 0.8)
                - top_k (int, optional): Top-k sampling parameter (default: 20)

    Returns:
        Task result (DetectResult, SegmentResult, ClassifyResult, DepthResult, or VisionResult)

    Raises:
        ValueError: If task is "track" (tracking requires stateful pipeline)
        TaskNotSupportedError: If task is not supported

    Examples:
        >>> # Detect objects in image
        >>> result = mata.run("detect", "image.jpg")

        >>> # Use specific model with custom threshold
        >>> result = mata.run(
        ...     "detect",
        ...     "image.jpg",
        ...     model="dino",
        ...     threshold=0.6
        ... )

        >>> # Get JSON output
        >>> print(result.to_json(indent=2))

        >>> # Vision-language model for image understanding
        >>> result = mata.run(
        ...     "vlm",
        ...     "image.jpg",
        ...     model="Qwen/Qwen3-VL-2B-Instruct",
        ...     prompt="Describe this image in detail."
        ... )
        >>> print(result.text)

        >>> # VLM with custom system prompt
        >>> result = mata.run(
        ...     "vlm",
        ...     "product.jpg",
        ...     model="Qwen/Qwen3-VL-2B-Instruct",
        ...     prompt="Describe any defects.",
        ...     system_prompt="You are a quality inspector."
        ... )
    """
    # Track task is stateful and requires pipeline
    if task == "track":
        raise ValueError(
            "Track task is stateful and cannot be used with run(). "
            "Use load('track', ...) and call update() in a loop instead."
        )

    # Load adapter
    adapter = load(task=task, model=model, model_type=model_type, **kwargs)

    # Run prediction
    if task in ("detect", "segment", "classify", "depth", "pose", "vlm", "ocr"):
        return adapter.predict(input, **kwargs)
    else:
        # Should not reach here due to earlier checks
        raise TaskNotSupportedError(task, ["detect", "segment", "classify", "depth", "pose", "vlm", "ocr"])


def track(
    source: str | Path | Image.Image | np.ndarray | int,
    model: str | None = None,
    tracker: str | dict | None = "botsort",
    persist: bool = True,
    conf: float = 0.25,
    iou: float = 0.7,
    show: bool = False,
    save: bool = False,
    save_dir: str | Path | None = None,
    stream: bool = False,
    classes: list[int] | None = None,
    frame_rate: int = 30,
    max_frames: int | None = None,
    show_track_ids: bool = True,
    show_trails: bool = False,
    trail_length: int = 30,
    reid_model: str | None = None,
    with_reid: bool = False,
    reid_bridge: Any | None = None,
    **kwargs: Any,
) -> list[VisionResult] | Generator[VisionResult, None, None]:
    """Run object detection + tracking on video, stream, or image sequence.

    Combines detection and multi-object tracking into a single call.
    Uses ByteTrack or BotSort for temporal association across frames.

    Args:
        source: Video file path (.mp4, .avi, etc.), RTSP URL,
            camera index (0 for webcam), PIL Image, numpy array,
            or directory path containing image sequence.
        model: Detection model identifier (HuggingFace ID, local path,
            or config alias). Default uses registry default for 'detect'.
        tracker: Tracker type ('bytetrack', 'botsort'), path to custom
            YAML config, or dict of tracker parameters.
        persist: Maintain tracker state across frames. Set False to
            reset tracker each frame (rarely useful).
        conf: Minimum detection confidence threshold.
        iou: IoU threshold for NMS.
        show: Display annotated frames in OpenCV window.
        save: Save annotated video/frames to disk.
        save_dir: Output directory for saved results (default: 'runs/track/').
        stream: If True, return a generator yielding results per frame
            (memory-efficient for long videos).
        classes: Filter detections by class IDs.
        frame_rate: Video frame rate (for track lifetime calculation).
        max_frames: Maximum frames to process (None = all).
        show_track_ids: Draw track IDs on annotated frames.
        show_trails: Draw trajectory trails on annotated frames.
        trail_length: Number of frames to keep in trail history.
        reid_model: HuggingFace model ID or local .onnx path for ReID encoder.
            When provided, appearance embeddings are extracted from detection
            crops and injected into the tracker for identity recovery.
        with_reid: Convenience flag — must be paired with reid_model.
            Raises ValueError if True but reid_model is None.
        reid_bridge: Optional :class:`~mata.trackers.reid_bridge.ReIDBridge`
            instance for cross-camera ReID publishing.  After each frame,
            confirmed track embeddings are published to the shared Valkey
            store so other camera instances can query them.
        **kwargs: Additional arguments passed to detection model.

    Returns:
        If stream=False: list[VisionResult] — one result per frame,
            each with Instance.track_id populated.
        If stream=True: Generator yielding VisionResult per frame.

    Raises:
        ValueError: If source type is unsupported.
        FileNotFoundError: If video file does not exist.

    Example:
        >>> import mata
        >>> # Track objects in a video
        >>> results = mata.track("video.mp4", model="facebook/detr-resnet-50")
        >>> for result in results:
        ...     for inst in result.instances:
        ...         print(f"Track #{inst.track_id}: {inst.label_name}")
        >>>
        >>> # Stream mode for long videos
        >>> for result in mata.track("video.mp4", stream=True):
        ...     print(f"Frame: {len(result.instances)} objects tracked")
        >>>
        >>> # Webcam tracking
        >>> mata.track(0, model="detr", show=True)
    """
    # Load adapter eagerly so it is ready before any generator is iterated.
    # This ensures load() runs immediately (not lazily) which is important for
    # stream=True callers who consume the generator outside any patch context.
    adapter = load(
        "track",
        model,
        tracker=tracker,
        frame_rate=frame_rate,
        reid_model=reid_model,
        with_reid=with_reid,
        reid_bridge=reid_bridge,
        **kwargs,
    )

    # Build the generator and either collect or return it
    gen = _track_generator(
        adapter=adapter,
        source=source,
        persist=persist,
        conf=conf,
        iou=iou,
        show=show,
        save=save,
        save_dir=save_dir,
        classes=classes,
        max_frames=max_frames,
        show_track_ids=show_track_ids,
        show_trails=show_trails,
        trail_length=trail_length,
        frame_rate=frame_rate,
    )
    if stream:
        return gen
    else:
        return list(gen)


# ---------------------------------------------------------------------------
# Source-type detection helpers
# ---------------------------------------------------------------------------

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpeg", ".mpg", ".ts", ".flv"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


def _detect_source_type(source: Any) -> str:
    """Classify a tracking source into a source-type string.

    Returns one of: 'webcam', 'stream', 'video_file', 'image_dir',
    'image_file', 'pil_image', 'numpy_array'.
    """
    if isinstance(source, int):
        return "webcam"
    if isinstance(source, np.ndarray):
        return "numpy_array"
    # PIL detection via duck-typing to avoid hard import
    if hasattr(source, "save") and hasattr(source, "tobytes"):
        return "pil_image"
    s = str(source)
    if any(s.lower().startswith(p) for p in _STREAM_PREFIXES):
        return "stream"
    p = Path(s)
    if p.is_dir():
        return "image_dir"
    ext = p.suffix.lower()
    if ext in _VIDEO_EXTENSIONS:
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        return "video_file"
    if ext in _IMAGE_EXTENSIONS:
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        return "image_file"
    # Fallback: treat as a video path (cv2 will fail gracefully)
    return "video_file"


def _make_output_dir(base: str = "runs/track", name: str = "exp") -> Path:
    """Create an auto-incrementing output directory (exp1, exp2, ...)."""
    base_path = Path(base)
    i = 1
    while True:
        candidate = base_path / f"{name}{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        i += 1


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------


def _annotate_frame_cv2(
    frame_bgr: np.ndarray,
    result: VisionResult,
    show_track_ids: bool,
    trail_history: dict[int, list[tuple[int, int]]] | None,
    trail_length: int,
) -> np.ndarray:
    """Draw bounding boxes, labels, and optional trails on a BGR numpy frame.

    This is a lightweight cv2-based renderer used when ``show=True`` or
    ``save=True`` inside :func:`track`.  It intentionally avoids loading the
    full PIL-based image exporter so that the video loop stays fast.

    Args:
        frame_bgr: HWC uint8 BGR numpy array (OpenCV native format).
        result: VisionResult with instances to draw.
        show_track_ids: Prepend ``#id`` to each label.
        trail_history: Mutable dict mapping track_id → list of (cx, cy).
            Pass ``None`` to skip trail drawing.
        trail_length: Maximum positions retained in each trail.

    Returns:
        Annotated copy of the frame (same dtype/shape).
    """
    try:
        import cv2
    except ImportError:
        return frame_bgr

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Update trail history
    if trail_history is not None:
        for inst in result.instances:
            if inst.track_id is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in inst.bbox]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            history = trail_history.setdefault(inst.track_id, [])
            history.append((cx, cy))
            if len(history) > trail_length:
                del history[: len(history) - trail_length]

        # Draw trails before boxes (so boxes appear on top)
        for tid, pts in trail_history.items():
            if len(pts) < 2:
                continue
            color = _track_color(tid)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(2 * alpha))
                cv2.line(out, pts[i - 1], pts[i], color, thickness)

    # Draw instances
    for inst in result.instances:
        x1, y1, x2, y2 = [int(v) for v in inst.bbox]
        tid = inst.track_id
        color = _track_color(tid) if tid is not None else (0, 255, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Build label string
        label = inst.label_name or str(inst.label)
        score_str = f"{inst.score:.2f}" if inst.score is not None else ""
        if show_track_ids and tid is not None:
            text = f"#{tid} {label} {score_str}".strip()
        else:
            text = f"{label} {score_str}".strip()

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 4, th + baseline)
        cv2.rectangle(out, (x1, ly - th - baseline), (x1 + tw, ly + baseline), color, cv2.FILLED)
        cv2.putText(out, text, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Generate a deterministic BGR color for a given track ID."""
    import hashlib

    h = hashlib.md5(str(track_id).encode()).hexdigest()
    # Use middle bytes to avoid predictable low-entropy values
    r = int(h[4:6], 16)
    g = int(h[8:10], 16)
    b = int(h[12:14], 16)
    # Boost saturation: ensure at least one channel is bright
    if max(r, g, b) < 128:
        r = min(r + 128, 255)
    return (b, g, r)  # BGR


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


def _track_generator(
    adapter: Any,
    source: Any,
    persist: bool,
    conf: float,
    iou: float,
    show: bool,
    save: bool,
    save_dir: str | Path | None,
    classes: list[int] | None,
    max_frames: int | None,
    show_track_ids: bool,
    show_trails: bool,
    trail_length: int,
    frame_rate: int = 30,
) -> Generator[VisionResult, None, None]:
    """Internal generator that drives the tracking loop."""

    source_type = _detect_source_type(source)

    # --- Set up video writer (lazy) ------------------------------------
    writer = None  # cv2.VideoWriter, opened on first frame
    out_dir: Path | None = None
    if save:
        base = str(save_dir) if save_dir is not None else "runs/track"
        out_dir = _make_output_dir(base)

    # --- Trail history (cv2 display/save path) -----------------------
    trail_history: dict[int, list[tuple[int, int]]] | None = {} if show_trails else None

    # --- PIL-based TrackTrailRenderer ---------------------------------
    # Also maintain a TrackTrailRenderer so callers can invoke
    # draw_trails() on any PIL image independently of cv2.
    if show_trails:
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        _trail_renderer: Any = TrackTrailRenderer(trail_length=trail_length)
    else:
        _trail_renderer = None
    # --- cv2 import (soft) -------------------------------------------
    cv2_available = False
    try:
        import cv2 as _cv2  # noqa: F401

        cv2_available = True
    except ImportError:
        if show:
            raise ImportError("OpenCV (cv2) is required for show=True. " "Install with: pip install opencv-python")
        if save and source_type in ("video_file", "stream", "webcam"):
            raise ImportError(
                "OpenCV (cv2) is required for save=True with video sources. " "Install with: pip install opencv-python"
            )

    try:
        # ==============================================================
        # VIDEO / STREAM / WEBCAM
        # ==============================================================
        if source_type in ("video_file", "stream", "webcam"):
            if not cv2_available:
                raise ImportError(
                    "OpenCV (cv2) is required for video sources. " "Install with: pip install opencv-python"
                )
            import cv2

            cap_arg = int(source) if source_type == "webcam" else str(source)
            cap = cv2.VideoCapture(cap_arg)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video source: {source!r}. " "Check the path, URL, or camera index.")

            try:
                # Query video metadata
                src_fps = cap.get(cv2.CAP_PROP_FPS) or float(frame_rate)
                src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                frame_idx = 0
                while True:
                    if max_frames is not None and frame_idx >= max_frames:
                        break
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    # Convert BGR → RGB for the detector
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)

                    result = adapter.update(
                        pil_frame,
                        persist=persist,
                        conf=conf,
                        iou=iou,
                        classes=classes,
                    )
                    result.meta["frame_idx"] = frame_idx
                    if _trail_renderer is not None:
                        _trail_renderer.update(result.instances)
                        result.meta["trail_renderer"] = _trail_renderer

                    # Annotate and show/save
                    should_quit = False
                    if show or save:
                        annotated = _annotate_frame_cv2(frame_bgr, result, show_track_ids, trail_history, trail_length)
                        if show:
                            cv2.imshow("MATA Track", annotated)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                should_quit = True

                        if save:
                            if writer is None and out_dir is not None:
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                out_path = str(out_dir / "track.mp4")
                                writer = cv2.VideoWriter(out_path, fourcc, src_fps, (src_w, src_h))
                            if writer is not None:
                                writer.write(annotated)

                    yield result
                    frame_idx += 1
                    if should_quit:
                        break

            finally:
                cap.release()
                if show and cv2_available:
                    import cv2

                    cv2.destroyAllWindows()

        # ==============================================================
        # IMAGE DIRECTORY
        # ==============================================================
        elif source_type == "image_dir":
            import cv2

            dir_path = Path(str(source))
            image_files = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS)
            if not image_files:
                raise ValueError(
                    f"No images found in directory: {source}. " f"Supported extensions: {sorted(_IMAGE_EXTENSIONS)}"
                )

            frame_idx = 0
            for img_path in image_files:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                pil_frame = Image.open(str(img_path)).convert("RGB")
                result = adapter.update(
                    pil_frame,
                    persist=persist,
                    conf=conf,
                    iou=iou,
                    classes=classes,
                )
                result.meta["frame_idx"] = frame_idx
                result.meta["image_path"] = str(img_path)
                if _trail_renderer is not None:
                    _trail_renderer.update(result.instances)
                    result.meta["trail_renderer"] = _trail_renderer

                should_quit = False
                if show or save:
                    if cv2_available:
                        frame_bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                        annotated = _annotate_frame_cv2(frame_bgr, result, show_track_ids, trail_history, trail_length)
                        if show:
                            cv2.imshow("MATA Track", annotated)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                should_quit = True
                        if save and out_dir is not None:
                            cv2.imwrite(str(out_dir / img_path.name), annotated)

                yield result
                frame_idx += 1
                if should_quit:
                    break

            if show and cv2_available:
                cv2.destroyAllWindows()

        # ==============================================================
        # SINGLE IMAGE (PIL / numpy / file path)
        # ==============================================================
        elif source_type in ("image_file", "pil_image", "numpy_array"):
            if source_type == "image_file":
                pil_frame = Image.open(str(source)).convert("RGB")
            elif source_type == "pil_image":
                pil_frame = source.convert("RGB")
            else:
                # numpy_array — assume HWC RGB or BGR
                arr = np.asarray(source)
                if arr.shape[2] == 3:
                    pil_frame = Image.fromarray(arr.astype(np.uint8))
                else:
                    pil_frame = Image.fromarray(arr[:, :, :3].astype(np.uint8))

            result = adapter.update(
                pil_frame,
                persist=persist,
                conf=conf,
                iou=iou,
                classes=classes,
            )
            result.meta["frame_idx"] = 0
            if _trail_renderer is not None:
                _trail_renderer.update(result.instances)
                result.meta["trail_renderer"] = _trail_renderer

            if (show or save) and cv2_available:
                import cv2

                frame_bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                annotated = _annotate_frame_cv2(frame_bgr, result, show_track_ids, trail_history, trail_length)
                if show:
                    cv2.imshow("MATA Track", annotated)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                if save and out_dir is not None:
                    fname = Path(str(source)).name if source_type == "image_file" else "track.jpg"
                    cv2.imwrite(str(out_dir / fname), annotated)

            yield result

        else:
            raise ValueError(
                f"Unsupported source type: {source_type!r} for source {source!r}. "
                "Supported: video files, RTSP/HTTP streams, webcam index (int), "
                "image directories, single image files, PIL Images, numpy arrays."
            )

    finally:
        if writer is not None:
            writer.release()


def list_models(
    task: str | None = None, limit: int = 20, sort: str = "downloads"
) -> dict[str, list[dict[str, Any]]] | list[dict[str, Any]]:
    """List available models from HuggingFace Hub.

    Queries the HuggingFace Hub for models matching task filters.
    Results are cached for performance.

    Args:
        task: Optional task filter ("detect", "segment", "classify")
            If None, returns models for all supported tasks
        limit: Maximum number of models per task (default: 20)
        sort: Sort order - "downloads", "likes", "updated" (default: "downloads")

    Returns:
        If task specified: list of model info dicts for that task
        If task is None: dict mapping task to list of model info dicts

        Each model info dict contains:
        - id: Model ID (e.g., "facebook/detr-resnet-50")
        - downloads: Download count
        - likes: Number of likes
        - tags: List of model tags

    Examples:
        >>> # List detection models
        >>> models = mata.list_models('detect')
        >>> for model in models[:5]:
        ...     print(f"{model['id']} ({model['downloads']} downloads)")

        >>> # List all models
        >>> all_models = mata.list_models()
        >>> print(f"Found {len(all_models['detect'])} detection models")
    """
    try:
        from huggingface_hub import list_models as hf_list_models
    except ImportError:
        raise ImportError("huggingface_hub is required for list_models(). " "Install with: pip install huggingface_hub")

    # Task to HuggingFace pipeline tag mapping
    TASK_TO_TAG = {  # noqa: N806
        "detect": "object-detection",
        "segment": "image-segmentation",
        "classify": "image-classification",
        "depth": "depth-estimation",
    }

    def _fetch_models(task_name: str) -> list[dict[str, Any]]:
        """Fetch models for a specific task."""
        tag = TASK_TO_TAG.get(task_name)
        if not tag:
            return []

        try:
            models = hf_list_models(filter=tag, sort=sort, limit=limit, full=False)

            results = []
            for model in models:
                results.append(
                    {
                        "id": model.id,
                        "downloads": getattr(model, "downloads", 0),
                        "likes": getattr(model, "likes", 0),
                        "tags": getattr(model, "tags", []),
                    }
                )
            return results
        except Exception as e:
            from .core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Failed to fetch models for {task_name}: {e}")
            return []

    if task:
        # Return models for specific task
        return _fetch_models(task)
    else:
        # Return models for all tasks
        return {task_name: _fetch_models(task_name) for task_name in TASK_TO_TAG.keys()}


def get_model_info(model_id: str) -> dict[str, Any]:
    """Get detailed information about a HuggingFace model.

    Fetches model card, metadata, and configuration from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "facebook/detr-resnet-50")

    Returns:
        Dictionary with model metadata including:
        - id: Model ID
        - author: Model author/organization
        - downloads: Total downloads
        - likes: Number of likes
        - tags: Model tags (tasks, libraries, etc.)
        - card_data: Model card metadata
        - library: ML library (e.g., "transformers", "timm")
        - pipeline_tag: Task type (e.g., "object-detection")

    Examples:
        >>> info = mata.get_model_info('PekingU/rtdetr_v2_r18vd')
        >>> print(f"Model: {info['id']}")
        >>> print(f"Downloads: {info['downloads']}")
        >>> print(f"License: {info.get('card_data', {}).get('license')}")
    """
    try:
        from huggingface_hub import model_info
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for get_model_info(). " "Install with: pip install huggingface_hub"
        )

    try:
        info = model_info(model_id)
        return {
            "id": info.id,
            "author": info.author,
            "downloads": info.downloads,
            "likes": info.likes,
            "tags": info.tags,
            "card_data": info.card_data,
            "library": getattr(info, "library_name", None),
            "pipeline_tag": info.pipeline_tag,
            "created_at": str(info.created_at) if info.created_at else None,
            "last_modified": str(info.last_modified) if info.last_modified else None,
        }
    except Exception as e:
        from .core.exceptions import ModelNotFoundError

        raise ModelNotFoundError(f"Failed to fetch model info for '{model_id}': {e}")


def register_model(task: str, alias: str, source: str, **config: Any) -> None:
    """Register a model alias at runtime.

    Allows programmatic registration of model aliases without modifying
    configuration files. Useful for dynamic model management.

    Args:
        task: Task type ("detect", "segment", "classify")
        alias: Alias name for the model
        source: Model source (HF ID, file path, or URL)
        **config: Additional configuration (threshold, device, etc.)

    Examples:
        >>> # Register local ONNX model
        >>> mata.register_model(
        ...     "detect",
        ...     "my-onnx-model",
        ...     "/path/to/model.onnx",
        ...     threshold=0.5,
        ...     device="cuda"
        ... )
        >>> # Use registered model
        >>> detector = mata.load("detect", "my-onnx-model")

        >>> # Register HuggingFace model
        >>> mata.register_model(
        ...     "detect",
        ...     "my-rtdetr",
        ...     "PekingU/rtdetr_v2_r101vd",
        ...     threshold=0.6
        ... )
    """
    loader = _get_universal_loader()
    full_config = {"source": source, **config}
    loader.registry.register(task, alias, full_config)


def infer(
    image: str | Path | Image.Image | np.ndarray,
    graph: Graph | list[Node],
    providers: dict[str, Any],
    scheduler: Any | None = None,
    device: str = "auto",
    **kwargs: Any,
) -> MultiResult:
    """Execute a multi-task graph on an image.

    This is the primary API for running multi-task computer vision workflows.
    It accepts an image in multiple formats, a graph (or list of nodes),
    and a provider dictionary mapping provider names to loaded adapters.

    Args:
        image: Input image. Accepts:
            - ``str`` or ``Path``: file path to an image on disk
            - ``PIL.Image.Image``: a Pillow image object
            - ``np.ndarray``: a numpy array (HWC, uint8, RGB or BGR)
        graph: Execution graph. Accepts:
            - ``Graph``: a pre-built MATA graph object
            - ``list[Node]``: a list of nodes (will be wrapped in a Graph automatically)
        providers: Provider instances keyed by name.
            Keys must match the ``using`` parameter of nodes in the graph.
            Values are loaded adapters (e.g. from ``mata.load()``).
            Accepts either:
            - Flat dict: ``{"detector": adapter}`` — auto-organized by inspecting
              nodes for the capability each provider fulfills.
            - Nested dict: ``{"detect": {"detr": adapter}}`` — passed through directly.
        scheduler: Optional scheduler instance for execution strategy.
            Defaults to ``SyncScheduler`` (sequential execution).
            Pass ``ParallelScheduler`` for concurrent independent stages.
        device: Device placement. One of:
            - ``"auto"``: auto-detect (CUDA if available, else CPU)
            - ``"cuda"``: force CUDA
            - ``"cpu"``: force CPU
        **kwargs: Additional keyword arguments (reserved for future use).

    Returns:
        MultiResult with all task outputs accessible as attributes
        (e.g. ``result.dets``, ``result.masks``, ``result.final``).

    Raises:
        ValueError: If image type is unsupported or graph is empty.
        ValidationError: If graph compilation fails.
        RuntimeError: If graph execution fails.

    Examples:
        >>> import mata
        >>> from mata.nodes import Detect, Filter, Fuse
        >>>
        >>> # Load providers
        >>> detector = mata.load("detect", "facebook/detr-resnet-50")
        >>>
        >>> # Run graph
        >>> result = mata.infer(
        ...     image="test.jpg",
        ...     graph=[
        ...         Detect(using="detector", out="dets"),
        ...         Filter(src="dets", score_gt=0.3, out="filtered"),
        ...         Fuse(dets="filtered", out="final"),
        ...     ],
        ...     providers={"detector": detector},
        ... )
        >>> result.final  # Access fused result

        >>> # Using a pre-built Graph object
        >>> from mata.core.graph import Graph
        >>> g = (Graph("my_pipeline")
        ...     .then(Detect(using="detector", out="dets"))
        ...     .then(Filter(src="dets", score_gt=0.5, out="filtered"))
        ... )
        >>> result = mata.infer("image.jpg", g, providers={"detector": detector})

        >>> # Parallel execution for speedup
        >>> from mata.core.graph import ParallelScheduler
        >>> result = mata.infer(
        ...     "scene.jpg",
        ...     graph=[...],
        ...     providers={...},
        ...     scheduler=ParallelScheduler(),
        ... )
    """
    from .core.artifacts.image import Image as ImageArtifact
    from .core.graph import ExecutionContext, SyncScheduler
    from .core.graph import Graph as GraphClass
    from .core.graph.node import Node

    # --- Convert image to Image artifact ---
    if isinstance(image, (str, Path)):
        image_artifact = ImageArtifact.from_path(str(image))
    elif isinstance(image, Image.Image):
        image_artifact = ImageArtifact.from_pil(image)
    elif isinstance(image, np.ndarray):
        image_artifact = ImageArtifact.from_numpy(image)
    else:
        raise ValueError(
            f"Unsupported image type: {type(image).__name__}. " f"Expected str, Path, PIL.Image.Image, or np.ndarray."
        )

    # --- Build graph if list of nodes provided ---
    if isinstance(graph, list):
        if not graph:
            raise ValueError("Node list cannot be empty.")
        g = GraphClass()
        for node in graph:
            if not isinstance(node, Node):
                raise ValueError(f"Expected Node instance in graph list, " f"got {type(node).__name__}.")
            g.add(node)
        graph = g
    elif not isinstance(graph, GraphClass):
        raise ValueError(f"Unsupported graph type: {type(graph).__name__}. " f"Expected Graph or list[Node].")

    # --- Normalize providers ---
    # ExecutionContext expects nested: {capability: {name: provider}}
    # graph.compile() validator expects flat: {name: provider}
    # We maintain both formats.
    flat_providers, nested_providers = _normalize_providers(providers, graph)

    # --- Compile graph ---
    compiled = graph.compile(flat_providers)

    # --- Create execution context ---
    context = ExecutionContext(nested_providers, device=device)

    # --- Execute ---
    if scheduler is None:
        scheduler = SyncScheduler()

    result = scheduler.execute(compiled, context, {"input.image": image_artifact})

    return result


def _normalize_providers(
    providers: dict[str, Any],
    graph: Graph,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Normalize a providers dict into both flat and nested formats.

    The graph validator expects a flat dict ``{name: adapter}``.
    The ExecutionContext expects a nested dict ``{capability: {name: adapter}}``.

    This function accepts either format and returns *both*.

    Args:
        providers: Flat or nested provider dict.
        graph: The graph being compiled (used to infer capabilities).

    Returns:
        Tuple of (flat_dict, nested_dict).
    """
    if not providers:
        return {}, {}

    # Detect whether already nested (values are dicts)
    first_value = next(iter(providers.values()))
    if isinstance(first_value, dict):
        # Already nested → flatten for validator
        nested = providers
        flat: dict[str, Any] = {}
        for _cap, provider_dict in nested.items():
            flat.update(provider_dict)
        return flat, nested

    # Flat format: {"name": adapter}
    flat = providers

    # Build nested dict by inspecting graph nodes
    _CAPABILITY_BY_NODE_CLASS = _build_capability_map()  # noqa: N806

    name_to_capability: dict[str, str] = {}
    for node in graph._nodes:
        provider_name = getattr(node, "provider_name", None)
        if provider_name and provider_name in flat:
            node_class = type(node).__name__
            capability = _CAPABILITY_BY_NODE_CLASS.get(node_class)
            if capability:
                name_to_capability[provider_name] = capability

    nested = {}
    for name, adapter in flat.items():
        capability = name_to_capability.get(name)
        if capability is None:
            capability = _infer_capability(adapter)
        if capability is None:
            capability = name
        nested.setdefault(capability, {})[name] = adapter

    return flat, nested


def _build_capability_map() -> dict[str, str]:
    """Map node class names to their capability (provider) types."""
    return {
        # Detection
        "Detect": "detect",
        # Classification
        "Classify": "classify",
        # Segmentation
        "SegmentImage": "segment",
        "PromptBoxes": "segment",
        "PromptPoints": "segment",
        "SegmentEverything": "segment",
        # Depth
        "EstimateDepth": "depth",
        # Tracking
        "Track": "track",
        # VLM
        "VLMDescribe": "vlm",
        "VLMDetect": "vlm",
        "VLMQuery": "vlm",
        # OCR
        "OCR": "ocr",
        # Annotate (uses backend, not a provider)
        # Filter, TopK, Fuse, Merge, etc. have no provider
    }


def _infer_capability(adapter: Any) -> str | None:
    """Infer capability from adapter class name or attributes."""
    cls_name = type(adapter).__name__.lower()

    if "detect" in cls_name:
        return "detect"
    if "classify" in cls_name or "clip" in cls_name:
        return "classify"
    if "segment" in cls_name or "sam" in cls_name:
        return "segment"
    if "depth" in cls_name:
        return "depth"
    if "track" in cls_name:
        return "track"
    if "vlm" in cls_name:
        return "vlm"
    if "ocr" in cls_name:
        return "ocr"

    return None


def val(
    task: str,
    *,
    model: str | Any | None = None,
    data: str | dict | None = None,
    predictions: list | None = None,
    ground_truth: str | list | None = None,
    conf: float = 0.001,
    iou: float = 0.50,
    device: str | None = None,
    verbose: bool = True,
    plots: bool = False,
    save_dir: str = "",
    split: str = "val",
    **kwargs,
) -> Any:
    """Run YOLO-style validation on a task model.

    Args:
        task: "detect", "segment", "classify", or "depth"
        model: Model ID, path, alias, or pre-loaded adapter.
        data: Path to dataset YAML file or config dict.
        predictions: Pre-run predictions (standalone mode, skips inference).
        ground_truth: COCO JSON path or annotation list (standalone mode).
        conf: Confidence threshold for filtering predictions.
        iou: IoU threshold for TP/FP matching.
        device: Inference device ("cpu", "cuda", etc.).
        verbose: Print per-class metrics table.
        plots: Save PR curve, F1 curve, and confusion matrix plots.
        save_dir: Directory for plot output files.
        split: Dataset split to evaluate ("val", "test", "train").

    Returns:
        DetMetrics | SegmentMetrics | ClassifyMetrics | DepthMetrics
    """
    from mata.eval.validator import Validator

    return Validator(
        task=task,
        model=model,
        data=data,
        predictions=predictions,
        ground_truth=ground_truth,
        conf=conf,
        iou=iou,
        device=device,
        verbose=verbose,
        plots=plots,
        save_dir=save_dir,
        split=split,
        **kwargs,
    ).run()


def verbose(level: int = 2) -> None:
    """Control MATA's output verbosity.

    Args:
        level: Verbosity level:
            - ``0`` (silent): Suppress *all* output — both MATA ``[INFO]``
              messages **and** third-party noise.
            - ``1`` (quiet, **default**): Show MATA logs, suppress third-party
              noise (tqdm progress bars, transformers warnings, etc.).
            - ``2`` (verbose): Show everything — useful for debugging model
              loading issues.

    Examples::

        import mata

        mata.verbose(0)   # total silence
        mata.verbose(1)   # only MATA logs (default)
        mata.verbose(2)   # MATA logs + third-party output
    """
    from .core.logging import verbose as _verbose

    _verbose(level)
