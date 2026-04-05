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
from .core.logging import get_logger
from .core.model_loader import UniversalLoader
from .core.types import ClassifyResult, DepthResult, DetectResult, ModelType, SegmentResult, VisionResult

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .core.artifacts.result import MultiResult
    from .core.graph.graph import Graph
    from .core.graph.node import Node
    from .training.result import TrainingResult

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
        task: Task type ("detect", "segment", "classify", "depth", "vlm", "ocr",
            "barcode", "embed", "recognize")
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
            For "recognize" task:
                - gallery (Gallery, required): Pre-populated Gallery instance
                - top_k (int, optional): Number of top matches to return (default: 1)
                - threshold (float, optional): Minimum cosine similarity (default: gallery default)

    Returns:
        Task result (DetectResult, SegmentResult, ClassifyResult, DepthResult,
        VisionResult, EmbedResult, or Matches for "recognize")

    Raises:
        ValueError: If task is "track" (tracking requires stateful pipeline)
        ValueError: If task is "recognize" but no gallery is provided
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

        >>> # Gallery recognition (embed + cosine search)
        >>> gallery = mata.Gallery()
        >>> gallery.add("alice", alice_embedding)
        >>> result = mata.run("recognize", "image.jpg",
        ...                   model="openai/clip-vit-base-patch32",
        ...                   gallery=gallery, top_k=1)
        >>> print(result.entries[0].label)
    """
    # Track task is stateful and requires pipeline
    if task == "track":
        raise ValueError(
            "Track task is stateful and cannot be used with run(). "
            "Use load('track', ...) and call update() in a loop instead."
        )

    # Recognize task: embed + gallery cosine search (no model adapter required)
    if task == "recognize":
        return _run_recognize(input, model=model, model_type=model_type, **kwargs)

    # Pop embed-specific kwargs before load() to prevent leaking into adapter constructors
    _embed_frames = kwargs.pop("frames", None) if task == "embed" else None
    _embed_text = kwargs.pop("text", None) if task == "embed" else None
    _embed_fps = kwargs.pop("fps", None) if task == "embed" else None
    _embed_max_frames = kwargs.pop("max_frames", None) if task == "embed" else None
    _embed_dim = kwargs.pop("embed_dim", None) if task == "embed" else None

    # For embed task, forward constructor kwargs to load() explicitly
    if task == "embed":
        _embed_load_kwargs = dict(kwargs)
        if _embed_dim is not None:
            _embed_load_kwargs["embed_dim"] = _embed_dim
        if _embed_fps is not None:
            _embed_load_kwargs["fps"] = _embed_fps
        if _embed_max_frames is not None:
            _embed_load_kwargs["max_frames"] = _embed_max_frames
        adapter = load(task=task, model=model, model_type=model_type, **_embed_load_kwargs)
    else:
        # Load adapter
        adapter = load(task=task, model=model, model_type=model_type, **kwargs)

    # Run prediction
    if task in ("detect", "segment", "classify", "depth", "pose", "vlm", "ocr", "barcode"):
        return adapter.predict(input, **kwargs)
    elif task == "embed":
        from .core.artifacts.image import Image as ImageArtifact

        # XClip video frames (list of numpy arrays) — unchanged
        if _embed_frames is not None:
            return adapter.embed(_embed_frames)

        # Text-only query (no image input)
        if _embed_text is not None and input is None:
            if hasattr(adapter._encoder, "predict_text"):
                return adapter._encoder.predict_text(_embed_text)
            # XClip / generic fallback via embed()
            return adapter.embed(_embed_text)

        # Multimodal dict input
        if isinstance(input, dict):
            if hasattr(adapter._encoder, "predict_multimodal"):
                return adapter._encoder.predict_multimodal(input)
            raise ValueError(
                "Multimodal embedding requires a model with predict_multimodal() support "
                "(e.g., Qwen/Qwen3-VL-Embedding-2B)"
            )

        # Video file path
        if isinstance(input, (str, Path)) and _is_video_path(str(input)):
            if hasattr(adapter._encoder, "predict_video"):
                frames = _extract_video_frames(str(input), fps=_embed_fps, max_frames=_embed_max_frames)
                return adapter._encoder.predict_video(frames)
            raise ValueError(
                "Video embedding requires a model with predict_video() support "
                "(e.g., Qwen/Qwen3-VL-Embedding-2B, microsoft/xclip-base-patch32)"
            )

        # Image + text mixed-modal (image provided together with text= kwarg)
        if _embed_text is not None and input is not None:
            if hasattr(adapter._encoder, "predict_multimodal"):
                mm_input: dict[str, Any] = {"text": _embed_text}
                if isinstance(input, (str, Path)):
                    mm_input["image"] = str(input)
                else:
                    mm_input["image"] = input
                return adapter._encoder.predict_multimodal(mm_input)

        # Standard image input — backward compatible
        if isinstance(input, (str, Path)):
            image_artifact = ImageArtifact.from_path(str(input))
        elif isinstance(input, Image.Image):
            image_artifact = ImageArtifact.from_pil(input)
        elif isinstance(input, np.ndarray):
            image_artifact = ImageArtifact.from_numpy(input)
        else:
            raise ValueError(
                f"Unsupported input type for embed task: {type(input).__name__}. "
                "Expected file path, PIL Image, numpy array, dict, or video path."
            )
        return adapter.embed(image_artifact)
    else:
        # Should not reach here due to earlier checks
        raise TaskNotSupportedError(
            task, ["detect", "segment", "classify", "depth", "pose", "vlm", "ocr", "barcode", "embed", "recognize"]
        )


def _run_recognize(
    input: Any,
    model: str | None = None,
    model_type: Any = None,
    gallery: Any = None,
    top_k: int = 1,
    threshold: float | None = None,
    **kwargs: Any,
) -> Any:
    """Internal implementation for mata.run('recognize', ...).

    Embeds the input image and performs cosine similarity search against a Gallery.

    Args:
        input: Image path, PIL Image, or numpy array.
        model: Optional embed model (HuggingFace ID, local path, or alias).
            Defaults to registry default for 'embed' task.
        model_type: Optional explicit model type (passed through to embed adapter).
        gallery: Required :class:`~mata.recognition.Gallery` with enrolled embeddings.
        top_k: Number of top gallery matches to return per image.
        threshold: Minimum cosine similarity. When None, uses the gallery's default.
        **kwargs: Additional arguments passed to the embed adapter constructor.

    Returns:
        :class:`~mata.core.artifacts.matches.Matches` with one entry per image.

    Raises:
        ValueError: If ``gallery`` is not provided or input type is unsupported.
    """
    from .core.artifacts.image import Image as ImageArtifact
    from .core.artifacts.matches import MatchEntry, Matches

    if gallery is None:
        raise ValueError(
            "mata.run('recognize', ...) requires a 'gallery' keyword argument.\n"
            "Create a Gallery and add embeddings before searching:\n"
            "    gallery = mata.Gallery()\n"
            "    gallery.add('alice', embedding)\n"
            "    result = mata.run('recognize', image, gallery=gallery)"
        )

    # Build embed image artifact
    if isinstance(input, (str, Path)):
        image_artifact = ImageArtifact.from_path(str(input))
    elif isinstance(input, Image.Image):
        image_artifact = ImageArtifact.from_pil(input)
    elif isinstance(input, np.ndarray):
        image_artifact = ImageArtifact.from_numpy(input)
    else:
        raise ValueError(
            f"Unsupported input type for recognize task: {type(input).__name__}. "
            "Expected file path, PIL Image, or numpy array."
        )

    # Embed using the embed task adapter
    embed_adapter = load(task="embed", model=model, model_type=model_type, **kwargs)
    embeddings_ndarray = embed_adapter.embed(image_artifact)  # (N, D) ndarray

    # Query the gallery with the first (and only) embedding vector
    query_vector = embeddings_ndarray[0]
    matches = gallery.search(query_vector, top_k=top_k, threshold=threshold)

    best = matches[0] if matches else None
    entry = MatchEntry(
        instance_id="query",
        label=best.label if best is not None else "unknown",
        similarity=best.similarity if best is not None else 0.0,
        all_matches=[m.to_dict() for m in matches],
    )
    return Matches(entries=[entry], meta={"model": str(model), "top_k": top_k})


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


def _is_video_path(path: str) -> bool:
    """Return True if *path* has a recognized video file extension."""
    return Path(path).suffix.lower() in _VIDEO_EXTENSIONS


def _extract_video_frames(
    video_path: str,
    fps: float | None = None,
    max_frames: int | None = None,
) -> list:
    """Extract frames from a video file using OpenCV.

    Args:
        video_path: Path to the video file.
        fps: Target frames-per-second to sample (default 1.0).
        max_frames: Maximum number of frames to return (default 64).

    Returns:
        List of BGR numpy arrays.
    """
    import cv2

    target_fps = fps if fps is not None else 1.0
    target_max = max_frames if max_frames is not None else 64

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(source_fps / target_fps))
    frames: list = []
    idx = 0
    while len(frames) < target_max:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")
    return frames


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
        logger.info(f"Saving output to: {out_dir.resolve()}")

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
                                logger.info(f"Video writer opened: {out_path}")
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
    image: str | Path | Image.Image | np.ndarray | None = None,
    graph: Graph | list[Node] = None,
    providers: dict[str, Any] = None,
    scheduler: Any | None = None,
    device: str = "auto",
    video: str | None = None,
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

    if graph is None:
        raise ValueError("'graph' must be provided.")
    if providers is None:
        providers = {}

    # --- Determine input artifact ---
    if image is None and video is not None:
        from .core.artifacts.video_path import VideoPath

        initial_artifact_key = "input.video"
        initial_artifact_value = VideoPath(path=str(video))
    elif image is not None:
        # --- Convert image to Image artifact ---
        if isinstance(image, (str, Path)):
            initial_artifact_value = ImageArtifact.from_path(str(image))
        elif isinstance(image, Image.Image):
            initial_artifact_value = ImageArtifact.from_pil(image)
        elif isinstance(image, np.ndarray):
            initial_artifact_value = ImageArtifact.from_numpy(image)
        else:
            raise ValueError(
                f"Unsupported image type: {type(image).__name__}. "
                f"Expected str, Path, PIL.Image.Image, or np.ndarray."
            )
        initial_artifact_key = "input.image"
    else:
        raise ValueError("Either 'image' or 'video' must be provided to infer().")

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

    result = scheduler.execute(compiled, context, {initial_artifact_key: initial_artifact_value})

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
        # Embedding
        "Embed": "embed",
        # ReID (cross-camera re-identification)
        "ReID": "reid",
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
    if "reid" in cls_name or "bridge" in cls_name:
        return "reid"
    if "embed" in cls_name:
        return "embed"

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


def train(
    task: str,
    *,
    model: str,
    data: str | dict,
    val_data: str | dict | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    optimizer: str = "adamw",
    weight_decay: float = 0.01,
    scheduler: str = "cosine",
    warmup_epochs: int = 1,
    device: str = "auto",
    amp: bool = True,
    save_dir: str = "runs/train",
    save_every: int = 0,
    val_every: int = 1,
    patience: int = 0,
    freeze_backbone: bool = False,
    freeze_layers: list[str] | None = None,
    augment: bool = True,
    augment_config: dict | None = None,
    resume: str | None = None,
    num_workers: int = 4,
    seed: int = 42,
    verbose: bool = True,
    **kwargs: Any,
) -> TrainingResult:
    """Train a model from scratch or continue training.

    Args:
        task: Task type — "detect", "classify", or "segment"
        model: Model source (HuggingFace ID, torchvision/*, config alias, local path)
        data: Training data (YAML config path, directory, or COCO JSON)
        val_data: Validation data (same formats as data). Defaults to None.
        epochs: Number of training epochs. Defaults to 10.
        batch_size: Batch size. Defaults to 8.
        lr: Learning rate. Defaults to 1e-4.
        optimizer: Optimizer name ("adamw", "adam", "sgd"). Defaults to "adamw".
        weight_decay: Weight decay for regularization. Defaults to 0.01.
        scheduler: LR scheduler ("cosine", "linear", "step", "none"). Defaults to "cosine".
        warmup_epochs: Epochs to warm up LR from 0. Defaults to 1.
        device: Device to train on ("auto", "cuda", "cpu"). Defaults to "auto".
        amp: Enable automatic mixed precision. Defaults to True.
        save_dir: Root directory for saving checkpoints. Defaults to "runs/train".
        save_every: Save checkpoint every N epochs (0 = only best/last). Defaults to 0.
        val_every: Run validation every N epochs. Defaults to 1.
        patience: Early stopping patience in epochs (0 = disabled). Defaults to 0.
        freeze_backbone: Freeze backbone weights during training. Defaults to False.
        freeze_layers: List of layer name patterns to freeze. Defaults to None.
        augment: Enable data augmentation. Defaults to True.
        augment_config: Augmentation configuration dict. Defaults to None.
        resume: Path to checkpoint directory to resume from. Defaults to None.
        num_workers: DataLoader worker processes. Defaults to 4.
        seed: Random seed for reproducibility. Defaults to 42.
        verbose: Print training progress. Defaults to True.
        **kwargs: Additional keyword arguments for future extensibility.

    Returns:
        TrainingResult with metrics, checkpoint paths, and training history.

    Example:
        >>> result = mata.train("detect", model="facebook/detr-resnet-50",
        ...     data="coco.yaml", epochs=10, lr=1e-4)
        >>> print(f"Best mAP50: {result.best_metrics.box.map50:.3f}")
    """
    from mata.training import TrainingConfig, TrainingOrchestrator

    config = TrainingConfig(
        task=task,
        model=model,
        data=data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay,
        scheduler=scheduler,
        warmup_epochs=warmup_epochs,
        device=device,
        amp=amp,
        save_dir=save_dir,
        save_every=save_every,
        val_every=val_every,
        patience=patience,
        freeze_backbone=freeze_backbone,
        freeze_layers=freeze_layers,
        augment=augment,
        augment_config=augment_config,
        resume=resume,
        num_workers=num_workers,
        seed=seed,
        verbose=verbose,
    )
    config.validate()
    return TrainingOrchestrator(config).train()


def finetune(
    task: str,
    *,
    model: str,
    data: str | dict,
    val_data: str | dict | None = None,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    freeze_backbone: bool = True,
    **kwargs: Any,
) -> TrainingResult:
    """Fine-tune a pre-trained model on custom data.

    Like train() but with fine-tuning defaults: lower LR, fewer epochs, frozen backbone.

    Args:
        task: Task type — "detect", "classify", or "segment"
        model: Model source (HuggingFace ID, torchvision/*, config alias, local path)
        data: Training data (YAML config path, directory, or COCO JSON)
        val_data: Validation data. Defaults to None.
        epochs: Number of fine-tuning epochs. Defaults to 5.
        batch_size: Batch size. Defaults to 16.
        lr: Learning rate. Defaults to 1e-5.
        freeze_backbone: Freeze backbone weights. Defaults to True.
        **kwargs: Additional keyword arguments forwarded to train().

    Returns:
        TrainingResult with metrics, checkpoint paths, and training history.

    Example:
        >>> result = mata.finetune("classify", model="microsoft/resnet-50",
        ...     data="/data/flowers/", epochs=5)
        >>> print(f"Top-1: {result.best_metrics.top1:.1%}")
    """
    return train(
        task,
        model=model,
        data=data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )


def annotate(
    data: str = "data",
    *,
    host: str = "127.0.0.1",
    port: int = 8710,
    open_browser: bool = True,
    block: bool = True,
    detect_model: str | None = None,
    vlm_model: str | None = None,
    embed_model: str | None = None,
    zeroshot_model: str | None = None,
    **kwargs: Any,
) -> Any:
    """Launch the MATA annotation web tool.

    Starts a browser-based annotation server for creating and editing
    datasets with AI-assisted labeling. Outputs COCO JSON annotations
    and YAML configs compatible with mata.train().

    Args:
        data: Root data directory to manage. Defaults to "data".
        host: Server bind address. Defaults to "127.0.0.1" (localhost only).
        port: Server port. Defaults to 8710.
        open_browser: Auto-open browser. Defaults to True.
        block: Block until server stops. Defaults to True.
        detect_model: Detection model for AI-assist pre-labeling.
        vlm_model: VLM model for AI-assist auto-annotation.
        embed_model: Embedding model for CLIP classify suggestions.
        zeroshot_model: Grounding DINO model for zero-shot detection AI-assist.
        **kwargs: Additional server configuration.

    Returns:
        AnnotateServer instance.
    """
    from .annotate import start_server

    return start_server(
        data=data,
        host=host,
        port=port,
        open_browser=open_browser,
        block=block,
        detect_model=detect_model,
        vlm_model=vlm_model,
        embed_model=embed_model,
        zeroshot_model=zeroshot_model,
        **kwargs,
    )


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
