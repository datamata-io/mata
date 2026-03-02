"""Video I/O utilities for tracking and temporal processing.

Handles:
- Video file reading (.mp4, .avi, .mkv, .mov, .wmv)
- RTSP/HTTP stream reading
- Webcam capture (device index)
- Image directory iteration (sorted by name)
- Video writing (annotated output)
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from mata.core.logging import get_logger

logger = get_logger(__name__)

# Supported image extensions for directory iteration
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Supported video file extensions
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}

# Stream URL prefixes
_STREAM_PREFIXES = ("rtsp://", "http://", "https://")


def _require_cv2() -> Any:
    """Import cv2 or raise a helpful ImportError."""
    try:
        import cv2  # type: ignore[import]

        return cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for video I/O but is not installed. "
            "Install it with:\n"
            "    pip install opencv-python\n"
            "or for headless environments:\n"
            "    pip install opencv-python-headless"
        ) from exc


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_source_type(source: Any) -> str:
    """Determine the input source type.

    Args:
        source: Input source — can be an integer (webcam), a string/Path
            (video file, stream URL, image directory, or image file), a
            PIL Image, or a numpy array.

    Returns:
        One of: ``'webcam'``, ``'stream'``, ``'video_file'``, ``'image_dir'``,
        ``'image_file'``, ``'pil_image'``, ``'numpy_array'``.

    Raises:
        ValueError: If the source type cannot be determined.
    """
    # Webcam — integer device index
    if isinstance(source, (int,)):
        return "webcam"

    # NumPy arrays (raw frames)
    if isinstance(source, np.ndarray):
        return "numpy_array"

    # PIL Images (lazy import to avoid hard dependency for non-image paths)
    try:
        from PIL import Image as _PILImage  # type: ignore[import]

        if isinstance(source, _PILImage.Image):
            return "pil_image"
    except ImportError:
        pass

    # Normalize to string path for the remaining checks
    if isinstance(source, Path):
        source_str = str(source)
    elif isinstance(source, str):
        source_str = source
    else:
        raise ValueError(
            f"Unsupported source type: {type(source).__name__!r}. "
            "Expected int (webcam), str/Path (file/URL), PIL.Image, or np.ndarray."
        )

    # Stream URLs
    if source_str.lower().startswith(_STREAM_PREFIXES):
        return "stream"

    path = Path(source_str)

    # Directory of images
    if path.is_dir():
        return "image_dir"

    suffix = path.suffix.lower()

    if suffix in _VIDEO_EXTENSIONS:
        return "video_file"

    if suffix in _IMAGE_EXTENSIONS:
        return "image_file"

    # Unknown extension — if the path looks like a file, treat as video_file
    # (cv2.VideoCapture handles many container formats).
    if path.exists():
        return "video_file"

    raise ValueError(
        f"Cannot determine source type for: {source_str!r}. "
        "Provide a video file, image file, image directory, stream URL, "
        "webcam index, PIL Image, or numpy array."
    )


def iter_frames(
    source: Any,
    max_frames: int | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Iterate frames from any supported source.

    Yields ``(frame_index, frame_bgr)`` tuples where ``frame_bgr`` is a
    ``uint8`` BGR numpy array matching the ``cv2`` convention.

    ``cv2.VideoCapture`` resources are always released on generator close or
    exhaustion — even if the caller abandons the generator mid-way.

    Args:
        source: Any source accepted by :func:`detect_source_type`.
        max_frames: Maximum number of frames to yield.  ``None`` = unlimited.

    Yields:
        ``(frame_index, frame_bgr)`` — 0-based integer index and BGR frame.

    Raises:
        ImportError: If cv2 is required but not installed.
        FileNotFoundError: If a video file or image path does not exist.
        ValueError: If the source cannot be opened.
    """
    source_type = detect_source_type(source)

    if source_type == "numpy_array":
        frame = _ensure_bgr(source)
        if max_frames is None or max_frames > 0:
            yield 0, frame
        return

    if source_type == "pil_image":
        frame = _pil_to_bgr(source)
        if max_frames is None or max_frames > 0:
            yield 0, frame
        return

    if source_type == "image_file":
        cv2 = _require_cv2()
        path = Path(source) if not isinstance(source, Path) else source
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"cv2.imread could not read: {path}")
        if max_frames is None or max_frames > 0:
            yield 0, frame
        return

    if source_type == "image_dir":
        yield from _iter_image_dir(source, max_frames)
        return

    # video_file, stream, webcam — all use cv2.VideoCapture
    yield from _iter_capture(source, source_type, max_frames)


def get_video_info(source: str | Path) -> dict[str, Any]:
    """Return metadata for a video file or stream.

    Args:
        source: Path to a video file or RTSP/HTTP stream URL.

    Returns:
        Dictionary with keys:
        - ``fps`` (float): Frames per second.
        - ``width`` (int): Frame width in pixels.
        - ``height`` (int): Frame height in pixels.
        - ``frame_count`` (int): Total frame count (-1 for live streams).

    Raises:
        ImportError: If cv2 is not installed.
        FileNotFoundError: If *source* is a path that does not exist.
        ValueError: If the source cannot be opened by cv2.
    """
    cv2 = _require_cv2()

    source_str = str(source)
    path = Path(source_str)

    if not source_str.lower().startswith(_STREAM_PREFIXES) and not path.exists():
        raise FileNotFoundError(f"Video file not found: {source_str}")

    cap = cv2.VideoCapture(source_str)
    try:
        if not cap.isOpened():
            raise ValueError(
                f"cv2.VideoCapture could not open: {source_str!r}. "
                "Ensure the file is a valid video or the stream URL is reachable."
            )
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


class VideoWriter:
    """Context manager for writing annotated video output.

    Selects the correct FourCC codec automatically based on the output
    file extension:

    - ``.mp4`` / ``.mkv`` / ``.mov`` → ``mp4v``
    - ``.avi`` → ``XVID``
    - ``.wmv`` → ``WMV2``
    - Other → ``mp4v`` (fallback)

    Usage::

        with VideoWriter("output.mp4", fps=30, size=(1920, 1080)) as writer:
            for frame in frames:
                writer.write(frame)

    Args:
        path: Output video file path.
        fps: Frames per second.
        size: ``(width, height)`` in pixels.
        codec: FourCC codec string override.  When ``None`` (default) the
            codec is inferred from the file extension.
    """

    # Mapping from lowercase extension to FourCC string
    _CODEC_MAP: dict[str, str] = {
        ".mp4": "mp4v",
        ".mkv": "mp4v",
        ".mov": "mp4v",
        ".avi": "XVID",
        ".wmv": "WMV2",
    }

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        codec: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._fps = fps
        self._size = size  # (width, height)
        self._codec = codec or self._infer_codec(self._path)
        self._writer: Any = None  # cv2.VideoWriter handle

    def _infer_codec(self, path: Path) -> str:
        return self._CODEC_MAP.get(path.suffix.lower(), "mp4v")

    def __enter__(self) -> VideoWriter:
        cv2 = _require_cv2()
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self._path),
            fourcc,
            self._fps,
            self._size,  # (W, H)
        )
        if not self._writer.isOpened():
            raise ValueError(
                f"cv2.VideoWriter could not open output path: {self._path!r} "
                f"(codec={self._codec!r}, fps={self._fps}, size={self._size})."
            )
        logger.debug(
            "VideoWriter opened: %s (codec=%s, fps=%.1f, size=%dx%d)",
            self._path,
            self._codec,
            self._fps,
            self._size[0],
            self._size[1],
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the output video.

        Args:
            frame: BGR numpy array of shape ``(H, W, 3)``.

        Raises:
            RuntimeError: If called outside the context manager.
        """
        if self._writer is None:
            raise RuntimeError(
                "VideoWriter.write() called outside 'with' block. " "Use VideoWriter as a context manager."
            )
        self._writer.write(frame)

    def __exit__(self, *args: Any) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.debug("VideoWriter released: %s", self._path)


def make_output_dir(base: str = "runs/track", name: str = "exp") -> Path:
    """Create an auto-incrementing output directory.

    Scans *base* for existing ``<name>N`` directories and creates the next one::

        runs/track/exp1/   ← first call
        runs/track/exp2/   ← second call
        ...

    Args:
        base: Parent directory.
        name: Prefix for the numbered sub-directory.

    Returns:
        Path to the newly created directory.
    """
    base_path = Path(base)
    idx = 1
    while True:
        candidate = base_path / f"{name}{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            logger.debug("Created output directory: %s", candidate)
            return candidate
        idx += 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pil_to_bgr(image: Any) -> np.ndarray:
    """Convert a PIL Image to a BGR numpy array."""
    img_rgb = np.array(image.convert("RGB"))
    return img_rgb[:, :, ::-1].copy()


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Ensure a numpy array is a valid BGR uint8 frame."""
    if frame.ndim == 2:
        # Grayscale → BGR
        cv2 = _require_cv2()
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        # BGRA → BGR
        cv2 = _require_cv2()
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _iter_image_dir(
    source: str | Path,
    max_frames: int | None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield frames from a sorted directory of image files."""
    cv2 = _require_cv2()
    dir_path = Path(source)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {dir_path}")

    image_files = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)

    if not image_files:
        logger.warning("No supported image files found in directory: %s", dir_path)
        return

    for idx, img_path in enumerate(image_files):
        if max_frames is not None and idx >= max_frames:
            break
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Skipping unreadable image: %s", img_path)
            continue
        yield idx, frame


def _iter_capture(
    source: Any,
    source_type: str,
    max_frames: int | None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield frames from cv2.VideoCapture (video file, stream, or webcam)."""
    cv2 = _require_cv2()

    if source_type == "video_file":
        source_arg: str | int = str(source)
        path = Path(source_arg)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {source_arg}")
    elif source_type == "webcam":
        source_arg = int(source)
    else:
        # stream
        source_arg = str(source)

    cap = cv2.VideoCapture(source_arg)
    try:
        if not cap.isOpened():
            raise ValueError(
                f"cv2.VideoCapture could not open source: {source!r}. "
                "For webcams, check the device index. "
                "For files, verify the path and codec. "
                "For streams, verify the URL is reachable."
            )

        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()
        logger.debug("VideoCapture released for source: %s", source)
