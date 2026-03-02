"""Temporal/video support for graph system.

Provides video file and real-time stream processing with configurable
frame policies and temporal window buffering.

Classes:
    FramePolicy: Abstract base for frame selection strategies.
    FramePolicyEveryN: Process every N-th frame.
    FramePolicyLatest: Drop old frames when processing is slow (real-time).
    FramePolicyQueue: Queue up to N frames.
    VideoProcessor: Process video files and streams with graph execution.
    Window: Node that buffers N frames for temporal operations.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.result import MultiResult
from mata.core.graph.node import Node
from mata.core.logging import get_logger

# Optional OpenCV import — needed only at runtime for video I/O
try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext
    from mata.core.graph.graph import CompiledGraph
    from mata.core.graph.scheduler import SyncScheduler

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Frame Policies
# ---------------------------------------------------------------------------


class FramePolicy(ABC):
    """Abstract base class for frame selection policies.

    A frame policy decides which frames from a video source should be
    processed by the graph.  Implementations must provide
    :meth:`should_process` and may optionally override :meth:`reset`.
    """

    @abstractmethod
    def should_process(self, frame_num: int) -> bool:
        """Return ``True`` if *frame_num* should be processed.

        Args:
            frame_num: Zero-based frame index.

        Returns:
            Whether the frame should be processed.
        """
        ...

    def reset(self) -> None:
        """Reset internal state (e.g. between videos)."""
        pass


class FramePolicyEveryN(FramePolicy):
    """Process every N-th frame.

    Useful for reducing workload on longer videos where real-time speed is
    not required but covering the full timeline is desirable.

    Args:
        n: Process every *n*-th frame.  Must be >= 1.

    Example::

        policy = FramePolicyEveryN(n=5)
        assert policy.should_process(0)   # True – 0 % 5 == 0
        assert not policy.should_process(1)
        assert policy.should_process(5)   # True
    """

    def __init__(self, n: int = 5):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n

    def should_process(self, frame_num: int) -> bool:
        return frame_num % self.n == 0

    def reset(self) -> None:
        pass  # Stateless


class FramePolicyLatest(FramePolicy):
    """Drop old frames if processing is slow (real-time).

    Only the most recently offered frame is kept; earlier frames that
    haven't been consumed are discarded.  This is suitable for live
    camera / RTSP feeds where latency matters more than coverage.

    The policy tracks the *latest* frame number that was offered and
    always returns ``True`` so the caller can decide to skip older ones.
    Typically used together with :class:`VideoProcessor` stream mode
    which handles the drop logic externally.
    """

    def __init__(self):
        self._latest: int = -1

    def should_process(self, frame_num: int) -> bool:
        self._latest = frame_num
        return True

    def reset(self) -> None:
        self._latest = -1

    @property
    def latest_frame(self) -> int:
        """Return latest offered frame number."""
        return self._latest


class FramePolicyQueue(FramePolicy):
    """Queue up to *max_queue* frames.

    Frames are accepted until the internal queue reaches capacity.  Once
    full, subsequent frames are dropped (newest is discarded) to bound
    memory usage.

    Args:
        max_queue: Maximum number of queued frames.  Must be >= 1.
    """

    def __init__(self, max_queue: int = 10):
        if max_queue < 1:
            raise ValueError(f"max_queue must be >= 1, got {max_queue}")
        self.max_queue = max_queue
        self._count: int = 0

    def should_process(self, frame_num: int) -> bool:
        if self._count < self.max_queue:
            self._count += 1
            return True
        return False

    def mark_processed(self) -> None:
        """Notify the policy that one frame has been consumed.

        Call this after the graph finishes processing a frame so the
        policy can free a slot for the next frame.
        """
        if self._count > 0:
            self._count -= 1

    def reset(self) -> None:
        self._count = 0


# ---------------------------------------------------------------------------
# Video Processor
# ---------------------------------------------------------------------------


class VideoProcessor:
    """Process video files and real-time streams with graph execution.

    ``VideoProcessor`` drives a compiled graph over successive video
    frames while respecting a :class:`FramePolicy` for frame selection.

    Args:
        graph: Compiled graph to execute per frame.
        providers: Provider dictionary (passed to context creation).
        frame_policy: Policy controlling which frames are processed.
        scheduler: Scheduler instance.  Defaults to :class:`SyncScheduler`.

    Example::

        from mata.core.graph import Graph, SyncScheduler
        from mata.core.graph.temporal import VideoProcessor, FramePolicyEveryN

        compiled = graph.compile(providers=...)
        vp = VideoProcessor(
            graph=compiled,
            providers=providers,
            frame_policy=FramePolicyEveryN(n=3),
        )
        results = vp.process_video("input.mp4")
    """

    def __init__(
        self,
        graph: CompiledGraph,
        providers: dict[str, Any],
        frame_policy: FramePolicy,
        scheduler: SyncScheduler | None = None,
    ):
        self.graph = graph
        self.providers = providers
        self.frame_policy = frame_policy

        # Lazy import to avoid circular dependency
        if scheduler is None:
            from mata.core.graph.scheduler import SyncScheduler

            scheduler = SyncScheduler()
        self.scheduler = scheduler

    # ---- public helpers ---------------------------------------------------

    def _make_context(self) -> ExecutionContext:
        """Create a fresh execution context for one frame."""
        from mata.core.graph.context import ExecutionContext

        return ExecutionContext(providers=self.providers)

    def _frame_to_image(self, frame: np.ndarray, frame_num: int, timestamp_ms: int | None = None) -> Image:
        """Convert an OpenCV BGR numpy frame to an Image artifact."""
        return Image.from_numpy(
            frame,
            color_space="BGR",
            frame_id=f"frame_{frame_num:06d}",
            timestamp_ms=timestamp_ms,
        )

    # ---- video file -------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_path: str | None = None,
        max_frames: int | None = None,
    ) -> list[MultiResult]:
        """Process a video file frame-by-frame.

        Reads frames using OpenCV, applies the frame policy, and executes
        the compiled graph on each selected frame.

        Args:
            video_path: Filesystem path to the video file.
            output_path: (Reserved) Optional path for annotated output video.
            max_frames: Stop after this many *total* frames (including skipped).
                        ``None`` means read until EOF.

        Returns:
            List of :class:`MultiResult` objects, one per processed frame.

        Raises:
            ImportError: If OpenCV (``cv2``) is not installed.
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: If the video cannot be opened.
        """
        if _cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required for video processing. " "Install with: pip install opencv-python"
            )

        import os

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = _cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # cv2.CAP_PROP_FPS = 5, cv2.CAP_PROP_FRAME_COUNT = 7
        fps = cap.get(5) or 30.0
        total_frames = int(cap.get(7))

        logger.info(f"Processing video '{video_path}' " f"({total_frames} frames, {fps:.1f} FPS)")

        self.frame_policy.reset()
        results: list[MultiResult] = []
        frame_num = 0
        processed_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames is not None and frame_num >= max_frames:
                    break

                if self.frame_policy.should_process(frame_num):
                    timestamp_ms = int((frame_num / fps) * 1000)
                    image = self._frame_to_image(frame, frame_num, timestamp_ms)

                    ctx = self._make_context()
                    result = self.scheduler.execute(
                        self.graph,
                        ctx,
                        {"input.image": image},
                    )
                    results.append(result)
                    processed_count += 1

                    # Notify queue policy if applicable
                    if hasattr(self.frame_policy, "mark_processed"):
                        self.frame_policy.mark_processed()

                frame_num += 1
        finally:
            cap.release()

        elapsed = time.time() - start_time
        avg_fps = processed_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"Processed {processed_count}/{frame_num} frames in {elapsed:.2f}s " f"({avg_fps:.1f} processed FPS)"
        )

        return results

    # ---- real-time stream -------------------------------------------------

    def process_stream(
        self,
        source: str | int,
        callback: Callable[[MultiResult, int], None],
        stop_event: threading.Event | None = None,
        max_frames: int | None = None,
    ) -> None:
        """Process a real-time stream (RTSP URL or camera index).

        Reads frames in a loop and invokes *callback* with each result.
        The loop continues until *stop_event* is set, *max_frames* is
        reached, or the stream ends.

        For :class:`FramePolicyLatest`, frames are dropped transparently
        so the callback always receives the freshest available result.

        Args:
            source: RTSP URL (``str``) or camera device index (``int``).
            callback: Called as ``callback(result, frame_num)`` for every
                      processed frame.
            stop_event: Optional :class:`threading.Event`.  Set it to stop
                        processing gracefully.
            max_frames: Maximum number of frames to read.  ``None`` = unlimited.

        Raises:
            ImportError: If OpenCV (``cv2``) is not installed.
            RuntimeError: If the stream cannot be opened.
        """
        if _cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required for stream processing. " "Install with: pip install opencv-python"
            )

        if stop_event is None:
            stop_event = threading.Event()

        cap = _cv2.VideoCapture(source if isinstance(source, str) else int(source))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open stream: {source}")

        fps = cap.get(5) or 30.0  # cv2.CAP_PROP_FPS = 5
        self.frame_policy.reset()
        frame_num = 0

        logger.info(f"Starting stream processing from '{source}'")

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames is not None and frame_num >= max_frames:
                    break

                if self.frame_policy.should_process(frame_num):
                    timestamp_ms = int((frame_num / fps) * 1000)
                    image = self._frame_to_image(frame, frame_num, timestamp_ms)

                    ctx = self._make_context()
                    result = self.scheduler.execute(
                        self.graph,
                        ctx,
                        {"input.image": image},
                    )
                    callback(result, frame_num)

                    if hasattr(self.frame_policy, "mark_processed"):
                        self.frame_policy.mark_processed()

                frame_num += 1
        finally:
            cap.release()
            logger.info(f"Stream processing stopped after {frame_num} frames")


# ---------------------------------------------------------------------------
# Window Node (temporal buffering)
# ---------------------------------------------------------------------------


class Window(Node):
    """Buffer the last *n* frames for temporal operations.

    This node maintains an internal ring-buffer of :class:`Image` artifacts
    and emits the current buffer contents on each invocation.  Downstream
    nodes can use the buffered images for temporal reasoning (e.g.
    action recognition, optical flow, video classification).

    .. note::
        Because :class:`Node` requires **stateless** execution per the
        frozen-artifact design, the buffer is stored on the node instance
        itself (mutable state).  Callers should create one ``Window`` per
        video sequence and call :meth:`reset` between sequences.

    Args:
        n: Buffer capacity (number of frames to keep).  Must be >= 1.
        image_src: Artifact name for the input image (default ``"image"``).
        out: Output artifact name (default ``"images"``).
        name: Optional human-readable node name.

    Inputs:
        image (Image): Current frame image.

    Outputs:
        images (list[Image]): Copy of the current buffer (up to *n* images).

    Example::

        window = Window(n=8, out="frame_buffer")
        # After 3 calls the output will contain 3 images
        # After 10 calls it will contain the 8 most recent
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {}  # Dynamic – returns list of Image

    def __init__(
        self,
        n: int = 8,
        image_src: str = "image",
        out: str = "images",
        name: str | None = None,
    ):
        if n < 1:
            raise ValueError(f"Window size n must be >= 1, got {n}")
        super().__init__(name=name or "Window")
        self.n = n
        self.image_src = image_src
        self.output_name = out
        self._buffer: list[Image] = []

    def run(self, ctx: ExecutionContext, image: Image, **kwargs: Any) -> dict[str, Any]:
        """Append *image* to the buffer and return the current window.

        Args:
            ctx: Execution context (unused but required by interface).
            image: Current frame image artifact.

        Returns:
            Dict with a single key (*out*) mapping to a list of up to
            *n* :class:`Image` artifacts.
        """
        self._buffer.append(image)
        if len(self._buffer) > self.n:
            self._buffer.pop(0)
        return {self.output_name: list(self._buffer)}

    def reset(self) -> None:
        """Clear the frame buffer."""
        self._buffer.clear()

    @property
    def buffer_size(self) -> int:
        """Current number of buffered frames."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached capacity."""
        return len(self._buffer) >= self.n
