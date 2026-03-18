"""Tests for Graph.run() video/stream/webcam dispatch (v1.9.4+).

Covers:
- _classify_run_source() helper
- Graph.run() with image sources (backward compat)
- Graph.run() with video files → list[MultiResult]
- Graph.run() with stream sources → generator
- Graph.run() with stream + callback → blocking / None
- Graph.run() with webcam → generator
- frame_policy required for temporal sources
- max_frames respected
- stop_event respected for streams
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.result import MultiResult
from mata.core.graph.graph import Graph, _classify_run_source
from mata.core.graph.temporal import (
    FramePolicyEveryN,
    FramePolicyLatest,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_frame(h: int = 60, w: int = 80) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_multi_result() -> MultiResult:
    arr = np.zeros((60, 80, 3), dtype=np.uint8)
    img = Image.from_numpy(arr, color_space="RGB")
    return MultiResult(channels={"image": img}, provenance={}, metrics={})


class _FakeCapture:
    """Minimal cv2.VideoCapture stub."""

    def __init__(self, num_frames: int = 6, fps: float = 30.0):
        self._frames = [_make_frame() for _ in range(num_frames)]
        self._idx = 0
        self._opened = True
        self._fps = fps

    def isOpened(self):  # noqa: N802
        return self._opened

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def get(self, prop_id):
        if prop_id == 5:  # CAP_PROP_FPS
            return self._fps
        if prop_id == 7:  # CAP_PROP_FRAME_COUNT
            return float(len(self._frames))
        return 0.0

    def release(self):
        self._opened = False


@pytest.fixture
def minimal_graph():
    """A Graph with no nodes — sufficient for mock-based tests."""
    from mata.core.graph.node import Node

    class _EchoNode(Node):
        inputs: dict[str, type[Artifact]] = {"image": Image}
        outputs: dict[str, type[Artifact]] = {}

        def __init__(self):
            super().__init__(name="Echo")

        def run(self, ctx, image: Image, **kw):
            return {"result": image}

    return Graph("test_video_graph").add(_EchoNode())


@pytest.fixture
def mock_processor():
    """A MagicMock VideoProcessor whose process_* return sensible data."""
    proc = MagicMock()
    # process_video returns a list
    proc.process_video.return_value = [_make_multi_result() for _ in range(4)]

    # process_stream calls callback once then returns
    def _fake_stream(source, callback, stop_event=None, max_frames=None):
        callback(_make_multi_result(), 0)

    proc.process_stream.side_effect = _fake_stream
    return proc


# ---------------------------------------------------------------------------
# _classify_run_source
# ---------------------------------------------------------------------------


class TestClassifyRunSource:
    def test_integer_is_webcam(self):
        assert _classify_run_source(0) == "webcam"

    def test_large_integer_is_webcam(self):
        assert _classify_run_source(2) == "webcam"

    def test_numpy_array_is_image(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        assert _classify_run_source(arr) == "image"

    def test_pil_like_is_image(self):
        mock_pil = MagicMock()
        mock_pil.save = MagicMock()
        mock_pil.tobytes = MagicMock()
        assert _classify_run_source(mock_pil) == "image"

    def test_rtsp_is_stream(self):
        assert _classify_run_source("rtsp://192.168.1.1/stream") == "stream"

    def test_rtsps_is_stream(self):
        assert _classify_run_source("rtsps://cam/feed") == "stream"

    def test_rtmp_is_stream(self):
        assert _classify_run_source("rtmp://live.example.com/app/key") == "stream"

    def test_http_is_stream(self):
        assert _classify_run_source("http://cam.example.com/mjpg") == "stream"

    def test_https_is_stream(self):
        assert _classify_run_source("https://cam.example.com/stream") == "stream"

    def test_mp4_is_video_file(self):
        assert _classify_run_source("video.mp4") == "video_file"

    def test_avi_is_video_file(self):
        assert _classify_run_source("clip.avi") == "video_file"

    def test_mov_is_video_file(self):
        assert _classify_run_source("footage.mov") == "video_file"

    def test_mkv_is_video_file(self):
        assert _classify_run_source("movie.mkv") == "video_file"

    def test_ts_is_video_file(self):
        assert _classify_run_source("segment.ts") == "video_file"

    def test_flv_is_video_file(self):
        assert _classify_run_source("old.flv") == "video_file"

    def test_jpg_is_image(self):
        assert _classify_run_source("photo.jpg") == "image"

    def test_png_is_image(self):
        assert _classify_run_source("frame.png") == "image"

    def test_path_object_jpg_is_image(self):
        from pathlib import Path

        assert _classify_run_source(Path("photo.jpg")) == "image"

    def test_path_object_mp4_is_video(self):
        from pathlib import Path

        assert _classify_run_source(Path("clip.mp4")) == "video_file"

    def test_unknown_extension_treated_as_image(self):
        # files without video extension → image (let infer() fail gracefully)
        assert _classify_run_source("file.xyz") == "image"

    def test_case_insensitive_extension(self):
        assert _classify_run_source("VIDEO.MP4") == "video_file"

    def test_stream_case_insensitive(self):
        assert _classify_run_source("RTSP://cam/stream") == "stream"


# ---------------------------------------------------------------------------
# Graph.run() — single image (backward compatibility)
# ---------------------------------------------------------------------------


class TestGraphRunImage:
    def test_image_path_delegates_to_infer(self, minimal_graph):
        expected = _make_multi_result()
        with patch("mata.api.infer", return_value=expected) as mock_infer:
            result = minimal_graph.run("photo.jpg", providers={})
        mock_infer.assert_called_once()
        assert result is expected

    def test_numpy_image_delegates_to_infer(self, minimal_graph):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        expected = _make_multi_result()
        with patch("mata.api.infer", return_value=expected) as mock_infer:
            result = minimal_graph.run(arr, providers={})
        mock_infer.assert_called_once()
        assert result is expected

    def test_extra_kwargs_forwarded_to_infer(self, minimal_graph):
        expected = _make_multi_result()
        with patch("mata.api.infer", return_value=expected) as mock_infer:
            minimal_graph.run("photo.jpg", providers={}, device="cpu", custom_arg=42)
        _, call_kwargs = mock_infer.call_args
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["custom_arg"] == 42


# ---------------------------------------------------------------------------
# Graph.run() — video file
# ---------------------------------------------------------------------------


class TestGraphRunVideoFile:
    def _run_with_mock_processor(self, graph, path, **kwargs):
        """Helper: patch VideoProcessor so no real cv2 needed."""
        fake_results = [_make_multi_result() for _ in range(4)]

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            instance = MagicMock()
            instance.process_video.return_value = fake_results
            mock_vp.return_value = instance
            result = graph.run(path, providers={}, frame_policy=FramePolicyEveryN(n=1), **kwargs)

        return result, mock_vp, instance

    def test_returns_list(self, minimal_graph):
        results, _, _ = self._run_with_mock_processor(minimal_graph, "clip.mp4")
        assert isinstance(results, list)
        assert len(results) == 4

    def test_video_processor_constructed_with_policy(self, minimal_graph):
        policy = FramePolicyEveryN(n=2)
        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.return_value = []
            minimal_graph.run("clip.mp4", providers={}, frame_policy=policy)
        _, kwargs = mock_vp.call_args
        assert kwargs["frame_policy"] is policy

    def test_max_frames_forwarded(self, minimal_graph):
        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.return_value = []
            minimal_graph.run("clip.mp4", providers={}, frame_policy=FramePolicyEveryN(n=1), max_frames=10)

        instance = mock_vp.return_value
        _, pkwargs = instance.process_video.call_args
        assert pkwargs.get("max_frames") == 10

    def test_output_path_forwarded(self, minimal_graph):
        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.return_value = []
            minimal_graph.run(
                "clip.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
                output_path="out.mp4",
            )

        instance = mock_vp.return_value
        _, pkwargs = instance.process_video.call_args
        assert pkwargs.get("output_path") == "out.mp4"

    def test_missing_frame_policy_raises(self, minimal_graph):
        with pytest.raises(ValueError, match="frame_policy is required"):
            minimal_graph.run("clip.mp4", providers={})

    def test_frame_policy_none_raises(self, minimal_graph):
        with pytest.raises(ValueError, match="frame_policy is required"):
            minimal_graph.run("clip.mp4", providers={}, frame_policy=None)

    # -- callback forwarding tests (Task E1) ---------------------------------

    def test_video_file_callback_forwarded(self, minimal_graph):
        """Graph.run() passes the callback down to process_video."""
        cb = MagicMock()
        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.return_value = []
            minimal_graph.run(
                "clip.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
                callback=cb,
            )

        instance = mock_vp.return_value
        _, pkwargs = instance.process_video.call_args
        assert pkwargs.get("callback") is cb

    def test_video_file_callback_invoked(self, minimal_graph):
        """Callback actually fires for video files via Graph.run()."""
        fake_result = _make_multi_result()
        fake_frame = np.zeros((60, 80, 3), dtype=np.uint8)
        received: list = []

        def _fake_process_video(path, output_path=None, max_frames=None, callback=None):
            if callback is not None:
                callback(fake_result, 0, fake_frame)
            return [fake_result]

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.side_effect = _fake_process_video
            minimal_graph.run(
                "clip.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
                callback=lambda r, n, f: received.append((r, n, f)),
            )

        assert len(received) == 1
        assert received[0][0] is fake_result
        assert received[0][1] == 0
        assert received[0][2] is fake_frame

    def test_video_file_callback_none_returns_list(self, minimal_graph):
        """No callback → returns list[MultiResult] (existing behaviour)."""
        results, _, _ = self._run_with_mock_processor(minimal_graph, "clip.mp4")
        assert isinstance(results, list)
        assert len(results) == 4

    def test_video_file_callback_returns_list_alongside(self, minimal_graph):
        """With callback, Graph.run() still returns the list[MultiResult]."""
        fake_result = _make_multi_result()
        cb_calls: list = []

        def _fake_process_video(path, output_path=None, max_frames=None, callback=None):
            if callback is not None:
                callback(fake_result, 0, np.zeros((60, 80, 3), dtype=np.uint8))
            return [fake_result]

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.side_effect = _fake_process_video
            result = minimal_graph.run(
                "clip.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
                callback=lambda r, n, f: cb_calls.append(r),
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(cb_calls) == 1
        assert result[0] is cb_calls[0]


# ---------------------------------------------------------------------------
# Graph.run() — stream (generator mode, no callback)
# ---------------------------------------------------------------------------


class TestGraphRunStreamGenerator:
    def test_returns_generator_for_rtsp(self, minimal_graph):
        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor"),
            patch("mata.core.graph.graph._stream_generator") as mock_gen,
        ):
            mock_gen.return_value = iter([])
            result = minimal_graph.run(
                "rtsp://cam/stream",
                providers={},
                frame_policy=FramePolicyLatest(),
            )
        mock_gen.assert_called_once()
        assert result is not None  # generator returned

    def test_returns_generator_for_webcam(self, minimal_graph):
        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor"),
            patch("mata.core.graph.graph._stream_generator") as mock_gen,
        ):
            mock_gen.return_value = iter([])
            minimal_graph.run(0, providers={}, frame_policy=FramePolicyLatest())
        mock_gen.assert_called_once()

    def test_stream_missing_frame_policy_raises(self, minimal_graph):
        with pytest.raises(ValueError, match="frame_policy is required"):
            minimal_graph.run("rtsp://cam/stream", providers={})

    def test_infer_not_called_for_stream(self, minimal_graph):
        with (
            patch("mata.api.infer") as mock_infer,
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor"),
            patch("mata.core.graph.graph._stream_generator", return_value=iter([])),
        ):
            minimal_graph.run("rtsp://cam/stream", providers={}, frame_policy=FramePolicyLatest())
        mock_infer.assert_not_called()


# ---------------------------------------------------------------------------
# Graph.run() — stream with callback (blocking mode)
# ---------------------------------------------------------------------------


class TestGraphRunStreamCallback:
    def test_callback_mode_returns_none(self, minimal_graph):
        received: list = []
        fake_result = _make_multi_result()

        def _fake_stream(source, callback, stop_event=None, max_frames=None):
            callback(fake_result, 0)

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_stream.side_effect = _fake_stream

            ret = minimal_graph.run(
                "rtsp://cam/stream",
                providers={},
                frame_policy=FramePolicyLatest(),
                callback=lambda r, n: received.append(r),
            )

        assert ret is None
        assert len(received) == 1
        assert received[0] is fake_result

    def test_stop_event_forwarded(self, minimal_graph):
        stop = threading.Event()

        def _fake_stream(source, callback, stop_event=None, max_frames=None):
            pass

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_stream.side_effect = _fake_stream
            minimal_graph.run(
                "rtsp://cam/stream",
                providers={},
                frame_policy=FramePolicyLatest(),
                callback=lambda r, n: None,
                stop_event=stop,
            )
            _, call_kw = mock_vp.return_value.process_stream.call_args
            assert call_kw["stop_event"] is stop

    def test_max_frames_forwarded_to_stream(self, minimal_graph):
        def _fake_stream(source, callback, stop_event=None, max_frames=None):
            pass

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_stream.side_effect = _fake_stream
            minimal_graph.run(
                "rtsp://cam/stream",
                providers={},
                frame_policy=FramePolicyLatest(),
                callback=lambda r, n: None,
                max_frames=100,
            )
            _, call_kw = mock_vp.return_value.process_stream.call_args
            assert call_kw["max_frames"] == 100

    def test_stream_generator_not_called_when_callback_given(self, minimal_graph):
        def _fake_stream(source, callback, stop_event=None, max_frames=None):
            pass

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
            patch("mata.core.graph.graph._stream_generator") as mock_gen,
        ):
            mock_vp.return_value.process_stream.side_effect = _fake_stream
            minimal_graph.run(
                "rtsp://cam/stream",
                providers={},
                frame_policy=FramePolicyLatest(),
                callback=lambda r, n: None,
            )
        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# _stream_generator integration (uses real threading)
# ---------------------------------------------------------------------------


class TestStreamGenerator:
    """Tests for _stream_generator function."""

    def test_yields_items_from_process_stream(self):
        from mata.core.graph.graph import _stream_generator

        fake_results = [_make_multi_result(), _make_multi_result()]
        collected: list = []

        proc = MagicMock()

        def _fake_stream(source, callback, stop_event=None, max_frames=None):
            for r in fake_results:
                callback(r, 0)

        proc.process_stream.side_effect = _fake_stream

        for item in _stream_generator(proc, "rtsp://cam/stream", max_frames=None):
            collected.append(item)

        assert len(collected) == 2
        assert collected[0] is fake_results[0]
        assert collected[1] is fake_results[1]

    def test_max_frames_forwarded(self):
        from mata.core.graph.graph import _stream_generator

        proc = MagicMock()

        def _fake_stream(source, callback, stop_event=None, max_frames=None):
            pass

        proc.process_stream.side_effect = _fake_stream

        list(_stream_generator(proc, "rtsp://cam/stream", max_frames=5))

        _, call_kw = proc.process_stream.call_args
        assert call_kw["max_frames"] == 5

    def test_empty_stream_yields_nothing(self):
        from mata.core.graph.graph import _stream_generator

        proc = MagicMock()
        proc.process_stream.side_effect = lambda src, callback, stop_event=None, max_frames=None: None

        results = list(_stream_generator(proc, "rtsp://cam/stream", max_frames=None))
        assert results == []


# ---------------------------------------------------------------------------
# Scheduler propagation
# ---------------------------------------------------------------------------


class TestSchedulerPropagation:
    def test_custom_scheduler_forwarded_to_processor(self, minimal_graph):
        from mata.core.graph import SyncScheduler

        sched = SyncScheduler()
        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(minimal_graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.return_value = []
            minimal_graph.run(
                "clip.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
                scheduler=sched,
            )
        _, call_kw = mock_vp.call_args
        assert call_kw["scheduler"] is sched
