"""Unit tests for mata.track() public API (Task B3).

Tests cover:
- Source type detection (_detect_source_type)
- Output directory creation (_make_output_dir)
- Single image input (PIL, numpy, file path)
- Image directory input
- Video file input via mocked cv2.VideoCapture
- Webcam input (integer source)
- RTSP stream input
- stream=True returns Generator, stream=False returns list
- persist= forwarded to adapter.update()
- conf/iou/classes forwarded to adapter.update()
- max_frames limits frame count
- show/save with mocked cv2
- 'q' key exits display loop
- Error handling: missing video file, empty image dir, unsupported source
- mata.track accessible from top-level import
- _annotate_frame_cv2 draws bboxes and labels
- _track_color returns deterministic BGR tuples
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

import mata
from mata.api import (
    _annotate_frame_cv2,
    _detect_source_type,
    _make_output_dir,
    _track_color,
)
from mata.core.types import Instance, VisionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(
    x1=10.0,
    y1=20.0,
    x2=110.0,
    y2=120.0,
    score=0.85,
    label=0,
    label_name="cat",
    track_id=1,
) -> Instance:
    return Instance(
        bbox=(x1, y1, x2, y2),
        score=score,
        label=label,
        label_name=label_name,
        track_id=track_id,
    )


def _make_result(*instances) -> VisionResult:
    return VisionResult(instances=list(instances))


def _make_adapter_mock(results=None) -> MagicMock:
    """Mock TrackingAdapter whose update() returns VisionResults in sequence."""
    mock = MagicMock()
    if results is None:
        results = [_make_result(_make_instance())]
    mock.update.side_effect = results
    return mock


# ---------------------------------------------------------------------------
# _detect_source_type
# ---------------------------------------------------------------------------


class TestDetectSourceType:
    def test_integer_is_webcam(self):
        assert _detect_source_type(0) == "webcam"
        assert _detect_source_type(2) == "webcam"

    def test_numpy_array(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        assert _detect_source_type(arr) == "numpy_array"

    def test_pil_image(self):
        img = PILImage.new("RGB", (100, 100))
        assert _detect_source_type(img) == "pil_image"

    def test_rtsp_stream(self):
        assert _detect_source_type("rtsp://192.168.1.1/stream") == "stream"

    def test_rtmp_stream(self):
        assert _detect_source_type("rtmp://server/live") == "stream"

    def test_http_stream(self):
        assert _detect_source_type("http://example.com/feed.mjpg") == "stream"

    def test_https_stream(self):
        assert _detect_source_type("https://example.com/cam") == "stream"

    def test_video_file_mp4(self, tmp_path):
        f = tmp_path / "test.mp4"
        f.touch()
        assert _detect_source_type(str(f)) == "video_file"

    def test_video_file_avi(self, tmp_path):
        f = tmp_path / "clip.avi"
        f.touch()
        assert _detect_source_type(str(f)) == "video_file"

    def test_video_file_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _detect_source_type(str(tmp_path / "ghost.mp4"))

    def test_image_file_jpg(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.touch()
        assert _detect_source_type(str(f)) == "image_file"

    def test_image_file_png(self, tmp_path):
        f = tmp_path / "photo.png"
        f.touch()
        assert _detect_source_type(str(f)) == "image_file"

    def test_image_file_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _detect_source_type(str(tmp_path / "ghost.png"))

    def test_image_directory(self, tmp_path):
        assert _detect_source_type(str(tmp_path)) == "image_dir"


# ---------------------------------------------------------------------------
# _make_output_dir
# ---------------------------------------------------------------------------


class TestMakeOutputDir:
    def test_creates_exp1_first(self, tmp_path):
        base = str(tmp_path / "runs" / "track")
        d = _make_output_dir(base)
        assert d.name == "exp1"
        assert d.exists()

    def test_increments_to_exp2(self, tmp_path):
        base = str(tmp_path / "runs" / "track")
        d1 = _make_output_dir(base)
        d2 = _make_output_dir(base)
        assert d1.name == "exp1"
        assert d2.name == "exp2"

    def test_custom_name(self, tmp_path):
        base = str(tmp_path / "out")
        d = _make_output_dir(base, name="run")
        assert d.name == "run1"


# ---------------------------------------------------------------------------
# _track_color
# ---------------------------------------------------------------------------


class TestTrackColor:
    def test_returns_three_tuple(self):
        color = _track_color(1)
        assert len(color) == 3

    def test_values_in_range(self):
        for tid in range(20):
            b, g, r = _track_color(tid)
            assert 0 <= b <= 255
            assert 0 <= g <= 255
            assert 0 <= r <= 255

    def test_deterministic(self):
        assert _track_color(42) == _track_color(42)

    def test_different_ids_give_different_colors(self):
        colors = {_track_color(tid) for tid in range(50)}
        assert len(colors) > 20  # most ids differ


# ---------------------------------------------------------------------------
# _annotate_frame_cv2
# ---------------------------------------------------------------------------


class TestAnnotateFrameCv2:
    """Verify annotation helper draws correctly when cv2 is available."""

    def _make_frame(self, h=200, w=200):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_returns_array_same_shape(self):
        pytest.importorskip("cv2")
        frame = self._make_frame()
        result = _make_result(_make_instance())
        out = _annotate_frame_cv2(frame, result, True, None, 30)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_does_not_modify_original(self):
        pytest.importorskip("cv2")
        frame = self._make_frame()
        original = frame.copy()
        result = _make_result(_make_instance())
        _annotate_frame_cv2(frame, result, True, None, 30)
        np.testing.assert_array_equal(frame, original)

    def test_empty_result_returns_unchanged(self):
        pytest.importorskip("cv2")
        frame = self._make_frame()
        result = _make_result()
        out = _annotate_frame_cv2(frame, result, True, None, 30)
        np.testing.assert_array_equal(out, frame)

    def test_trail_history_updated(self):
        pytest.importorskip("cv2")
        frame = self._make_frame()
        result = _make_result(_make_instance(track_id=7))
        history: dict = {}
        _annotate_frame_cv2(frame, result, True, history, 30)
        assert 7 in history
        assert len(history[7]) == 1

    def test_trail_capped_at_trail_length(self):
        pytest.importorskip("cv2")
        frame = self._make_frame()
        history: dict = {5: [(i, i) for i in range(50)]}
        result = _make_result(_make_instance(track_id=5))
        _annotate_frame_cv2(frame, result, True, history, 30)
        assert len(history[5]) <= 30

    def test_no_track_id_skips_trail(self):
        pytest.importorskip("cv2")
        frame = self._make_frame()
        inst = Instance(bbox=(10, 10, 50, 50), score=0.9, label=0, label_name="dog", track_id=None)
        result = _make_result(inst)
        history: dict = {}
        _annotate_frame_cv2(frame, result, True, history, 30)
        assert len(history) == 0

    def test_show_track_ids_false_omits_hash(self):
        """Smoke test: with show_track_ids=False the function runs without error."""
        pytest.importorskip("cv2")
        frame = self._make_frame()
        result = _make_result(_make_instance(track_id=3))
        out = _annotate_frame_cv2(frame, result, False, None, 30)
        assert out.shape == frame.shape


# ---------------------------------------------------------------------------
# mata.track — top-level import
# ---------------------------------------------------------------------------


def test_track_accessible_from_mata():
    assert hasattr(mata, "track")
    assert callable(mata.track)


# ---------------------------------------------------------------------------
# stream=False returns list, stream=True returns generator
# ---------------------------------------------------------------------------


class TestTrackReturnType:
    """stream= controls whether we get a list or a generator."""

    def _patch_load(self, adapter):
        return patch("mata.api.load", return_value=adapter)

    def test_stream_false_returns_list(self, tmp_path):
        img = tmp_path / "a.jpg"
        PILImage.new("RGB", (100, 100)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            result = mata.track(str(img), stream=False)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_stream_true_returns_generator(self, tmp_path):
        img = tmp_path / "a.jpg"
        PILImage.new("RGB", (100, 100)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            result = mata.track(str(img), stream=True)
        assert isinstance(result, Generator)
        frames = list(result)
        assert len(frames) == 1

    def test_single_image_returns_one_result(self, tmp_path):
        img = tmp_path / "photo.png"
        PILImage.new("RGB", (64, 64)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            results = mata.track(str(img))
        assert len(results) == 1
        assert isinstance(results[0], VisionResult)


# ---------------------------------------------------------------------------
# Single image sources
# ---------------------------------------------------------------------------


class TestSingleImageSources:
    def _patch_load(self, adapter):
        return patch("mata.api.load", return_value=adapter)

    def test_pil_image_input(self):
        pil_img = PILImage.new("RGB", (80, 80), color=(128, 0, 0))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            results = mata.track(pil_img)
        assert len(results) == 1
        call_args = adapter.update.call_args
        # First positional arg is the frame
        assert isinstance(call_args[0][0], PILImage.Image)

    def test_numpy_array_input(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            results = mata.track(arr)
        assert len(results) == 1

    def test_image_file_path_input(self, tmp_path):
        img_path = tmp_path / "frame.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img_path))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            results = mata.track(str(img_path))
        assert len(results) == 1
        assert results[0].meta["frame_idx"] == 0

    def test_missing_image_file_raises(self, tmp_path):
        with patch("mata.api.load", return_value=MagicMock()):
            with pytest.raises(FileNotFoundError):
                mata.track(str(tmp_path / "ghost.jpg"))


# ---------------------------------------------------------------------------
# Image directory
# ---------------------------------------------------------------------------


class TestImageDirectory:
    def _patch_load(self, adapter):
        return patch("mata.api.load", return_value=adapter)

    def _make_dir_with_images(self, tmp_path, n=3):
        for i in range(n):
            PILImage.new("RGB", (50, 50)).save(str(tmp_path / f"frame_{i:04d}.jpg"))
        return tmp_path

    def test_processes_all_images(self, tmp_path):
        d = self._make_dir_with_images(tmp_path, 3)
        results = [_make_result(_make_instance(track_id=i + 1)) for i in range(3)]
        adapter = _make_adapter_mock(results)
        with self._patch_load(adapter):
            out = mata.track(str(d))
        assert len(out) == 3
        assert adapter.update.call_count == 3

    def test_frame_idx_in_meta(self, tmp_path):
        d = self._make_dir_with_images(tmp_path, 2)
        results = [_make_result(_make_instance()) for _ in range(2)]
        adapter = _make_adapter_mock(results)
        with self._patch_load(adapter):
            out = mata.track(str(d))
        assert out[0].meta["frame_idx"] == 0
        assert out[1].meta["frame_idx"] == 1

    def test_image_path_in_meta(self, tmp_path):
        d = self._make_dir_with_images(tmp_path, 1)
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            out = mata.track(str(d))
        assert "image_path" in out[0].meta

    def test_max_frames_limits_directory(self, tmp_path):
        d = self._make_dir_with_images(tmp_path, 5)
        results = [_make_result(_make_instance()) for _ in range(5)]
        adapter = _make_adapter_mock(results)
        with self._patch_load(adapter):
            out = mata.track(str(d), max_frames=2)
        assert len(out) == 2

    def test_empty_directory_raises(self, tmp_path):
        # No image files in dir
        adapter = MagicMock()
        with self._patch_load(adapter):
            with pytest.raises(ValueError, match="No images found"):
                mata.track(str(tmp_path))


# ---------------------------------------------------------------------------
# Video / webcam / stream (mocked cv2)
# ---------------------------------------------------------------------------


def _make_cv2_mock(frames_bgr: list[np.ndarray], fps=30.0, width=320, height=240) -> MagicMock:
    """Build a minimal cv2 module mock for VideoCapture."""
    cv2 = MagicMock()

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        4: fps,  # CAP_PROP_FPS
        3: width,  # CAP_PROP_FRAME_WIDTH
        8: height,  # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0)

    # read() returns (True, frame) for each frame, then (False, None)
    side_effects = [(True, f) for f in frames_bgr] + [(False, None)]
    cap.read.side_effect = side_effects

    cv2.VideoCapture.return_value = cap
    cv2.COLOR_BGR2RGB = 4  # dummy constant
    cv2.COLOR_RGB2BGR = 4
    cv2.CAP_PROP_FPS = 4
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 8
    cv2.cvtColor.side_effect = lambda img, _: img  # identity
    cv2.waitKey.return_value = 0xFF & ord("a")  # not 'q'
    cv2.imshow = MagicMock()
    cv2.destroyAllWindows = MagicMock()
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.LINE_AA = 16
    cv2.FILLED = -1
    cv2.VideoWriter_fourcc = MagicMock(return_value=0x7634706D)
    cv2.getTextSize = MagicMock(return_value=((50, 12), 2))
    cv2.VideoWriter = MagicMock()

    return cv2


class TestVideoSource:
    """Tests for video file / stream / webcam sources with mocked cv2."""

    def _make_frames(self, n=3):
        return [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(n)]

    def _patch_load(self, adapter):
        return patch("mata.api.load", return_value=adapter)

    def _patch_cv2(self, cv2_mock):
        return patch.dict("sys.modules", {"cv2": cv2_mock})

    def test_video_file_returns_n_results(self, tmp_path):
        frames = self._make_frames(3)
        cv2_mock = _make_cv2_mock(frames)
        results = [_make_result(_make_instance(track_id=i + 1)) for i in range(3)]
        adapter = _make_adapter_mock(results)
        vid = tmp_path / "test.mp4"
        vid.touch()

        with self._patch_cv2(cv2_mock), self._patch_load(adapter):
            # Reload the module so it picks up mocked cv2
            out = mata.track(str(vid))

        assert len(out) == 3

    def test_video_max_frames(self, tmp_path):
        frames = self._make_frames(5)
        cv2_mock = _make_cv2_mock(frames)
        results = [_make_result(_make_instance()) for _ in range(5)]
        adapter = _make_adapter_mock(results)
        vid = tmp_path / "clip.avi"
        vid.touch()

        with self._patch_cv2(cv2_mock), self._patch_load(adapter):
            out = mata.track(str(vid), max_frames=2)

        assert len(out) == 2

    def test_webcam_integer_source(self, tmp_path):
        frames = self._make_frames(2)
        cv2_mock = _make_cv2_mock(frames)
        results = [_make_result(_make_instance()) for _ in range(2)]
        adapter = _make_adapter_mock(results)

        with self._patch_cv2(cv2_mock), self._patch_load(adapter):
            out = mata.track(0)  # webcam index

        assert len(out) == 2
        # VideoCapture should be called with integer 0
        cv2_mock.VideoCapture.assert_called_once_with(0)

    def test_rtsp_stream_source(self):
        frames = self._make_frames(2)
        cv2_mock = _make_cv2_mock(frames)
        results = [_make_result(_make_instance()) for _ in range(2)]
        adapter = _make_adapter_mock(results)

        with self._patch_cv2(cv2_mock), self._patch_load(adapter):
            out = mata.track("rtsp://cam.local/live")

        assert len(out) == 2

    def test_invalid_video_raises_value_error(self, tmp_path):
        cv2_mock = _make_cv2_mock([])
        # Make isOpened() return False
        cap = cv2_mock.VideoCapture.return_value
        cap.isOpened.return_value = False

        vid = tmp_path / "broken.mp4"
        vid.touch()

        with self._patch_cv2(cv2_mock), patch("mata.api.load", return_value=MagicMock()):
            with pytest.raises(ValueError, match="Failed to open"):
                mata.track(str(vid))

    def test_frame_idx_in_meta(self, tmp_path):
        frames = self._make_frames(3)
        cv2_mock = _make_cv2_mock(frames)
        results = [_make_result(_make_instance()) for _ in range(3)]
        adapter = _make_adapter_mock(results)
        vid = tmp_path / "movie.mp4"
        vid.touch()

        with self._patch_cv2(cv2_mock), self._patch_load(adapter):
            out = mata.track(str(vid))

        assert [r.meta["frame_idx"] for r in out] == [0, 1, 2]

    def test_cap_released_after_loop(self, tmp_path):
        frames = self._make_frames(2)
        cv2_mock = _make_cv2_mock(frames)
        cap = cv2_mock.VideoCapture.return_value

        results = [_make_result(_make_instance()) for _ in range(2)]
        adapter = _make_adapter_mock(results)
        vid = tmp_path / "vid.mp4"
        vid.touch()

        with self._patch_cv2(cv2_mock), self._patch_load(adapter):
            mata.track(str(vid))

        cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# Parameter forwarding: conf, iou, classes, persist
# ---------------------------------------------------------------------------


class TestParameterForwarding:
    def _patch_load(self, adapter):
        return patch("mata.api.load", return_value=adapter)

    def test_conf_iou_classes_forwarded(self, tmp_path):
        img = tmp_path / "t.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            mata.track(str(img), conf=0.4, iou=0.55, classes=[1, 2])

        call_kwargs = adapter.update.call_args[1]
        assert call_kwargs["conf"] == 0.4
        assert call_kwargs["iou"] == 0.55
        assert call_kwargs["classes"] == [1, 2]

    def test_persist_true_forwarded(self, tmp_path):
        img = tmp_path / "t.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            mata.track(str(img), persist=True)
        assert adapter.update.call_args[1]["persist"] is True

    def test_persist_false_forwarded(self, tmp_path):
        img = tmp_path / "t.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with self._patch_load(adapter):
            mata.track(str(img), persist=False)
        assert adapter.update.call_args[1]["persist"] is False

    def test_tracker_and_frame_rate_forwarded_to_load(self, tmp_path):
        img = tmp_path / "t.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with patch("mata.api.load", return_value=adapter) as mock_load:
            mata.track(str(img), tracker="bytetrack", frame_rate=25)

        mock_load.assert_called_once()
        _, kw = mock_load.call_args
        assert kw.get("tracker") == "bytetrack"
        assert kw.get("frame_rate") == 25

    def test_default_tracker_is_botsort(self, tmp_path):
        img = tmp_path / "t.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with patch("mata.api.load", return_value=adapter) as mock_load:
            mata.track(str(img))

        _, kw = mock_load.call_args
        assert kw.get("tracker") == "botsort"

    def test_task_is_track_in_load_call(self, tmp_path):
        img = tmp_path / "t.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with patch("mata.api.load", return_value=adapter) as mock_load:
            mata.track(str(img))

        args, _ = mock_load.call_args
        assert args[0] == "track"


# ---------------------------------------------------------------------------
# show=True with mocked cv2
# ---------------------------------------------------------------------------


class TestShowMode:
    def _make_frames(self, n=2):
        return [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(n)]

    def test_show_calls_imshow(self, tmp_path):
        frames = self._make_frames(2)
        cv2_mock = _make_cv2_mock(frames)
        results = [_make_result(_make_instance()) for _ in range(2)]
        adapter = _make_adapter_mock(results)
        vid = tmp_path / "show.mp4"
        vid.touch()

        with patch.dict("sys.modules", {"cv2": cv2_mock}), patch("mata.api.load", return_value=adapter):
            mata.track(str(vid), show=True, save=False)

        assert cv2_mock.imshow.call_count >= 1

    def test_q_key_exits_loop(self, tmp_path):
        # Simulate pressing 'q' on the first frame
        frames = self._make_frames(5)
        cv2_mock = _make_cv2_mock(frames)
        # Return 'q' key code on first waitKey call
        cv2_mock.waitKey.return_value = ord("q")
        results = [_make_result(_make_instance()) for _ in range(5)]
        adapter = _make_adapter_mock(results)
        vid = tmp_path / "quit.mp4"
        vid.touch()

        with patch.dict("sys.modules", {"cv2": cv2_mock}), patch("mata.api.load", return_value=adapter):
            out = mata.track(str(vid), show=True)

        # Should have exited after the first frame (q pressed after first waitKey)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# save=True with mocked cv2
# ---------------------------------------------------------------------------


class TestSaveMode:
    def _make_frames(self, n=2):
        return [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(n)]

    def test_save_creates_output_dir(self, tmp_path):
        frames = self._make_frames(2)
        cv2_mock = _make_cv2_mock(frames)
        mock_writer_instance = MagicMock()
        cv2_mock.VideoWriter.return_value = mock_writer_instance
        results = [_make_result(_make_instance()) for _ in range(2)]
        adapter = _make_adapter_mock(results)
        vid = tmp_path / "save.mp4"
        vid.touch()
        out_base = str(tmp_path / "output")

        with patch.dict("sys.modules", {"cv2": cv2_mock}), patch("mata.api.load", return_value=adapter):
            mata.track(str(vid), save=True, save_dir=out_base)

        # Check that at least one exp* dir was created under out_base
        created = list(Path(out_base).iterdir())
        assert len(created) >= 1

    def test_writer_released_after_save(self, tmp_path):
        frames = self._make_frames(2)
        cv2_mock = _make_cv2_mock(frames)
        mock_writer_instance = MagicMock()
        cv2_mock.VideoWriter.return_value = mock_writer_instance
        results = [_make_result(_make_instance()) for _ in range(2)]
        adapter = _make_adapter_mock(results)
        vid = tmp_path / "saveit.mp4"
        vid.touch()

        with patch.dict("sys.modules", {"cv2": cv2_mock}), patch("mata.api.load", return_value=adapter):
            mata.track(str(vid), save=True)

        mock_writer_instance.release.assert_called_once()

    def test_save_image_dir_writes_files(self, tmp_path):
        pytest.importorskip("cv2")

        # Create image dir
        img_dir = tmp_path / "frames"
        img_dir.mkdir()
        for i in range(2):
            PILImage.new("RGB", (50, 50)).save(str(img_dir / f"f{i}.jpg"))

        results = [_make_result(_make_instance()) for _ in range(2)]
        adapter = _make_adapter_mock(results)
        out_base = str(tmp_path / "out")

        with patch("mata.api.load", return_value=adapter):
            mata.track(str(img_dir), save=True, save_dir=out_base)

        # Check files written under out/exp1/
        exp_dir = Path(out_base) / "exp1"
        assert exp_dir.exists()
        written = list(exp_dir.iterdir())
        assert len(written) == 2


# ---------------------------------------------------------------------------
# VisionResult metadata
# ---------------------------------------------------------------------------


class TestVisionResultMeta:
    def test_result_is_vision_result(self, tmp_path):
        img = tmp_path / "m.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        adapter = _make_adapter_mock([_make_result(_make_instance())])
        with patch("mata.api.load", return_value=adapter):
            results = mata.track(str(img))
        assert isinstance(results[0], VisionResult)

    def test_track_id_preserved_in_result(self, tmp_path):
        img = tmp_path / "tid.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        inst = _make_instance(track_id=99)
        adapter = _make_adapter_mock([_make_result(inst)])
        with patch("mata.api.load", return_value=adapter):
            results = mata.track(str(img))
        assert results[0].instances[0].track_id == 99

    def test_multiple_instances_per_frame(self, tmp_path):
        img = tmp_path / "multi.jpg"
        PILImage.new("RGB", (50, 50)).save(str(img))
        instances = [_make_instance(track_id=i) for i in range(5)]
        adapter = _make_adapter_mock([_make_result(*instances)])
        with patch("mata.api.load", return_value=adapter):
            results = mata.track(str(img))
        assert len(results[0].instances) == 5
