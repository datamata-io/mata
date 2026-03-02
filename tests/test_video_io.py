"""Unit tests for mata.core.video_io (Task B4).

Tests cover:
- detect_source_type(): all supported source types
- iter_frames(): video file, webcam, stream, image dir, image file,
  PIL image, numpy array
- get_video_info(): fps, width, height, frame_count
- VideoWriter: context manager, codec selection, write, closed-state error
- make_output_dir(): auto-incrementing directory creation
- Resource cleanup on generator close
- Graceful ImportError when cv2 is not installed
- Helpful error messages for unsupported sources
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_bgr_frame(h: int = 4, w: int = 6) -> np.ndarray:
    """Create a small dummy BGR uint8 frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_mock_cv2(
    cap_is_opened: bool = True,
    read_frames: list[np.ndarray] | None = None,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
    frame_count: int = 100,
) -> MagicMock:
    """Build a mock cv2 module with a configurable VideoCapture."""
    if read_frames is None:
        read_frames = [_make_bgr_frame()]

    cv2 = MagicMock()

    # VideoCapture mock
    cap = MagicMock()
    cap.isOpened.return_value = cap_is_opened

    # read() returns (True, frame) for each frame then (False, None)
    read_returns = [(True, f) for f in read_frames] + [(False, None)]
    cap.read.side_effect = read_returns

    # CAP_PROP constants
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.CAP_PROP_FRAME_COUNT = 7

    def _cap_get(prop: int) -> float:
        return {5: fps, 3: float(width), 4: float(height), 7: float(frame_count)}.get(prop, 0.0)

    cap.get.side_effect = _cap_get
    cv2.VideoCapture.return_value = cap

    # VideoWriter mock
    writer = MagicMock()
    writer.isOpened.return_value = True
    cv2.VideoWriter.return_value = writer
    cv2.VideoWriter_fourcc.return_value = 0x58564944  # arbitrary

    # imread mock — returns a valid frame by default
    cv2.imread.return_value = _make_bgr_frame()

    # cvtColor pass-through
    cv2.cvtColor.side_effect = lambda img, *_: img

    # Color constants
    cv2.COLOR_GRAY2BGR = 8
    cv2.COLOR_BGRA2BGR = 1

    return cv2


# ---------------------------------------------------------------------------
# detect_source_type
# ---------------------------------------------------------------------------


class TestDetectSourceType:
    """Tests for detect_source_type()."""

    def test_integer_is_webcam(self):
        from mata.core.video_io import detect_source_type

        assert detect_source_type(0) == "webcam"
        assert detect_source_type(1) == "webcam"

    def test_numpy_array(self):
        from mata.core.video_io import detect_source_type

        assert detect_source_type(np.zeros((4, 4, 3), dtype=np.uint8)) == "numpy_array"

    def test_pil_image(self):
        from PIL import Image

        from mata.core.video_io import detect_source_type

        img = Image.new("RGB", (10, 10))
        assert detect_source_type(img) == "pil_image"

    def test_rtsp_stream(self):
        from mata.core.video_io import detect_source_type

        assert detect_source_type("rtsp://192.168.1.1/stream") == "stream"

    def test_http_stream(self):
        from mata.core.video_io import detect_source_type

        assert detect_source_type("http://example.com/video.m3u8") == "stream"

    def test_https_stream(self):
        from mata.core.video_io import detect_source_type

        assert detect_source_type("https://example.com/stream") == "stream"

    def test_video_file_mp4(self, tmp_path):
        from mata.core.video_io import detect_source_type

        f = tmp_path / "clip.mp4"
        f.touch()
        assert detect_source_type(str(f)) == "video_file"

    def test_video_file_avi(self, tmp_path):
        from mata.core.video_io import detect_source_type

        f = tmp_path / "clip.avi"
        f.touch()
        assert detect_source_type(f) == "video_file"

    def test_video_file_mkv(self, tmp_path):
        from mata.core.video_io import detect_source_type

        f = tmp_path / "clip.mkv"
        f.touch()
        assert detect_source_type(str(f)) == "video_file"

    def test_image_dir(self, tmp_path):
        from mata.core.video_io import detect_source_type

        assert detect_source_type(str(tmp_path)) == "image_dir"
        assert detect_source_type(tmp_path) == "image_dir"

    def test_image_file_jpg(self, tmp_path):
        from mata.core.video_io import detect_source_type

        f = tmp_path / "frame.jpg"
        f.touch()
        assert detect_source_type(str(f)) == "image_file"

    def test_image_file_png(self, tmp_path):
        from mata.core.video_io import detect_source_type

        f = tmp_path / "frame.png"
        f.touch()
        assert detect_source_type(str(f)) == "image_file"

    def test_image_file_bmp(self, tmp_path):
        from mata.core.video_io import detect_source_type

        f = tmp_path / "frame.bmp"
        f.touch()
        assert detect_source_type(str(f)) == "image_file"

    def test_image_file_tiff(self, tmp_path):
        from mata.core.video_io import detect_source_type

        f = tmp_path / "frame.tiff"
        f.touch()
        assert detect_source_type(str(f)) == "image_file"

    def test_unsupported_type_raises(self):
        from mata.core.video_io import detect_source_type

        with pytest.raises(ValueError, match="Unsupported source type"):
            detect_source_type(12.5)

    def test_unknown_nonexistent_path_raises(self):
        from mata.core.video_io import detect_source_type

        with pytest.raises(ValueError):
            detect_source_type("/nonexistent/path/unknown.xyz")


# ---------------------------------------------------------------------------
# iter_frames — numpy array
# ---------------------------------------------------------------------------


class TestIterFramesNumpyArray:
    def test_yields_single_frame(self):
        from mata.core.video_io import iter_frames

        frame = _make_bgr_frame(4, 6)
        results = list(iter_frames(frame))
        assert len(results) == 1
        idx, f = results[0]
        assert idx == 0
        assert f.shape == (4, 6, 3)

    def test_max_frames_zero_yields_nothing(self):
        from mata.core.video_io import iter_frames

        results = list(iter_frames(_make_bgr_frame(), max_frames=0))
        assert results == []


# ---------------------------------------------------------------------------
# iter_frames — PIL image
# ---------------------------------------------------------------------------


class TestIterFramesPILImage:
    def test_yields_single_bgr_frame(self):
        from PIL import Image

        from mata.core.video_io import iter_frames

        img = Image.new("RGB", (8, 6), color=(100, 150, 200))
        results = list(iter_frames(img))
        assert len(results) == 1
        idx, frame = results[0]
        assert idx == 0
        assert frame.shape == (6, 8, 3)


# ---------------------------------------------------------------------------
# iter_frames — image directory
# ---------------------------------------------------------------------------


class TestIterFramesImageDir:
    @patch("mata.core.video_io._require_cv2")
    def test_yields_sorted_frames(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        cv2 = _make_mock_cv2()
        mock_req_cv2.return_value = cv2

        # Create dummy image files — they don't need real content (imread is mocked)
        for name in ["c.jpg", "a.png", "b.bmp"]:
            (tmp_path / name).touch()

        results = list(iter_frames(str(tmp_path)))
        # Should be 3 frames in sorted order (a.png, b.bmp, c.jpg)
        assert len(results) == 3
        assert results[0][0] == 0
        assert results[1][0] == 1
        assert results[2][0] == 2

    @patch("mata.core.video_io._require_cv2")
    def test_max_frames_limits_output(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        cv2 = _make_mock_cv2()
        mock_req_cv2.return_value = cv2

        for name in ["a.jpg", "b.jpg", "c.jpg"]:
            (tmp_path / name).touch()

        results = list(iter_frames(str(tmp_path), max_frames=2))
        assert len(results) == 2

    @patch("mata.core.video_io._require_cv2")
    def test_non_image_files_skipped(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        cv2 = _make_mock_cv2()
        mock_req_cv2.return_value = cv2

        (tmp_path / "video.mp4").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "image.jpg").touch()

        results = list(iter_frames(str(tmp_path)))
        assert len(results) == 1  # only image.jpg

    @patch("mata.core.video_io._require_cv2")
    def test_empty_dir_yields_nothing(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        cv2 = _make_mock_cv2()
        mock_req_cv2.return_value = cv2

        results = list(iter_frames(str(tmp_path)))
        assert results == []

    def test_missing_dir_raises(self):
        from mata.core.video_io import iter_frames

        with pytest.raises(Exception):
            # Trigger by using a non-existent path that looks like a dir
            # detect_source_type will raise ValueError for non-existent paths
            list(iter_frames("/nonexistent/dir_that_does_not_exist"))


# ---------------------------------------------------------------------------
# iter_frames — image file
# ---------------------------------------------------------------------------


class TestIterFramesImageFile:
    @patch("mata.core.video_io._require_cv2")
    def test_yields_single_frame(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        cv2 = _make_mock_cv2()
        mock_req_cv2.return_value = cv2

        img_path = tmp_path / "frame.jpg"
        img_path.touch()
        results = list(iter_frames(str(img_path)))
        assert len(results) == 1
        assert results[0][0] == 0

    def test_missing_image_file_raises(self, tmp_path):
        from mata.core.video_io import iter_frames

        with pytest.raises(FileNotFoundError):
            list(iter_frames(str(tmp_path / "missing.jpg")))


# ---------------------------------------------------------------------------
# iter_frames — video file
# ---------------------------------------------------------------------------


class TestIterFramesVideoFile:
    @patch("mata.core.video_io._require_cv2")
    def test_yields_all_frames(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        frames = [_make_bgr_frame() for _ in range(5)]
        cv2 = _make_mock_cv2(read_frames=frames)
        mock_req_cv2.return_value = cv2

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        results = list(iter_frames(str(video_path)))
        assert len(results) == 5
        indices = [r[0] for r in results]
        assert indices == list(range(5))

    @patch("mata.core.video_io._require_cv2")
    def test_max_frames_limits(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        frames = [_make_bgr_frame() for _ in range(10)]
        cv2 = _make_mock_cv2(read_frames=frames)
        mock_req_cv2.return_value = cv2

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        results = list(iter_frames(str(video_path), max_frames=3))
        assert len(results) == 3

    @patch("mata.core.video_io._require_cv2")
    def test_cap_released_after_exhaustion(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        cv2 = _make_mock_cv2(read_frames=[_make_bgr_frame()])
        mock_req_cv2.return_value = cv2
        cap = cv2.VideoCapture.return_value

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        list(iter_frames(str(video_path)))
        cap.release.assert_called_once()

    @patch("mata.core.video_io._require_cv2")
    def test_cap_released_on_early_close(self, mock_req_cv2, tmp_path):
        """VideoCapture must be released even if generator is abandoned."""
        from mata.core.video_io import iter_frames

        frames = [_make_bgr_frame() for _ in range(5)]
        cv2 = _make_mock_cv2(read_frames=frames)
        mock_req_cv2.return_value = cv2
        cap = cv2.VideoCapture.return_value

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        gen = iter_frames(str(video_path))
        next(gen)  # consume one frame
        gen.close()  # abandon the rest

        cap.release.assert_called_once()

    @patch("mata.core.video_io._require_cv2")
    def test_unopenable_video_raises(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import iter_frames

        cv2 = _make_mock_cv2(cap_is_opened=False)
        mock_req_cv2.return_value = cv2

        video_path = tmp_path / "bad.mp4"
        video_path.touch()

        with pytest.raises(ValueError, match="could not open"):
            list(iter_frames(str(video_path)))

    def test_missing_video_file_raises(self, tmp_path):
        from mata.core.video_io import iter_frames

        with pytest.raises(FileNotFoundError):
            list(iter_frames(str(tmp_path / "missing.mp4")))


# ---------------------------------------------------------------------------
# iter_frames — webcam
# ---------------------------------------------------------------------------


class TestIterFramesWebcam:
    @patch("mata.core.video_io._require_cv2")
    def test_webcam_yields_frames(self, mock_req_cv2):
        from mata.core.video_io import iter_frames

        frames = [_make_bgr_frame() for _ in range(3)]
        cv2 = _make_mock_cv2(read_frames=frames)
        mock_req_cv2.return_value = cv2

        results = list(iter_frames(0))
        assert len(results) == 3

    @patch("mata.core.video_io._require_cv2")
    def test_webcam_cap_released(self, mock_req_cv2):
        from mata.core.video_io import iter_frames

        cv2 = _make_mock_cv2(read_frames=[_make_bgr_frame()])
        mock_req_cv2.return_value = cv2
        cap = cv2.VideoCapture.return_value

        list(iter_frames(0))
        cap.release.assert_called_once()

    @patch("mata.core.video_io._require_cv2")
    def test_webcam_max_frames(self, mock_req_cv2):
        from mata.core.video_io import iter_frames

        frames = [_make_bgr_frame() for _ in range(10)]
        cv2 = _make_mock_cv2(read_frames=frames)
        mock_req_cv2.return_value = cv2

        results = list(iter_frames(0, max_frames=2))
        assert len(results) == 2


# ---------------------------------------------------------------------------
# iter_frames — stream
# ---------------------------------------------------------------------------


class TestIterFramesStream:
    @patch("mata.core.video_io._require_cv2")
    def test_stream_yields_frames(self, mock_req_cv2):
        from mata.core.video_io import iter_frames

        frames = [_make_bgr_frame() for _ in range(4)]
        cv2 = _make_mock_cv2(read_frames=frames)
        mock_req_cv2.return_value = cv2

        results = list(iter_frames("rtsp://host/stream"))
        assert len(results) == 4


# ---------------------------------------------------------------------------
# get_video_info
# ---------------------------------------------------------------------------


class TestGetVideoInfo:
    @patch("mata.core.video_io._require_cv2")
    def test_returns_correct_metadata(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import get_video_info

        cv2 = _make_mock_cv2(fps=25.0, width=1920, height=1080, frame_count=500)
        mock_req_cv2.return_value = cv2

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        info = get_video_info(str(video_path))
        assert info["fps"] == 25.0
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["frame_count"] == 500

    @patch("mata.core.video_io._require_cv2")
    def test_cap_released_after_info(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import get_video_info

        cv2 = _make_mock_cv2()
        mock_req_cv2.return_value = cv2
        cap = cv2.VideoCapture.return_value

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        get_video_info(str(video_path))
        cap.release.assert_called_once()

    def test_missing_file_raises(self, tmp_path):
        from mata.core.video_io import get_video_info

        with pytest.raises(FileNotFoundError):
            get_video_info(str(tmp_path / "missing.mp4"))

    @patch("mata.core.video_io._require_cv2")
    def test_unopenable_raises(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import get_video_info

        cv2 = _make_mock_cv2(cap_is_opened=False)
        mock_req_cv2.return_value = cv2

        video_path = tmp_path / "bad.mp4"
        video_path.touch()

        with pytest.raises(ValueError, match="could not open"):
            get_video_info(str(video_path))


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------


class TestVideoWriter:
    def _make_writer_mock_cv2(self, writer_opened: bool = True):
        cv2 = MagicMock()
        cv2.VideoWriter_fourcc.return_value = 0x58564944
        writer = MagicMock()
        writer.isOpened.return_value = writer_opened
        cv2.VideoWriter.return_value = writer
        return cv2, writer

    @patch("mata.core.video_io._require_cv2")
    def test_mp4_uses_mp4v_codec(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, _ = self._make_writer_mock_cv2()
        mock_req_cv2.return_value = cv2

        output = tmp_path / "out.mp4"
        with VideoWriter(str(output), fps=30, size=(640, 480)):
            pass

        cv2.VideoWriter_fourcc.assert_called_with(*"mp4v")

    @patch("mata.core.video_io._require_cv2")
    def test_avi_uses_xvid_codec(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, _ = self._make_writer_mock_cv2()
        mock_req_cv2.return_value = cv2

        output = tmp_path / "out.avi"
        with VideoWriter(str(output), fps=30, size=(640, 480)):
            pass

        cv2.VideoWriter_fourcc.assert_called_with(*"XVID")

    @patch("mata.core.video_io._require_cv2")
    def test_wmv_uses_wmv2_codec(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, _ = self._make_writer_mock_cv2()
        mock_req_cv2.return_value = cv2

        output = tmp_path / "out.wmv"
        with VideoWriter(str(output), fps=30, size=(640, 480)):
            pass

        cv2.VideoWriter_fourcc.assert_called_with(*"WMV2")

    @patch("mata.core.video_io._require_cv2")
    def test_codec_override(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, _ = self._make_writer_mock_cv2()
        mock_req_cv2.return_value = cv2

        output = tmp_path / "out.mp4"
        with VideoWriter(str(output), fps=30, size=(640, 480), codec="H264"):
            pass

        cv2.VideoWriter_fourcc.assert_called_with(*"H264")

    @patch("mata.core.video_io._require_cv2")
    def test_write_delegates_to_cv2(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, writer = self._make_writer_mock_cv2()
        mock_req_cv2.return_value = cv2

        frame = _make_bgr_frame()
        output = tmp_path / "out.mp4"
        with VideoWriter(str(output), fps=30, size=(640, 480)) as vw:
            vw.write(frame)

        writer.write.assert_called_once_with(frame)

    @patch("mata.core.video_io._require_cv2")
    def test_writer_released_on_exit(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, writer = self._make_writer_mock_cv2()
        mock_req_cv2.return_value = cv2

        output = tmp_path / "out.mp4"
        with VideoWriter(str(output), fps=30, size=(640, 480)):
            pass

        writer.release.assert_called_once()

    @patch("mata.core.video_io._require_cv2")
    def test_writer_released_on_exception(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, writer = self._make_writer_mock_cv2()
        mock_req_cv2.return_value = cv2

        output = tmp_path / "out.mp4"
        with pytest.raises(RuntimeError):
            with VideoWriter(str(output), fps=30, size=(640, 480)):
                raise RuntimeError("test error")

        writer.release.assert_called_once()

    @patch("mata.core.video_io._require_cv2")
    def test_write_outside_context_raises(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, _ = self._make_writer_mock_cv2()
        mock_req_cv2.return_value = cv2

        vw = VideoWriter(str(tmp_path / "out.mp4"), fps=30, size=(640, 480))
        with pytest.raises(RuntimeError, match="context manager"):
            vw.write(_make_bgr_frame())

    @patch("mata.core.video_io._require_cv2")
    def test_unopenable_writer_raises(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import VideoWriter

        cv2, _ = self._make_writer_mock_cv2(writer_opened=False)
        mock_req_cv2.return_value = cv2

        with pytest.raises(ValueError, match="could not open"):
            output = tmp_path / "out.mp4"
            VideoWriter(str(output), fps=30, size=(640, 480)).__enter__()


# ---------------------------------------------------------------------------
# make_output_dir
# ---------------------------------------------------------------------------


class TestMakeOutputDir:
    def test_first_call_creates_exp1(self, tmp_path):
        from mata.core.video_io import make_output_dir

        result = make_output_dir(base=str(tmp_path / "runs"), name="exp")
        assert result.name == "exp1"
        assert result.exists()

    def test_second_call_creates_exp2(self, tmp_path):
        from mata.core.video_io import make_output_dir

        base = str(tmp_path / "runs")
        d1 = make_output_dir(base=base, name="exp")
        d2 = make_output_dir(base=base, name="exp")
        assert d1.name == "exp1"
        assert d2.name == "exp2"

    def test_skips_existing(self, tmp_path):
        from mata.core.video_io import make_output_dir

        base = tmp_path / "runs"
        (base / "exp1").mkdir(parents=True)
        (base / "exp2").mkdir(parents=True)

        result = make_output_dir(base=str(base), name="exp")
        assert result.name == "exp3"

    def test_custom_name(self, tmp_path):
        from mata.core.video_io import make_output_dir

        result = make_output_dir(base=str(tmp_path), name="run")
        assert result.name == "run1"

    def test_creates_intermediate_dirs(self, tmp_path):
        from mata.core.video_io import make_output_dir

        result = make_output_dir(base=str(tmp_path / "deep" / "nested" / "runs"), name="exp")
        assert result.exists()


# ---------------------------------------------------------------------------
# cv2 not installed
# ---------------------------------------------------------------------------


class TestCv2NotInstalled:
    def test_require_cv2_raises_import_error(self):
        """_require_cv2 raises ImportError with install hint when cv2 is missing."""
        # Temporarily hide cv2 from sys.modules
        cv2_backup = sys.modules.pop("cv2", None)
        try:
            # Also remove from mata.core.video_io namespace if cached
            import mata.core.video_io as vio_mod

            # Patch builtins.__import__ to raise for cv2
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
            import builtins

            original = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "cv2":
                    raise ImportError("No module named 'cv2'")
                return original(name, *args, **kwargs)

            builtins.__import__ = fake_import
            try:
                # Re-run _require_cv2 directly
                with pytest.raises(ImportError, match="opencv"):
                    vio_mod._require_cv2()
            finally:
                builtins.__import__ = original
        finally:
            if cv2_backup is not None:
                sys.modules["cv2"] = cv2_backup

    @patch("mata.core.video_io._require_cv2")
    def test_get_video_info_propagates_import_error(self, mock_req_cv2, tmp_path):
        from mata.core.video_io import get_video_info

        mock_req_cv2.side_effect = ImportError("OpenCV is required but not installed. pip install opencv-python")
        video_path = tmp_path / "video.mp4"
        video_path.touch()

        with pytest.raises(ImportError, match="opencv"):
            get_video_info(str(video_path))
