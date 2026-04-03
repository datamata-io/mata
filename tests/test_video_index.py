"""Unit tests for video indexing helpers.

Run independently: pytest tests/test_video_index.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

import mata
from mata.recognition import Gallery, VideoIndex, VideoMatch, index_video


def _fake_frames(count: int = 10) -> list[tuple[int, np.ndarray]]:
    return [(idx, np.full((4, 4, 3), idx, dtype=np.uint8)) for idx in range(count)]


class _FrameAdapter:
    def __init__(self, dim: int = 4):
        self._encoder = object()
        self.dim = dim

    def embed(self, input_value):
        pixel_value = int(input_value.to_numpy()[0, 0, 0])
        vec = np.zeros((1, self.dim), dtype=np.float32)
        vec[0, pixel_value % self.dim] = 1.0
        return vec


class _ChunkEncoder:
    def predict_video(self, frames):
        return frames


class _ChunkAdapter:
    def __init__(self, dim: int = 6):
        self._encoder = _ChunkEncoder()
        self.dim = dim

    def embed(self, frames):
        return np.full((1, self.dim), float(len(frames)), dtype=np.float32)


class TestIndexVideoFrameMode:
    def test_frame_mode_indexes_sampled_frames(self):
        adapter = _FrameAdapter()
        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 4.0, "frame_count": 8}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(8))):
                result = index_video("video.mp4", adapter=adapter, mode="frame", sample_fps=2.0)

        assert isinstance(result, VideoIndex)
        assert result.mode == "frame"
        assert result.indexed_count == 4
        assert len(result.gallery) == 4
        assert list(result.frame_map.keys()) == [
            "frame_000000",
            "frame_000002",
            "frame_000004",
            "frame_000006",
        ]
        assert result.frame_map["frame_000004"] == 1.0
        assert result.end_map["frame_000004"] == 1.0

    def test_frame_mode_search_resolves_timestamps(self):
        adapter = _FrameAdapter(dim=3)
        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 2.0, "frame_count": 4}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(4))):
                result = index_video("video.mp4", adapter=adapter, mode="frame", sample_fps=1.0)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matches = result.search(query, top_k=1, threshold=-1.0)
        assert len(matches) == 1
        assert isinstance(matches[0], VideoMatch)
        assert matches[0].start_s == 0.0
        assert matches[0].end_s == 0.0


class TestIndexVideoChunkMode:
    def test_chunk_mode_indexes_windows_and_tail(self):
        adapter = _ChunkAdapter()
        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 5.0, "frame_count": 7}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(7))):
                result = index_video(
                    "video.mp4",
                    adapter=adapter,
                    mode="chunk",
                    chunk_stride=3,
                    chunk_frames=2,
                )

        assert result.mode == "chunk"
        assert result.indexed_count == 3
        assert list(result.frame_map.keys()) == [
            "chunk_000000",
            "chunk_000003",
            "chunk_000006",
        ]
        assert result.frame_map["chunk_000003"] == 0.6
        assert result.end_map["chunk_000003"] == 1.0
        assert result.end_map["chunk_000006"] == 1.2

    def test_auto_mode_uses_chunk_for_video_capable_adapter(self):
        adapter = _ChunkAdapter()
        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 4.0, "frame_count": 4}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(4))):
                result = index_video("video.mp4", adapter=adapter, chunk_stride=2, chunk_frames=2)

        assert result.mode == "chunk"
        assert result.indexed_count == 2


class TestIndexVideoIncremental:
    def test_incremental_indexing_appends_to_existing_index(self):
        adapter = _FrameAdapter()
        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 2.0, "frame_count": 4}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(4))):
                first = index_video("video.mp4", adapter=adapter, mode="frame", sample_fps=1.0)

        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 2.0, "frame_count": 4}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(4))):
                second = index_video("video.mp4", adapter=adapter, mode="frame", sample_fps=1.0, index=first)

        assert second.indexed_count == 4
        assert len(second.gallery) == 4
        assert "frame_000000_1" in second.frame_map
        assert second.frame_map["frame_000000_1"] == 0.0

    def test_incremental_indexing_accepts_existing_gallery_and_maps(self):
        adapter = _FrameAdapter()
        gallery = Gallery()
        gallery.add("seed", np.array([1.0, 0.0], dtype=np.float32))
        frame_map = {"seed": 9.5}
        end_map = {"seed": 9.5}

        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 2.0, "frame_count": 2}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(2))):
                result = index_video(
                    "video.mp4",
                    adapter=adapter,
                    mode="frame",
                    sample_fps=1.0,
                    gallery=gallery,
                    frame_map=frame_map,
                    end_map=end_map,
                )

        assert len(result.gallery) == 2
        assert result.frame_map["seed"] == 9.5
        assert "frame_000000" in result.frame_map


class TestIndexVideoPersistence:
    def test_save_and_load_json_round_trip(self, tmp_path: Path):
        adapter = _ChunkAdapter()
        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 3.0, "frame_count": 3}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(3))):
                result = index_video("video.mp4", adapter=adapter, mode="chunk", chunk_stride=2, chunk_frames=2)

        path = tmp_path / "video_index.json"
        result.save(str(path))
        loaded = VideoIndex.load(str(path))

        assert loaded.mode == result.mode
        assert loaded.indexed_count == result.indexed_count
        assert loaded.frame_map == result.frame_map
        assert loaded.end_map == result.end_map
        assert len(loaded.gallery) == len(result.gallery)

    def test_save_and_load_npz_round_trip(self, tmp_path: Path):
        adapter = _ChunkAdapter()
        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 3.0, "frame_count": 3}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(3))):
                result = index_video("video.mp4", adapter=adapter, mode="chunk", chunk_stride=2, chunk_frames=2)

        path = tmp_path / "video_index.npz"
        result.save(str(path))
        loaded = VideoIndex.load(str(path))

        assert loaded.mode == result.mode
        assert loaded.frame_map == result.frame_map


class TestIndexVideoModelLoading:
    def test_model_argument_uses_mata_load(self):
        adapter = _FrameAdapter()
        with patch.object(mata, "load", return_value=adapter) as mocked_load:
            with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 2.0, "frame_count": 2}):
                with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(2))):
                    result = index_video("video.mp4", model="org/model", mode="frame")

        mocked_load.assert_called_once_with("embed", "org/model")
        assert result.indexed_count == 1

    def test_progress_callback_receives_updates(self):
        adapter = _FrameAdapter()
        calls: list[tuple[int, int]] = []

        with patch("mata.recognition.video_index.get_video_info", return_value={"fps": 2.0, "frame_count": 4}):
            with patch("mata.recognition.video_index.iter_frames", return_value=iter(_fake_frames(4))):
                index_video(
                    "video.mp4",
                    adapter=adapter,
                    mode="frame",
                    sample_fps=1.0,
                    progress=lambda current, total: calls.append((current, total)),
                )

        assert calls == [(1, 2), (2, 2)]
