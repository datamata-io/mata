"""Unit tests for Task 5.5: Tracking Node.

Tests cover:
- Track node with mock tracker provider
- BYTETrack wrapper (with and without external dependency)
- Simple IoU tracker fallback
- Track state management (active/lost/terminated)
- Frame ID handling (manual and auto-generated)
- Track association and lifecycle
- Metrics recording
- Real video sequence tracking
- Error handling (missing provider)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.tracks import Track as TrackArtifact
from mata.core.artifacts.tracks import Tracks
from mata.core.graph.context import ExecutionContext
from mata.core.types import Instance
from mata.nodes.track import (
    BotSortWrapper,
    ByteTrackWrapper,
    SimpleIOUTracker,
    Track,
    _detections_to_detection_results,
    _tracker_array_to_tracks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_detections_frame1():
    """Detections for frame 1 - 2 objects."""
    instances = [
        Instance(
            bbox=(10, 20, 100, 150),
            score=0.95,
            label=0,
            label_name="person",
        ),
        Instance(
            bbox=(200, 50, 300, 200),
            score=0.82,
            label=1,
            label_name="car",
        ),
    ]
    return Detections(instances=instances, meta={"frame": 1})


@pytest.fixture
def sample_detections_frame2():
    """Detections for frame 2 - moved objects."""
    instances = [
        Instance(
            bbox=(15, 25, 105, 155),  # Slightly moved person
            score=0.90,
            label=0,
            label_name="person",
        ),
        Instance(
            bbox=(210, 55, 310, 205),  # Slightly moved car
            score=0.85,
            label=1,
            label_name="car",
        ),
    ]
    return Detections(instances=instances, meta={"frame": 2})


@pytest.fixture
def sample_detections_frame3():
    """Detections for frame 3 - one object missing (lost track)."""
    instances = [
        Instance(
            bbox=(12, 22, 102, 152),  # Person moved slightly (better IoU match)
            score=0.88,
            label=0,
            label_name="person",
        ),
        # Car is missing (lost track)
    ]
    return Detections(instances=instances, meta={"frame": 3})


@pytest.fixture
def sample_tracks_frame1():
    """Expected tracks for frame 1."""
    tracks = [
        TrackArtifact(track_id=1, bbox=(10, 20, 100, 150), score=0.95, label="person", age=1, state="active"),
        TrackArtifact(track_id=2, bbox=(200, 50, 300, 200), score=0.82, label="car", age=1, state="active"),
    ]
    return Tracks(tracks=tracks, frame_id="frame_0001")


def _make_ctx(providers: dict[str, dict[str, Any]] | None = None) -> ExecutionContext:
    """Helper to build an ExecutionContext with given providers."""
    return ExecutionContext(providers=providers or {}, device="cpu")


# ===========================================================================
# Track node tests
# ===========================================================================


class TestTrackNode:
    """Tests for the Track node."""

    def test_basic_tracking_with_mock(self, sample_detections_frame1, sample_tracks_frame1):
        """Track node with mock tracker provider."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = sample_tracks_frame1

        ctx = _make_ctx({"track": {"mock_tracker": mock_tracker}})
        node = Track(using="mock_tracker")

        result = node.run(ctx, detections=sample_detections_frame1, frame_id="frame_0001")

        assert "tracks" in result
        tracks = result["tracks"]
        assert isinstance(tracks, Tracks)
        assert len(tracks.tracks) == 2
        assert tracks.frame_id == "frame_0001"

        # Verify tracker was called with correct parameters
        mock_tracker.update.assert_called_once_with(
            sample_detections_frame1,
            frame_id="frame_0001",
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
        )

    def test_custom_output_name(self, sample_detections_frame1, sample_tracks_frame1):
        """Track node respects custom output name."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = sample_tracks_frame1

        ctx = _make_ctx({"track": {"tracker": mock_tracker}})
        node = Track(using="tracker", out="my_tracks")

        result = node.run(ctx, detections=sample_detections_frame1)

        assert "my_tracks" in result
        assert isinstance(result["my_tracks"], Tracks)

    def test_auto_frame_id_generation(self, sample_detections_frame1):
        """Frame ID auto-generation when not provided."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = Tracks(tracks=[], frame_id="auto_frame")

        ctx = _make_ctx({"track": {"tracker": mock_tracker}})
        node = Track(using="tracker")

        # First call - frame_0000
        node.run(ctx, detections=sample_detections_frame1)
        mock_tracker.update.assert_called_with(
            sample_detections_frame1,
            frame_id="frame_0000",
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
        )

        # Second call - frame_0001
        node.run(ctx, detections=sample_detections_frame1)
        mock_tracker.update.assert_called_with(
            sample_detections_frame1,
            frame_id="frame_0001",
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
        )

    def test_frame_id_template(self, sample_detections_frame1):
        """Frame ID template formatting."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = Tracks(tracks=[], frame_id="video_frame")

        ctx = _make_ctx({"track": {"tracker": mock_tracker}})
        node = Track(using="tracker", frame_id="video_{frame:05d}")

        node.run(ctx, detections=sample_detections_frame1)

        mock_tracker.update.assert_called_with(
            sample_detections_frame1,
            frame_id="video_00000",
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
        )

    def test_custom_tracking_parameters(self, sample_detections_frame1):
        """Custom tracking thresholds and parameters."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = Tracks(tracks=[], frame_id="frame_test")

        ctx = _make_ctx({"track": {"tracker": mock_tracker}})
        node = Track(using="tracker", track_thresh=0.7, track_buffer=50, match_thresh=0.9)

        node.run(ctx, detections=sample_detections_frame1, frame_id="test")

        mock_tracker.update.assert_called_with(
            sample_detections_frame1,
            frame_id="test",
            track_thresh=0.7,
            track_buffer=50,
            match_thresh=0.9,
        )

    def test_metrics_recording(self, sample_detections_frame1):
        """Metrics are recorded correctly."""
        tracks = Tracks(
            tracks=[
                TrackArtifact(track_id=1, bbox=(0, 0, 10, 10), score=0.9, label="obj", state="active"),
                TrackArtifact(track_id=2, bbox=(0, 0, 10, 10), score=0.8, label="obj", state="lost"),
            ],
            frame_id="frame_test",
        )

        mock_tracker = MagicMock()

        # Add a small delay to ensure measurable latency
        def mock_update(*args, **kwargs):
            time.sleep(0.001)  # 1ms delay
            return tracks

        mock_tracker.update.side_effect = mock_update

        ctx = _make_ctx({"track": {"tracker": mock_tracker}})
        node = Track(using="tracker")

        node.run(ctx, detections=sample_detections_frame1, frame_id="test")

        # Check metrics were recorded
        metrics = ctx.get_metrics()
        assert "Track" in metrics
        node_metrics = metrics["Track"]

        assert "latency_ms" in node_metrics
        assert node_metrics["latency_ms"] > 0
        assert node_metrics["num_tracks"] == 2
        assert node_metrics["num_active_tracks"] == 1
        assert node_metrics["num_lost_tracks"] == 1

    def test_missing_provider_error(self, sample_detections_frame1):
        """Error when tracker provider not found."""
        ctx = _make_ctx({})
        node = Track(using="nonexistent")

        with pytest.raises(KeyError):
            node.run(ctx, detections=sample_detections_frame1)


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_detections_to_detection_results_basic(self, sample_detections_frame1):
        """_detections_to_detection_results converts Detections to DetectionResults."""
        dr = _detections_to_detection_results(sample_detections_frame1)
        assert len(dr) == 2
        assert dr.xyxy.shape == (2, 4)
        assert dr.xywh.shape == (2, 4)
        assert dr.conf.shape == (2,)
        assert dr.cls.shape == (2,)
        # Validate xyxy values
        np.testing.assert_allclose(dr.xyxy[0], [10, 20, 100, 150], atol=1e-4)
        np.testing.assert_allclose(dr.xyxy[1], [200, 50, 300, 200], atol=1e-4)
        # Validate xywh (center format)
        np.testing.assert_allclose(dr.xywh[0], [55, 85, 90, 130], atol=1e-4)
        # Confidence scores
        np.testing.assert_allclose(dr.conf[0], 0.95, atol=1e-4)
        np.testing.assert_allclose(dr.conf[1], 0.82, atol=1e-4)

    def test_detections_to_detection_results_empty(self):
        """Handles empty Detections gracefully."""
        empty_dets = Detections(instances=[])
        dr = _detections_to_detection_results(empty_dets)
        assert len(dr) == 0
        assert dr.xyxy.shape == (0, 4)
        assert dr.conf.shape == (0,)

    def test_detections_to_detection_results_no_bbox(self):
        """Skips instances without bbox (e.g. mask-only instances)."""
        mask = np.ones((50, 50), dtype=bool)
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="person"),
            # mask-only instance — no bbox
            Instance(
                bbox=None,
                mask=mask,
                score=0.8,
                label=1,
                label_name="entity",
            ),
        ]
        dets = Detections(instances=instances)
        dr = _detections_to_detection_results(dets)
        assert len(dr) == 1  # Only the instance with bbox

    def test_detections_to_detection_results_none_score(self):
        """Uses 1.0 as default score when Instance.score is None."""
        instances = [Instance(bbox=(10, 20, 100, 150), score=None, label=0, label_name="obj")]
        dets = Detections(instances=instances)
        dr = _detections_to_detection_results(dets)
        assert dr.conf[0] == pytest.approx(1.0)

    def test_tracker_array_to_tracks_basic(self):
        """_tracker_array_to_tracks converts (N,8) array to Tracks artifact."""
        tracked = np.array(
            [
                [10.0, 20.0, 100.0, 150.0, 1.0, 0.9, 0.0, 0.0],
                [200.0, 50.0, 300.0, 200.0, 2.0, 0.8, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        tracks = _tracker_array_to_tracks(tracked, "frame_42")
        assert isinstance(tracks, Tracks)
        assert tracks.frame_id == "frame_42"
        assert len(tracks.tracks) == 2
        assert tracks.tracks[0].track_id == 1
        assert tracks.tracks[0].state == "active"
        assert tracks.tracks[0].score == pytest.approx(0.9)
        assert tracks.tracks[1].track_id == 2
        assert tracks.tracks[1].label_id == 1

    def test_tracker_array_to_tracks_empty(self):
        """Handles empty array (no active tracks)."""
        tracked = np.empty((0, 8), dtype=np.float32)
        tracks = _tracker_array_to_tracks(tracked, "frame_00")
        assert len(tracks.tracks) == 0
        assert tracks.frame_id == "frame_00"

    def test_tracker_array_to_tracks_degenerate_box_skipped(self):
        """Degenerate bboxes (x2<=x1 or y2<=y1) are silently dropped."""
        tracked = np.array(
            [
                [10.0, 20.0, 100.0, 150.0, 1.0, 0.9, 0.0, 0.0],  # valid
                [50.0, 50.0, 40.0, 60.0, 2.0, 0.8, 0.0, 1.0],  # degenerate x2<x1
            ],
            dtype=np.float32,
        )
        tracks = _tracker_array_to_tracks(tracked, "frame_01")
        assert len(tracks.tracks) == 1  # Only valid row kept
        assert tracks.tracks[0].track_id == 1


# ===========================================================================
# ByteTrackWrapper tests
# ===========================================================================


class TestByteTrackWrapper:
    """Tests for the vendored BYTETracker wrapper."""

    def test_construction_uses_vendored_tracker(self):
        """ByteTrackWrapper initialises a vendored BYTETracker."""
        from mata.trackers.byte_tracker import BYTETracker

        wrapper = ByteTrackWrapper()
        assert isinstance(wrapper.tracker, BYTETracker)

    def test_no_yolox_import(self):
        """ByteTrackWrapper never imports yolox."""
        import sys

        # Ensure yolox is NOT in sys.modules after construction
        ByteTrackWrapper()
        assert "yolox" not in sys.modules
        assert "yolox.tracker.byte_tracker" not in sys.modules

    def test_construction_params_forwarded(self):
        """Construction parameters are forwarded to the vendored tracker config."""
        wrapper = ByteTrackWrapper(
            track_buffer=60,
            frame_rate=24,
            track_thresh=0.6,
            low_thresh=0.15,
            new_track_thresh=0.7,
            match_thresh=0.9,
            fuse_score=False,
        )
        args = wrapper.tracker.args
        assert args.track_high_thresh == pytest.approx(0.6)
        assert args.track_low_thresh == pytest.approx(0.15)
        assert args.new_track_thresh == pytest.approx(0.7)
        assert args.track_buffer == 60
        assert args.match_thresh == pytest.approx(0.9)
        assert args.fuse_score is False

    def test_update_returns_tracks_artifact(self, sample_detections_frame1):
        """update() delegates to the vendored tracker and returns Tracks."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.array(
            [[10.0, 20.0, 100.0, 150.0, 1.0, 0.9, 0.0, 0.0]],
            dtype=np.float32,
        )

        wrapper = ByteTrackWrapper()
        wrapper.tracker = mock_tracker

        tracks = wrapper.update(sample_detections_frame1, "frame_0001")

        assert isinstance(tracks, Tracks)
        assert tracks.frame_id == "frame_0001"
        assert len(tracks.tracks) == 1
        assert tracks.tracks[0].track_id == 1
        # Verify vendored tracker was called once with a DetectionResults object
        mock_tracker.update.assert_called_once()

    def test_update_empty_detections(self):
        """update() with empty detections still calls the tracker (keeps lost tracks)."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.empty((0, 8), dtype=np.float32)

        wrapper = ByteTrackWrapper()
        wrapper.tracker = mock_tracker

        empty_dets = Detections(instances=[])
        tracks = wrapper.update(empty_dets, "frame_0001")

        assert isinstance(tracks, Tracks)
        assert len(tracks.tracks) == 0
        mock_tracker.update.assert_called_once()

    def test_update_signature_accepts_override_kwargs(self, sample_detections_frame1):
        """update() accepts track_thresh/track_buffer/match_thresh without error."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.empty((0, 8), dtype=np.float32)

        wrapper = ByteTrackWrapper()
        wrapper.tracker = mock_tracker

        # Must not raise even though overrides are ignored for vendored tracker
        wrapper.update(
            sample_detections_frame1,
            "frame_0001",
            track_thresh=0.7,
            track_buffer=50,
            match_thresh=0.85,
        )

    def test_reset_delegates_to_vendored_tracker(self):
        """reset() calls reset() on the vendored BYTETracker."""
        mock_tracker = MagicMock()
        wrapper = ByteTrackWrapper()
        wrapper.tracker = mock_tracker

        wrapper.reset()

        mock_tracker.reset.assert_called_once()

    def test_multi_frame_integration(self, sample_detections_frame1, sample_detections_frame2):
        """Multi-frame update with the real vendored BYTETracker."""
        wrapper = ByteTrackWrapper(track_thresh=0.5, new_track_thresh=0.5)

        tracks1 = wrapper.update(sample_detections_frame1, "frame_0001")
        assert isinstance(tracks1, Tracks)
        assert tracks1.frame_id == "frame_0001"

        tracks2 = wrapper.update(sample_detections_frame2, "frame_0002")
        assert isinstance(tracks2, Tracks)
        assert tracks2.frame_id == "frame_0002"

    def test_reset_clears_tracker_state(self, sample_detections_frame1):
        """After reset(), tracker state is cleared."""
        wrapper = ByteTrackWrapper(track_thresh=0.5, new_track_thresh=0.5)
        wrapper.update(sample_detections_frame1, "frame_0001")
        wrapper.reset()
        # frame_id should be back to 0 inside the vendored tracker
        assert wrapper.tracker.frame_id == 0


# ===========================================================================
# BotSortWrapper tests
# ===========================================================================


class TestBotSortWrapper:
    """Tests for the vendored BOTSORT wrapper."""

    def test_construction_uses_vendored_botsort(self):
        """BotSortWrapper initialises a vendored BOTSORT."""
        from mata.trackers.bot_sort import BOTSORT

        wrapper = BotSortWrapper()
        assert isinstance(wrapper.tracker, BOTSORT)

    def test_no_yolox_import(self):
        """BotSortWrapper never imports yolox."""
        import sys

        BotSortWrapper()
        assert "yolox" not in sys.modules

    def test_construction_params_forwarded(self):
        """BotSort-specific params are forwarded to the vendored tracker config."""
        wrapper = BotSortWrapper(
            track_buffer=45,
            frame_rate=25,
            track_thresh=0.55,
            gmc_method=None,
            proximity_thresh=0.4,
            appearance_thresh=0.3,
            with_reid=False,
        )
        # BotSort inherits _TrackerArgs so check common fields
        args = wrapper.tracker.args
        assert args.track_high_thresh == pytest.approx(0.55)
        assert args.track_buffer == 45

    def test_update_returns_tracks_artifact(self, sample_detections_frame1):
        """update() delegates to the vendored BOTSORT and returns Tracks."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.array(
            [[10.0, 20.0, 100.0, 150.0, 1.0, 0.9, 0.0, 0.0]],
            dtype=np.float32,
        )

        wrapper = BotSortWrapper()
        wrapper.tracker = mock_tracker

        tracks = wrapper.update(sample_detections_frame1, "frame_0005")

        assert isinstance(tracks, Tracks)
        assert tracks.frame_id == "frame_0005"
        assert len(tracks.tracks) == 1
        mock_tracker.update.assert_called_once()

    def test_update_empty_detections(self):
        """update() with empty detections still calls the tracker."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.empty((0, 8), dtype=np.float32)

        wrapper = BotSortWrapper()
        wrapper.tracker = mock_tracker

        empty_dets = Detections(instances=[])
        tracks = wrapper.update(empty_dets, "frame_0001")

        assert len(tracks.tracks) == 0
        mock_tracker.update.assert_called_once()

    def test_update_signature_accepts_override_kwargs(self, sample_detections_frame1):
        """update() accepts track_thresh/track_buffer/match_thresh without error."""
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.empty((0, 8), dtype=np.float32)

        wrapper = BotSortWrapper()
        wrapper.tracker = mock_tracker

        wrapper.update(
            sample_detections_frame1,
            "frame_0001",
            track_thresh=0.6,
            track_buffer=50,
            match_thresh=0.85,
        )

    def test_reset_delegates_to_vendored_tracker(self):
        """reset() calls reset() on the vendored BOTSORT."""
        mock_tracker = MagicMock()
        wrapper = BotSortWrapper()
        wrapper.tracker = mock_tracker

        wrapper.reset()

        mock_tracker.reset.assert_called_once()

    def test_bytetrack_and_botsort_produce_same_output_format(self, sample_detections_frame1):
        """Both wrappers return Tracks with identical structure."""
        mock_array = np.array(
            [[10.0, 20.0, 100.0, 150.0, 1.0, 0.9, 0.0, 0.0]],
            dtype=np.float32,
        )

        bt_wrapper = ByteTrackWrapper()
        bt_wrapper.tracker = MagicMock()
        bt_wrapper.tracker.update.return_value = mock_array

        bs_wrapper = BotSortWrapper()
        bs_wrapper.tracker = MagicMock()
        bs_wrapper.tracker.update.return_value = mock_array

        bt_tracks = bt_wrapper.update(sample_detections_frame1, "frame_0001")
        bs_tracks = bs_wrapper.update(sample_detections_frame1, "frame_0001")

        # Both should return Tracks with the same track count and structure
        assert len(bt_tracks.tracks) == len(bs_tracks.tracks)
        assert bt_tracks.tracks[0].track_id == bs_tracks.tracks[0].track_id
        assert bt_tracks.tracks[0].state == bs_tracks.tracks[0].state
        assert bt_tracks.tracks[0].score == pytest.approx(bs_tracks.tracks[0].score)

    def test_multi_frame_integration(self, sample_detections_frame1, sample_detections_frame2):
        """Multi-frame update with the real vendored BOTSORT."""
        # Disable GMC (no cv2 needed in CI)
        wrapper = BotSortWrapper(track_thresh=0.5, new_track_thresh=0.5, gmc_method=None)

        tracks1 = wrapper.update(sample_detections_frame1, "frame_0001")
        assert isinstance(tracks1, Tracks)

        tracks2 = wrapper.update(sample_detections_frame2, "frame_0002")
        assert isinstance(tracks2, Tracks)

    def test_reset_clears_tracker_state(self, sample_detections_frame1):
        """After reset(), BOTSORT internal state is cleared."""
        wrapper = BotSortWrapper(track_thresh=0.5, new_track_thresh=0.5, gmc_method=None)
        wrapper.update(sample_detections_frame1, "frame_0001")
        wrapper.reset()
        assert wrapper.tracker.frame_id == 0


# ===========================================================================
# SimpleIOUTracker tests
# ===========================================================================


class TestSimpleIOUTracker:
    """Tests for standalone SimpleIOUTracker."""

    def test_basic_functionality(self, sample_detections_frame1, sample_detections_frame2):
        """Basic tracking with SimpleIOUTracker."""
        tracker = SimpleIOUTracker(match_thresh=0.7)  # Lower threshold for better matching

        # Frame 1
        tracks1 = tracker.update(sample_detections_frame1, "frame_0001")
        assert len(tracks1.tracks) == 2
        assert all(t.age == 1 for t in tracks1.tracks)

        # Frame 2
        tracks2 = tracker.update(sample_detections_frame2, "frame_0002")
        assert len(tracks2.tracks) == 2
        assert all(t.age == 2 for t in tracks2.tracks)

    def test_parameter_override(self, sample_detections_frame1):
        """Test parameter override in update call."""
        tracker = SimpleIOUTracker(track_thresh=0.98)  # Very high threshold above 0.95

        # Should create no tracks with very high threshold
        tracks1 = tracker.update(sample_detections_frame1, "frame_0001")
        assert len(tracks1.tracks) == 0

        # Override threshold to lower value
        tracks2 = tracker.update(sample_detections_frame1, "frame_0002", track_thresh=0.5)
        assert len(tracks2.tracks) == 2

    def test_reset(self, sample_detections_frame1):
        """Test reset functionality."""
        tracker = SimpleIOUTracker()

        # Create tracks
        tracks1 = tracker.update(sample_detections_frame1, "frame_0001")
        assert len(tracks1.tracks) == 2

        # Reset
        tracker.reset()

        # Next update should create new track IDs
        tracks2 = tracker.update(sample_detections_frame1, "frame_0001")
        assert len(tracks2.tracks) == 2
        assert tracks2.tracks[0].track_id == 1  # Reset to 1
        assert tracks2.tracks[1].track_id == 2


# ===========================================================================
# Integration tests
# ===========================================================================


class TestTrackingIntegration:
    """Integration tests with real video sequence simulation."""

    def test_video_sequence_simulation(self):
        """Simulate tracking across a video sequence."""
        # Simulate 5 frames of a moving car
        frames = []
        for i in range(5):
            # Car moves right by 10 pixels each frame
            x_offset = i * 10
            instance = Instance(
                bbox=(10 + x_offset, 20, 100 + x_offset, 150),
                score=0.9 - i * 0.05,  # Slightly decreasing confidence
                label=1,
                label_name="car",
            )
            detections = Detections(instances=[instance])
            frames.append(detections)

        # Track across frames
        tracker = SimpleIOUTracker()
        track_id = None

        for frame_idx, detections in enumerate(frames):
            tracks = tracker.update(detections, f"frame_{frame_idx:04d}")

            assert len(tracks.tracks) == 1
            track = tracks.tracks[0]

            if track_id is None:
                track_id = track.track_id
            else:
                # Should maintain same track ID
                assert track.track_id == track_id

            # Age should increment
            assert track.age == frame_idx + 1
            assert track.state == "active"
            assert track.label == "car"

    def test_node_with_simple_tracker_integration(self):
        """Integration test: Track node with SimpleIOUTracker."""
        tracker = SimpleIOUTracker()
        ctx = _make_ctx({"track": {"simple": tracker}})
        node = Track(using="simple")

        # Create test sequence
        detections1 = Detections(
            instances=[
                Instance(
                    bbox=(10, 20, 100, 150),
                    score=0.95,
                    label=0,
                    label_name="person",
                )
            ]
        )

        detections2 = Detections(
            instances=[
                Instance(
                    bbox=(15, 25, 105, 155),  # Moved slightly
                    score=0.90,
                    label=0,
                    label_name="person",
                )
            ]
        )

        # Process frames
        result1 = node.run(ctx, detections=detections1, frame_id="frame_0001")
        result2 = node.run(ctx, detections=detections2, frame_id="frame_0002")

        tracks1 = result1["tracks"]
        tracks2 = result2["tracks"]

        assert len(tracks1.tracks) == 1
        assert len(tracks2.tracks) == 1
        assert tracks1.tracks[0].track_id == tracks2.tracks[0].track_id  # Same track
        assert tracks2.tracks[0].age == 2

    def test_multi_object_multi_frame_scenario(self):
        """Complex scenario with multiple objects across multiple frames."""
        tracker = SimpleIOUTracker(match_thresh=0.6)  # Lower threshold for better tracking

        # Frame 1: 2 objects
        frame1 = Detections(
            instances=[
                Instance(bbox=(10, 10, 50, 50), score=0.9, label=0, label_name="A"),
                Instance(bbox=(100, 100, 140, 140), score=0.8, label=1, label_name="B"),
            ]
        )

        # Frame 2: Both objects moved slightly
        frame2 = Detections(
            instances=[
                Instance(bbox=(12, 12, 52, 52), score=0.85, label=0, label_name="A"),  # Better match
                Instance(bbox=(102, 102, 142, 142), score=0.75, label=1, label_name="B"),  # Better match
            ]
        )

        # Frame 3: One object disappeared
        frame3 = Detections(
            instances=[
                Instance(bbox=(14, 14, 54, 54), score=0.8, label=0, label_name="A"),  # Better match
            ]
        )

        # Frame 4: Object reappears with new one
        frame4 = Detections(
            instances=[
                Instance(bbox=(16, 16, 56, 56), score=0.82, label=0, label_name="A"),  # Better match
                Instance(bbox=(200, 200, 240, 240), score=0.9, label=2, label_name="C"),
            ]
        )

        # Process sequence
        tracks1 = tracker.update(frame1, "frame_0001")
        tracks2 = tracker.update(frame2, "frame_0002")
        tracks3 = tracker.update(frame3, "frame_0003")
        tracks4 = tracker.update(frame4, "frame_0004")

        # Validate results
        assert len(tracks1.tracks) == 2  # 2 new tracks
        assert len(tracks2.tracks) == 2  # 2 continued tracks
        assert len(tracks3.tracks) == 2  # 1 active, 1 lost
        assert len(tracks4.tracks) == 3  # 1 continued, 1 recovered/lost, 1 new

        # Check track states
        frame3_active = [t for t in tracks3.tracks if t.state == "active"]
        frame3_lost = [t for t in tracks3.tracks if t.state == "lost"]
        assert len(frame3_active) == 1
        assert len(frame3_lost) == 1


if __name__ == "__main__":
    pytest.main([__file__])
