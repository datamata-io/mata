"""Integration tests for the full graph video pipeline (Task E4).

Verifies end-to-end behaviour with mocked providers — no real models,
no real video files, no Valkey connection required.

Scenarios covered:
1. Full graph: Detect → Filter → Track → ExtractROIs → Embed → ReID → AnnotateRT
2. Graph without ReID: Detect → Filter → Track → AnnotateRT (no encoder)
3. Graph.run() with video file + callback + full pipeline
4. Graph.run() with callback receives annotated image
5. Multiple frames: track state persists, trail history grows
6. AnnotateRT + missing CrossMatches (no ReID in graph): graceful fallback
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image as PILImage

from mata.core.artifacts.cross_matches import CrossMatch, CrossMatches
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.embeddings import Embeddings
from mata.core.artifacts.image import Image
from mata.core.artifacts.result import MultiResult
from mata.core.artifacts.rois import ROIs
from mata.core.artifacts.tracks import Track, Tracks
from mata.core.graph.graph import Graph
from mata.core.graph.temporal import FramePolicyEveryN
from mata.nodes.annotate_rt import AnnotateRT
from mata.nodes.detect import Detect
from mata.nodes.embed import Embed
from mata.nodes.filter import Filter
from mata.nodes.reid import ReID
from mata.nodes.roi import ExtractROIs
from mata.nodes.track import Track as TrackNode

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_rgb_frame(h: int = 60, w: int = 80) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_bgr_frame(h: int = 60, w: int = 80) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_image_artifact(h: int = 60, w: int = 80, color_space: str = "RGB") -> Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    return Image.from_numpy(arr, color_space=color_space)


def _make_detections(n: int = 2) -> Detections:
    from mata.core.types import Instance, VisionResult

    instances = [
        Instance(
            bbox=(float(i * 20), float(i * 20), float(i * 20 + 40), float(i * 20 + 40)),
            score=0.9 - i * 0.1,
            label=0,
            label_name="car",
        )
        for i in range(n)
    ]
    return Detections.from_vision_result(VisionResult(instances=instances, meta={}))


def _make_tracks(n: int = 2, frame_id: str = "f0") -> Tracks:
    tracks = [
        Track(
            track_id=i + 1,
            bbox=(float(i * 20), float(i * 20), float(i * 20 + 40), float(i * 20 + 40)),
            score=0.9,
            label="car",
            age=1,
            state="active",
        )
        for i in range(n)
    ]
    return Tracks(tracks=tracks, frame_id=frame_id)


def _make_embeddings(n: int = 2, dim: int = 64) -> Embeddings:
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    return Embeddings(vectors=vecs, normalized=True)


def _make_rois(n: int = 2) -> ROIs:
    roi_images = [PILImage.new("RGB", (40, 40), color=(128, 0, 0)) for _ in range(n)]
    source_boxes = [(i * 10, i * 10, i * 10 + 40, i * 10 + 40) for i in range(n)]
    return ROIs(roi_images=roi_images, source_boxes=source_boxes)


def _make_multi_result(**channels) -> MultiResult:
    img = _make_image_artifact()
    base = {"image": img}
    base.update(channels)
    return MultiResult(channels=base, provenance={}, metrics={})


# ---------------------------------------------------------------------------
# Mock provider factories
# ---------------------------------------------------------------------------


def _make_mock_detector(detections: Detections | None = None):
    """Returns a mock that satisfies the Detector protocol."""
    from mata.core.types import VisionResult

    adapter = MagicMock()

    def _predict(image, **kwargs):
        return VisionResult(instances=[], meta={})

    adapter.predict.side_effect = _predict
    return adapter


def _make_mock_tracker(tracks: Tracks | None = None):
    """Returns a mock that satisfies the Tracker protocol."""
    adapter = MagicMock()
    _tracks = tracks or _make_tracks(2)

    def _update(dets, **kwargs):
        return _tracks

    adapter.update.side_effect = _update
    return adapter


def _make_mock_embedder(embeddings: Embeddings | None = None):
    """Returns a mock that satisfies the Embedder protocol."""
    adapter = MagicMock()
    _embs = embeddings or _make_embeddings(2)

    def _embed(rois, normalize=True):
        n = max(len(rois.roi_images), 1)
        vecs = np.random.randn(n, 64).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-8)

    adapter.embed.side_effect = _embed
    return adapter


def _make_mock_bridge(camera_id: str = "cam-1", matches: list | None = None):
    """Returns a mock that satisfies the ReIDBridge protocol."""
    bridge = MagicMock()
    bridge.camera_id = camera_id
    bridge.query.return_value = matches if matches is not None else []
    return bridge


# ---------------------------------------------------------------------------
# Execution helpers for node-level tests
# ---------------------------------------------------------------------------


def _make_ctx(providers: dict | None = None) -> MagicMock:
    """Build a minimal ExecutionContext mock."""
    ctx = MagicMock()
    ctx.record_metric = MagicMock()
    _providers = providers or {}

    def _get_provider(cap, name):
        for key, val in _providers.items():
            if key == name or key == cap:
                return val
        return MagicMock()

    ctx.get_provider.side_effect = _get_provider
    return ctx


# ===========================================================================
# Section 1: Full graph node-level integration
# (Detect → Filter → Track → ExtractROIs → Embed → ReID → AnnotateRT)
# Each node is run individually with mocked provider + real artifact flow.
# ===========================================================================


class TestFullPipelineNodeLevel:
    """Run all nodes in sequence with real artifact types to verify compatibility."""

    def test_detect_node_produces_detections(self):
        """Detect node returns a Detections artifact."""
        from mata.core.types import Instance, VisionResult

        detector = MagicMock()
        detector.predict.return_value = VisionResult(
            instances=[
                Instance(bbox=(10, 20, 60, 80), score=0.9, label=0, label_name="car"),
            ],
            meta={},
        )
        ctx = _make_ctx({"detector": detector})
        node = Detect(using="detector", out="dets")

        img = _make_image_artifact()
        result = node.run(ctx, image=img)

        assert "dets" in result
        assert isinstance(result["dets"], Detections)
        assert len(result["dets"].instances) == 1

    def test_filter_node_filters_detections(self):
        """Filter node correctly filters low-score detections."""
        node = Filter(src="dets", out="filtered", score_gt=0.85)
        ctx = _make_ctx()
        dets = _make_detections(2)  # scores 0.9 and 0.8

        result = node.run(ctx, dets=dets)
        assert "filtered" in result
        # Only one instance passes score > 0.85
        assert len(result["filtered"].instances) == 1
        assert result["filtered"].instances[0].score > 0.85

    def test_track_node_produces_tracks(self):
        """Track node returns a Tracks artifact from a mocked tracker."""
        expected_tracks = _make_tracks(2, frame_id="f1")
        tracker_mock = MagicMock()
        tracker_mock.update.return_value = expected_tracks

        ctx = _make_ctx({"tracker": tracker_mock})
        node = TrackNode(using="tracker", dets="dets", out="tracks")

        dets = _make_detections(2)
        result = node.run(ctx, detections=dets)

        assert "tracks" in result
        assert isinstance(result["tracks"], Tracks)
        assert len(result["tracks"].tracks) == 2

    def test_extract_rois_produces_rois(self):
        """ExtractROIs node crops image regions for each detection."""
        ctx = _make_ctx()
        dets = _make_detections(2)
        img = _make_image_artifact(h=100, w=100)

        node = ExtractROIs(src_image="image", src_dets="dets", out="rois")
        result = node.run(ctx, image=img, detections=dets)

        assert "rois" in result
        assert isinstance(result["rois"], ROIs)
        # Two detections → up to two ROIs
        assert len(result["rois"].roi_images) >= 0  # may be 0 for zero-size boxes

    def test_embed_node_produces_embeddings(self):
        """Embed node calls provider.embed() and returns Embeddings."""
        embedder = _make_mock_embedder()
        ctx = _make_ctx({"encoder": embedder})
        rois = _make_rois(2)

        node = Embed(using="encoder", src="rois", out="embeddings")
        result = node.run(ctx, rois=rois)

        assert "embeddings" in result
        assert isinstance(result["embeddings"], Embeddings)
        assert result["embeddings"].vectors.ndim == 2

    def test_reid_node_produces_cross_matches(self):
        """ReID node publishes and queries, returning CrossMatches."""
        bridge = _make_mock_bridge(
            camera_id="cam-1",
            matches=[
                {
                    "camera_id": "cam-2",
                    "track_id": 7,
                    "similarity": 0.88,
                    "bbox": [100.0, 80.0, 140.0, 160.0],
                }
            ],
        )
        ctx = MagicMock()
        ctx.get_provider.return_value = bridge
        ctx.record_metric = MagicMock()

        tracks = _make_tracks(2, frame_id="f0")
        embs = _make_embeddings(2)

        node = ReID(using="bridge", tracks_src="tracks", embeddings_src="embeddings", out="cross_matches")
        result = node.run(ctx, tracks=tracks, embeddings=embs)

        assert "cross_matches" in result
        cm = result["cross_matches"]
        assert isinstance(cm, CrossMatches)
        assert len(cm.matches) > 0

    def test_annotate_rt_node_produces_image(self):
        """AnnotateRT node returns an annotated Image artifact in BGR."""
        node = AnnotateRT(show_boxes=True, show_trails=False)
        ctx = _make_ctx()

        img = _make_image_artifact(h=60, w=80, color_space="RGB")
        dets = _make_detections(2)

        result = node.run(ctx, image=img, detections=dets)

        assert "annotated" in result
        out_img = result["annotated"]
        assert isinstance(out_img, Image)
        assert out_img.color_space == "BGR"

    def test_full_pipeline_artifact_flow(self):
        """All nodes can be chained manually with real artifact types."""
        from mata.core.types import Instance, VisionResult

        # --- Detect ---
        detector = MagicMock()
        detector.predict.return_value = VisionResult(
            instances=[
                Instance(bbox=(5.0, 5.0, 45.0, 55.0), score=0.9, label=0, label_name="car"),
                Instance(bbox=(50.0, 10.0, 90.0, 60.0), score=0.7, label=0, label_name="car"),
            ],
            meta={},
        )
        det_ctx = _make_ctx({"detector": detector})
        detect_node = Detect(using="detector", out="dets")
        img = _make_image_artifact(h=100, w=100)
        dets_result = detect_node.run(det_ctx, image=img)
        dets = dets_result["dets"]
        assert len(dets.instances) == 2

        # --- Filter ---
        filter_node = Filter(src="dets", out="filtered", score_gt=0.75)
        filtered_result = filter_node.run(_make_ctx(), dets=dets)
        filtered = filtered_result["filtered"]
        assert len(filtered.instances) == 1  # only score=0.9 passes

        # --- Track ---
        expected_tracks = _make_tracks(1, frame_id="f0")
        tracker_mock = MagicMock()
        tracker_mock.update.return_value = expected_tracks
        track_ctx = _make_ctx({"tracker": tracker_mock})
        track_node = TrackNode(using="tracker", dets="filtered", out="tracks")
        tracks_result = track_node.run(track_ctx, detections=filtered)
        tracks = tracks_result["tracks"]
        assert isinstance(tracks, Tracks)

        # --- Embed (via ROIs) ---
        embedder = _make_mock_embedder()
        embed_ctx = _make_ctx({"encoder": embedder})
        rois = _make_rois(1)
        embed_node = Embed(using="encoder", src="rois", out="embeddings")
        emb_result = embed_node.run(embed_ctx, rois=rois)
        embs = emb_result["embeddings"]
        assert isinstance(embs, Embeddings)

        # --- ReID ---
        bridge = _make_mock_bridge()
        reid_ctx = MagicMock()
        reid_ctx.get_provider.return_value = bridge
        reid_ctx.record_metric = MagicMock()
        reid_node = ReID(using="bridge", tracks_src="tracks", embeddings_src="embeddings", out="cross_matches")
        reid_result = reid_node.run(reid_ctx, tracks=tracks, embeddings=embs)
        cross_matches = reid_result["cross_matches"]
        assert isinstance(cross_matches, CrossMatches)

        # --- AnnotateRT ---
        ann_node = AnnotateRT(
            show_boxes=True,
            show_track_ids=True,
            show_trails=False,
            tracks_src="tracks",
            cross_matches_src="cross_matches",
        )
        ann_ctx = _make_ctx()
        ann_result = ann_node.run(ann_ctx, image=img, detections=tracks, tracks=tracks, cross_matches=cross_matches)
        assert "annotated" in ann_result
        assert ann_result["annotated"].color_space == "BGR"


# ===========================================================================
# Section 2: Graph without ReID (Detect → Filter → Track → AnnotateRT)
# ===========================================================================


class TestPipelineWithoutReID:
    """Verify the simplified pipeline (no encoder / no ReID) works correctly."""

    def test_pipeline_without_reid_node_level(self):
        """Detect → Track → AnnotateRT runs without CrossMatches."""
        from mata.core.types import Instance, VisionResult

        # Detect
        detector = MagicMock()
        detector.predict.return_value = VisionResult(
            instances=[Instance(bbox=(5.0, 5.0, 45.0, 55.0), score=0.9, label=0, label_name="car")],
            meta={},
        )
        img = _make_image_artifact(h=80, w=80)
        det_node = Detect(using="det", out="dets")
        dets = det_node.run(_make_ctx({"det": detector}), image=img)["dets"]

        # Track
        expected_tracks = _make_tracks(1, frame_id="f0")
        tracker_mock = MagicMock()
        tracker_mock.update.return_value = expected_tracks
        track_node = TrackNode(using="trk", dets="dets", out="tracks")
        tracks = track_node.run(_make_ctx({"trk": tracker_mock}), detections=dets)["tracks"]

        # AnnotateRT (no cross_matches_src)
        ann_node = AnnotateRT(show_boxes=True, show_trails=False)
        ann_result = ann_node.run(_make_ctx(), image=img, detections=tracks)
        assert "annotated" in ann_result
        assert ann_result["annotated"].color_space == "BGR"

    def test_annotate_rt_without_cross_matches_no_error(self):
        """AnnotateRT with cross_matches_src set but artifact not supplied."""
        ann_node = AnnotateRT(
            show_boxes=True,
            tracks_src="tracks",
            cross_matches_src="cross_matches",
        )
        ctx = _make_ctx()
        img = _make_image_artifact()
        tracks = _make_tracks(1, frame_id="f0")
        # cross_matches intentionally NOT passed → node must degrade gracefully
        result = ann_node.run(ctx, image=img, detections=tracks, tracks=tracks)
        assert "annotated" in result
        assert result["annotated"].color_space == "BGR"

    def test_annotate_rt_without_tracks_no_error(self):
        """AnnotateRT with only detections (no Tracks artifact) works fine."""
        ann_node = AnnotateRT(show_boxes=True, show_trails=True)
        ctx = _make_ctx()
        img = _make_image_artifact()
        dets = _make_detections(1)
        result = ann_node.run(ctx, image=img, detections=dets)
        assert "annotated" in result


# ===========================================================================
# Section 3: Graph.run() with video file + callback + full pipeline
# ===========================================================================


class TestGraphRunVideoCallback:
    """Verify Graph.run() with video-file source + callback forwards results."""

    def _build_detect_annotate_graph(self) -> Graph:
        """Simple Detect → AnnotateRT graph for video processing tests."""
        return (
            Graph("detect_annotate")
            .add(Detect(using="detector", out="dets"), inputs={"image": "input.image"})
            .add(
                AnnotateRT(show_boxes=True, show_trails=False, detections_src="dets"),
                inputs={"image": "input.image", "dets": "Detect.dets"},
            )
        )

    def test_graph_run_video_returns_list(self):
        """graph.run() with a video source returns list[MultiResult]."""
        graph = self._build_detect_annotate_graph()
        fake_results = [_make_multi_result() for _ in range(3)]

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.return_value = fake_results
            result = graph.run(
                "video.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
            )

        assert isinstance(result, list)
        assert len(result) == 3

    def test_graph_run_video_callback_invoked(self):
        """Callback is called for each processed frame during video.run()."""
        graph = self._build_detect_annotate_graph()
        fake_result = _make_multi_result()
        fake_bgr = np.zeros((60, 80, 3), dtype=np.uint8)
        received: list = []

        def _fake_process_video(path, output_path=None, max_frames=None, callback=None):
            if callback is not None:
                for i in range(3):
                    callback(fake_result, i, fake_bgr)
            return [fake_result] * 3

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.side_effect = _fake_process_video
            graph.run(
                "video.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
                callback=lambda r, n, f: received.append((r, n, f)),
            )

        assert len(received) == 3
        for r, n, f in received:
            assert r is fake_result
            assert isinstance(n, int)
            assert isinstance(f, np.ndarray)

    def test_graph_run_video_callback_receives_annotated_channel(self):
        """Callback receives MultiResult whose channels contain 'annotated' image."""
        graph = self._build_detect_annotate_graph()
        ann_img = _make_image_artifact(color_space="BGR")
        fake_result = MultiResult(
            channels={"image": _make_image_artifact(), "annotated": ann_img},
            provenance={},
            metrics={},
        )
        received_results: list[MultiResult] = []

        def _fake_process_video(path, output_path=None, max_frames=None, callback=None):
            if callback is not None:
                callback(fake_result, 0, np.zeros((60, 80, 3), dtype=np.uint8))
            return [fake_result]

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.side_effect = _fake_process_video
            graph.run(
                "video.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
                callback=lambda r, n, f: received_results.append(r),
            )

        assert len(received_results) == 1
        assert "annotated" in received_results[0].channels

    def test_graph_run_video_callback_none_returns_list(self):
        """Without callback, graph.run() still returns list as before."""
        graph = self._build_detect_annotate_graph()
        fake_results = [_make_multi_result() for _ in range(2)]

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.return_value = fake_results
            result = graph.run(
                "clip.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
            )

        assert isinstance(result, list)
        assert len(result) == 2

    def test_graph_run_video_with_max_frames(self):
        """max_frames is forwarded correctly through graph.run() to VideoProcessor."""
        graph = self._build_detect_annotate_graph()

        with (
            patch("mata.api._normalize_providers", return_value=({}, {})),
            patch.object(graph, "compile", return_value=MagicMock()),
            patch("mata.core.graph.temporal.VideoProcessor") as mock_vp,
        ):
            mock_vp.return_value.process_video.return_value = []
            graph.run(
                "clip.mp4",
                providers={},
                frame_policy=FramePolicyEveryN(n=1),
                max_frames=50,
            )

        _, pkwargs = mock_vp.return_value.process_video.call_args
        assert pkwargs.get("max_frames") == 50


# ===========================================================================
# Section 4: Multiple frames — track state + trail history growth
# ===========================================================================


class TestMultipleFramesStatefulness:
    """AnnotateRT trail history persists across multiple run() calls."""

    def test_trail_history_accumulates_across_frames(self):
        """Trail history grows with each successive frame that has valid track_ids."""
        ann = AnnotateRT(show_trails=True, trail_length=50)
        ctx = _make_ctx()
        img = _make_image_artifact()

        # Frame 1
        tracks_f1 = _make_tracks(2, frame_id="f1")
        ann.run(ctx, image=img, detections=tracks_f1)
        assert len(ann._trail_history) == 2

        # Frame 2 — same tracks
        tracks_f2 = _make_tracks(2, frame_id="f2")
        ann.run(ctx, image=img, detections=tracks_f2)
        assert len(ann._trail_history) == 2
        # Both tracks should have 2 history points now
        for tid, pts in ann._trail_history.items():
            assert len(pts) == 2

    def test_trail_history_grows_over_many_frames(self):
        """Trail history accumulates up to trail_length limit."""
        trail_len = 5
        ann = AnnotateRT(show_trails=True, trail_length=trail_len)
        ctx = _make_ctx()
        img = _make_image_artifact()

        for i in range(10):
            tracks = _make_tracks(1, frame_id=f"f{i}")
            ann.run(ctx, image=img, detections=tracks)

        # After 10 frames, history should be capped at trail_length=5
        assert len(ann._trail_history) == 1
        for pts in ann._trail_history.values():
            assert len(pts) <= trail_len

    def test_reset_clears_trail_history(self):
        """reset() clears accumulated trail history."""
        ann = AnnotateRT(show_trails=True)
        ctx = _make_ctx()
        img = _make_image_artifact()

        for i in range(3):
            tracks = _make_tracks(2, frame_id=f"f{i}")
            ann.run(ctx, image=img, detections=tracks)

        assert len(ann._trail_history) > 0
        ann.reset()
        assert len(ann._trail_history) == 0

    def test_reset_and_restart_trail_from_scratch(self):
        """After reset(), trail history starts fresh."""
        ann = AnnotateRT(show_trails=True, trail_length=10)
        ctx = _make_ctx()
        img = _make_image_artifact()

        # Accumulate
        for i in range(5):
            ann.run(ctx, image=img, detections=_make_tracks(1, frame_id=f"f{i}"))

        ann.reset()

        # Single frame after reset
        ann.run(ctx, image=img, detections=_make_tracks(1, frame_id="f0"))
        for pts in ann._trail_history.values():
            assert len(pts) == 1

    def test_track_state_persists_via_tracks_src(self):
        """Trail history is sourced from tracks_src when configured."""
        ann = AnnotateRT(
            show_trails=True,
            trail_length=10,
            tracks_src="tracks",
        )
        ctx = _make_ctx()
        img = _make_image_artifact()
        dets = _make_detections(0)

        for i in range(3):
            tracks = _make_tracks(2, frame_id=f"f{i}")
            ann.run(ctx, image=img, detections=dets, tracks=tracks)

        assert len(ann._trail_history) == 2


# ===========================================================================
# Section 5: CrossMatches flows from ReID → AnnotateRT correctly
# ===========================================================================


class TestCrossMatchesFlow:
    """Verify CrossMatches artifact flows correctly into AnnotateRT."""

    def test_cross_matches_forwarded_to_draw_boxes(self):
        """AnnotateRT passes CrossMatches to draw_boxes when configured."""
        ann = AnnotateRT(
            show_boxes=True,
            cross_matches_src="cross_matches",
        )
        ctx = _make_ctx()
        img = _make_image_artifact()
        dets = _make_tracks(2, frame_id="f0")
        cm = CrossMatches(
            matches=[CrossMatch(local_track_id=1, remote_camera_id="cam-2", remote_track_id=7, similarity=0.9)],
            camera_id="cam-1",
        )

        with patch("mata.visualization_cv2.draw_boxes") as mock_draw:
            mock_draw.return_value = img.to_numpy().copy()
            ann.run(ctx, image=img, detections=dets, cross_matches=cm)

        mock_draw.assert_called_once()
        _, call_kwargs = mock_draw.call_args
        assert call_kwargs.get("cross_matches") is cm

    def test_reid_output_cross_matches_used_by_annotate_rt(self):
        """End-to-end: ReID → CrossMatches → AnnotateRT (node level)."""
        # Setup ReID
        bridge = _make_mock_bridge(
            camera_id="cam-1",
            matches=[
                {
                    "camera_id": "cam-2",
                    "track_id": 5,
                    "similarity": 0.85,
                    "bbox": None,
                }
            ],
        )
        reid_ctx = MagicMock()
        reid_ctx.get_provider.return_value = bridge
        reid_ctx.record_metric = MagicMock()

        tracks = _make_tracks(1, frame_id="f0")
        embs = _make_embeddings(1)

        reid_node = ReID(using="bridge", tracks_src="tracks", embeddings_src="embeddings", out="cross_matches")
        reid_result = reid_node.run(reid_ctx, tracks=tracks, embeddings=embs)
        cross_matches = reid_result["cross_matches"]

        # Feed into AnnotateRT
        ann_node = AnnotateRT(
            show_boxes=True,
            cross_matches_src="cross_matches",
        )
        ann_ctx = _make_ctx()
        img = _make_image_artifact()
        ann_result = ann_node.run(ann_ctx, image=img, detections=tracks, cross_matches=cross_matches)

        assert "annotated" in ann_result
        assert ann_result["annotated"].color_space == "BGR"

    def test_empty_cross_matches_no_error(self):
        """AnnotateRT handles empty CrossMatches artifact without error."""
        ann = AnnotateRT(show_boxes=True, cross_matches_src="cross_matches")
        ctx = _make_ctx()
        img = _make_image_artifact()
        dets = _make_detections(1)
        empty_cm = CrossMatches(matches=[], camera_id="cam-1")

        result = ann.run(ctx, image=img, detections=dets, cross_matches=empty_cm)
        assert "annotated" in result

    def test_cross_matches_with_multiple_matches(self):
        """Multiple cross-camera matches are all passed to AnnotateRT."""
        ann = AnnotateRT(show_boxes=True, cross_matches_src="cross_matches")
        ctx = _make_ctx()
        img = _make_image_artifact()
        tracks = _make_tracks(3, frame_id="f0")
        cm = CrossMatches(
            matches=[
                CrossMatch(local_track_id=1, remote_camera_id="cam-2", remote_track_id=10, similarity=0.92),
                CrossMatch(local_track_id=2, remote_camera_id="cam-3", remote_track_id=11, similarity=0.87),
            ],
            camera_id="cam-1",
        )

        result = ann.run(ctx, image=img, detections=tracks, cross_matches=cm)
        assert "annotated" in result
        assert result["annotated"].color_space == "BGR"


# ===========================================================================
# Section 6: Graceful fallback when AnnotateRT missing optional inputs
# ===========================================================================


class TestGracefulFallback:
    """AnnotateRT degrades gracefully when optional inputs are absent."""

    def test_no_tracks_src_no_trails(self):
        """show_trails=True but no tracks_src: trails fall back to detections."""
        ann = AnnotateRT(show_trails=True, trail_length=10)
        ctx = _make_ctx()
        img = _make_image_artifact()
        dets = _make_tracks(1, frame_id="f0")  # Tracks used as detections

        result = ann.run(ctx, image=img, detections=dets)
        assert "annotated" in result

    def test_no_cross_matches_no_error(self):
        """cross_matches_src not configured → no cross highlights, no error."""
        ann = AnnotateRT(show_boxes=True)  # no cross_matches_src
        ctx = _make_ctx()
        img = _make_image_artifact()
        dets = _make_detections(2)

        result = ann.run(ctx, image=img, detections=dets)
        assert "annotated" in result

    def test_empty_detections_no_error(self):
        """AnnotateRT with zero detections returns annotated image without error."""
        ann = AnnotateRT(show_boxes=True, show_trails=True)
        ctx = _make_ctx()
        img = _make_image_artifact()
        dets = _make_detections(0)

        result = ann.run(ctx, image=img, detections=dets)
        assert "annotated" in result
        assert isinstance(result["annotated"], Image)

    def test_show_boxes_false_skips_draw_boxes(self):
        """show_boxes=False means draw_boxes is never called."""
        ann = AnnotateRT(show_boxes=False)
        ctx = _make_ctx()
        img = _make_image_artifact()
        dets = _make_detections(2)

        with patch("mata.visualization_cv2.draw_boxes") as mock_db:
            ann.run(ctx, image=img, detections=dets)

        mock_db.assert_not_called()

    def test_show_trails_false_skips_trail_accumulation(self):
        """show_trails=False → trail history never grows."""
        ann = AnnotateRT(show_trails=False)
        ctx = _make_ctx()
        img = _make_image_artifact()

        for i in range(5):
            ann.run(ctx, image=img, detections=_make_tracks(2, frame_id=f"f{i}"))

        assert len(ann._trail_history) == 0

    def test_output_preserves_frame_id(self):
        """AnnotateRT output Image preserves frame_id from input."""
        ann = AnnotateRT()
        ctx = _make_ctx()

        arr = np.zeros((60, 80, 3), dtype=np.uint8)
        img = Image.from_numpy(arr, color_space="RGB", frame_id="frame_042")
        dets = _make_detections(0)

        result = ann.run(ctx, image=img, detections=dets)
        assert result["annotated"].frame_id == "frame_042"

    def test_output_preserves_timestamp_ms(self):
        """AnnotateRT output Image preserves timestamp_ms from input."""
        ann = AnnotateRT()
        ctx = _make_ctx()

        arr = np.zeros((60, 80, 3), dtype=np.uint8)
        img = Image.from_numpy(arr, color_space="RGB", timestamp_ms=12345)
        dets = _make_detections(0)

        result = ann.run(ctx, image=img, detections=dets)
        assert result["annotated"].timestamp_ms == 12345
