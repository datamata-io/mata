"""Tests for the AnnotateRT node and visualization_cv2 helpers (Task E3).

Tests cover:
=== visualization_cv2 helpers ===
1.  test_track_color_deterministic           — same ID → same color
2.  test_track_color_different_ids           — different IDs → different colors
3.  test_track_color_returns_bgr_tuple       — returns 3-tuple in [0,255]
4.  test_draw_boxes_basic                    — passes through frame when cv2 absent
5.  test_draw_boxes_calls_cv2_rectangle      — cv2.rectangle called per instance
6.  test_draw_boxes_with_labels              — label text built from instance attrs
7.  test_draw_boxes_no_labels               — show_labels=False omits label
8.  test_draw_boxes_no_scores               — show_scores=False omits score
9.  test_draw_boxes_no_track_ids            — show_track_ids=False omits #id
10. test_draw_boxes_with_cross_matches       — yellow highlight rect for matched
11. test_draw_boxes_empty_instances         — no crash on empty list
12. test_draw_boxes_returns_frame           — return value is same array
13. test_draw_trails_basic                  — cv2.line called for segments
14. test_draw_trails_empty_history          — no crash / no cv2.line on empty
15. test_draw_trails_single_pt_per_track    — single point → no line drawn
16. test_draw_trails_returns_frame          — return value is same array
17. test_draw_camera_label_basic            — cv2.rectangle + putText called
18. test_draw_camera_label_returns_frame    — return value is same array
19. test_draw_boxes_no_cv2_returns_frame    — ImportError → frame returned
20. test_draw_trails_no_cv2_returns_frame   — ImportError → frame returned
21. test_draw_camera_label_no_cv2_returns_frame — ImportError → frame returned

=== AnnotateRT node ===
22. test_annotate_rt_default_inputs_outputs           — protocol check
23. test_annotate_rt_init_stores_config               — all params stored
24. test_annotate_rt_dynamic_inputs_with_tracks_src   — tracks_src adds to inputs
25. test_annotate_rt_dynamic_inputs_with_cross_matches_src — cross_matches_src
26. test_annotate_rt_dynamic_inputs_both_optional     — both optional srcs added
27. test_annotate_rt_default_outputs                  — default out key is annotated
28. test_annotate_rt_custom_out_key                   — custom out → outputs updated
29. test_annotate_rt_basic_run                        — image + detections → annotated Image
30. test_annotate_rt_output_color_space_bgr            — output.color_space == "BGR"
31. test_annotate_rt_preserves_frame_id               — frame_id propagated
32. test_annotate_rt_preserves_timestamp_ms           — timestamp_ms propagated
33. test_annotate_rt_with_tracks_src                  — track IDs added to history
34. test_annotate_rt_trail_history_accumulates        — history grows across run() calls
35. test_annotate_rt_reset_clears_trail_history       — reset() empties dict
36. test_annotate_rt_show_trails_false_no_history     — show_trails=False → no accumulation
37. test_annotate_rt_with_cross_matches_input         — cross_matches forwarded to draw_boxes
38. test_annotate_rt_with_camera_label                — draw_camera_label called
39. test_annotate_rt_no_camera_label                  — draw_camera_label NOT called
40. test_annotate_rt_missing_optional_tracks          — no crash if tracks key absent
41. test_annotate_rt_missing_optional_cross_matches   — no crash if cross_matches absent
42. test_annotate_rt_show_boxes_false                 — draw_boxes NOT called
43. test_annotate_rt_empty_instances                  — no crash when detections empty
44. test_annotate_rt_records_num_instances_metric     — ctx.record_metric called
45. test_annotate_rt_rgb_input_converted_to_bgr       — RGB input flipped
46. test_annotate_rt_exported_from_nodes_package      — importable via mata.nodes
47. test_annotate_rt_repr                             — __repr__ contains key fields
48. test_annotate_rt_trail_trimmed_to_trail_length    — old points pruned
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from mata.core.artifacts.cross_matches import CrossMatch, CrossMatches
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.tracks import Track, Tracks
from mata.core.graph.context import ExecutionContext
from mata.nodes.annotate_rt import AnnotateRT
from mata.visualization_cv2 import draw_boxes, draw_camera_label, draw_trails, track_color

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> ExecutionContext:
    """Minimal ExecutionContext — no providers needed for AnnotateRT."""
    return ExecutionContext(providers={}, device="cpu")


def _make_image(
    w: int = 120,
    h: int = 80,
    color_space: str = "BGR",
    frame_id: str | None = "f001",
    timestamp_ms: int | None = 1000,
) -> Image:
    data = np.zeros((h, w, 3), dtype=np.uint8)
    return Image(data=data, width=w, height=h, color_space=color_space, frame_id=frame_id, timestamp_ms=timestamp_ms)


def _make_detections(n: int = 2) -> Detections:
    """Create a minimal Detections artifact with n instances."""
    from mata.core.artifacts.detections import Instance

    instances = [
        Instance(
            bbox=(float(i * 10), float(i * 10), float(i * 10 + 20), float(i * 10 + 20)),
            score=0.9 - i * 0.1,
            label=0,
            label_name="car",
        )
        for i in range(n)
    ]
    return Detections(
        instances=instances,
        instance_ids=[f"i{i:04d}" for i in range(n)],
    )


def _make_tracks(n: int = 2, frame_id: str = "f001") -> Tracks:
    """Create a minimal Tracks artifact with n active tracks."""
    tracks = [
        Track(
            track_id=i + 1,
            bbox=(float(i * 30), 10.0, float(i * 30 + 20), 30.0),
            score=0.85,
            label="person",
            state="active",
        )
        for i in range(n)
    ]
    return Tracks(tracks=tracks, frame_id=frame_id)


def _make_cross_matches() -> CrossMatches:
    """CrossMatches with one match for track_id=1."""
    return CrossMatches(
        matches=[CrossMatch(local_track_id=1, remote_camera_id="cam-2", remote_track_id=5, similarity=0.88)]
    )


def _blank_frame(h: int = 80, w: int = 120) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Mock cv2 factory
# ---------------------------------------------------------------------------


def _make_mock_cv2() -> MagicMock:
    """Return a mock cv2 module with the constants AnnotateRT uses."""
    cv2 = MagicMock(name="cv2")
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.FILLED = -1
    cv2.LINE_AA = 16
    cv2.COLOR_RGB2BGR = 4
    # getTextSize returns (size_tuple, baseline)
    cv2.getTextSize.return_value = ((40, 12), 2)
    # cvtColor / rectangle / putText / line operate in-place; return None
    cv2.cvtColor.side_effect = lambda frame, *a, **kw: frame
    return cv2


# ===========================================================================
# Part 1 — visualization_cv2 helpers
# ===========================================================================


class TestTrackColor:
    """Tests for track_color()."""

    def test_track_color_deterministic(self):
        """Same track_id → identical color on repeated calls."""
        assert track_color(7) == track_color(7)
        assert track_color(42) == track_color(42)

    def test_track_color_different_ids(self):
        """Different IDs should produce different colors (high probability)."""
        colors = {track_color(i) for i in range(20)}
        assert len(colors) > 1

    def test_track_color_returns_bgr_tuple(self):
        """Result is a 3-tuple with values in [0, 255]."""
        color = track_color(3)
        assert isinstance(color, tuple)
        assert len(color) == 3
        for ch in color:
            assert 0 <= ch <= 255

    def test_track_color_saturation_boost(self):
        """At least one channel should be >= 128 (saturation boost applied)."""
        # The implementation boosts R when max(r,g,b) < 128
        color = track_color(0)
        assert max(color) >= 128


class TestDrawBoxes:
    """Tests for draw_boxes()."""

    def test_draw_boxes_returns_frame(self):
        """Return value is the same array object."""
        frame = _blank_frame()
        instances: list[Any] = []
        with patch.dict(sys.modules, {"cv2": _make_mock_cv2()}):
            result = draw_boxes(frame, instances)
        assert result is frame

    def test_draw_boxes_empty_instances(self):
        """No crash and frame unchanged for empty instances list."""
        frame = _blank_frame()
        with patch.dict(sys.modules, {"cv2": _make_mock_cv2()}):
            result = draw_boxes(frame, [])
        assert result is frame

    def test_draw_boxes_calls_cv2_rectangle(self):
        """cv2.rectangle should be called at least once per instance."""
        frame = _blank_frame()
        inst = MagicMock()
        inst.bbox = (10.0, 10.0, 50.0, 50.0)
        inst.track_id = 1
        inst.score = 0.9
        inst.label_name = "car"

        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_boxes(frame, [inst])
        assert mock_cv2.rectangle.called

    def test_draw_boxes_with_labels(self):
        """Label text passed to putText."""
        frame = _blank_frame()
        inst = MagicMock()
        inst.bbox = (5.0, 5.0, 40.0, 40.0)
        inst.track_id = None
        inst.score = None
        inst.label_name = "dog"

        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_boxes(frame, [inst], show_labels=True, show_scores=False, show_track_ids=False)

        # putText should have been called with text containing "dog"
        assert mock_cv2.putText.called
        call_args = mock_cv2.putText.call_args
        assert "dog" in call_args[0][1]

    def test_draw_boxes_no_labels(self):
        """show_labels=False: label not included in text (only score present)."""
        frame = _blank_frame()
        inst = MagicMock()
        inst.bbox = (5.0, 5.0, 40.0, 40.0)
        inst.track_id = None
        inst.score = 0.75
        inst.label_name = "cat"

        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_boxes(frame, [inst], show_labels=False, show_scores=True, show_track_ids=False)

        if mock_cv2.putText.called:
            text = mock_cv2.putText.call_args[0][1]
            assert "cat" not in text

    def test_draw_boxes_no_scores(self):
        """show_scores=False: score not in overlay text."""
        frame = _blank_frame()
        inst = MagicMock()
        inst.bbox = (5.0, 5.0, 40.0, 40.0)
        inst.track_id = None
        inst.score = 0.75
        inst.label_name = "cat"

        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_boxes(frame, [inst], show_labels=True, show_scores=False, show_track_ids=False)

        if mock_cv2.putText.called:
            text = mock_cv2.putText.call_args[0][1]
            assert "0.75" not in text

    def test_draw_boxes_no_track_ids(self):
        """show_track_ids=False: #id not in overlay text."""
        frame = _blank_frame()
        inst = MagicMock()
        inst.bbox = (5.0, 5.0, 40.0, 40.0)
        inst.track_id = 42
        inst.score = None
        inst.label_name = "person"

        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_boxes(frame, [inst], show_labels=True, show_scores=False, show_track_ids=False)

        if mock_cv2.putText.called:
            text = mock_cv2.putText.call_args[0][1]
            assert "#42" not in text

    def test_draw_boxes_with_cross_matches(self):
        """Yellow double-border rectangle drawn for matched track."""
        frame = _blank_frame()
        inst = MagicMock()
        inst.bbox = (10.0, 10.0, 50.0, 50.0)
        inst.track_id = 1
        inst.score = 0.9
        inst.label_name = "person"

        cross = _make_cross_matches()  # track_id=1 matched to cam-2#5
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_boxes(frame, [inst], cross_matches=cross)

        # rectangle should be called at least twice (highlight + main border)
        assert mock_cv2.rectangle.call_count >= 2
        # One call should use the yellow color (0, 255, 255)
        colors_used = [c[0][3] for c in mock_cv2.rectangle.call_args_list]
        assert (0, 255, 255) in colors_used

    def test_draw_boxes_no_cv2_returns_frame(self):
        """When cv2 is not installed, frame is returned unchanged."""
        frame = _blank_frame()
        inst = MagicMock()
        inst.bbox = (0.0, 0.0, 10.0, 10.0)
        # Simulate cv2 import failure by removing it from sys.modules
        with patch.dict(sys.modules, {"cv2": None}, clear=False):
            # patch builtins __import__ for cv2
            import builtins

            real_import = builtins.__import__

            def _import(name, *args, **kwargs):
                if name == "cv2":
                    raise ImportError("mocked absence")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_import):
                from mata.visualization_cv2 import draw_boxes as _draw_boxes

                result = _draw_boxes(frame, [inst])
        assert result is frame


class TestDrawTrails:
    """Tests for draw_trails()."""

    def test_draw_trails_returns_frame(self):
        """Return value is the same array object."""
        frame = _blank_frame()
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = draw_trails(frame, {1: [(10, 10), (20, 20)]})
        assert result is frame

    def test_draw_trails_empty_history(self):
        """No crash and no cv2.line call for empty trail history."""
        frame = _blank_frame()
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_trails(frame, {})
        mock_cv2.line.assert_not_called()

    def test_draw_trails_single_pt_per_track(self):
        """Single point per track → no line drawn."""
        frame = _blank_frame()
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_trails(frame, {1: [(50, 50)]})
        mock_cv2.line.assert_not_called()

    def test_draw_trails_basic(self):
        """cv2.line called for each consecutive point pair."""
        frame = _blank_frame()
        pts = [(10, 10), (20, 15), (30, 20)]
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_trails(frame, {1: pts})
        # 3 points → 2 segments → 2 line calls
        assert mock_cv2.line.call_count == 2

    def test_draw_trails_no_cv2_returns_frame(self):
        """When cv2 absent, frame returned unchanged."""
        frame = _blank_frame()

        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("mocked absence")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            from mata.visualization_cv2 import draw_trails as _dt

            result = _dt(frame, {1: [(0, 0), (10, 10)]})
        assert result is frame


class TestDrawCameraLabel:
    """Tests for draw_camera_label()."""

    def test_draw_camera_label_returns_frame(self):
        """Return value is the same array object."""
        frame = _blank_frame()
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = draw_camera_label(frame, "CAM-1")
        assert result is frame

    def test_draw_camera_label_basic(self):
        """cv2.rectangle and cv2.putText are called with the label text."""
        frame = _blank_frame()
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            draw_camera_label(frame, "CAM-1")
        assert mock_cv2.rectangle.called
        assert mock_cv2.putText.called
        text = mock_cv2.putText.call_args[0][1]
        assert "CAM-1" in text

    def test_draw_camera_label_no_cv2_returns_frame(self):
        """When cv2 absent, frame returned unchanged."""
        frame = _blank_frame()

        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("mocked absence")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            from mata.visualization_cv2 import draw_camera_label as _dcl

            result = _dcl(frame, "CAM-X")
        assert result is frame


# ===========================================================================
# Part 2 — AnnotateRT node unit tests
# ===========================================================================


class TestAnnotateRTInit:
    """Initialisation and protocol tests."""

    def test_default_inputs_outputs(self):
        """Default inputs/outputs declared correctly."""
        node = AnnotateRT()
        assert "image" in node.inputs
        assert "detections" in node.inputs
        assert node.inputs["image"] is Image
        assert "annotated" in node.outputs
        assert node.outputs["annotated"] is Image

    def test_init_stores_config(self):
        """Config attributes stored as expected."""
        node = AnnotateRT(
            show_boxes=False,
            show_labels=False,
            show_scores=False,
            show_track_ids=False,
            show_trails=True,
            trail_length=50,
            camera_label="CAM-X",
            camera_color=(0, 128, 0),
            line_width=3,
            out="out_img",
            image_src="img",
            detections_src="dets",
            tracks_src="trk",
            cross_matches_src="cm",
            name="MyAnnotate",
        )
        assert node.show_boxes is False
        assert node.show_trails is True
        assert node.trail_length == 50
        assert node.camera_label == "CAM-X"
        assert node.camera_color == (0, 128, 0)
        assert node.line_width == 3
        assert node.out == "out_img"
        assert node.image_src == "img"
        assert node.detections_src == "dets"
        assert node.tracks_src == "trk"
        assert node.cross_matches_src == "cm"
        assert node.name == "MyAnnotate"

    def test_dynamic_inputs_with_tracks_src(self):
        """tracks_src is registered in self.inputs when provided."""
        node = AnnotateRT(tracks_src="my_tracks")
        assert "my_tracks" in node.inputs
        assert node.inputs["my_tracks"] is Tracks

    def test_dynamic_inputs_with_cross_matches_src(self):
        """cross_matches_src is registered in self.inputs when provided."""
        from mata.core.artifacts.cross_matches import CrossMatches

        node = AnnotateRT(cross_matches_src="cm")
        assert "cm" in node.inputs
        assert node.inputs["cm"] is CrossMatches

    def test_dynamic_inputs_both_optional(self):
        """Both optional srcs registered together."""
        node = AnnotateRT(tracks_src="trk", cross_matches_src="cm")
        assert "trk" in node.inputs
        assert "cm" in node.inputs

    def test_default_out_key_in_outputs(self):
        """Default output key is 'annotated'."""
        node = AnnotateRT()
        assert "annotated" in node.outputs

    def test_custom_out_key_updates_outputs(self):
        """Custom out key replaces 'annotated' in outputs."""
        node = AnnotateRT(out="frame_out")
        assert "frame_out" in node.outputs
        assert "annotated" not in node.outputs

    def test_trail_history_initially_empty(self):
        """_trail_history starts empty."""
        node = AnnotateRT()
        assert node._trail_history == {}


class TestAnnotateRTRun:
    """run() method tests."""

    def _run(self, node: AnnotateRT, **extra_inputs: Any):
        """Helper: run node with mock cv2, returning (result_dict, mock_cv2)."""
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(2)
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = node.run(ctx, image=image, detections=dets, **extra_inputs)
        return result, mock_cv2

    def test_basic_run_returns_annotated_key(self):
        """run() returns dict with 'annotated' key."""
        node = AnnotateRT()
        result, _ = self._run(node)
        assert "annotated" in result

    def test_output_is_image_artifact(self):
        """Output value is an Image artifact."""
        node = AnnotateRT()
        result, _ = self._run(node)
        assert isinstance(result["annotated"], Image)

    def test_output_color_space_bgr(self):
        """Annotated image has color_space='BGR'."""
        node = AnnotateRT()
        result, _ = self._run(node)
        assert result["annotated"].color_space == "BGR"

    def test_preserves_frame_id(self):
        """frame_id from input image propagated to output."""
        node = AnnotateRT()
        ctx = _make_ctx()
        image = _make_image(frame_id="frame_99")
        dets = _make_detections(1)
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = node.run(ctx, image=image, detections=dets)
        assert result["annotated"].frame_id == "frame_99"

    def test_preserves_timestamp_ms(self):
        """timestamp_ms from input image propagated to output."""
        node = AnnotateRT()
        ctx = _make_ctx()
        image = _make_image(timestamp_ms=9999)
        dets = _make_detections(1)
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = node.run(ctx, image=image, detections=dets)
        assert result["annotated"].timestamp_ms == 9999

    def test_show_boxes_false_no_draw_boxes(self):
        """show_boxes=False: draw_boxes NOT called (cv2.rectangle not called for boxes)."""
        node = AnnotateRT(show_boxes=False)
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(2)
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_boxes") as mock_db,
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=image, detections=dets)
        mock_db.assert_not_called()

    def test_with_camera_label(self):
        """camera_label set → draw_camera_label called."""
        node = AnnotateRT(camera_label="CAM-1")
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(1)
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_camera_label") as mock_dcl,
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_trails"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=image, detections=dets)
        mock_dcl.assert_called_once()

    def test_no_camera_label(self):
        """camera_label=None → draw_camera_label NOT called."""
        node = AnnotateRT(camera_label=None)
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(1)
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_camera_label") as mock_dcl,
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_trails"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=image, detections=dets)
        mock_dcl.assert_not_called()

    def test_cross_matches_forwarded_to_draw_boxes(self):
        """CrossMatches artifact is forwarded to draw_boxes."""
        cross = _make_cross_matches()
        node = AnnotateRT(cross_matches_src="cm")
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(2)
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_boxes") as mock_db,
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=image, detections=dets, cm=cross)
        # draw_boxes called; cross_matches kwarg is our artifact
        assert mock_db.called
        kw = mock_db.call_args[1]
        assert kw.get("cross_matches") is cross

    def test_missing_optional_cross_matches_no_crash(self):
        """cross_matches_src configured but key not passed → no error."""
        node = AnnotateRT(cross_matches_src="cm")
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(1)
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            # cm key not provided — should not raise
            node.run(ctx, image=image, detections=dets)

    def test_trail_history_accumulates(self):
        """_trail_history grows across successive run() calls."""
        node = AnnotateRT(show_trails=True)
        ctx = _make_ctx()
        dets = _make_detections(2)
        mock_cv2 = _make_mock_cv2()

        with (
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            for frame_num in range(3):
                image = _make_image(frame_id=f"f{frame_num:03d}")
                node.run(ctx, image=image, detections=dets)

        # Detections have instances without track_id (None), so trail may
        # not accumulate for them — but no crash is the key guarantee.
        # Using a Tracks artifact with track IDs:
        node2 = AnnotateRT(show_trails=True, tracks_src="trk")
        with (
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            for frame_num in range(3):
                image = _make_image(frame_id=f"f{frame_num:03d}")
                tracks = _make_tracks(1, frame_id=f"f{frame_num:03d}")
                node2.run(ctx, image=image, detections=dets, trk=tracks)

        # track_id=1 should have accumulated 3 centre points
        assert 1 in node2._trail_history
        assert len(node2._trail_history[1]) == 3

    def test_trail_history_trimmed_to_trail_length(self):
        """Trail points are capped at trail_length."""
        trail_length = 5
        node = AnnotateRT(show_trails=True, trail_length=trail_length, tracks_src="trk")
        ctx = _make_ctx()
        dets = _make_detections(0)
        mock_cv2 = _make_mock_cv2()

        with (
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            for i in range(10):
                image = _make_image(frame_id=f"f{i:03d}")
                tracks = _make_tracks(1, frame_id=f"f{i:03d}")
                node.run(ctx, image=image, detections=dets, trk=tracks)

        assert len(node._trail_history[1]) <= trail_length

    def test_reset_clears_trail_history(self):
        """reset() empties _trail_history."""
        node = AnnotateRT(show_trails=True, tracks_src="trk")
        ctx = _make_ctx()
        dets = _make_detections(0)
        mock_cv2 = _make_mock_cv2()

        with (
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            image = _make_image()
            node.run(ctx, image=image, detections=dets, trk=_make_tracks(1))

        assert node._trail_history  # non-empty before reset
        node.reset()
        assert node._trail_history == {}

    def test_show_trails_false_no_accumulation(self):
        """show_trails=False: trail history not updated."""
        node = AnnotateRT(show_trails=False, tracks_src="trk")
        ctx = _make_ctx()
        dets = _make_detections(0)
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=_make_image(), detections=dets, trk=_make_tracks(2))

        assert node._trail_history == {}

    def test_records_num_instances_metric(self):
        """ctx.record_metric called with correct num_instances count."""
        node = AnnotateRT()
        ctx = _make_ctx()
        image = _make_image()
        n = 3
        dets = _make_detections(n)
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=image, detections=dets)

        metrics = ctx.get_metrics()
        node_metrics = metrics.get(node.name, {})
        assert node_metrics.get("num_instances") == n

    def test_rgb_input_converted_to_bgr(self):
        """RGB input image is colour-converted before annotation."""
        node = AnnotateRT()
        ctx = _make_ctx()
        image = _make_image(color_space="RGB")
        dets = _make_detections(0)
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=image, detections=dets)
        mock_cv2.cvtColor.assert_called_once()

    def test_empty_detections_no_crash(self):
        """Detections with zero instances does not crash run()."""
        node = AnnotateRT()
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(0)
        mock_cv2 = _make_mock_cv2()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = node.run(ctx, image=image, detections=dets)
        assert "annotated" in result

    def test_with_tracks_artifact_uses_active_tracks_for_trails(self):
        """When tracks_src is set, active tracks drive trail history updates."""
        node = AnnotateRT(show_trails=True, tracks_src="trk")
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(0)
        tracks = _make_tracks(1)  # track_id=1, active
        mock_cv2 = _make_mock_cv2()
        with (
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=image, detections=dets, trk=tracks)
        assert 1 in node._trail_history

    def test_missing_optional_tracks_src_no_crash(self):
        """tracks_src configured but key absent from inputs → no error."""
        node = AnnotateRT(show_trails=True, tracks_src="trk")
        ctx = _make_ctx()
        image = _make_image()
        dets = _make_detections(1)
        mock_cv2 = _make_mock_cv2()
        # Note: "trk" not passed → should fall back to instances from dets
        with (
            patch("mata.visualization_cv2.draw_boxes"),
            patch("mata.visualization_cv2.draw_trails"),
            patch("mata.visualization_cv2.draw_camera_label"),
            patch.dict(sys.modules, {"cv2": mock_cv2}),
        ):
            node.run(ctx, image=image, detections=dets)  # no "trk" kwarg


class TestAnnotateRTExportAndRepr:
    """Export and repr tests."""

    def test_exported_from_nodes_package(self):
        """AnnotateRT importable from mata.nodes."""
        from mata.nodes import AnnotateRT as _AnnotateRT

        assert _AnnotateRT is AnnotateRT

    def test_repr_contains_key_fields(self):
        """__repr__ includes show_boxes, show_trails, camera_label, out."""
        node = AnnotateRT(show_boxes=False, show_trails=True, camera_label="CAM-A", out="out")
        r = repr(node)
        assert "show_boxes" in r
        assert "show_trails" in r
        assert "camera_label" in r
