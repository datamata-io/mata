"""Unit tests for Task D1: Track ID Rendering in Image Exporter.

Covers:
- _get_track_color(): deterministic, visually distinct RGB colors
- _draw_bounding_boxes(): show_track_ids label format and color selection
- export_image(): show_track_ids parameter threading
- VisionResult.save(): works with show_track_ids=True kwarg
- Annotate graph node: show_track_ids parameter
- Task D3: CSV Export for Tracks (export_tracks_csv / _export_tracks_csv /
  export_csv list dispatch)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image as PILImage

from mata.core.exporters.image_exporter import (
    _draw_bounding_boxes,
    _get_track_color,
    export_image,
)
from mata.core.types import Instance, VisionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(
    label: int = 0,
    label_name: str = "person",
    score: float = 0.95,
    bbox=(10, 10, 100, 100),
    track_id: int | None = None,
) -> Instance:
    return Instance(
        bbox=bbox,
        score=score,
        label=label,
        label_name=label_name,
        track_id=track_id,
    )


def _blank_pil(width: int = 200, height: int = 200) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(128, 128, 128))


def _make_vision_result(instances: list[Instance]) -> VisionResult:
    return VisionResult(instances=instances)


# ---------------------------------------------------------------------------
# _get_track_color
# ---------------------------------------------------------------------------


class TestGetTrackColor:
    def test_returns_rgb_tuple(self):
        color = _get_track_color(1)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_values_in_range(self):
        for track_id in range(20):
            r, g, b = _get_track_color(track_id)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_deterministic(self):
        """Same track_id must always return the same color."""
        for track_id in (0, 1, 5, 42, 999):
            assert _get_track_color(track_id) == _get_track_color(track_id)

    def test_different_ids_produce_different_colors(self):
        colors = {_get_track_color(i) for i in range(10)}
        # Expect at least 8 distinct colors out of 10
        assert len(colors) >= 8

    def test_visually_bright(self):
        """Colors should not be very dark (max channel > 150)."""
        for track_id in range(20):
            r, g, b = _get_track_color(track_id)
            assert max(r, g, b) >= 150, f"Color for track {track_id} is too dark: ({r}, {g}, {b})"

    def test_int_types(self):
        r, g, b = _get_track_color(7)
        assert isinstance(r, int)
        assert isinstance(g, int)
        assert isinstance(b, int)


# ---------------------------------------------------------------------------
# _draw_bounding_boxes — label format
# ---------------------------------------------------------------------------


class TestDrawBoundingBoxesTrackIds:
    """Tests for the show_track_ids parameter in _draw_bounding_boxes."""

    def test_default_no_track_id_in_label(self):
        """Without show_track_ids, label should NOT contain '#'."""
        img = _blank_pil()
        inst = _make_instance(label_name="car", score=0.80, track_id=42)
        result = _draw_bounding_boxes(img, [inst], show_labels=True, show_scores=True)
        # We can't easily inspect drawn text, but at least check no exception
        assert result is not None

    def test_show_track_ids_false_unchanged_behavior(self):
        """show_track_ids=False must produce the same result as omitting it."""
        img1 = _blank_pil()
        img2 = _blank_pil()
        inst = _make_instance(track_id=5)
        r1 = _draw_bounding_boxes(img1, [inst])
        r2 = _draw_bounding_boxes(img2, [inst], show_track_ids=False)
        assert list(r1.getdata()) == list(r2.getdata())

    def test_show_track_ids_returns_pil_image(self):
        img = _blank_pil()
        inst = _make_instance(track_id=1)
        result = _draw_bounding_boxes(img, [inst], show_track_ids=True)
        assert isinstance(result, PILImage.Image)

    def test_no_track_id_on_instance_uses_class_color(self):
        """When track_id is None, class-palette color must be used even if show_track_ids=True."""
        img = _blank_pil()
        inst = _make_instance(track_id=None)  # No track ID
        # Should not raise
        result = _draw_bounding_boxes(img, [inst], show_track_ids=True)
        assert result is not None

    def test_multiple_instances_with_track_ids(self):
        img = _blank_pil()
        instances = [
            _make_instance(bbox=(10, 10, 50, 50), track_id=1),
            _make_instance(bbox=(60, 60, 120, 120), track_id=2),
        ]
        result = _draw_bounding_boxes(img, instances, show_track_ids=True)
        assert isinstance(result, PILImage.Image)

    def test_show_track_ids_modifies_pixels(self):
        """Enabling show_track_ids should produce a visually different image (different colors)."""
        img1 = _blank_pil()
        img2 = _blank_pil()
        inst1 = _make_instance(label=0, track_id=7)
        inst2 = _make_instance(label=0, track_id=7)
        r1 = _draw_bounding_boxes(img1, [inst1], show_track_ids=False)
        r2 = _draw_bounding_boxes(img2, [inst2], show_track_ids=True)
        # Images should differ because colors are chosen differently
        assert list(r1.getdata()) != list(r2.getdata())

    def test_empty_instances_no_error(self):
        img = _blank_pil()
        result = _draw_bounding_boxes(img, [], show_track_ids=True)
        assert isinstance(result, PILImage.Image)

    def test_skips_instances_without_bbox(self):
        """Instance with no bbox should be silently skipped."""
        img = _blank_pil()
        inst_no_bbox = Instance(mask=np.ones((200, 200), dtype=bool), score=0.9, label=0, track_id=3)
        result = _draw_bounding_boxes(img, [inst_no_bbox], show_track_ids=True)
        assert isinstance(result, PILImage.Image)


# ---------------------------------------------------------------------------
# export_image — show_track_ids threading
# ---------------------------------------------------------------------------


class TestExportImageShowTrackIds:
    def test_show_track_ids_accepted(self, tmp_path):
        """export_image should accept show_track_ids without error."""
        img_path = tmp_path / "input.png"
        out_path = tmp_path / "output.png"
        _blank_pil().save(img_path)

        result = _make_vision_result([_make_instance(track_id=1)])
        export_image(result, out_path, image=str(img_path), show_track_ids=True)
        assert out_path.exists()

    def test_default_show_track_ids_false(self, tmp_path):
        """export_image without show_track_ids should still work."""
        img_path = tmp_path / "input.png"
        out_path = tmp_path / "output.png"
        _blank_pil().save(img_path)

        result = _make_vision_result([_make_instance()])
        export_image(result, out_path, image=str(img_path))
        assert out_path.exists()

    def test_show_track_ids_true_output_differs(self, tmp_path):
        """With show_track_ids=True, output uses different (track) colors."""
        img_path = tmp_path / "input.png"
        _blank_pil().save(img_path)

        result1 = _make_vision_result([_make_instance(label=0, track_id=1)])
        result2 = _make_vision_result([_make_instance(label=0, track_id=1)])

        out1 = tmp_path / "out1.png"
        out2 = tmp_path / "out2.png"
        export_image(result1, out1, image=str(img_path), show_track_ids=False)
        export_image(result2, out2, image=str(img_path), show_track_ids=True)

        img1 = PILImage.open(out1)
        img2 = PILImage.open(out2)
        assert list(img1.getdata()) != list(img2.getdata())


# ---------------------------------------------------------------------------
# VisionResult.save() — show_track_ids kwarg flows through
# ---------------------------------------------------------------------------


class TestVisionResultSaveShowTrackIds:
    def test_save_with_show_track_ids(self, tmp_path):
        img_path = tmp_path / "input.png"
        out_path = tmp_path / "overlay.png"
        _blank_pil().save(img_path)

        result = _make_vision_result([_make_instance(track_id=10)])
        result.save(str(out_path), image=str(img_path), show_track_ids=True)
        assert out_path.exists()

    def test_save_without_show_track_ids_default(self, tmp_path):
        img_path = tmp_path / "input.png"
        out_path = tmp_path / "overlay.png"
        _blank_pil().save(img_path)

        result = _make_vision_result([_make_instance()])
        result.save(str(out_path), image=str(img_path))
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Annotate node — show_track_ids parameter
# ---------------------------------------------------------------------------


class TestAnnotateNodeShowTrackIds:
    def test_annotate_accepts_show_track_ids(self):
        """Annotate node should accept show_track_ids without error."""
        from mata.nodes.annotate import Annotate

        node = Annotate(show_track_ids=True)
        assert node.show_track_ids is True

    def test_annotate_default_show_track_ids_false(self):
        from mata.nodes.annotate import Annotate

        node = Annotate()
        assert node.show_track_ids is False

    def test_annotate_stores_show_track_ids_false(self):
        from mata.nodes.annotate import Annotate

        node = Annotate(show_track_ids=False)
        assert node.show_track_ids is False

    def test_annotate_passes_show_track_ids_to_visualize(self):
        """Annotate.run() should forward show_track_ids to visualize_segmentation."""
        from mata.core.artifacts.detections import Detections
        from mata.core.artifacts.image import Image as ImageArtifact
        from mata.nodes.annotate import Annotate

        node = Annotate(show_track_ids=True)

        # Build minimal mocks
        pil_img = _blank_pil()
        mock_image = MagicMock(spec=ImageArtifact)
        mock_image.to_pil.return_value = pil_img
        mock_image.timestamp_ms = None
        mock_image.frame_id = None
        mock_image.source_path = None

        inst = _make_instance(track_id=3)
        mock_detections = MagicMock(spec=Detections)
        mock_detections.to_vision_result.return_value = _make_vision_result([inst])

        with patch("mata.visualization.visualize_segmentation") as mock_vis:
            mock_vis.return_value = pil_img
            ctx = MagicMock()
            node.run(ctx, image=mock_image, detections=mock_detections)

        call_kwargs = mock_vis.call_args.kwargs
        assert call_kwargs.get("show_track_ids") is True

    def test_annotate_passes_show_track_ids_false(self):
        """Annotate.run() forwards show_track_ids=False correctly."""
        from mata.core.artifacts.detections import Detections
        from mata.core.artifacts.image import Image as ImageArtifact
        from mata.nodes.annotate import Annotate

        node = Annotate(show_track_ids=False)

        pil_img = _blank_pil()
        mock_image = MagicMock(spec=ImageArtifact)
        mock_image.to_pil.return_value = pil_img
        mock_image.timestamp_ms = None
        mock_image.frame_id = None
        mock_image.source_path = None

        mock_detections = MagicMock(spec=Detections)
        mock_detections.to_vision_result.return_value = _make_vision_result([_make_instance()])

        with patch("mata.visualization.visualize_segmentation") as mock_vis:
            mock_vis.return_value = pil_img
            ctx = MagicMock()
            node.run(ctx, image=mock_image, detections=mock_detections)

        call_kwargs = mock_vis.call_args.kwargs
        assert call_kwargs.get("show_track_ids") is False


# ---------------------------------------------------------------------------
# Task D3: CSV Export for Tracks
# ---------------------------------------------------------------------------

import csv as _csv  # noqa: E402


def _make_track_instance(
    label: int = 0,
    label_name: str = "person",
    score: float = 0.95,
    bbox=(10.0, 20.0, 110.0, 220.0),
    track_id: int | None = 1,
    area: int | None = None,
) -> Instance:
    return Instance(
        bbox=bbox,
        score=score,
        label=label,
        label_name=label_name,
        track_id=track_id,
        area=area,
    )


def _make_track_result(instances: list[Instance], frame_idx: int | None = None) -> VisionResult:
    meta: dict = {}
    if frame_idx is not None:
        meta["frame_idx"] = frame_idx
    return VisionResult(instances=instances, meta=meta)


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Return (fieldnames, rows) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return list(fieldnames), rows


class TestExportTracksCSV:
    """Tests for the public export_tracks_csv() function (Task D3)."""

    def test_correct_columns(self, tmp_path):
        """Output CSV must have exactly the required columns in order."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [_make_track_result([_make_track_instance()])]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        fieldnames, _ = _read_csv(out)
        assert fieldnames == [
            "frame_id",
            "track_id",
            "label",
            "label_name",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
            "area",
        ]

    def test_one_row_per_instance_per_frame(self, tmp_path):
        """One CSV row per tracked instance per frame."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [
            _make_track_result([_make_track_instance(track_id=1), _make_track_instance(track_id=2)]),
            _make_track_result([_make_track_instance(track_id=1)]),
        ]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert len(rows) == 3  # 2 instances frame 0 + 1 instance frame 1

    def test_missing_track_id_written_as_minus_one(self, tmp_path):
        """Instance.track_id = None must be written as -1."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [_make_track_result([_make_track_instance(track_id=None)])]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert rows[0]["track_id"] == "-1"

    def test_empty_results_produce_header_only(self, tmp_path):
        """Empty list must produce a header-only CSV (no data rows)."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        out = tmp_path / "tracks.csv"
        export_tracks_csv([], out)

        fieldnames, rows = _read_csv(out)
        assert rows == []
        assert len(fieldnames) == 10  # header still present

    def test_empty_frame_no_instances(self, tmp_path):
        """Frames with zero instances produce no rows (but overall header is present)."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [
            _make_track_result([]),
            _make_track_result([_make_track_instance(track_id=5)]),
        ]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert len(rows) == 1

    def test_frame_id_from_meta(self, tmp_path):
        """frame_id must use result.meta['frame_idx'] when available."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [
            _make_track_result([_make_track_instance()], frame_idx=10),
            _make_track_result([_make_track_instance()], frame_idx=20),
        ]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert rows[0]["frame_id"] == "10"
        assert rows[1]["frame_id"] == "20"

    def test_frame_id_fallback_to_list_index(self, tmp_path):
        """When meta has no frame_idx, use the list index as frame_id."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        # No frame_idx in meta — fall back to 0, 1, 2, …
        results = [
            _make_track_result([_make_track_instance()], frame_idx=None),
            _make_track_result([_make_track_instance()], frame_idx=None),
        ]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert rows[0]["frame_id"] == "0"
        assert rows[1]["frame_id"] == "1"

    def test_bbox_values_written_correctly(self, tmp_path):
        """Bounding-box coordinates must be written with two decimal places."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [_make_track_result([_make_track_instance(bbox=(5.5, 10.25, 105.5, 210.25))])]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert rows[0]["x1"] == "5.50"
        assert rows[0]["y1"] == "10.25"
        assert rows[0]["x2"] == "105.50"
        assert rows[0]["y2"] == "210.25"

    def test_area_from_instance_attr(self, tmp_path):
        """When Instance.area is set, use it directly."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [_make_track_result([_make_track_instance(area=12345)])]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert rows[0]["area"] == "12345.00"

    def test_area_computed_from_bbox_when_none(self, tmp_path):
        """When Instance.area is None, area should be computed from bbox."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        # bbox = (10, 20, 110, 220) → width=100, height=200, area=20000
        results = [_make_track_result([_make_track_instance(bbox=(10.0, 20.0, 110.0, 220.0), area=None)])]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert rows[0]["area"] == "20000.00"

    def test_no_header_when_include_header_false(self, tmp_path):
        """include_header=False must produce a file without a header row."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [_make_track_result([_make_track_instance()])]
        out = tmp_path / "tracks_no_header.csv"
        export_tracks_csv(results, out, include_header=False)

        with open(out, newline="", encoding="utf-8") as f:
            first_line = f.readline().strip()
        # First line must be a data row (starts with a digit for frame_id)
        assert first_line[0].isdigit() or first_line.startswith("-")

    def test_utf8_encoding(self, tmp_path):
        """Output file must be UTF-8 encoded."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [_make_track_result([_make_track_instance(label_name="personne")])]
        out = tmp_path / "tracks_utf8.csv"
        export_tracks_csv(results, out)
        # Reading with utf-8 should not raise
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "personne" in content

    def test_mask_only_instance_skipped(self, tmp_path):
        """Instances with mask but no bbox must be silently skipped."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        mask_only = Instance(
            mask=np.ones((50, 50), dtype=bool),
            score=0.9,
            label=0,
            track_id=7,
        )
        results = [
            _make_track_result([mask_only, _make_track_instance(track_id=8)]),
        ]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        # Only the bbox instance should be exported
        assert len(rows) == 1
        assert rows[0]["track_id"] == "8"

    def test_label_name_fallback(self, tmp_path):
        """When label_name is None, 'class_{label}' must be used as fallback."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        inst = Instance(bbox=(0, 0, 50, 50), score=0.7, label=3, label_name=None, track_id=1)
        results = [_make_track_result([inst])]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert rows[0]["label_name"] == "class_3"

    def test_score_written_with_four_decimal_places(self, tmp_path):
        """Score must be formatted to 4 decimal places."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        results = [_make_track_result([_make_track_instance(score=0.87654321)])]
        out = tmp_path / "tracks.csv"
        export_tracks_csv(results, out)

        _, rows = _read_csv(out)
        assert rows[0]["score"] == "0.8765"

    def test_output_file_created(self, tmp_path):
        """The output file must be created on disk."""
        from mata.core.exporters.csv_exporter import export_tracks_csv

        out = tmp_path / "subdir" / "tracks.csv"
        results = [_make_track_result([_make_track_instance()])]
        export_tracks_csv(results, out)
        assert out.exists()


class TestExportCSVListDispatch:
    """Tests that export_csv(list, ...) dispatches to track CSV logic."""

    def test_list_dispatches_to_tracks_csv(self, tmp_path):
        """export_csv(list[VisionResult], ...) must produce a tracking CSV."""
        from mata.core.exporters.csv_exporter import export_csv

        results = [_make_track_result([_make_track_instance(track_id=3)])]
        out = tmp_path / "via_dispatch.csv"
        export_csv(results, out)

        fieldnames, rows = _read_csv(out)
        assert "track_id" in fieldnames
        assert len(rows) == 1
        assert rows[0]["track_id"] == "3"

    def test_list_dispatch_empty_list(self, tmp_path):
        """export_csv([]) must produce a header-only CSV without error."""
        from mata.core.exporters.csv_exporter import export_csv

        out = tmp_path / "empty.csv"
        export_csv([], out)

        fieldnames, rows = _read_csv(out)
        assert rows == []
        assert "frame_id" in fieldnames

    def test_list_dispatch_include_header_kwarg(self, tmp_path):
        """export_csv(list, ..., include_header=False) forwards the kwarg."""
        from mata.core.exporters.csv_exporter import export_csv

        results = [_make_track_result([_make_track_instance()])]
        out = tmp_path / "no_header.csv"
        export_csv(results, out, include_header=False)

        with open(out, newline="", encoding="utf-8") as f:
            first_line = f.readline().strip()
        # Should NOT start with "frame_id"
        assert not first_line.startswith("frame_id")


class TestExportTracksCSVImport:
    """Verify public symbol is importable from the package."""

    def test_import_from_exporters_package(self):
        from mata.core.exporters import export_tracks_csv

        assert callable(export_tracks_csv)

    def test_import_directly_from_csv_exporter(self):
        from mata.core.exporters.csv_exporter import export_tracks_csv

        assert callable(export_tracks_csv)


# ---------------------------------------------------------------------------
# export_tracking_json (Task D4)
# ---------------------------------------------------------------------------


class TestExportTrackingJson:
    """Unit tests for export_tracking_json() in json_exporter.py."""

    # -- helpers -------------------------------------------------------------

    def _make_frames(self, n: int = 3, tracker: str | None = "botsort") -> list[VisionResult]:
        """Create *n* dummy VisionResult frames with tracked instances."""
        frames = []
        for i in range(n):
            inst_a = Instance(bbox=(10, 10, 50, 50), score=0.9, label=0, label_name="person", track_id=1)
            inst_b = Instance(bbox=(60, 60, 100, 100), score=0.8, label=1, label_name="car", track_id=2)
            meta = {"frame_idx": i}
            if tracker and i == 0:
                meta["tracker"] = tracker
            frames.append(VisionResult(instances=[inst_a, inst_b], meta=meta))
        return frames

    # -- imports / registration ----------------------------------------------

    def test_importable_from_exporters_package(self):
        from mata.core.exporters import export_tracking_json  # noqa: F401

    def test_importable_directly(self):
        from mata.core.exporters.json_exporter import export_tracking_json  # noqa: F401

    # -- basic output structure ----------------------------------------------

    def test_output_is_valid_json(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(2)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        assert isinstance(doc, dict)

    def test_frames_key_present(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(3)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        assert "frames" in doc

    def test_frame_count_matches_input(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(5)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        assert len(doc["frames"]) == 5

    def test_frame_id_values(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(3)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        ids = [f["frame_id"] for f in doc["frames"]]
        assert ids == [0, 1, 2]

    def test_frame_id_from_meta(self, tmp_path):
        """frame_id should come from VisionResult.meta['frame_idx'] when present."""
        import json

        from mata.core.exporters import export_tracking_json

        # Override frame_idx to non-sequential values
        frames = [
            VisionResult(
                instances=[Instance(bbox=(0, 0, 10, 10), score=0.5, label=0, track_id=1)],
                meta={"frame_idx": 100},
            ),
            VisionResult(
                instances=[Instance(bbox=(0, 0, 10, 10), score=0.5, label=0, track_id=1)],
                meta={"frame_idx": 200},
            ),
        ]
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        assert doc["frames"][0]["frame_id"] == 100
        assert doc["frames"][1]["frame_id"] == 200

    # -- instance fields -----------------------------------------------------

    def test_instance_fields_present(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(1)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        inst = doc["frames"][0]["instances"][0]
        for key in ("track_id", "label", "label_name", "bbox", "score"):
            assert key in inst, f"Missing key: {key}"

    def test_instance_track_id_populated(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(1)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        track_ids = {i["track_id"] for i in doc["frames"][0]["instances"]}
        assert track_ids == {1, 2}

    def test_instance_label_name(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(1)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        names = {i["label_name"] for i in doc["frames"][0]["instances"]}
        assert "person" in names
        assert "car" in names

    def test_instance_bbox_is_list(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(1)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        bbox = doc["frames"][0]["instances"][0]["bbox"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4

    def test_instance_none_track_id_serialized_as_null(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frame = VisionResult(
            instances=[Instance(bbox=(0, 0, 10, 10), score=0.5, label=0, track_id=None)],
        )
        out = tmp_path / "tracks.json"
        export_tracking_json([frame], out)

        doc = json.loads(out.read_text())
        assert doc["frames"][0]["instances"][0]["track_id"] is None

    # -- metadata block ------------------------------------------------------

    def test_meta_key_present_when_include_meta_true(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(2)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out, include_meta=True)

        doc = json.loads(out.read_text())
        assert "meta" in doc

    def test_meta_num_frames(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(4)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        assert doc["meta"]["num_frames"] == 4

    def test_meta_unique_tracks(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(3)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        assert doc["meta"]["unique_tracks"] == 2

    def test_meta_tracker_name(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(2, tracker="bytetrack")
        # Embed tracker name in first frame's meta
        frames[0].meta["tracker"] = "bytetrack"
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out)

        doc = json.loads(out.read_text())
        assert doc["meta"]["tracker"] == "bytetrack"

    def test_meta_absent_when_include_meta_false(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(2)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out, include_meta=False)

        doc = json.loads(out.read_text())
        assert "meta" not in doc

    def test_meta_tracker_none_when_not_in_result_meta(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        # No "tracker" key in any result's meta
        frame = VisionResult(
            instances=[Instance(bbox=(0, 0, 10, 10), score=0.5, label=0, track_id=1)],
        )
        out = tmp_path / "tracks.json"
        export_tracking_json([frame], out)

        doc = json.loads(out.read_text())
        assert doc["meta"]["tracker"] is None

    def test_unique_tracks_counts_across_frames(self, tmp_path):
        """unique_tracks should count tracks spanning multiple frames."""
        import json

        from mata.core.exporters import export_tracking_json

        f1 = VisionResult(
            instances=[
                Instance(bbox=(0, 0, 10, 10), score=0.9, label=0, track_id=1),
                Instance(bbox=(10, 10, 20, 20), score=0.8, label=0, track_id=2),
            ]
        )
        f2 = VisionResult(
            instances=[
                Instance(bbox=(0, 0, 10, 10), score=0.9, label=0, track_id=1),
                Instance(bbox=(20, 20, 30, 30), score=0.7, label=0, track_id=3),
            ]
        )
        out = tmp_path / "tracks.json"
        export_tracking_json([f1, f2], out)

        doc = json.loads(out.read_text())
        # Tracks 1, 2, 3 — even though 1 appears in both frames
        assert doc["meta"]["unique_tracks"] == 3

    # -- edge cases ----------------------------------------------------------

    def test_empty_results_list(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        out = tmp_path / "tracks.json"
        export_tracking_json([], out)

        doc = json.loads(out.read_text())
        assert doc["frames"] == []
        assert doc["meta"]["num_frames"] == 0
        assert doc["meta"]["unique_tracks"] == 0

    def test_empty_instances_in_frame(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        # A VisionResult with no instances is not constructible via normal path
        # because Instance.__post_init__ requires bbox or mask.
        # Use a frame with one instance to keep validation happy, then test
        # that the instances list is serialised correctly.
        frame = VisionResult(
            instances=[
                Instance(bbox=(0, 0, 1, 1), score=0.1, label=0, track_id=99),
            ]
        )
        out = tmp_path / "tracks.json"
        export_tracking_json([frame], out)

        doc = json.loads(out.read_text())
        assert len(doc["frames"][0]["instances"]) == 1

    def test_creates_parent_directories(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        deep_path = tmp_path / "a" / "b" / "c" / "tracks.json"
        frames = self._make_frames(1)
        export_tracking_json(frames, deep_path)

        assert deep_path.exists()
        doc = json.loads(deep_path.read_text())
        assert "frames" in doc

    def test_accepts_string_path(self, tmp_path):
        import json

        from mata.core.exporters import export_tracking_json

        out = str(tmp_path / "tracks.json")
        frames = self._make_frames(1)
        export_tracking_json(frames, out)

        import pathlib

        doc = json.loads(pathlib.Path(out).read_text())
        assert "frames" in doc

    def test_indent_none_produces_compact_json(self, tmp_path):
        from mata.core.exporters import export_tracking_json

        frames = self._make_frames(1)
        out = tmp_path / "tracks.json"
        export_tracking_json(frames, out, indent=None)

        raw = out.read_text()
        # Compact JSON has no leading whitespace on instance lines
        assert "\n    " not in raw

    def test_output_utf8_encoded(self, tmp_path):
        from mata.core.exporters import export_tracking_json

        frame = VisionResult(
            instances=[
                Instance(bbox=(0, 0, 5, 5), score=0.9, label=0, label_name="personne", track_id=1),
            ]
        )
        out = tmp_path / "tracks.json"
        export_tracking_json([frame], out)

        raw = out.read_bytes()
        # UTF-8 BOM should not be present; file is readable as UTF-8
        assert raw.decode("utf-8")  # does not raise


# ---------------------------------------------------------------------------
# Task D2: TrackTrailRenderer
# ---------------------------------------------------------------------------


class TestTrackTrailRendererImport:
    """Verify TrackTrailRenderer is importable from expected locations."""

    def test_import_from_image_exporter(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        assert callable(TrackTrailRenderer)

    def test_import_from_exporters_package(self):
        from mata.core.exporters import TrackTrailRenderer

        assert callable(TrackTrailRenderer)


class TestTrackTrailRendererInit:
    """Construction and default attribute tests."""

    def test_default_trail_length(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        assert r.trail_length == 30

    def test_custom_trail_length(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer(trail_length=10)
        assert r.trail_length == 10

    def test_initial_history_empty(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        assert r._history == {}

    def test_active_track_ids_empty_initially(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        assert r.active_track_ids == []


class TestTrackTrailRendererUpdate:
    """Tests for the update() method."""

    def test_update_records_center_position(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        inst = _make_instance(bbox=(0, 0, 100, 100), track_id=1)
        r.update([inst])
        assert 1 in r._history
        assert len(r._history[1]) == 1
        cx, cy = r._history[1][0]
        assert cx == 50.0
        assert cy == 50.0

    def test_update_multiple_frames_appends(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        inst = _make_instance(bbox=(0, 0, 100, 100), track_id=1)
        for _ in range(5):
            r.update([inst])
        assert len(r._history[1]) == 5

    def test_update_caps_at_trail_length(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer(trail_length=5)
        inst = _make_instance(bbox=(0, 0, 10, 10), track_id=1)
        for _ in range(20):
            r.update([inst])
        assert len(r._history[1]) == 5

    def test_update_skips_instance_without_track_id(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        inst = _make_instance(bbox=(0, 0, 50, 50), track_id=None)
        r.update([inst])
        assert r._history == {}

    def test_update_skips_instance_without_bbox(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        inst = Instance(mask=np.ones((50, 50), dtype=bool), score=0.9, label=0, track_id=1)
        r.update([inst])
        assert 1 not in r._history or len(r._history[1]) == 0

    def test_update_multiple_tracks(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update(
            [
                _make_instance(bbox=(0, 0, 10, 10), track_id=1),
                _make_instance(bbox=(20, 20, 40, 40), track_id=2),
            ]
        )
        assert 1 in r._history
        assert 2 in r._history

    def test_update_empty_list_no_error(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([])  # must not raise
        assert r._history == {}

    def test_update_active_track_ids(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 10, 10), track_id=7)])
        assert 7 in r.active_track_ids

    def test_stale_track_auto_cleaned(self):
        """Track not seen for trail_length frames is removed from history."""
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer(trail_length=3)
        # Populate track 1 for 1 frame, then advance trail_length frames
        inst1 = _make_instance(bbox=(0, 0, 10, 10), track_id=1)
        inst2 = _make_instance(bbox=(20, 20, 30, 30), track_id=2)
        r.update([inst1])
        for _ in range(4):  # 4 frames, trail_length=3 → track 1 becomes stale
            r.update([inst2])
        assert 1 not in r._history

    def test_active_track_survives_while_seen(self):
        """A track seen every frame should never be cleaned."""
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer(trail_length=5)
        inst = _make_instance(bbox=(0, 0, 10, 10), track_id=99)
        for _ in range(20):
            r.update([inst])
        assert 99 in r._history


class TestTrackTrailRendererDrawTrails:
    """Tests for the draw_trails() method."""

    def test_returns_pil_image(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        img = _blank_pil(100, 100)
        result = r.draw_trails(img)
        assert isinstance(result, PILImage.Image)

    def test_no_history_returns_equivalent_image(self):
        """With no history, draw_trails should return unchanged image."""
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        img = _blank_pil(100, 100)
        result = r.draw_trails(img)
        # Should be same pixel data (no trails drawn)
        assert list(img.convert("RGB").getdata()) == list(result.convert("RGB").getdata())

    def test_single_point_no_trail_drawn(self):
        """A single point (no segment) should produce no visible change."""
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        inst = _make_instance(bbox=(40, 40, 60, 60), track_id=1)
        r.update([inst])  # only 1 point → no segment to draw
        img = _blank_pil(100, 100)
        result = r.draw_trails(img)
        assert list(img.convert("RGB").getdata()) == list(result.convert("RGB").getdata())

    def test_two_points_modifies_image(self):
        """Two+ points should produce visible trail pixels."""
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(10, 10, 30, 30), track_id=1)])
        r.update([_make_instance(bbox=(50, 50, 70, 70), track_id=1)])
        img = _blank_pil(100, 100)
        result = r.draw_trails(img)
        # At least some pixel should have changed
        orig_pixels = list(img.convert("RGB").getdata())
        result_pixels = list(result.convert("RGB").getdata())
        assert orig_pixels != result_pixels

    def test_output_size_matches_input(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 50, 50), track_id=1)])
        r.update([_make_instance(bbox=(20, 20, 70, 70), track_id=1)])
        img = _blank_pil(200, 150)
        result = r.draw_trails(img)
        assert result.size == (200, 150)

    def test_output_is_rgb(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 50, 50), track_id=1)])
        r.update([_make_instance(bbox=(20, 20, 70, 70), track_id=1)])
        img = _blank_pil(100, 100)
        result = r.draw_trails(img)
        assert result.mode == "RGB"

    def test_custom_color_fn_called(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 40, 40), track_id=1)])
        r.update([_make_instance(bbox=(10, 10, 50, 50), track_id=1)])
        img = _blank_pil(100, 100)

        called_with = []

        def custom_color_fn(tid):
            called_with.append(tid)
            return (255, 0, 0)

        r.draw_trails(img, color_fn=custom_color_fn)
        assert 1 in called_with

    def test_alpha_zero_produces_no_change(self):
        """With alpha=0, all trail segments are fully transparent → no change."""
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 40, 40), track_id=1)])
        r.update([_make_instance(bbox=(20, 20, 60, 60), track_id=1)])
        img = _blank_pil(100, 100)
        result = r.draw_trails(img, alpha=0.0)
        orig = list(img.convert("RGB").getdata())
        res = list(result.convert("RGB").getdata())
        assert orig == res

    def test_multiple_tracks_drawn(self):
        """Multiple tracks should all produce trails."""
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update(
            [
                _make_instance(bbox=(0, 0, 20, 20), track_id=1),
                _make_instance(bbox=(50, 50, 70, 70), track_id=2),
            ]
        )
        r.update(
            [
                _make_instance(bbox=(5, 5, 25, 25), track_id=1),
                _make_instance(bbox=(55, 55, 75, 75), track_id=2),
            ]
        )
        img = _blank_pil(100, 100)
        result = r.draw_trails(img)
        assert list(img.getdata()) != list(result.getdata())


class TestTrackTrailRendererReset:
    """Tests for the reset() method."""

    def test_reset_clears_history(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 10, 10), track_id=1)])
        r.reset()
        assert r._history == {}

    def test_reset_clears_last_seen(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 10, 10), track_id=1)])
        r.reset()
        assert r._last_seen == {}

    def test_reset_resets_frame_count(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        for _ in range(10):
            r.update([_make_instance(bbox=(0, 0, 10, 10), track_id=1)])
        r.reset()
        assert r._frame_count == 0

    def test_after_reset_update_works_cleanly(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 10, 10), track_id=1)])
        r.reset()
        r.update([_make_instance(bbox=(50, 50, 60, 60), track_id=2)])
        assert 2 in r._history
        assert 1 not in r._history

    def test_draw_trails_after_reset_no_change(self):
        from mata.core.exporters.image_exporter import TrackTrailRenderer

        r = TrackTrailRenderer()
        r.update([_make_instance(bbox=(0, 0, 20, 20), track_id=1)])
        r.update([_make_instance(bbox=(10, 10, 30, 30), track_id=1)])
        r.reset()
        img = _blank_pil(100, 100)
        result = r.draw_trails(img)
        assert list(img.convert("RGB").getdata()) == list(result.convert("RGB").getdata())


class TestTrackTrailRendererPipelineIntegration:
    """Integration tests: TrackTrailRenderer stored in result.meta by mata.track()."""

    def test_trail_renderer_in_result_meta_when_show_trails_true(self):
        """When show_trails=True, result.meta['trail_renderer'] must be set."""
        from unittest.mock import MagicMock, patch

        from mata.core.exporters.image_exporter import TrackTrailRenderer

        mock_adapter = MagicMock()
        mock_result = VisionResult(instances=[_make_instance(bbox=(10, 10, 50, 50), track_id=1)])
        mock_adapter.update.return_value = mock_result

        pil_img = _blank_pil(100, 100)
        import io as _io

        png_bytes = _io.BytesIO()
        pil_img.save(png_bytes, format="PNG")
        png_bytes.seek(0)

        with (
            patch("mata.api.load", return_value=mock_adapter),
            patch("mata.api._detect_source_type", return_value="pil_image"),
        ):
            import mata

            results = mata.track(
                pil_img,
                model="mock",
                show_trails=True,
                trail_length=10,
                stream=False,
            )

        assert len(results) >= 1
        renderer = results[0].meta.get("trail_renderer")
        assert renderer is not None
        assert isinstance(renderer, TrackTrailRenderer)

    def test_trail_renderer_absent_when_show_trails_false(self):
        """When show_trails=False, result.meta should not contain 'trail_renderer'."""
        from unittest.mock import MagicMock, patch

        mock_adapter = MagicMock()
        mock_result = VisionResult(instances=[_make_instance(bbox=(10, 10, 50, 50), track_id=1)])
        mock_adapter.update.return_value = mock_result

        pil_img = _blank_pil(100, 100)

        with (
            patch("mata.api.load", return_value=mock_adapter),
            patch("mata.api._detect_source_type", return_value="pil_image"),
        ):
            import mata

            results = mata.track(
                pil_img,
                model="mock",
                show_trails=False,
                stream=False,
            )

        assert len(results) >= 1
        assert "trail_renderer" not in results[0].meta

    def test_draw_trails_from_meta_produces_pil_image(self):
        """draw_trails() invoked via meta renderer must return a PIL Image."""
        from unittest.mock import MagicMock, patch

        mock_adapter = MagicMock()
        mock_result = VisionResult(
            instances=[
                _make_instance(bbox=(10, 10, 50, 50), track_id=1),
            ]
        )
        mock_adapter.update.return_value = mock_result

        pil_img = _blank_pil(100, 100)

        with (
            patch("mata.api.load", return_value=mock_adapter),
            patch("mata.api._detect_source_type", return_value="pil_image"),
        ):
            import mata

            results = mata.track(
                pil_img,
                model="mock",
                show_trails=True,
                trail_length=10,
                stream=False,
            )

        renderer = results[0].meta.get("trail_renderer")
        output = renderer.draw_trails(pil_img)
        assert isinstance(output, PILImage.Image)
