"""Unit tests for TrackingAdapter, TrackerConfig, and DetectionResults (B1).

Tests cover:
- TrackerConfig construction, from_dict(), from_yaml() with built-in names and
  custom YAML paths.
- DetectionResults re-export / import from tracking_adapter.
- TrackingAdapter construction (bytetrack, botsort, defaults, invalid type).
- update() pipeline: detect → class filter → convert → track → VisionResult.
- persist=True maintains tracker state across calls.
- persist=False resets tracker before each call.
- Empty detection frames (zero detections, Kalman-predicted output).
- Class filtering via classes= parameter.
- reset() clears tracker state.
- Properties: tracker_type, id2label, config.
- _convert_tracker_output() with and without id2label.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from mata.adapters.tracking_adapter import (
    _BUILTIN_CONFIG_NAMES,
    _BUILTIN_CONFIGS_DIR,
    DetectionResults,
    TrackerConfig,
    TrackingAdapter,
    _resolve_config,
)
from mata.core.types import Instance, VisionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(
    x1: float = 10,
    y1: float = 20,
    x2: float = 110,
    y2: float = 120,
    score: float = 0.9,
    label: int = 0,
    label_name: str | None = "cat",
    track_id: int | None = None,
) -> Instance:
    return Instance(
        bbox=(x1, y1, x2, y2),
        score=score,
        label=label,
        label_name=label_name,
        track_id=track_id,
    )


def _make_vision_result(*instances: Instance) -> VisionResult:
    return VisionResult(instances=list(instances))


def _make_mock_detector(vision_result: VisionResult | None = None) -> MagicMock:
    """Return a mock detector whose predict() returns *vision_result*."""
    detector = MagicMock()
    detector.id2label = {0: "cat", 1: "dog"}
    if vision_result is None:
        vision_result = _make_vision_result(_make_instance())
    detector.predict.return_value = vision_result
    return detector


# ---------------------------------------------------------------------------
# TrackerConfig — construction
# ---------------------------------------------------------------------------


class TestTrackerConfigDefaults:
    def test_default_tracker_type(self):
        cfg = TrackerConfig()
        assert cfg.tracker_type == "botsort"

    def test_default_thresholds(self):
        cfg = TrackerConfig()
        assert cfg.track_high_thresh == 0.5
        assert cfg.track_low_thresh == 0.1
        assert cfg.new_track_thresh == 0.6
        assert cfg.match_thresh == 0.8

    def test_default_buffer(self):
        assert TrackerConfig().track_buffer == 30

    def test_default_fuse_score(self):
        assert TrackerConfig().fuse_score is True

    def test_default_botsort_fields(self):
        cfg = TrackerConfig()
        assert cfg.gmc_method == "sparseOptFlow"
        assert cfg.proximity_thresh == 0.5
        assert cfg.appearance_thresh == 0.25
        assert cfg.with_reid is False

    def test_custom_construction(self):
        cfg = TrackerConfig(tracker_type="bytetrack", track_high_thresh=0.7)
        assert cfg.tracker_type == "bytetrack"
        assert cfg.track_high_thresh == 0.7


# ---------------------------------------------------------------------------
# TrackerConfig.from_dict
# ---------------------------------------------------------------------------


class TestTrackerConfigFromDict:
    def test_empty_dict_yields_defaults(self):
        cfg = TrackerConfig.from_dict({})
        assert cfg == TrackerConfig()

    def test_partial_dict(self):
        cfg = TrackerConfig.from_dict({"tracker_type": "bytetrack"})
        assert cfg.tracker_type == "bytetrack"
        assert cfg.track_high_thresh == 0.5  # default preserved

    def test_all_fields(self):
        d = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.2,
            "new_track_thresh": 0.7,
            "track_buffer": 50,
            "match_thresh": 0.7,
            "fuse_score": False,
            "gmc_method": None,
            "proximity_thresh": 0.4,
            "appearance_thresh": 0.3,
            "with_reid": False,
        }
        cfg = TrackerConfig.from_dict(d)
        assert cfg.tracker_type == "bytetrack"
        assert cfg.track_buffer == 50
        assert cfg.fuse_score is False
        assert cfg.gmc_method is None

    def test_unknown_keys_ignored(self):
        cfg = TrackerConfig.from_dict({"tracker_type": "bytetrack", "unknown_key": 42})
        assert cfg.tracker_type == "bytetrack"

    def test_returns_config_instance(self):
        assert isinstance(TrackerConfig.from_dict({}), TrackerConfig)


# ---------------------------------------------------------------------------
# TrackerConfig.from_yaml — built-in configs
# ---------------------------------------------------------------------------


class TestTrackerConfigFromYamlBuiltin:
    @pytest.mark.parametrize("name", ["bytetrack", "botsort"])
    def test_builtin_config_loads(self, name):
        cfg = TrackerConfig.from_yaml(name)
        assert isinstance(cfg, TrackerConfig)
        assert cfg.tracker_type == name

    @pytest.mark.parametrize("name", ["BYTETRACK", "ByteTrack"])
    def test_case_insensitive_name(self, name):
        cfg = TrackerConfig.from_yaml(name)
        assert cfg.tracker_type in ("bytetrack", "botsort")

    def test_bytetrack_gmc_method_none(self):
        cfg = TrackerConfig.from_yaml("bytetrack")
        assert cfg.gmc_method is None

    def test_botsort_gmc_method_sparse(self):
        cfg = TrackerConfig.from_yaml("botsort")
        assert cfg.gmc_method == "sparseOptFlow"

    def test_name_with_extension(self):
        cfg = TrackerConfig.from_yaml("bytetrack.yaml")
        assert cfg.tracker_type == "bytetrack"

    def test_unknown_name_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TrackerConfig.from_yaml("unknowntracker")


# ---------------------------------------------------------------------------
# TrackerConfig.from_yaml — custom YAML file
# ---------------------------------------------------------------------------


class TestTrackerConfigFromYamlFile:
    def test_load_from_custom_file(self):
        data = "tracker_type: bytetrack\ntrack_high_thresh: 0.75\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            cfg = TrackerConfig.from_yaml(path)
            assert cfg.tracker_type == "bytetrack"
            assert cfg.track_high_thresh == 0.75
        finally:
            os.unlink(path)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TrackerConfig.from_yaml("/tmp/nonexistent_tracker_config_abc123.yaml")


# ---------------------------------------------------------------------------
# _resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_none_returns_default_botsort(self):
        cfg = _resolve_config(None)
        assert cfg.tracker_type == "botsort"

    def test_config_instance_passthrough(self):
        orig = TrackerConfig(tracker_type="bytetrack")
        assert _resolve_config(orig) is orig

    def test_string_bytetrack(self):
        cfg = _resolve_config("bytetrack")
        assert cfg.tracker_type == "bytetrack"

    def test_dict(self):
        cfg = _resolve_config({"tracker_type": "bytetrack"})
        assert cfg.tracker_type == "bytetrack"

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_config(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DetectionResults import
# ---------------------------------------------------------------------------


class TestDetectionResultsImport:
    def test_import_from_tracking_adapter(self):
        """DetectionResults is importable from tracking_adapter module."""
        from mata.adapters.tracking_adapter import DetectionResults as DR  # noqa: N817
        from mata.trackers.byte_tracker import DetectionResults as DR2  # noqa: N814

        assert DR is DR2  # same object (re-export)

    def test_from_vision_result_round_trip(self):
        inst = _make_instance(10, 20, 110, 120, score=0.8, label=1)
        vr = _make_vision_result(inst)
        dr = DetectionResults.from_vision_result(vr)
        assert len(dr) == 1
        np.testing.assert_allclose(dr.conf, [0.8], atol=1e-4)
        np.testing.assert_allclose(dr.xyxy[0], [10, 20, 110, 120], atol=1e-4)

    def test_from_empty_vision_result(self):
        vr = _make_vision_result()
        dr = DetectionResults.from_vision_result(vr)
        assert len(dr) == 0
        assert dr.xyxy.shape == (0, 4)


# ---------------------------------------------------------------------------
# TrackingAdapter — construction
# ---------------------------------------------------------------------------


class TestTrackingAdapterConstruction:
    def test_default_botsort(self):
        detector = _make_mock_detector()
        adapter = TrackingAdapter(detector)
        assert adapter.tracker_type == "botsort"

    def test_bytetrack_string(self):
        detector = _make_mock_detector()
        adapter = TrackingAdapter(detector, tracker_config="bytetrack")
        assert adapter.tracker_type == "bytetrack"

    def test_botsort_string(self):
        detector = _make_mock_detector()
        adapter = TrackingAdapter(detector, tracker_config="botsort")
        assert adapter.tracker_type == "botsort"

    def test_config_instance(self):
        detector = _make_mock_detector()
        cfg = TrackerConfig(tracker_type="bytetrack")
        adapter = TrackingAdapter(detector, tracker_config=cfg)
        assert adapter.tracker_type == "bytetrack"

    def test_config_dict(self):
        detector = _make_mock_detector()
        adapter = TrackingAdapter(detector, tracker_config={"tracker_type": "bytetrack"})
        assert adapter.tracker_type == "bytetrack"

    def test_none_config_defaults_to_botsort(self):
        adapter = TrackingAdapter(_make_mock_detector(), tracker_config=None)
        assert adapter.tracker_type == "botsort"

    def test_invalid_tracker_type_raises(self):
        detector = _make_mock_detector()
        cfg = TrackerConfig(tracker_type="unknowntracker")
        with pytest.raises(ValueError, match="Unsupported tracker_type"):
            TrackingAdapter(detector, tracker_config=cfg)

    def test_custom_frame_rate(self):
        adapter = TrackingAdapter(_make_mock_detector(), frame_rate=60)
        assert adapter._frame_rate == 60

    def test_repr(self):
        detector = _make_mock_detector()
        adapter = TrackingAdapter(detector, tracker_config="bytetrack")
        r = repr(adapter)
        assert "bytetrack" in r
        assert "TrackingAdapter" in r


# ---------------------------------------------------------------------------
# TrackingAdapter — properties
# ---------------------------------------------------------------------------


class TestTrackingAdapterProperties:
    def test_id2label_from_detector(self):
        detector = _make_mock_detector()
        adapter = TrackingAdapter(detector)
        assert adapter.id2label == {0: "cat", 1: "dog"}

    def test_id2label_none_when_detector_has_none(self):
        detector = MagicMock()
        del detector.id2label  # no attribute
        adapter = TrackingAdapter(detector)
        assert adapter.id2label is None

    def test_config_property(self):
        detector = _make_mock_detector()
        cfg = TrackerConfig(tracker_type="bytetrack")
        adapter = TrackingAdapter(detector, tracker_config=cfg)
        assert adapter.config is cfg


# ---------------------------------------------------------------------------
# TrackingAdapter — update() pipeline
# ---------------------------------------------------------------------------


class TestTrackingAdapterUpdate:
    def _make_adapter_with_tracker_mock(
        self, tracker_output: np.ndarray | None = None
    ) -> tuple[TrackingAdapter, MagicMock]:
        """Return adapter + detector mock; tracker.update returns tracker_output."""
        detector = _make_mock_detector()
        adapter = TrackingAdapter(detector, tracker_config="bytetrack")
        # Replace the inner tracker with a mock
        tracker_mock = MagicMock()
        if tracker_output is None:
            # Typical single-detection output: [x1,y1,x2,y2,track_id,score,cls,idx]
            tracker_output = np.array([[10, 20, 110, 120, 1, 0.9, 0, 0]], dtype=np.float32)
        tracker_mock.update.return_value = tracker_output
        adapter._tracker = tracker_mock
        return adapter, detector

    def test_update_calls_detector_predict(self):
        adapter, detector = self._make_adapter_with_tracker_mock()
        adapter.update("dummy_image.jpg")
        detector.predict.assert_called_once_with("dummy_image.jpg")

    def test_update_returns_vision_result(self):
        adapter, _ = self._make_adapter_with_tracker_mock()
        result = adapter.update("image.jpg")
        assert isinstance(result, VisionResult)

    def test_update_populates_track_id(self):
        adapter, _ = self._make_adapter_with_tracker_mock()
        result = adapter.update("image.jpg")
        assert len(result.instances) == 1
        assert result.instances[0].track_id == 1

    def test_update_populates_bbox(self):
        adapter, _ = self._make_adapter_with_tracker_mock()
        result = adapter.update("image.jpg")
        assert result.instances[0].bbox == pytest.approx((10, 20, 110, 120), abs=1e-4)

    def test_update_populates_label_name_from_id2label(self):
        adapter, _ = self._make_adapter_with_tracker_mock()
        result = adapter.update("image.jpg")
        assert result.instances[0].label_name == "cat"

    def test_update_with_conf_kwarg(self):
        adapter, detector = self._make_adapter_with_tracker_mock()
        adapter.update("image.jpg", conf=0.6)
        _, call_kwargs = detector.predict.call_args
        assert call_kwargs.get("conf") == 0.6

    def test_update_with_iou_kwarg(self):
        adapter, detector = self._make_adapter_with_tracker_mock()
        adapter.update("image.jpg", iou=0.5)
        _, call_kwargs = detector.predict.call_args
        assert call_kwargs.get("iou") == 0.5

    def test_extra_kwargs_forwarded_to_detector(self):
        adapter, detector = self._make_adapter_with_tracker_mock()
        adapter.update("image.jpg", custom_param="value")
        _, call_kwargs = detector.predict.call_args
        assert call_kwargs.get("custom_param") == "value"

    def test_empty_tracker_output_returns_empty_result(self):
        adapter, _ = self._make_adapter_with_tracker_mock(tracker_output=np.empty((0, 8), dtype=np.float32))
        result = adapter.update("image.jpg")
        assert result.instances == []

    def test_none_tracker_output_returns_empty_result(self):
        adapter, _ = self._make_adapter_with_tracker_mock(tracker_output=None)
        # Override tracker to return None
        adapter._tracker.update.return_value = None
        result = adapter.update("image.jpg")
        assert result.instances == []

    def test_multi_detection_output(self):
        output = np.array(
            [
                [10, 20, 110, 120, 1, 0.9, 0, 0],
                [200, 50, 350, 200, 2, 0.7, 1, 1],
            ],
            dtype=np.float32,
        )
        adapter, _ = self._make_adapter_with_tracker_mock(tracker_output=output)
        result = adapter.update("image.jpg")
        assert len(result.instances) == 2
        track_ids = {inst.track_id for inst in result.instances}
        assert track_ids == {1, 2}

    def test_id2label_none_uses_fallback_label(self):
        """When detector has no id2label, label_name is like 'class_0'."""
        detector = MagicMock()
        del detector.id2label
        detector.predict.return_value = _make_vision_result(_make_instance())
        adapter = TrackingAdapter(detector, tracker_config="bytetrack")
        tracker_mock = MagicMock()
        tracker_mock.update.return_value = np.array([[10, 20, 110, 120, 1, 0.9, 0, 0]], dtype=np.float32)
        adapter._tracker = tracker_mock
        result = adapter.update("image.jpg")
        assert result.instances[0].label_name == "class_0"


# ---------------------------------------------------------------------------
# TrackingAdapter — class filtering
# ---------------------------------------------------------------------------


class TestTrackingAdapterClassFilter:
    def test_classes_filter_removes_wrong_class(self):
        """Only detections matching 'classes' are fed to tracker."""
        inst_cat = _make_instance(label=0, label_name="cat")
        inst_dog = _make_instance(x1=200, y1=50, x2=350, y2=200, label=1, label_name="dog")
        detector = _make_mock_detector(_make_vision_result(inst_cat, inst_dog))
        adapter = TrackingAdapter(detector, tracker_config="bytetrack")

        # Capture what the tracker receives
        received: list[int] = []

        def fake_tracker_update(det_results, img=None):
            received.extend(det_results.cls.astype(int).tolist())
            # Return empty so we don't need to parse output
            return np.empty((0, 8), dtype=np.float32)

        adapter._tracker.update = fake_tracker_update

        adapter.update("image.jpg", classes=[0])  # only cat
        assert all(c == 0 for c in received)

    def test_no_classes_filter_passes_all(self):
        inst_cat = _make_instance(label=0)
        inst_dog = _make_instance(x1=200, y1=50, x2=350, y2=200, label=1)
        detector = _make_mock_detector(_make_vision_result(inst_cat, inst_dog))
        adapter = TrackingAdapter(detector, tracker_config="bytetrack")

        received: list[int] = []

        def fake_update(det_results, img=None):
            received.extend(det_results.cls.astype(int).tolist())
            return np.empty((0, 8), dtype=np.float32)

        adapter._tracker.update = fake_update
        adapter.update("image.jpg")  # no classes filter
        assert sorted(received) == [0, 1]

    def test_classes_filter_empty_after_filtering(self):
        inst_cat = _make_instance(label=0)
        detector = _make_mock_detector(_make_vision_result(inst_cat))
        adapter = TrackingAdapter(detector, tracker_config="bytetrack")

        call_count = [0]

        def fake_update(det_results, img=None):
            call_count[0] += 1
            assert len(det_results) == 0  # nothing passed through
            return np.empty((0, 8), dtype=np.float32)

        adapter._tracker.update = fake_update
        adapter.update("image.jpg", classes=[99])  # class 99 doesn't exist
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# TrackingAdapter — persist and state
# ---------------------------------------------------------------------------


class TestTrackingAdapterPersist:
    def test_persist_true_no_reset_called(self):
        adapter = TrackingAdapter(_make_mock_detector(), tracker_config="bytetrack")
        adapter._tracker = MagicMock()
        adapter._tracker.update.return_value = np.empty((0, 8), dtype=np.float32)
        adapter.update("image.jpg", persist=True)
        adapter._tracker.reset.assert_not_called()

    def test_persist_false_calls_reset_before_update(self):
        adapter = TrackingAdapter(_make_mock_detector(), tracker_config="bytetrack")
        calls: list[str] = []

        def fake_reset():
            calls.append("reset")

        def fake_update(det, img=None):
            calls.append("update")
            return np.empty((0, 8), dtype=np.float32)

        adapter._tracker.reset = fake_reset
        adapter._tracker.update = fake_update
        adapter.update("image.jpg", persist=False)
        assert calls == ["reset", "update"]

    def test_multiple_updates_persist_true(self):
        """Successive updates with persist=True both call tracker.update."""
        adapter = TrackingAdapter(_make_mock_detector(), tracker_config="bytetrack")
        adapter._tracker = MagicMock()
        adapter._tracker.update.return_value = np.empty((0, 8), dtype=np.float32)
        adapter.update("frame1.jpg", persist=True)
        adapter.update("frame2.jpg", persist=True)
        assert adapter._tracker.update.call_count == 2
        adapter._tracker.reset.assert_not_called()


# ---------------------------------------------------------------------------
# TrackingAdapter — reset()
# ---------------------------------------------------------------------------


class TestTrackingAdapterReset:
    def test_reset_calls_tracker_reset(self):
        adapter = TrackingAdapter(_make_mock_detector(), tracker_config="bytetrack")
        adapter._tracker = MagicMock()
        adapter.reset()
        adapter._tracker.reset.assert_called_once()

    def test_reset_botsort(self):
        adapter = TrackingAdapter(_make_mock_detector(), tracker_config="botsort")
        adapter._tracker = MagicMock()
        adapter.reset()
        adapter._tracker.reset.assert_called_once()


# ---------------------------------------------------------------------------
# TrackingAdapter — _convert_tracker_output
# ---------------------------------------------------------------------------


class TestConvertTrackerOutput:
    def _make_adapter(self) -> TrackingAdapter:
        return TrackingAdapter(_make_mock_detector(), tracker_config="bytetrack")

    def test_empty_array(self):
        adapter = self._make_adapter()
        result = adapter._convert_tracker_output(np.empty((0, 8), dtype=np.float32), None)
        assert result.instances == []

    def test_none_input(self):
        adapter = self._make_adapter()
        result = adapter._convert_tracker_output(None, None)  # type: ignore[arg-type]
        assert result.instances == []

    def test_single_row(self):
        adapter = self._make_adapter()
        row = np.array([[5, 10, 105, 110, 7, 0.85, 2, 0]], dtype=np.float32)
        result = adapter._convert_tracker_output(row, {2: "bird"})
        assert len(result.instances) == 1
        inst = result.instances[0]
        assert inst.track_id == 7
        assert inst.label == 2
        assert inst.label_name == "bird"
        assert inst.score == pytest.approx(0.85, abs=1e-4)
        assert inst.bbox == pytest.approx((5, 10, 105, 110), abs=1e-4)

    def test_no_id2label_uses_fallback(self):
        adapter = self._make_adapter()
        row = np.array([[0, 0, 100, 100, 1, 0.9, 3, 0]], dtype=np.float32)
        result = adapter._convert_tracker_output(row, None)
        assert result.instances[0].label_name == "class_3"

    def test_multiple_rows(self):
        adapter = self._make_adapter()
        rows = np.array(
            [
                [0, 0, 50, 50, 1, 0.9, 0, 0],
                [60, 60, 120, 120, 2, 0.8, 1, 1],
                [200, 200, 300, 300, 3, 0.7, 0, 2],
            ],
            dtype=np.float32,
        )
        result = adapter._convert_tracker_output(rows, {0: "cat", 1: "dog"})
        assert len(result.instances) == 3
        assert {inst.track_id for inst in result.instances} == {1, 2, 3}

    def test_meta_source_tag(self):
        adapter = self._make_adapter()
        result = adapter._convert_tracker_output(np.empty((0, 8)), None)
        assert result.meta.get("source") == "tracking_adapter"


# ---------------------------------------------------------------------------
# TrackingAdapter — numpy image conversion
# ---------------------------------------------------------------------------


class TestToNumpyImage:
    def test_numpy_array_passthrough(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = TrackingAdapter._to_numpy_image(arr)
        assert result is arr

    def test_pil_image_converted(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow not installed")
        pil_img = PILImage.new("RGB", (64, 64), color=(128, 0, 0))
        result = TrackingAdapter._to_numpy_image(pil_img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)

    def test_string_path_returns_none(self):
        result = TrackingAdapter._to_numpy_image("path/to/image.jpg")
        assert result is None


# ---------------------------------------------------------------------------
# Builtin config files exist
# ---------------------------------------------------------------------------


class TestBuiltinConfigFiles:
    @pytest.mark.parametrize("name", _BUILTIN_CONFIG_NAMES.keys())
    def test_builtin_config_file_exists(self, name):
        path = _BUILTIN_CONFIGS_DIR / _BUILTIN_CONFIG_NAMES[name]
        assert path.is_file(), f"Built-in config not found: {path}"

    def test_configs_dir_exists(self):
        assert _BUILTIN_CONFIGS_DIR.is_dir()
