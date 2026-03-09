"""Integration tests for Tracking + ReID pipeline (Task D2).

Tests verify that ``mata.track()`` and ``mata.load("track", ...)`` with
``reid_model`` / ``with_reid`` kwargs correctly:

- Wire the ReID encoder through the loader and into ``TrackingAdapter``
- Populate ``Instance.embedding`` in ``VisionResult`` output
- Activate the BOTSORT ``get_dists()`` ReID appearance-distance branch
- Preserve backward compatibility when ``with_reid=False`` (default)
- Publish track embeddings via ``ReIDBridge`` when provided

All external dependencies (actual model loading, Valkey connections) are
mocked so these tests run fully offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mata
from mata.adapters.tracking_adapter import TrackingAdapter
from mata.core.types import Instance, VisionResult
from mata.trackers.basetrack import BaseTrack
from mata.trackers.bot_sort import BOTSORT, BOTrack

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_track_ids():
    """Reset global track-ID counter before and after every test."""
    BaseTrack.reset_id()
    yield
    BaseTrack.reset_id()


def _make_instance(
    x1: float = 10.0,
    y1: float = 20.0,
    x2: float = 110.0,
    y2: float = 120.0,
    score: float = 0.9,
    label: int = 0,
    label_name: str | None = "person",
    track_id: int | None = None,
    embedding: np.ndarray | None = None,
) -> Instance:
    return Instance(
        bbox=(x1, y1, x2, y2),
        score=score,
        label=label,
        label_name=label_name,
        track_id=track_id,
        embedding=embedding,
    )


def _make_vision_result(*instances: Instance) -> VisionResult:
    return VisionResult(instances=list(instances))


def _make_mock_detector(vision_result: VisionResult | None = None) -> MagicMock:
    """Return a mock detector whose ``predict()`` returns *vision_result*."""
    detector = MagicMock()
    detector.id2label = {0: "person", 1: "car"}
    if vision_result is None:
        vision_result = _make_vision_result(_make_instance())
    detector.predict.return_value = vision_result
    return detector


def _make_mock_reid_encoder(
    embedding_dim: int = 128,
    n_detections: int = 1,
) -> MagicMock:
    """Return a mock ReID encoder returning an (n, D) L2-normed float32 array."""
    encoder = MagicMock()
    # Produce unit-length embeddings
    raw = np.random.randn(n_detections, embedding_dim).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    encoder.predict.return_value = raw / norms
    return encoder


def _make_mock_strack(track_id: int, smooth_feat: np.ndarray | None) -> MagicMock:
    """Mock BOTrack-like strack with ``track_id``, ``smooth_feat``, ``is_activated``."""
    st = MagicMock()
    st.track_id = track_id
    st.smooth_feat = smooth_feat
    st.is_activated = True
    return st


def _make_mock_tracker(
    tracked_output: np.ndarray | None = None,
    tracked_stracks: list | None = None,
) -> MagicMock:
    """Return a mock tracker with configured ``update()`` return and stracks."""
    tracker = MagicMock()
    if tracked_output is None:
        # Default: 1 tracked object — [x1,y1,x2,y2,tid,score,cls,idx]
        tracked_output = np.array([[10.0, 20.0, 110.0, 120.0, 1.0, 0.9, 0.0, 0.0]])
    tracker.update.return_value = tracked_output
    tracker.tracked_stracks = tracked_stracks if tracked_stracks is not None else []
    return tracker


# ---------------------------------------------------------------------------
# Group 1: TrackingAdapter constructor with ReID
# ---------------------------------------------------------------------------


class TestTrackingAdapterReIDInit:
    def test_no_reid_by_default(self):
        """Default construction has no ReID encoder or bridge."""
        detector = _make_mock_detector()
        with patch.object(TrackingAdapter, "_build_tracker", return_value=MagicMock()):
            adapter = TrackingAdapter(detector)
        assert adapter._reid_encoder is None
        assert adapter._reid_bridge is None

    def test_reid_encoder_stored_when_provided(self):
        """Providing ``reid_encoder`` stores it on the adapter."""
        detector = _make_mock_detector()
        encoder = _make_mock_reid_encoder()
        mock_tracker = MagicMock(spec=[])  # no 'encoder' attribute
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_encoder=encoder)
        assert adapter._reid_encoder is encoder

    def test_botsort_encoder_set_when_reid_active(self):
        """When ``reid_encoder`` is given, ``BOTSORT.encoder`` is wired."""
        detector = _make_mock_detector()
        encoder = _make_mock_reid_encoder()
        mock_tracker = MagicMock()
        mock_tracker.encoder = None
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            TrackingAdapter(detector, reid_encoder=encoder)
        assert mock_tracker.encoder is encoder

    def test_botsort_encoder_not_set_when_no_reid(self):
        """Without ``reid_encoder``, ``BOTSORT.encoder`` stays as-is."""
        detector = _make_mock_detector()
        mock_tracker = MagicMock()
        mock_tracker.encoder = None
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            TrackingAdapter(detector)
        # encoder attribute on tracker should not have been mutated
        assert mock_tracker.encoder is None

    def test_bytetrack_no_encoder_attribute_no_crash(self):
        """Adapter with reid_encoder works even when tracker has no .encoder."""
        detector = _make_mock_detector()
        encoder = _make_mock_reid_encoder()
        # Tracker with no 'encoder' attribute
        mock_tracker = MagicMock(spec=["update", "tracked_stracks", "reset"])
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            # Must not raise
            adapter = TrackingAdapter(detector, reid_encoder=encoder)
        assert adapter._reid_encoder is encoder

    def test_reid_bridge_stored_when_provided(self):
        """Providing ``reid_bridge`` stores it on the adapter."""
        detector = _make_mock_detector()
        bridge = MagicMock()
        with patch.object(TrackingAdapter, "_build_tracker", return_value=MagicMock()):
            adapter = TrackingAdapter(detector, reid_bridge=bridge)
        assert adapter._reid_bridge is bridge


# ---------------------------------------------------------------------------
# Group 2: update() pipeline with ReID active
# ---------------------------------------------------------------------------


class TestTrackingAdapterUpdateWithReID:
    def _make_adapter_with_reid(
        self,
        n_detections: int = 1,
        embedding_dim: int = 8,
        smooth_feat: np.ndarray | None = None,
    ):
        """Build an adapter with a mock detector/tracker/encoder."""
        vr = _make_vision_result(*[_make_instance() for _ in range(n_detections)])
        detector = _make_mock_detector(vr)
        encoder = _make_mock_reid_encoder(embedding_dim=embedding_dim, n_detections=n_detections)

        if smooth_feat is None:
            embed = np.ones(embedding_dim, dtype=np.float32)
            smooth_feat = embed / np.linalg.norm(embed)

        strack = _make_mock_strack(track_id=1, smooth_feat=smooth_feat)
        tracked_out = np.array([[10.0, 20.0, 110.0, 120.0, 1.0, 0.9, 0.0, 0.0]])
        mock_tracker = _make_mock_tracker(tracked_out, [strack])

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_encoder=encoder)
        # Replace tracker directly (patch already returned mock_tracker)
        adapter._tracker = mock_tracker
        return adapter, encoder, smooth_feat

    def test_update_without_reid_unchanged(self):
        """``update()`` without reid_encoder returns VisionResult normally."""
        detector = _make_mock_detector()
        mock_tracker = _make_mock_tracker()
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector)
        result = adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(result, VisionResult)
        assert result.instances[0].track_id == 1
        assert result.instances[0].embedding is None

    def test_update_with_reid_populates_embeddings(self):
        """VisionResult instances carry embeddings when ReID is active."""
        feat = np.array([0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        adapter, _, smooth_feat = self._make_adapter_with_reid(smooth_feat=feat)
        result = adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.instances[0].embedding is not None
        np.testing.assert_array_equal(result.instances[0].embedding, feat)

    def test_embedding_shape_in_vision_result(self):
        """``Instance.embedding`` is a 1-D float32 numpy array of expected dim."""
        dim = 16
        adapter, _, smooth_feat = self._make_adapter_with_reid(embedding_dim=dim)
        result = adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        emb = result.instances[0].embedding
        assert emb is not None
        assert emb.ndim == 1
        assert emb.shape[0] == dim
        assert emb.dtype == np.float32

    def test_reid_encoder_called_with_crops(self):
        """The reid encoder's ``predict()`` is invoked with image crops."""
        adapter, encoder, _ = self._make_adapter_with_reid()
        adapter.update(np.zeros((200, 200, 3), dtype=np.uint8))
        encoder.predict.assert_called_once()
        crops_arg = encoder.predict.call_args[0][0]
        assert len(crops_arg) == 1
        assert isinstance(crops_arg[0], np.ndarray)

    def test_update_with_reid_skipped_when_np_image_none(self):
        """ReID crop extraction is skipped gracefully when image can't be converted."""
        adapter, encoder, _ = self._make_adapter_with_reid()
        # Pass a string URL — _to_numpy_image returns None, encoder must not be called
        with patch.object(adapter, "_to_numpy_image", return_value=None):
            result = adapter.update("http://example.com/frame.jpg")
        encoder.predict.assert_not_called()
        assert isinstance(result, VisionResult)

    def test_update_empty_frame_no_crash_with_reid(self):
        """Zero detections + reid_encoder doesn't crash."""
        vr = _make_vision_result()  # empty
        detector = _make_mock_detector(vr)
        encoder = _make_mock_reid_encoder()
        mock_tracker = _make_mock_tracker(
            tracked_output=np.empty((0, 8)),
            tracked_stracks=[],
        )
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_encoder=encoder)
        adapter._tracker = mock_tracker
        result = adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        encoder.predict.assert_not_called()
        assert result.instances == []

    def test_zero_area_bbox_skipped_in_reid(self):
        """Zero-area bboxes produce empty placeholder crops (encoder not called for them)."""
        zero_inst = _make_instance(x1=50, y1=50, x2=50, y2=50)  # degenerate
        vr = _make_vision_result(zero_inst)
        detector = _make_mock_detector(vr)
        encoder = _make_mock_reid_encoder()
        mock_tracker = _make_mock_tracker(
            tracked_output=np.empty((0, 8)),
            tracked_stracks=[],
        )
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_encoder=encoder)
        adapter._tracker = mock_tracker
        adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        # No valid crops → encoder predict never called
        encoder.predict.assert_not_called()

    def test_reid_encoder_not_called_without_reid(self):
        """Without reid_encoder, no encode call is made during update."""
        detector = _make_mock_detector()
        encoder = _make_mock_reid_encoder()
        mock_tracker = _make_mock_tracker()
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector)  # no reid_encoder
        adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        encoder.predict.assert_not_called()

    def test_multiple_detections_all_get_embeddings(self):
        """All tracked instances receive their respective embeddings."""
        n = 3
        vr = _make_vision_result(*[_make_instance(x1=i * 30, x2=i * 30 + 20) for i in range(n)])
        detector = _make_mock_detector(vr)
        encoder = _make_mock_reid_encoder(n_detections=n, embedding_dim=4)

        stracks = [
            _make_mock_strack(track_id=i + 1, smooth_feat=np.array([float(i)] * 4, dtype=np.float32)) for i in range(n)
        ]
        tracked_out = np.array(
            [[i * 30.0, 0.0, i * 30.0 + 20.0, 100.0, float(i + 1), 0.9, 0.0, float(i)] for i in range(n)]
        )
        mock_tracker = _make_mock_tracker(tracked_out, stracks)

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_encoder=encoder)
        adapter._tracker = mock_tracker
        result = adapter.update(np.zeros((200, 200, 3), dtype=np.uint8))
        assert len(result.instances) == n
        for inst in result.instances:
            assert inst.embedding is not None


# ---------------------------------------------------------------------------
# Group 3: BOTSORT get_dists() ReID branch
# ---------------------------------------------------------------------------


class TestBOTSORTGetDistsReIDBranch:
    def _default_args(self, **overrides) -> dict:
        base = {
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "fuse_score": False,
            "gmc_method": None,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "with_reid": False,
        }
        base.update(overrides)
        return base

    def test_botsort_encoder_none_by_default(self):
        """``BOTSORT.encoder`` is ``None`` before any ReID wiring."""
        tracker = BOTSORT(self._default_args(), frame_rate=30)
        assert tracker.encoder is None

    def test_botsort_with_reid_false_skips_embedding_distance(self):
        """``get_dists()`` does NOT call embedding distance when ``with_reid=False``."""
        tracker = BOTSORT(self._default_args(with_reid=False), frame_rate=30)
        tracker.encoder = MagicMock()  # encoder set but with_reid=False

        t = BOTrack([50, 50, 40, 60, 0, 0], 0.9, 0)
        d = BOTrack([50, 50, 40, 60, 0, 0], 0.9, 0)

        with patch("mata.trackers.utils.matching.embedding_distance") as mock_emb:
            tracker.get_dists([t], [d])
        mock_emb.assert_not_called()

    def test_botsort_get_dists_uses_embedding_when_encoder_and_with_reid(self):
        """``get_dists()`` uses embedding distance when both ``with_reid=True`` and ``encoder`` is set."""
        tracker = BOTSORT(self._default_args(with_reid=True), frame_rate=30)
        tracker.encoder = MagicMock()

        feat = np.array([1.0, 0.0], dtype=np.float32)
        t = BOTrack([50, 60, 40, 60, 0, 0], 0.9, 0)
        t.update_features(feat)
        d = BOTrack([52, 60, 40, 60, 0, 0], 0.9, 0)
        d.update_features(feat)

        with patch("mata.trackers.utils.matching.embedding_distance", return_value=np.array([[0.1]])) as mock_emb:
            cost = tracker.get_dists([t], [d])
        mock_emb.assert_called_once()
        assert cost.shape == (1, 1)

    def test_botsort_get_dists_no_encoder_falls_back_to_iou(self):
        """``get_dists()`` falls back to IoU gating when ``encoder=None`` even with ``with_reid=True``."""
        tracker = BOTSORT(self._default_args(with_reid=True), frame_rate=30)
        # encoder stays None

        t = BOTrack([50, 60, 40, 60, 0, 0], 0.9, 0)
        d = BOTrack([52, 60, 40, 60, 0, 0], 0.9, 0)

        with patch("mata.trackers.utils.matching.embedding_distance") as mock_emb:
            tracker.get_dists([t], [d])
        mock_emb.assert_not_called()

    def test_tracking_adapter_sets_botsort_encoder(self):
        """``TrackingAdapter`` wires the encoder into a real BOTSORT instance."""
        encoder = _make_mock_reid_encoder()
        detector = _make_mock_detector()
        # Use real BOTSORT — no patching
        adapter = TrackingAdapter(detector, tracker_config="botsort", reid_encoder=encoder)
        assert adapter._tracker.encoder is encoder


# ---------------------------------------------------------------------------
# Group 4: mata.load() and mata.track() public API
# ---------------------------------------------------------------------------


class TestPublicAPIReID:
    def test_load_track_without_reid_returns_adapter_no_encoder(self):
        """``mata.load("track", ...)`` without ReID returns adapter without encoder."""
        mock_detect = _make_mock_detector()
        with patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter", return_value=mock_detect):
            adapter = mata.load("track", "facebook/detr-resnet-50")
        assert isinstance(adapter, TrackingAdapter)
        assert adapter._reid_encoder is None

    def test_load_track_with_reid_model_sets_encoder(self):
        """``mata.load("track", ..., reid_model=...)`` creates adapter with encoder."""
        mock_detect = _make_mock_detector()
        mock_encoder = _make_mock_reid_encoder()

        with (
            patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter", return_value=mock_detect),
            patch(
                "mata.core.model_loader.UniversalLoader._load_reid_encoder",
                return_value=mock_encoder,
            ),
        ):
            adapter = mata.load("track", "facebook/detr-resnet-50", reid_model="org/reid-model")

        assert isinstance(adapter, TrackingAdapter)
        assert adapter._reid_encoder is mock_encoder

    def test_with_reid_true_without_model_raises_value_error(self):
        """``with_reid=True`` without ``reid_model`` raises ``ValueError``."""
        mock_detect = _make_mock_detector()
        with patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter", return_value=mock_detect):
            with pytest.raises(ValueError, match="reid_model"):
                mata.load("track", "facebook/detr-resnet-50", with_reid=True)

    def test_load_track_onnx_reid_loads_onnx_adapter(self):
        """An ``.onnx`` path for ``reid_model`` creates an ``ONNXReIDAdapter``."""
        from mata.adapters.reid_adapter import ONNXReIDAdapter

        mock_detect = _make_mock_detector()
        with (
            patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter", return_value=mock_detect),
            patch.object(ONNXReIDAdapter, "_load_model"),  # skip actual ONNX loading
        ):
            adapter = mata.load("track", "facebook/detr-resnet-50", reid_model="reid.onnx")

        assert isinstance(adapter, TrackingAdapter)
        assert isinstance(adapter._reid_encoder, ONNXReIDAdapter)

    def test_load_track_hf_reid_loads_hf_adapter(self):
        """A HuggingFace ID for ``reid_model`` creates a ``HuggingFaceReIDAdapter``."""
        from mata.adapters.reid_adapter import HuggingFaceReIDAdapter

        mock_detect = _make_mock_detector()
        with (
            patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter", return_value=mock_detect),
            patch.object(HuggingFaceReIDAdapter, "_load_model"),  # skip real download
        ):
            adapter = mata.load("track", "facebook/detr-resnet-50", reid_model="org/reid-model")

        assert isinstance(adapter, TrackingAdapter)
        assert isinstance(adapter._reid_encoder, HuggingFaceReIDAdapter)

    def test_track_api_backward_compat_no_reid(self):
        """``mata.track()`` without any ReID kwargs has identical behaviour to before."""
        import numpy as np

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_adapter = MagicMock()
        mock_adapter.update.return_value = _make_vision_result(_make_instance(track_id=1))

        with patch("mata.api.load", return_value=mock_adapter):
            results = mata.track(frame, model="facebook/detr-resnet-50")

        assert isinstance(results, list)
        assert len(results) == 1

    def test_track_api_reid_model_forwarded_to_load(self):
        """``mata.track(..., reid_model=...)`` forwards ``reid_model`` to ``load()``."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_adapter = MagicMock()
        mock_adapter.update.return_value = _make_vision_result()

        with patch("mata.api.load", return_value=mock_adapter) as mock_load:
            mata.track(frame, model="facebook/detr-resnet-50", reid_model="org/reid")

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs.get("reid_model") == "org/reid"

    def test_track_api_with_reid_forwarded_to_load(self):
        """``mata.track(..., with_reid=True)`` forwards ``with_reid`` to ``load()``."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_adapter = MagicMock()
        mock_adapter.update.return_value = _make_vision_result()

        with patch("mata.api.load", return_value=mock_adapter) as mock_load:
            # with_reid=True but we allow it through since load is mocked
            mata.track(frame, model="facebook/detr-resnet-50", with_reid=True)

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs.get("with_reid") is True


# ---------------------------------------------------------------------------
# Group 5: Config / YAML support for ReID
# ---------------------------------------------------------------------------


class TestConfigReIDSupport:
    def test_resolve_tracker_kwargs_pops_reid_model(self):
        """``_resolve_tracker_kwargs`` pops and returns ``reid_model``."""
        from mata.core.model_loader import UniversalLoader

        loader = UniversalLoader.__new__(UniversalLoader)
        kwargs = {"tracker": "botsort", "frame_rate": 30, "reid_model": "org/reid"}
        _, _, reid_model, with_reid, reid_bridge = loader._resolve_tracker_kwargs(kwargs)
        assert reid_model == "org/reid"
        assert "reid_model" not in kwargs

    def test_resolve_tracker_kwargs_pops_with_reid(self):
        """``_resolve_tracker_kwargs`` pops and returns ``with_reid``."""
        from mata.core.model_loader import UniversalLoader

        loader = UniversalLoader.__new__(UniversalLoader)
        kwargs = {"with_reid": True, "reid_model": "org/reid"}
        _, _, _, with_reid, _ = loader._resolve_tracker_kwargs(kwargs)
        assert with_reid is True

    def test_config_alias_reid_model_extracted(self):
        """Config alias with ``reid_model`` key wires the encoder."""
        mock_detect = _make_mock_detector()
        mock_encoder = _make_mock_reid_encoder()

        config_entry = {
            "source": "facebook/detr-resnet-50",
            "tracker": "botsort",
            "reid_model": "org/reid-model",
        }

        # has_alias returns True only for the alias (not for the recursive source call)
        def has_alias_side_effect(task, name):
            return name == "my-config-alias"

        with (
            patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter", return_value=mock_detect),
            patch("mata.core.model_loader.UniversalLoader._load_reid_encoder", return_value=mock_encoder),
            patch("mata.core.model_registry.ModelRegistry.has_alias", side_effect=has_alias_side_effect),
            patch("mata.core.model_registry.ModelRegistry.get_config", return_value=config_entry),
        ):
            adapter = mata.load("track", "my-config-alias")

        assert isinstance(adapter, TrackingAdapter)
        assert adapter._reid_encoder is mock_encoder

    def test_runtime_kwarg_overrides_config_reid_model(self):
        """Runtime ``reid_model`` kwarg overrides the one in config."""
        mock_detect = _make_mock_detector()
        mock_encoder = _make_mock_reid_encoder()

        config_entry = {
            "source": "facebook/detr-resnet-50",
            "reid_model": "config/reid-model",  # should be overridden by runtime kwarg
        }

        def has_alias_side_effect(task, name):
            return name == "my-config-alias"

        with (
            patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter", return_value=mock_detect),
            patch(
                "mata.core.model_loader.UniversalLoader._load_reid_encoder",
                return_value=mock_encoder,
            ) as mock_load_reid,
            patch("mata.core.model_registry.ModelRegistry.has_alias", side_effect=has_alias_side_effect),
            patch("mata.core.model_registry.ModelRegistry.get_config", return_value=config_entry),
        ):
            mata.load("track", "my-config-alias", reid_model="runtime/reid-model")

        # Verify the runtime value was used (not the config value)
        mock_load_reid.assert_called_once_with("runtime/reid-model")

    def test_config_without_reid_unchanged(self):
        """Config aliases without ``reid_model`` continue to work unchanged."""
        mock_detect = _make_mock_detector()

        config_entry = {
            "source": "facebook/detr-resnet-50",
            "tracker": "botsort",
            # no reid_model key
        }

        def has_alias_side_effect(task, name):
            return name == "my-config-alias"

        with (
            patch("mata.adapters.huggingface_adapter.HuggingFaceDetectAdapter", return_value=mock_detect),
            patch("mata.core.model_registry.ModelRegistry.has_alias", side_effect=has_alias_side_effect),
            patch("mata.core.model_registry.ModelRegistry.get_config", return_value=config_entry),
        ):
            adapter = mata.load("track", "my-config-alias")

        assert isinstance(adapter, TrackingAdapter)
        assert adapter._reid_encoder is None


# ---------------------------------------------------------------------------
# Group 6: ReIDBridge integration in TrackingAdapter.update()
# ---------------------------------------------------------------------------


class TestReIDBridgeIntegration:
    def _make_adapter_with_bridge(self, embed_feat: np.ndarray | None = None):
        """Build a TrackingAdapter with both ReID encoder and bridge."""
        dim = 4
        if embed_feat is None:
            embed_feat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        vr = _make_vision_result(_make_instance())
        detector = _make_mock_detector(vr)
        encoder = _make_mock_reid_encoder(embedding_dim=dim, n_detections=1)
        bridge = MagicMock()

        strack = _make_mock_strack(track_id=1, smooth_feat=embed_feat)
        tracked_out = np.array([[10.0, 20.0, 110.0, 120.0, 1.0, 0.9, 0.0, 0.0]])
        mock_tracker = _make_mock_tracker(tracked_out, [strack])

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_encoder=encoder, reid_bridge=bridge)
        adapter._tracker = mock_tracker
        return adapter, bridge, embed_feat

    def test_reid_bridge_publish_called_after_update(self):
        """``ReIDBridge.publish()`` is called for each tracked instance with embedding."""
        feat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        adapter, bridge, _ = self._make_adapter_with_bridge(embed_feat=feat)
        adapter.update(np.zeros((200, 200, 3), dtype=np.uint8))
        bridge.publish.assert_called_once()
        call_kwargs = bridge.publish.call_args[1]
        assert call_kwargs["track_id"] == 1
        np.testing.assert_array_equal(call_kwargs["embedding"], feat)

    def test_reid_bridge_not_called_when_none(self):
        """No bridge calls occur when ``reid_bridge=None``."""
        vr = _make_vision_result(_make_instance())
        detector = _make_mock_detector(vr)
        mock_tracker = _make_mock_tracker()
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector)  # no bridge
        adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        # No bridge object → no publish() calls (no AttributeError)

    def test_reid_bridge_publish_skipped_without_embedding(self):
        """If embedding is ``None``, ``bridge.publish()`` is NOT called."""
        vr = _make_vision_result(_make_instance())
        detector = _make_mock_detector(vr)
        bridge = MagicMock()

        # strack has smooth_feat=None → embedding won't be in output
        strack = _make_mock_strack(track_id=1, smooth_feat=None)
        tracked_out = np.array([[10.0, 20.0, 110.0, 120.0, 1.0, 0.9, 0.0, 0.0]])
        mock_tracker = _make_mock_tracker(tracked_out, [strack])

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_bridge=bridge)
        adapter._tracker = mock_tracker
        adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        bridge.publish.assert_not_called()

    def test_reid_bridge_publish_called_for_multiple_instances(self):
        """``bridge.publish()`` is called once per tracked instance with an embedding."""
        dim = 4
        n = 2
        feats = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        ]

        vr = _make_vision_result(*[_make_instance(x1=i * 60, x2=i * 60 + 50) for i in range(n)])
        detector = _make_mock_detector(vr)
        encoder = _make_mock_reid_encoder(embedding_dim=dim, n_detections=n)
        bridge = MagicMock()

        stracks = [_make_mock_strack(track_id=i + 1, smooth_feat=feats[i]) for i in range(n)]
        tracked_out = np.array(
            [[i * 60.0, 0.0, i * 60.0 + 50.0, 100.0, float(i + 1), 0.9, 0.0, float(i)] for i in range(n)]
        )
        mock_tracker = _make_mock_tracker(tracked_out, stracks)

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_encoder=encoder, reid_bridge=bridge)
        adapter._tracker = mock_tracker
        adapter.update(np.zeros((300, 300, 3), dtype=np.uint8))

        assert bridge.publish.call_count == n


# ---------------------------------------------------------------------------
# Group 7: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_adapter_update_identical_without_reid_encoder(self):
        """``update()`` result is unchanged when ``reid_encoder=None``."""
        track_row = np.array([[10.0, 20.0, 110.0, 120.0, 1.0, 0.9, 0.0, 0.0]])
        vr = _make_vision_result(_make_instance())
        detector = _make_mock_detector(vr)
        mock_tracker = _make_mock_tracker(track_row, [])

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector)

        result = adapter.update(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(result.instances) == 1
        inst = result.instances[0]
        assert inst.track_id == 1
        assert inst.embedding is None
        assert inst.bbox == (10.0, 20.0, 110.0, 120.0)

    def test_with_reid_false_default_path_no_overhead(self):
        """Default ``with_reid=False`` path processes no embeddings."""
        vr = _make_vision_result(_make_instance())
        detector = _make_mock_detector(vr)
        mock_tracker = _make_mock_tracker()

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector)

        assert adapter._reid_encoder is None
        assert adapter._reid_bridge is None

    def test_tracker_type_property_unchanged(self):
        """``tracker_type`` property continues to work after ReID changes."""
        detector = _make_mock_detector()
        mock_tracker = MagicMock()
        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, tracker_config="botsort")
        assert adapter.tracker_type == "botsort"

    def test_persist_false_resets_tracker(self):
        """``persist=False`` still resets the tracker (ReID changes don't break this)."""
        vr = _make_vision_result(_make_instance())
        detector = _make_mock_detector(vr)
        mock_tracker = _make_mock_tracker()

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector)

        adapter.update(np.zeros((100, 100, 3), dtype=np.uint8), persist=False)
        mock_tracker.reset.assert_called_once()

    def test_class_filter_still_works_with_reid_encoder(self):
        """``classes=`` filtering still runs correctly when ReID is active."""
        # Provide two detections of different classes
        inst0 = _make_instance(x1=10, x2=60, label=0)
        inst1 = _make_instance(x1=70, x2=120, label=1)
        vr = _make_vision_result(inst0, inst1)
        detector = _make_mock_detector(vr)
        encoder = _make_mock_reid_encoder(n_detections=1, embedding_dim=4)

        # Tracker returns empty (filtered down to class 0 only)
        strack = _make_mock_strack(track_id=1, smooth_feat=np.ones(4, dtype=np.float32))
        tracked_out = np.array([[10.0, 20.0, 60.0, 120.0, 1.0, 0.9, 0.0, 0.0]])
        mock_tracker = _make_mock_tracker(tracked_out, [strack])

        with patch.object(TrackingAdapter, "_build_tracker", return_value=mock_tracker):
            adapter = TrackingAdapter(detector, reid_encoder=encoder)
        adapter._tracker = mock_tracker

        adapter.update(
            np.zeros((200, 200, 3), dtype=np.uint8),
            classes=[0],  # keep only class 0
        )
        # Encoder should only be called for the 1 filtered detection (class 0)
        encoder.predict.assert_called_once()
        crops = encoder.predict.call_args[0][0]
        assert len(crops) == 1
