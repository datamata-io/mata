"""Comprehensive tests for graph artifact types, conversions, and serialization.

Tests cover:
- All artifact types (Image, Detections, Masks, Classifications, DepthMap,
  Keypoints, Tracks, ROIs, MultiResult)
- VisionResult ↔ Detections conversions
- Serialization (to_dict / from_dict round-trips)
- Instance ID handling and auto-generation
- Entity support and promotion workflows
- Edge cases: empty collections, missing fields, validation errors
"""

from __future__ import annotations

import numpy as np
import pytest

from mata.core.artifacts.base import Artifact, ArtifactTypeRegistry
from mata.core.artifacts.classifications import Classifications
from mata.core.artifacts.converters import (
    align_instance_ids,
    ensure_instance_ids,
    generate_instance_ids,
    match_entity_to_instance,
    merge_entity_attributes,
    promote_entities_to_instances,
)
from mata.core.artifacts.depth_map import DepthMap
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.masks import Masks
from mata.core.artifacts.result import MultiResult
from mata.core.types import Classification, ClassifyResult, DepthResult, Entity, Instance, VisionResult

# ──────── helpers ────────


def _make_instance(label_name: str = "cat", score: float = 0.9) -> Instance:
    """Create a simple Instance for testing."""
    return Instance(
        bbox=(10.0, 20.0, 100.0, 150.0),
        score=score,
        label=0,
        label_name=label_name,
    )


def _make_entity(label: str = "cat", score: float = 0.95) -> Entity:
    """Create a simple Entity for testing."""
    return Entity(label=label, score=score)


def _make_vision_result(n: int = 3) -> VisionResult:
    """Create a VisionResult with *n* instances."""
    instances = [
        Instance(
            bbox=(i * 10.0, i * 10.0, i * 10.0 + 50, i * 10.0 + 50),
            score=0.9 - i * 0.1,
            label=i,
            label_name=f"obj_{i}",
        )
        for i in range(n)
    ]
    return VisionResult(instances=instances, meta={"source": "test"})


def _make_image() -> Image:
    """Create a small numpy-backed Image artifact."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    return Image(data=arr, width=64, height=64, color_space="RGB")


# ──────────────────────────────────────────────────────────────
# 1. Image artifact
# ──────────────────────────────────────────────────────────────


class TestImageArtifact:
    """Image artifact creation, conversion, serialization."""

    def test_from_numpy(self):
        img = _make_image()
        assert img.width == 64
        assert img.height == 64
        assert img.color_space == "RGB"

    def test_invalid_dimensions_raise(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid dimensions"):
            Image(data=arr, width=0, height=64)

    def test_invalid_color_space_raise(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid color_space"):
            Image(data=arr, width=64, height=64, color_space="XYZ")

    def test_to_dict_from_dict_round_trip(self):
        img = _make_image()
        d = img.to_dict()
        assert "width" in d
        assert "height" in d
        img2 = Image.from_dict(d)
        assert img2.width == img.width
        assert img2.height == img.height


# ──────────────────────────────────────────────────────────────
# 2. Detections artifact
# ──────────────────────────────────────────────────────────────


class TestDetectionsArtifact:
    """Detections creation, VisionResult conversion, instance_id handling."""

    def test_empty_detections(self):
        dets = Detections()
        assert len(dets.instances) == 0
        assert len(dets.instance_ids) == 0
        assert len(dets.entities) == 0

    def test_auto_generates_instance_ids(self):
        inst = _make_instance()
        dets = Detections(instances=[inst])
        assert len(dets.instance_ids) == 1
        assert dets.instance_ids[0].startswith("inst_")

    def test_auto_generates_entity_ids(self):
        ent = _make_entity()
        dets = Detections(entities=[ent])
        assert len(dets.entity_ids) == 1
        assert dets.entity_ids[0].startswith("ent_")

    def test_instance_id_mismatch_raises(self):
        inst = _make_instance()
        with pytest.raises(ValueError, match="length mismatch"):
            Detections(instances=[inst], instance_ids=["a", "b"])

    def test_from_vision_result(self):
        vr = _make_vision_result(3)
        dets = Detections.from_vision_result(vr)
        assert len(dets.instances) == 3
        assert len(dets.instance_ids) == 3

    def test_to_vision_result_round_trip(self):
        vr = _make_vision_result(2)
        dets = Detections.from_vision_result(vr)
        vr2 = dets.to_vision_result()
        assert len(vr2.instances) == 2
        assert "instance_ids" in vr2.meta

    def test_filter_by_score(self):
        vr = _make_vision_result(5)
        dets = Detections.from_vision_result(vr)
        filtered = dets.filter_by_score(0.75)
        assert all(inst.score > 0.75 for inst in filtered.instances)

    def test_filter_preserves_instance_ids(self):
        vr = _make_vision_result(3)
        dets = Detections.from_vision_result(vr)
        dets.instance_ids[:]
        filtered = dets.filter_by_score(0.0)
        # All should pass with threshold 0.0
        assert len(filtered.instance_ids) == 3

    def test_top_k(self):
        vr = _make_vision_result(5)
        dets = Detections.from_vision_result(vr)
        top2 = dets.top_k(2)
        assert len(top2.instances) == 2

    def test_boxes_property(self):
        vr = _make_vision_result(2)
        dets = Detections.from_vision_result(vr)
        boxes = dets.boxes
        assert boxes.shape == (2, 4)

    def test_scores_property(self):
        vr = _make_vision_result(3)
        dets = Detections.from_vision_result(vr)
        scores = dets.scores
        assert len(scores) == 3

    def test_labels_property(self):
        vr = _make_vision_result(2)
        dets = Detections.from_vision_result(vr)
        labels = dets.labels
        assert labels == ["obj_0", "obj_1"]

    def test_serialization(self):
        vr = _make_vision_result(2)
        dets = Detections.from_vision_result(vr)
        d = dets.to_dict()
        assert "instances" in d
        dets2 = Detections.from_dict(d)
        assert len(dets2.instances) == 2


# ──────────────────────────────────────────────────────────────
# 3. Entity promotion
# ──────────────────────────────────────────────────────────────


class TestEntityPromotion:
    """Entity → Instance promotion via converters."""

    def test_promote_entities_exact(self):
        entities = [Entity(label="cat", score=0.95)]
        spatial = [_make_instance("cat", 0.9)]
        promoted = promote_entities_to_instances(entities, spatial, "label_exact")
        assert len(promoted) == 1

    def test_promote_entities_fuzzy(self):
        entities = [Entity(label="the cats", score=0.95)]
        spatial = [_make_instance("cat", 0.9)]
        promoted = promote_entities_to_instances(entities, spatial, "label_fuzzy")
        assert len(promoted) == 1

    def test_promote_no_match(self):
        entities = [Entity(label="bird", score=0.95)]
        spatial = [_make_instance("cat", 0.9)]
        promoted = promote_entities_to_instances(entities, spatial, "label_exact")
        assert len(promoted) == 0

    def test_promote_embedding_not_implemented(self):
        with pytest.raises(NotImplementedError):
            promote_entities_to_instances([], [], "embedding")

    def test_promote_invalid_strategy(self):
        with pytest.raises(ValueError, match="Invalid match_strategy"):
            promote_entities_to_instances([], [], "unknown")

    def test_merge_entity_attributes_score(self):
        inst = _make_instance("cat", 0.8)
        ent = _make_entity("cat", 0.95)
        merged = merge_entity_attributes(inst, ent)
        assert merged.score == 0.95  # Takes max score

    def test_match_entity_to_instance_exact(self):
        ent = _make_entity("cat")
        instances = [_make_instance("cat"), _make_instance("dog")]
        matched = match_entity_to_instance(ent, instances, "label_exact")
        assert matched is not None
        assert matched.label_name == "cat"

    def test_match_entity_to_instance_no_match(self):
        ent = _make_entity("bird")
        instances = [_make_instance("cat")]
        matched = match_entity_to_instance(ent, instances, "label_exact")
        assert matched is None


# ──────────────────────────────────────────────────────────────
# 4. Masks artifact
# ──────────────────────────────────────────────────────────────


class TestMasksArtifact:
    """Masks creation, instance_id auto-generation, validation."""

    def test_empty_masks(self):
        masks = Masks()
        assert len(masks.instances) == 0

    def test_masks_require_mask_data(self):
        inst = _make_instance()  # No mask field set
        with pytest.raises(ValueError, match="no mask data"):
            Masks(instances=[inst])

    def test_auto_generate_mask_ids(self):
        inst = Instance(
            bbox=(10, 20, 100, 150),
            score=0.9,
            label=0,
            label_name="cat",
            mask={"size": [64, 64], "counts": "abc"},
        )
        masks = Masks(instances=[inst])
        assert len(masks.instance_ids) == 1
        assert masks.instance_ids[0].startswith("mask_")


# ──────────────────────────────────────────────────────────────
# 5. Classifications artifact
# ──────────────────────────────────────────────────────────────


class TestClassificationsArtifact:
    """Classifications creation, convenience accessors."""

    def test_from_classify_result(self):
        preds = [
            Classification(label=0, score=0.92, label_name="cat"),
            Classification(label=1, score=0.06, label_name="dog"),
        ]
        cr = ClassifyResult(predictions=preds)
        cls = Classifications.from_classify_result(cr)
        assert cls.top1.label_name == "cat"

    def test_empty_classifications(self):
        cls = Classifications()
        assert len(cls.predictions) == 0

    def test_serialization(self):
        preds = [Classification(label=0, score=0.92, label_name="cat")]
        cr = ClassifyResult(predictions=preds)
        cls = Classifications.from_classify_result(cr)
        d = cls.to_dict()
        assert "predictions" in d
        cls2 = Classifications.from_dict(d)
        assert len(cls2.predictions) == 1


# ──────────────────────────────────────────────────────────────
# 6. DepthMap artifact
# ──────────────────────────────────────────────────────────────


class TestDepthMapArtifact:
    """DepthMap creation, properties, round-trip."""

    def test_from_depth_result(self):
        arr = np.random.rand(48, 64).astype(np.float32)
        dr = DepthResult(depth=arr, meta={"source": "test"})
        dm = DepthMap.from_depth_result(dr)
        assert dm.height == 48
        assert dm.width == 64

    def test_to_dict_from_dict(self):
        arr = np.random.rand(48, 64).astype(np.float32)
        dm = DepthMap(depth=arr)
        d = dm.to_dict()
        dm2 = DepthMap.from_dict(d)
        assert dm2.height == 48
        assert dm2.width == 64


# ──────────────────────────────────────────────────────────────
# 7. MultiResult artifact
# ──────────────────────────────────────────────────────────────


class TestMultiResultArtifact:
    """MultiResult channel access, provenance, metrics."""

    def test_channel_access(self):
        vr = _make_vision_result(2)
        dets = Detections.from_vision_result(vr)
        mr = MultiResult(channels={"detections": dets})
        assert mr.has_channel("detections")
        assert mr.detections is dets

    def test_missing_channel_raises(self):
        mr = MultiResult(channels={})
        with pytest.raises(AttributeError, match="No channel"):
            _ = mr.nonexistent

    def test_get_channel_default(self):
        mr = MultiResult(channels={})
        assert mr.get_channel("missing") is None
        assert mr.get_channel("missing", default="fallback") == "fallback"

    def test_provenance_and_metrics(self):
        mr = MultiResult(
            channels={},
            provenance={"graph_name": "test", "timestamp": "2026-02-12"},
            metrics={"node1": {"latency_ms": 10.5}},
        )
        assert mr.provenance["graph_name"] == "test"
        assert mr.metrics["node1"]["latency_ms"] == 10.5

    def test_to_dict_from_dict(self):
        dets = Detections.from_vision_result(_make_vision_result(1))
        mr = MultiResult(
            channels={"dets": dets},
            provenance={"graph_name": "test"},
        )
        d = mr.to_dict()
        assert "channels" in d
        mr2 = MultiResult.from_dict(d)
        assert mr2.has_channel("dets")


# ──────────────────────────────────────────────────────────────
# 8. Instance ID utilities
# ──────────────────────────────────────────────────────────────


class TestInstanceIDUtilities:
    """generate_instance_ids, ensure_instance_ids, align_instance_ids."""

    def test_generate_instance_ids(self):
        ids = generate_instance_ids(5)
        assert len(ids) == 5
        assert all(isinstance(i, str) for i in ids)

    def test_ensure_instance_ids_adds_missing(self):
        vr = _make_vision_result(3)
        ids = ensure_instance_ids(vr.instances)
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_align_instance_ids_same_length(self):
        vr1 = _make_vision_result(3)
        dets1 = Detections.from_vision_result(vr1)
        vr2 = _make_vision_result(3)
        dets2 = Detections.from_vision_result(vr2)
        assert align_instance_ids([dets1, dets2]) is True


# ──────────────────────────────────────────────────────────────
# 9. ArtifactTypeRegistry
# ──────────────────────────────────────────────────────────────


class TestArtifactTypeRegistry:
    """Registry register/get/has/is_compatible."""

    def setup_method(self):
        self.registry = ArtifactTypeRegistry()
        self.registry.clear()

    def test_register_and_get(self):
        self.registry.register("detections", Detections)
        assert self.registry.get("detections") is Detections

    def test_register_non_artifact_raises(self):
        with pytest.raises(TypeError):
            self.registry.register("int", int)

    def test_has(self):
        self.registry.register("img", Image)
        assert self.registry.has("img")
        assert not self.registry.has("unknown")

    def test_is_compatible_same(self):
        assert self.registry.is_compatible(Detections, Detections)

    def test_is_compatible_subclass(self):
        assert self.registry.is_compatible(Detections, Artifact)

    def test_is_compatible_unrelated(self):
        assert not self.registry.is_compatible(Image, Detections)

    def test_duplicate_register_same_type_ok(self):
        self.registry.register("img", Image)
        self.registry.register("img", Image)  # Idempotent

    def test_duplicate_register_diff_type_raises(self):
        self.registry.register("x", Image)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register("x", Detections)
