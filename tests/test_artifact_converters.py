"""Tests for artifact conversion utilities (Task 1.6).

This module tests bidirectional conversions between VisionResult/ClassifyResult/DepthResult
and artifact types, plus instance ID management utilities.
"""

import numpy as np
import pytest

from mata.core.artifacts import (
    Detections,
    Masks,
    align_instance_ids,
    artifact_to_classify_result,
    artifact_to_depth_result,
    classify_result_to_artifact,
    depth_result_to_artifact,
    detections_to_vision_result,
    ensure_instance_ids,
    # Instance ID management
    generate_instance_ids,
    masks_to_vision_result,
    # Converters
    vision_result_to_detections,
    vision_result_to_masks,
)
from mata.core.types import ClassifyResult, DepthResult, Entity, Instance, VisionResult

# =============================================================================
# VisionResult ↔ Detections Conversions
# =============================================================================


class TestVisionResultDetectionsConversions:
    """Test VisionResult ↔ Detections bidirectional conversions."""

    def test_vision_result_to_detections_basic(self):
        """Test converting VisionResult to Detections."""
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
        ]
        result = VisionResult(instances=instances)

        dets = vision_result_to_detections(result)

        assert isinstance(dets, Detections)
        assert len(dets.instances) == 2
        assert len(dets.instance_ids) == 2
        assert dets.instance_ids[0].startswith("inst_")
        assert dets.instances[0].label_name == "cat"
        assert dets.instances[1].label_name == "dog"

    def test_vision_result_to_detections_with_entities(self):
        """Test VisionResult with entities preserves them."""
        entities = [Entity("person", 0.95), Entity("car", 0.87)]
        result = VisionResult(instances=[], entities=entities)

        dets = vision_result_to_detections(result)

        assert len(dets.entities) == 2
        assert len(dets.entity_ids) == 2
        assert dets.entities[0].label == "person"
        assert dets.entity_ids[0].startswith("ent_")

    def test_vision_result_to_detections_mixed(self):
        """Test VisionResult with both instances and entities."""
        instances = [Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat")]
        entities = [Entity("dog", 0.85)]
        result = VisionResult(instances=instances, entities=entities)

        dets = vision_result_to_detections(result)

        assert len(dets.instances) == 1
        assert len(dets.entities) == 1
        assert len(dets.instance_ids) == 1
        assert len(dets.entity_ids) == 1

    def test_detections_to_vision_result_basic(self):
        """Test converting Detections back to VisionResult."""
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
        ]
        dets = Detections(instances=instances)

        result = detections_to_vision_result(dets)

        assert isinstance(result, VisionResult)
        assert len(result.instances) == 1
        assert result.instances[0].label_name == "cat"
        assert "instance_ids" in result.meta
        assert len(result.meta["instance_ids"]) == 1

    def test_detections_to_vision_result_preserves_entities(self):
        """Test entities are preserved in round-trip."""
        entities = [Entity("person", 0.95)]
        dets = Detections(instances=[], entities=entities)

        result = detections_to_vision_result(dets)

        assert len(result.entities) == 1
        assert result.entities[0].label == "person"
        assert "entity_ids" in result.meta

    def test_vision_result_detections_roundtrip(self):
        """Test VisionResult → Detections → VisionResult preserves data."""
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
        ]
        entities = [Entity("bird", 0.8)]
        original = VisionResult(instances=instances, entities=entities, meta={"model": "test"})

        # Round-trip
        dets = vision_result_to_detections(original)
        restored = detections_to_vision_result(dets)

        assert len(restored.instances) == len(original.instances)
        assert len(restored.entities) == len(original.entities)
        assert restored.instances[0].label_name == "cat"
        assert restored.entities[0].label == "bird"
        assert "instance_ids" in restored.meta
        assert "entity_ids" in restored.meta

    def test_vision_result_to_detections_empty(self):
        """Test converting empty VisionResult."""
        result = VisionResult(instances=[])

        dets = vision_result_to_detections(result)

        assert len(dets.instances) == 0
        assert len(dets.instance_ids) == 0
        assert len(dets.entities) == 0


# =============================================================================
# VisionResult ↔ Masks Conversions
# =============================================================================


class TestVisionResultMasksConversions:
    """Test VisionResult ↔ Masks bidirectional conversions."""

    def test_vision_result_to_masks_basic(self):
        """Test converting VisionResult with masks to Masks artifact."""
        rle_mask1 = {"size": [100, 100], "counts": "test1"}
        rle_mask2 = {"size": [100, 100], "counts": "test2"}
        instances = [
            Instance(mask=rle_mask1, score=0.9, label=0, label_name="cat"),
            Instance(mask=rle_mask2, score=0.85, label=1, label_name="dog"),
        ]
        result = VisionResult(instances=instances)

        masks = vision_result_to_masks(result)

        assert isinstance(masks, Masks)
        assert len(masks.instances) == 2
        assert len(masks.instance_ids) == 2
        assert masks.instance_ids[0].startswith("mask_")
        assert masks.instances[0].mask == rle_mask1

    def test_vision_result_to_masks_filters_no_masks(self):
        """Test Masks extracts only instances with mask data."""
        rle_mask = {"size": [100, 100], "counts": "test"}
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),  # No mask
            Instance(mask=rle_mask, score=0.85, label=1, label_name="dog"),  # Has mask
        ]
        result = VisionResult(instances=instances)

        masks = vision_result_to_masks(result)

        assert len(masks.instances) == 1
        assert masks.instances[0].label_name == "dog"

    def test_vision_result_to_masks_no_masks_error(self):
        """Test error when VisionResult has no masks."""
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
        ]
        result = VisionResult(instances=instances)

        with pytest.raises(ValueError, match="no instances with masks"):
            vision_result_to_masks(result)

    def test_masks_to_vision_result_basic(self):
        """Test converting Masks back to VisionResult."""
        rle_mask = {"size": [100, 100], "counts": "test"}
        instances = [Instance(mask=rle_mask, score=0.9, label=0, label_name="cat")]
        masks = Masks(instances=instances)

        result = masks_to_vision_result(masks)

        assert isinstance(result, VisionResult)
        assert len(result.instances) == 1
        assert result.instances[0].mask == rle_mask
        assert "instance_ids" in result.meta

    def test_vision_result_masks_roundtrip(self):
        """Test VisionResult → Masks → VisionResult preserves data."""
        rle_mask1 = {"size": [100, 100], "counts": "test1"}
        rle_mask2 = {"size": [100, 100], "counts": "test2"}
        instances = [
            Instance(mask=rle_mask1, score=0.9, label=0, label_name="cat"),
            Instance(mask=rle_mask2, score=0.85, label=1, label_name="dog"),
        ]
        original = VisionResult(instances=instances, meta={"model": "sam"})

        # Round-trip
        masks = vision_result_to_masks(original)
        restored = masks_to_vision_result(masks)

        assert len(restored.instances) == len(original.instances)
        assert restored.instances[0].mask == rle_mask1
        assert restored.instances[1].mask == rle_mask2
        assert "instance_ids" in restored.meta


# =============================================================================
# ClassifyResult Conversions (Stubs)
# =============================================================================


class TestClassifyResultConversions:
    """Test ClassifyResult ↔ Classifications artifact conversions."""

    def test_classify_result_to_artifact(self):
        """Test converting ClassifyResult to Classifications artifact."""
        from mata.core.artifacts.classifications import Classifications
        from mata.core.types import Classification

        result = ClassifyResult(
            predictions=[Classification(label=0, score=0.9, label_name="cat")],
            meta={"model": "resnet"},
        )
        artifact = classify_result_to_artifact(result)

        assert isinstance(artifact, Classifications)
        assert len(artifact) == 1
        assert artifact.top1.label_name == "cat"

    def test_artifact_to_classify_result(self):
        """Test converting Classifications artifact back to ClassifyResult."""
        from mata.core.artifacts.classifications import Classifications
        from mata.core.types import Classification

        cls = Classifications(
            predictions=(Classification(label=0, score=0.9, label_name="cat"),),
            meta={"model": "resnet"},
        )
        restored = artifact_to_classify_result(cls)

        assert isinstance(restored, ClassifyResult)
        assert len(restored.predictions) == 1
        assert restored.predictions[0].label_name == "cat"


# =============================================================================
# DepthResult Conversions (Stubs)
# =============================================================================


class TestDepthResultConversions:
    """Test DepthResult ↔ DepthMap artifact conversions."""

    def test_depth_result_to_artifact(self):
        """Test converting DepthResult to DepthMap artifact."""
        from mata.core.artifacts.depth_map import DepthMap

        depth_arr = np.random.rand(100, 100).astype(np.float32)
        result = DepthResult(depth=depth_arr, meta={"model": "midas"})
        artifact = depth_result_to_artifact(result)

        assert isinstance(artifact, DepthMap)
        assert artifact.height == 100
        assert artifact.width == 100

    def test_artifact_to_depth_result(self):
        """Test converting DepthMap artifact back to DepthResult."""
        from mata.core.artifacts.depth_map import DepthMap

        depth_arr = np.random.rand(50, 80).astype(np.float32)
        dm = DepthMap(depth=depth_arr, meta={"model": "midas"})
        restored = artifact_to_depth_result(dm)

        assert isinstance(restored, DepthResult)
        np.testing.assert_array_equal(restored.depth, depth_arr)


# =============================================================================
# Instance ID Management
# =============================================================================


class TestInstanceIDManagement:
    """Test instance ID generation and management utilities."""

    def test_generate_instance_ids_basic(self):
        """Test basic instance ID generation."""
        ids = generate_instance_ids(5)

        assert len(ids) == 5
        assert ids[0] == "obj_0000"
        assert ids[1] == "obj_0001"
        assert ids[4] == "obj_0004"

    def test_generate_instance_ids_custom_prefix(self):
        """Test ID generation with custom prefix."""
        ids = generate_instance_ids(3, prefix="det")

        assert ids[0] == "det_0000"
        assert ids[1] == "det_0001"
        assert ids[2] == "det_0002"

    def test_generate_instance_ids_zero(self):
        """Test generating zero IDs."""
        ids = generate_instance_ids(0)

        assert ids == []

    def test_generate_instance_ids_deterministic(self):
        """Test ID generation is deterministic."""
        ids1 = generate_instance_ids(5)
        ids2 = generate_instance_ids(5)

        assert ids1 == ids2

    def test_ensure_instance_ids_basic(self):
        """Test ensure_instance_ids generates IDs for instances."""
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1),
        ]

        ids = ensure_instance_ids(instances)

        assert len(ids) == 2
        assert ids[0] == "inst_0000"
        assert ids[1] == "inst_0001"

    def test_ensure_instance_ids_empty(self):
        """Test ensure_instance_ids with empty list."""
        ids = ensure_instance_ids([])

        assert ids == []

    def test_align_instance_ids_matching(self):
        """Test align_instance_ids detects matching IDs."""
        instances1 = [Instance(bbox=(10, 20, 100, 150), score=0.9, label=0)]
        instances2 = [Instance(mask={"size": [100, 100], "counts": "test"}, score=0.9, label=0)]

        dets = Detections(instances=instances1, instance_ids=["obj_0000"])
        masks = Masks(instances=instances2, instance_ids=["obj_0000"])

        aligned = align_instance_ids([dets, masks])

        assert aligned is True

    def test_align_instance_ids_mismatched(self):
        """Test align_instance_ids detects mismatched IDs."""
        instances1 = [Instance(bbox=(10, 20, 100, 150), score=0.9, label=0)]
        instances2 = [Instance(mask={"size": [100, 100], "counts": "test"}, score=0.9, label=0)]

        dets = Detections(instances=instances1, instance_ids=["obj_0000"])
        masks = Masks(instances=instances2, instance_ids=["obj_0001"])

        aligned = align_instance_ids([dets, masks])

        assert aligned is False

    def test_align_instance_ids_different_lengths(self):
        """Test align_instance_ids detects different lengths."""
        instances1 = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1),
        ]
        instances2 = [Instance(mask={"size": [100, 100], "counts": "test"}, score=0.9, label=0)]

        dets = Detections(instances=instances1)
        masks = Masks(instances=instances2)

        aligned = align_instance_ids([dets, masks])

        assert aligned is False

    def test_align_instance_ids_different_order(self):
        """Test align_instance_ids detects different order."""
        instances1 = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1),
        ]
        instances2 = [
            Instance(mask={"size": [100, 100], "counts": "test1"}, score=0.9, label=0),
            Instance(mask={"size": [100, 100], "counts": "test2"}, score=0.85, label=1),
        ]

        dets = Detections(instances=instances1, instance_ids=["obj_0000", "obj_0001"])
        masks = Masks(instances=instances2, instance_ids=["obj_0001", "obj_0000"])

        aligned = align_instance_ids([dets, masks])

        assert aligned is False

    def test_align_instance_ids_empty_list(self):
        """Test align_instance_ids with empty list."""
        aligned = align_instance_ids([])

        assert aligned is True

    def test_align_instance_ids_single_artifact(self):
        """Test align_instance_ids with single artifact."""
        instances = [Instance(bbox=(10, 20, 100, 150), score=0.9, label=0)]
        dets = Detections(instances=instances)

        aligned = align_instance_ids([dets])

        assert aligned is True


# =============================================================================
# Edge Cases & Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_vision_result_to_detections_with_metadata(self):
        """Test metadata is preserved in conversions."""
        instances = [Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat")]
        result = VisionResult(instances=instances, meta={"model": "detr", "threshold": 0.5})

        dets = vision_result_to_detections(result)

        assert dets.meta["model"] == "detr"
        assert dets.meta["threshold"] == 0.5

    def test_vision_result_to_masks_with_metadata(self):
        """Test metadata is preserved for masks."""
        rle_mask = {"size": [100, 100], "counts": "test"}
        instances = [Instance(mask=rle_mask, score=0.9, label=0, label_name="cat")]
        result = VisionResult(instances=instances, meta={"model": "sam", "device": "cuda"})

        masks = vision_result_to_masks(result)

        assert masks.meta["model"] == "sam"
        assert masks.meta["device"] == "cuda"

    def test_detections_preserves_instance_order(self):
        """Test instance order is preserved in conversions."""
        instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.5, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.9, label=1, label_name="dog"),
            Instance(bbox=(300, 100, 400, 300), score=0.7, label=2, label_name="bird"),
        ]
        result = VisionResult(instances=instances)

        dets = vision_result_to_detections(result)
        restored = detections_to_vision_result(dets)

        assert restored.instances[0].label_name == "cat"
        assert restored.instances[1].label_name == "dog"
        assert restored.instances[2].label_name == "bird"

    def test_masks_preserves_mask_data(self):
        """Test mask data integrity in conversions."""
        rle_mask = {"size": [256, 256], "counts": "RLE_DATA_HERE_VERY_LONG_STRING"}
        instances = [Instance(mask=rle_mask, score=0.95, label=0, label_name="person")]
        result = VisionResult(instances=instances)

        masks = vision_result_to_masks(result)
        restored = masks_to_vision_result(masks)

        assert restored.instances[0].mask == rle_mask
        assert restored.instances[0].mask["counts"] == "RLE_DATA_HERE_VERY_LONG_STRING"


# =============================================================================
# Integration Tests
# =============================================================================


class TestConverterIntegration:
    """Integration tests with real workflows."""

    def test_detection_segmentation_pipeline(self):
        """Test detection → segmentation pipeline with ID alignment."""
        # Detection stage
        det_instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
        ]
        det_result = VisionResult(instances=det_instances)
        dets = vision_result_to_detections(det_result)

        # Segmentation stage (uses same IDs)
        rle1 = {"size": [100, 100], "counts": "mask1"}
        rle2 = {"size": [100, 100], "counts": "mask2"}
        seg_instances = [
            Instance(mask=rle1, score=0.9, label=0, label_name="cat"),
            Instance(mask=rle2, score=0.85, label=1, label_name="dog"),
        ]
        masks = Masks(instances=seg_instances, instance_ids=dets.instance_ids)

        # Verify alignment
        assert align_instance_ids([dets, masks])

    def test_vlm_grounding_fusion(self):
        """Test VLM entity + GroundingDINO fusion workflow."""
        # VLM detects entities (semantic)
        vlm_result = VisionResult(instances=[], entities=[Entity("cat", 0.95), Entity("dog", 0.87)])
        vlm_dets = vision_result_to_detections(vlm_result)

        # GroundingDINO provides spatial data
        spatial_instances = [
            Instance(bbox=(10, 20, 100, 150), score=0.9, label=0, label_name="cat"),
            Instance(bbox=(200, 50, 300, 200), score=0.85, label=1, label_name="dog"),
        ]
        spatial_result = VisionResult(instances=spatial_instances)
        spatial_dets = vision_result_to_detections(spatial_result)

        # Promote entities to instances
        promoted = vlm_dets.promote_entities(spatial_dets, match_strategy="label_fuzzy")

        assert len(promoted.instances) == 2
        assert promoted.instances[0].label_name == "cat"
        assert promoted.instances[0].bbox is not None

    def test_multiple_conversions_preserve_data(self):
        """Test multiple conversions don't lose data."""
        # Original data
        instances = [
            Instance(
                bbox=(10, 20, 100, 150),
                mask={"size": [100, 100], "counts": "rle"},
                score=0.95,
                label=0,
                label_name="cat",
                area=13500,
            )
        ]
        entities = [Entity("dog", 0.87, attributes={"color": "brown"})]
        original = VisionResult(instances=instances, entities=entities, meta={"model": "test"})

        # Convert multiple times
        dets1 = vision_result_to_detections(original)
        result1 = detections_to_vision_result(dets1)
        dets2 = vision_result_to_detections(result1)
        result2 = detections_to_vision_result(dets2)

        # Verify data preserved
        assert result2.instances[0].bbox == original.instances[0].bbox
        assert result2.instances[0].mask == original.instances[0].mask
        assert result2.instances[0].area == original.instances[0].area
        assert result2.entities[0].label == original.entities[0].label
        assert result2.entities[0].attributes["color"] == "brown"
