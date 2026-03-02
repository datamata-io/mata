"""Unit tests for Task 5.6: Fusion Nodes.

Tests cover:
- Fuse node bundling artifacts into MultiResult
- Merge node aligning instances by instance_id
- Provenance collection and metrics aggregation
- Channel mapping and missing artifact handling
- Instance merging with masks and keypoints
- Error handling and edge cases
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.artifacts.keypoints import Keypoints
from mata.core.artifacts.masks import Masks
from mata.core.artifacts.result import MultiResult
from mata.core.graph.context import ExecutionContext
from mata.core.types import Instance, VisionResult
from mata.nodes.fuse import Fuse
from mata.nodes.merge import Merge

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image():
    """Create a sample Image artifact."""
    import PIL.Image

    pil_img = PIL.Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    return Image.from_pil(pil_img, source_path="test.jpg")


@pytest.fixture
def sample_detections():
    """Create sample Detections artifact with 3 instances."""
    instances = [
        Instance(
            bbox=(10, 20, 100, 120),
            score=0.95,
            label=0,
            label_name="person",
        ),
        Instance(
            bbox=(150, 30, 200, 80),
            score=0.87,
            label=1,
            label_name="car",
        ),
        Instance(
            bbox=(50, 150, 120, 200),
            score=0.75,
            label=0,
            label_name="person",
        ),
    ]
    instance_ids = ["det_001", "det_002", "det_003"]

    return Detections(instances=instances, instance_ids=instance_ids, meta={"model": "detr", "threshold": 0.5})


@pytest.fixture
def sample_masks():
    """Create sample Masks artifact aligned with detections."""
    # Create simple RLE masks for the instances
    instances = [
        Instance(
            score=0.95,
            label=0,
            mask={"size": [224, 224], "counts": b"dummy_rle_1"},
            bbox=(10, 20, 100, 120),
        ),
        Instance(
            score=0.87,
            label=1,
            mask={"size": [224, 224], "counts": b"dummy_rle_2"},
            bbox=(150, 30, 200, 80),
        ),
        Instance(
            score=0.75,
            label=0,
            mask={"size": [224, 224], "counts": b"dummy_rle_3"},
            bbox=(50, 150, 120, 200),
        ),
    ]
    instance_ids = ["det_001", "det_002", "det_003"]

    return Masks(instances=instances, instance_ids=instance_ids, meta={"model": "sam", "mode": "prompt_boxes"})


@pytest.fixture
def sample_keypoints():
    """Create sample Keypoints artifact for first 2 instances only."""
    # COCO-style 17 keypoints with (x, y, score)
    kp1 = np.random.rand(17, 3)  # Random keypoints with scores in [0, 1]
    kp1[:, :2] *= 100  # Scale x, y to [0, 100] pixel range

    kp2 = np.random.rand(17, 3)  # Random keypoints with scores in [0, 1]
    kp2[:, :2] *= 100  # Scale x, y to [0, 100] pixel range

    keypoints_list = [kp1, kp2]
    instance_ids = ["det_001", "det_002"]  # Only covers first 2 detections

    return Keypoints(keypoints=keypoints_list, instance_ids=instance_ids, meta={"model": "hrnet", "skeleton": "coco"})


@pytest.fixture
def mock_context():
    """Create mock execution context."""
    context = MagicMock(spec=ExecutionContext)
    context.providers = {}
    context.device = "cpu"
    context.get_metrics.return_value = {
        "detect_node": {"latency_ms": 45.2, "memory_mb": 512},
        "segment_node": {"latency_ms": 120.5, "memory_mb": 1024},
    }

    # Configure retrieve to raise KeyError for non-existent artifacts
    # This simulates the real ExecutionContext behavior
    stored_artifacts = {}

    def mock_retrieve(name: str):
        if name not in stored_artifacts:
            raise KeyError(f"Artifact '{name}' not found in context")
        return stored_artifacts[name]

    context.retrieve = mock_retrieve
    context._stored_artifacts = stored_artifacts  # For test access if needed

    return context


# ---------------------------------------------------------------------------
# Test Fuse Node
# ---------------------------------------------------------------------------


class TestFuseNode:
    """Test cases for Fuse node."""

    def test_basic_fusion(self, mock_context, sample_image, sample_detections, sample_masks):
        """Test basic fusion with multiple artifacts."""
        fuse = Fuse(out="complete", image="img", detections="dets", masks="masks")

        result = fuse.run(mock_context, img=sample_image, dets=sample_detections, masks=sample_masks)

        assert "complete" in result
        multi_result = result["complete"]
        assert isinstance(multi_result, MultiResult)

        # Check channels
        assert "image" in multi_result.channels
        assert "detections" in multi_result.channels
        assert "masks" in multi_result.channels
        assert multi_result.channels["image"] == sample_image
        assert multi_result.channels["detections"] == sample_detections
        assert multi_result.channels["masks"] == sample_masks

        # Check dynamic access
        assert multi_result.image == sample_image
        assert multi_result.detections == sample_detections
        assert multi_result.masks == sample_masks

    def test_partial_artifacts(self, mock_context, sample_image, sample_detections):
        """Test fusion with missing artifacts (should continue without error)."""
        fuse = Fuse(detections="dets", masks="missing_masks", image="img")  # This artifact won't be provided

        result = fuse.run(
            mock_context,
            dets=sample_detections,
            img=sample_image,
            # Note: missing_masks not provided
        )

        multi_result = result["final"]  # Default output name

        # Should have available artifacts but not missing ones
        assert "detections" in multi_result.channels
        assert "image" in multi_result.channels
        assert "masks" not in multi_result.channels

        assert multi_result.detections == sample_detections
        assert multi_result.image == sample_image

    def test_provenance_collection(self, mock_context, sample_detections):
        """Test that provenance is collected properly."""
        fuse = Fuse(detections="dets")

        result = fuse.run(mock_context, dets=sample_detections)
        multi_result = result["final"]

        # Check provenance structure
        assert "provenance" in multi_result.__dict__
        provenance = multi_result.provenance

        assert "timestamp" in provenance
        assert "node_type" in provenance
        assert provenance["node_type"] == "Fuse"
        assert "framework_version" in provenance
        assert "device" in provenance
        assert provenance["device"] == "cpu"
        assert "channel_mapping" in provenance
        assert provenance["channel_mapping"]["detections"] == "dets"

    def test_metrics_collection(self, mock_context, sample_detections):
        """Test that metrics are aggregated and fusion timing recorded."""
        fuse = Fuse(detections="dets")

        result = fuse.run(mock_context, dets=sample_detections)
        multi_result = result["final"]

        # Check metrics are passed through
        assert "metrics" in multi_result.__dict__
        metrics = multi_result.metrics
        assert "detect_node" in metrics
        assert "segment_node" in metrics

        # Check fusion timing was recorded
        mock_context.record_metric.assert_called()
        call_args = [call.args for call in mock_context.record_metric.call_args_list]
        metric_names = [args[1] for args in call_args]
        assert "fusion_latency_ms" in metric_names
        assert "num_channels" in metric_names

    def test_custom_output_name(self, mock_context, sample_detections):
        """Test custom output artifact name."""
        fuse = Fuse(out="my_bundle", detections="dets")

        result = fuse.run(mock_context, dets=sample_detections)

        assert "my_bundle" in result
        assert "final" not in result

    def test_repr(self):
        """Test string representation."""
        fuse = Fuse(out="result", detections="dets", masks="masks")
        repr_str = repr(fuse)

        assert "Fuse" in repr_str
        assert "out='result'" in repr_str
        assert "detections=dets" in repr_str
        assert "masks=masks" in repr_str


# ---------------------------------------------------------------------------
# Test Merge Node
# ---------------------------------------------------------------------------


class TestMergeNode:
    """Test cases for Merge node."""

    def test_basic_merge_with_masks(self, mock_context, sample_detections, sample_masks):
        """Test basic merging of detections with masks."""
        merge = Merge(dets="dets", masks="masks", out="merged")

        result = merge.run(mock_context, detections=sample_detections, masks=sample_masks)

        assert "merged" in result
        vision_result = result["merged"]
        assert isinstance(vision_result, VisionResult)

        # Check that instances were merged with masks
        assert len(vision_result.instances) == 3
        for i, instance in enumerate(vision_result.instances):
            # Original detection data preserved
            assert instance.bbox == sample_detections.instances[i].bbox
            assert instance.score == sample_detections.instances[i].score
            assert instance.label == sample_detections.instances[i].label
            assert instance.label_name == sample_detections.instances[i].label_name

            # Mask data merged in
            assert instance.mask == sample_masks.instances[i].mask

    def test_merge_with_keypoints(self, mock_context, sample_detections, sample_keypoints):
        """Test merging detections with keypoints (partial coverage)."""
        merge = Merge(dets="dets", keypoints="keypoints", out="merged")

        result = merge.run(mock_context, detections=sample_detections, keypoints=sample_keypoints)

        vision_result = result["merged"]

        # Check keypoints merged for first 2 instances, not third
        assert len(vision_result.instances) == 3

        # First two should have keypoints
        assert np.array_equal(vision_result.instances[0].keypoints, sample_keypoints.keypoints[0])
        assert np.array_equal(vision_result.instances[1].keypoints, sample_keypoints.keypoints[1])

        # Third should not have keypoints (no matching instance_id)
        assert vision_result.instances[2].keypoints is None

    def test_full_merge_all_modalities(self, mock_context, sample_detections, sample_masks, sample_keypoints):
        """Test merging detections with both masks and keypoints."""
        merge = Merge(dets="dets", masks="masks", keypoints="keypoints")

        result = merge.run(mock_context, detections=sample_detections, masks=sample_masks, keypoints=sample_keypoints)

        vision_result = result["merged"]  # Default output name

        # Check all modalities merged appropriately
        assert len(vision_result.instances) == 3

        # First two instances should have all modalities
        for i in range(2):
            instance = vision_result.instances[i]
            assert instance.mask == sample_masks.instances[i].mask
            assert np.array_equal(instance.keypoints, sample_keypoints.keypoints[i])

        # Third instance should only have mask (no keypoints data)
        third_instance = vision_result.instances[2]
        assert third_instance.mask == sample_masks.instances[2].mask
        assert third_instance.keypoints is None

    def test_merge_missing_optional_artifacts(self, mock_context, sample_detections):
        """Test merge with only detections (no optional artifacts)."""
        merge = Merge(dets="dets", masks="masks", keypoints="keypoints")

        result = merge.run(
            mock_context, detections=sample_detections, masks=None, keypoints=None  # Optional artifacts not provided
        )

        vision_result = result["merged"]

        # Should preserve original detection instances unchanged
        assert len(vision_result.instances) == 3
        for i, instance in enumerate(vision_result.instances):
            assert instance.bbox == sample_detections.instances[i].bbox
            assert instance.score == sample_detections.instances[i].score
            assert instance.label == sample_detections.instances[i].label
            assert instance.label_name == sample_detections.instances[i].label_name
            assert instance.mask is None
            assert instance.keypoints is None

    def test_metadata_merging(self, mock_context, sample_detections, sample_masks, sample_keypoints):
        """Test that metadata from all artifacts is merged."""
        merge = Merge(dets="dets", masks="masks", keypoints="keypoints")

        result = merge.run(mock_context, detections=sample_detections, masks=sample_masks, keypoints=sample_keypoints)

        vision_result = result["merged"]
        meta = vision_result.meta

        # Should contain original detection meta
        assert meta["model"] == "detr"
        assert meta["threshold"] == 0.5

        # Should contain prefixed masks meta
        assert meta["masks_model"] == "sam"
        assert meta["masks_mode"] == "prompt_boxes"

        # Should contain prefixed keypoints meta
        assert meta["keypoints_model"] == "hrnet"
        assert meta["keypoints_skeleton"] == "coco"

    def test_metrics_recording(self, mock_context, sample_detections, sample_masks):
        """Test that merge metrics are recorded."""
        merge = Merge(dets="dets", masks="masks")

        merge.run(mock_context, detections=sample_detections, masks=sample_masks)

        # Check metrics were recorded
        mock_context.record_metric.assert_called()
        call_args = [call.args for call in mock_context.record_metric.call_args_list]
        metric_names = [args[1] for args in call_args]

        assert "merge_latency_ms" in metric_names
        assert "total_instances" in metric_names
        assert "masks_merged" in metric_names

    def test_custom_artifact_names(self, mock_context, sample_detections, sample_masks):
        """Test custom artifact source names."""
        merge = Merge(dets="filtered_dets", masks="sam_masks", out="final_result")

        # Test that it accepts the custom parameter names in constructor
        assert merge.dets_src == "filtered_dets"
        assert merge.masks_src == "sam_masks"
        assert merge.output_name == "final_result"

    def test_repr(self):
        """Test string representation."""
        merge = Merge(dets="my_dets", masks="my_masks", out="merged")
        repr_str = repr(merge)

        assert "Merge" in repr_str
        assert "dets=my_dets" in repr_str
        assert "masks=my_masks" in repr_str
        assert "out='merged'" in repr_str


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestFusionIntegration:
    """Integration tests for fusion nodes working together."""

    def test_merge_then_fuse_workflow(
        self, mock_context, sample_image, sample_detections, sample_masks, sample_keypoints
    ):
        """Test complete workflow: merge instances, then fuse into MultiResult."""
        # Step 1: Merge multi-modal data
        merge = Merge(dets="dets", masks="masks", keypoints="keypoints", out="merged_instances")

        merge_result = merge.run(
            mock_context, detections=sample_detections, masks=sample_masks, keypoints=sample_keypoints
        )

        # Step 2: Fuse everything into final bundle
        fuse = Fuse(out="complete_result", image="img", detections="merged_instances", raw_detections="raw_dets")

        fuse_result = fuse.run(
            mock_context,
            img=sample_image,
            merged_instances=merge_result["merged_instances"],
            raw_dets=sample_detections,
        )

        # Verify complete result
        multi_result = fuse_result["complete_result"]
        assert isinstance(multi_result, MultiResult)

        # Check channels
        assert "image" in multi_result.channels
        assert "detections" in multi_result.channels
        assert "raw_detections" in multi_result.channels

        # Check that merged instances have all modalities
        merged_vision_result = multi_result.detections
        first_instance = merged_vision_result.instances[0]
        assert first_instance.mask is not None
        assert first_instance.keypoints is not None

        # Check provenance chain is preserved
        assert multi_result.provenance["node_type"] == "Fuse"

    def test_empty_detection_handling(self, mock_context):
        """Test fusion/merge with empty detections."""
        empty_detections = Detections(instances=[], instance_ids=[], meta={"model": "detr", "no_detections": True})

        # Test merge with empty detections
        merge = Merge()
        merge_result = merge.run(mock_context, detections=empty_detections)

        vision_result = merge_result["merged"]
        assert len(vision_result.instances) == 0

        # Test fuse with empty result
        fuse = Fuse(detections="dets")
        fuse_result = fuse.run(mock_context, dets=empty_detections)

        multi_result = fuse_result["final"]
        assert len(multi_result.detections.instances) == 0
        assert multi_result.detections.meta["no_detections"] is True


if __name__ == "__main__":
    pytest.main([__file__])
